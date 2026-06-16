"""Periodic Lennard-Jones fluid/solid with minimum-image boundary conditions.

Companion to the free-cluster :class:`LennardJones`. This target lives in a
cubic periodic box and is the regime used for atomic-solid Boltzmann
generators (Wirnsberger et al. 2022; bgmat). The pairwise potential and the
soft-core / cutoff / shift conventions match ``bgmat.systems.lennard_jones``
so that downstream flows can be compared against that reference.
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array


@struct.dataclass
class PeriodicLennardJones:
    """Lennard-Jones in a cubic periodic box (Boltzmann distribution).

    The energy is a sum over unique minimum-image pairs of a soft-core,
    radially truncated and (optionally) shifted Lennard-Jones potential:

        g(r) = (r / sigma)^6 + 0.5 * (1 - lambda_lj)^2
        u(r) = 4 * lambda_lj * epsilon * [1/g^2 - 1/g]            (r <= cutoff)
        u(r) = 0                                                  (r >  cutoff)

    For ``lambda_lj=1`` this is the standard 12-6 potential
    ``4*epsilon*[(sigma/r)^12 - (sigma/r)^6]``; ``lambda_lj`` in (0, 1)
    softens the r -> 0 singularity for training stability. With
    ``shift_energy=True`` the potential is shifted to vanish at the cutoff.

    Distances use the minimum-image convention; this is exact only when
    ``cutoff <= box_length / 2``. Construction does not enforce that (some
    callers deliberately use a larger cutoff), but a warning-worthy regime.

    The log-density is ``log p(x) = -beta * U(x) + const``.

    Attributes:
        n_particles: Number of particles. Default: 256.
        spatial_dim: Spatial dimension per particle. Default: 3.
        box_length: Cubic box side length (same on every axis).
        epsilon: LJ well depth. Default: 1.0.
        sigma: LJ length scale. Default: 1.0.
        cutoff: Radial cutoff; pairs beyond it do not interact. Default: 2.7.
        beta: Inverse temperature. Default: 0.5 (bgmat LJ-solid default).
        lambda_lj: Soft-core parameter in (0, 1]. Default: 1.0 (hard LJ).
        min_distance: Squared distance is clipped to ``min_distance**2``,
            making the potential constant below it. Default: 0.0.
        linearize_below: If set, the potential is linear in r below this
            value (training-stability option). Default: None.
        shift_energy: Shift so u(cutoff) == 0. Default: True.
    """

    n_particles: int = 256
    spatial_dim: int = 3
    box_length: float = 5.848035476425733  # (256 / 1.28) ** (1/3)
    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float = 2.7
    beta: float = 0.5
    lambda_lj: float = 1.0
    min_distance: float = 0.0
    linearize_below: float = None
    shift_energy: bool = True

    def __post_init__(self):
        if self.n_particles < 2:
            raise ValueError(f"n_particles must be >= 2, got {self.n_particles}")
        if self.spatial_dim < 1:
            raise ValueError(f"spatial_dim must be >= 1, got {self.spatial_dim}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.box_length <= 0:
            raise ValueError(f"box_length must be positive, got {self.box_length}")
        if self.cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {self.cutoff}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if not (0.0 < self.lambda_lj <= 1.0):
            raise ValueError(f"lambda_lj must be in (0, 1], got {self.lambda_lj}")

    @classmethod
    def from_density(
        cls, n_particles: int, density: float, spatial_dim: int = 3, **kwargs
    ) -> "PeriodicLennardJones":
        """Build with the cubic box length implied by ``(N / density)^(1/d)``."""
        if density <= 0:
            raise ValueError(f"density must be positive, got {density}")
        box_length = float((n_particles / density) ** (1.0 / spatial_dim))
        return cls(
            n_particles=n_particles,
            spatial_dim=spatial_dim,
            box_length=box_length,
            **kwargs,
        )

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim

    def _soft_core(self, r2: Array) -> Array:
        """Unshifted, untruncated soft-core LJ from squared distance."""
        g = r2**3 / self.sigma**6 + 0.5 * (1.0 - self.lambda_lj) ** 2
        ginv = 1.0 / g
        return 4.0 * self.lambda_lj * self.epsilon * ginv * (ginv - 1.0)

    def _pair_potential(self, r2: Array) -> Array:
        """Clipped, optionally linearized, truncated-and-shifted pair potential."""
        r2 = jnp.clip(r2, self.min_distance**2)
        shift = self._soft_core(jnp.asarray(self.cutoff**2)) if self.shift_energy else 0.0
        if self.linearize_below is None:
            u = self._soft_core(r2)
        else:
            e0, g0 = jax.value_and_grad(lambda r: self._soft_core(r**2))(
                jnp.asarray(self.linearize_below)
            )
            u = jnp.where(
                r2 < self.linearize_below**2,
                e0 + (jnp.sqrt(r2) - self.linearize_below) * g0,
                self._soft_core(r2),
            )
        return jnp.where(r2 <= self.cutoff**2, u - shift, 0.0)

    def __call__(self, x: Array) -> Array:
        """Evaluate unnormalized log-density.

        Args:
            x: Particle positions of shape (..., n_particles, spatial_dim).

        Returns:
            Log-density of shape (...).
        """
        n, d = self.n_particles, self.spatial_dim
        if x.shape[-2:] != (n, d):
            raise ValueError(
                f"PeriodicLennardJones expected x.shape[-2:] == ({n}, {d}), "
                f"got shape {tuple(x.shape)}."
            )

        diff = x[..., :, None, :] - x[..., None, :, :]          # (..., n, n, d)
        diff = diff - self.box_length * jnp.round(diff / self.box_length)
        r2 = jnp.sum(diff**2, axis=-1)                           # (..., n, n)
        # Avoid the zero self-distance feeding the potential; the diagonal is
        # dropped by the strict upper triangle below.
        r2 = r2 + jnp.eye(n, dtype=r2.dtype)

        u = self._pair_potential(r2)
        energy = jnp.sum(jnp.triu(u, k=1), axis=(-2, -1))
        return -self.beta * energy

    def log_normalization(self) -> Array:
        """Log normalizing constant (intractable)."""
        raise NotImplementedError(
            "PeriodicLennardJones normalizing constant is intractable."
        )
