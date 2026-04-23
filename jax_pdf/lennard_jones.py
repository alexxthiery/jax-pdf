"""Lennard-Jones cluster with harmonic trap."""

import jax.numpy as jnp
from flax import struct
from jax import Array


@struct.dataclass
class LennardJones:
    """Lennard-Jones cluster with harmonic trap (Boltzmann distribution).

    The energy has two terms:

        U(x) = U_LJ(x) + U_trap(x)

    LJ 12-6 pair potential summed over unique pairs (i < j):
        U_LJ = sum_{i<j} epsilon * [(rm/r_ij)^12 - 2*(rm/r_ij)^6]

    The minimum of each pair interaction is -epsilon at r = rm.

    Harmonic trap (controlled by ``trap_mode``):

        'individual' (default):
            U_trap = (trap_scale/2) * sum_i ||x_i||^2
            Penalizes each particle's distance from the origin. This
            affects both the center of mass and the internal spread.
            Used in ML benchmarks (e.g., FAB, normalizing flows).

        'com':
            U_trap = (trap_scale/2) * N * ||COM||^2
            Penalizes only the center of mass position. The pair
            potential is unaffected, and the distribution factorizes
            as p(x) = p_LJ(internal) * p_trap(COM). This means the
            statistics of relative particle positions are identical
            to those of the pure potential with COM fixed to zero.
            Standard in physics/chemistry simulations.

    Set trap_scale=0 to disable (distribution becomes improper with
    spatial_dim flat directions from translational invariance).

    The log-density is log p(x) = -beta * U(x) + const.

    Standard configurations: LJ13 (n_particles=13, dim=39) and
    LJ55 (n_particles=55, dim=165).

    Attributes:
        n_particles: Number of particles. Default: 13.
        spatial_dim: Spatial dimension per particle. Default: 3.
        epsilon: LJ well depth. Default: 1.0.
        rm: LJ equilibrium distance (potential minimum). Default: 1.0.
        trap_scale: Harmonic trap strength. Default: 1.0.
        trap_mode: 'individual' or 'com'. Default: 'individual'.
        beta: Inverse temperature. Default: 1.0.
    """

    n_particles: int = 13
    """Number of particles."""

    spatial_dim: int = 3
    """Spatial dimension per particle (2D or 3D)."""

    epsilon: float = 1.0
    """LJ well depth."""

    rm: float = 1.0
    """LJ equilibrium distance (potential minimum at r = rm)."""

    trap_scale: float = 1.0
    """Harmonic trap strength. Set to 0.0 to disable."""

    trap_mode: str = 'individual'
    """Trap mode: 'individual' penalizes each particle's distance from the
    origin; 'com' penalizes only the center of mass."""

    beta: float = 1.0
    """Inverse temperature."""

    def __post_init__(self):
        if self.n_particles < 2:
            raise ValueError(
                f"n_particles must be >= 2, got {self.n_particles}"
            )
        if self.spatial_dim < 1:
            raise ValueError(
                f"spatial_dim must be >= 1, got {self.spatial_dim}"
            )
        if self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be positive, got {self.epsilon}"
            )
        if self.rm <= 0:
            raise ValueError(f"rm must be positive, got {self.rm}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.trap_mode not in ('individual', 'com'):
            raise ValueError(
                f"trap_mode must be 'individual' or 'com', got '{self.trap_mode}'"
            )

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim

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
                f"LennardJones expected x.shape[-2:] == ({n}, {d}), "
                f"got shape {tuple(x.shape)}."
            )
        pos = x

        # Pairwise distances, unique pairs i < j
        diff = pos[..., :, None, :] - pos[..., None, :, :]
        r = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
        i, j = jnp.triu_indices(n, k=1)
        r_pairs = r[..., i, j]

        # LJ 12-6 potential
        inv_r = self.rm / r_pairs
        u_lj = jnp.sum(
            self.epsilon * (inv_r**12 - 2.0 * inv_r**6), axis=-1
        )

        # Harmonic trap
        if self.trap_mode == 'individual':
            u_trap = 0.5 * self.trap_scale * jnp.sum(pos**2, axis=(-2, -1))
        else:  # 'com'
            com = pos.mean(axis=-2)  # (..., d)
            u_trap = 0.5 * self.trap_scale * n * jnp.sum(com**2, axis=-1)

        return -self.beta * (u_lj + u_trap)

    def log_normalization(self) -> Array:
        """Log normalizing constant (intractable).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "LennardJones normalizing constant is intractable."
        )
