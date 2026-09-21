"""Monatomic (mW) water in a periodic box (Stillinger-Weber potential).

mW (Molinero & Moore 2009, arXiv:0809.2811) is a single-site water model: a
soft two-body term plus a **three-body angular term** that penalises deviation
from the tetrahedral angle 109.47 deg. Minimum-image PBC. The functional form
follows the Stillinger-Weber pair and angular potentials. Tests compare the
vectorized energy and its derivatives with independent scalar pair/triplet
sums and analytic configurations; no external reference package is required.
"""
import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_pdf._validation import check_event_shape, is_concrete

# Molinero-Moore mW parameters (kcal/mol, Angstrom).
MW_A = 7.049556277
MW_B = 0.6022245584
MW_GAMMA = 1.2
MW_EPSILON = 6.189
MW_SIGMA = 2.3925
MW_REDUCED_CUTOFF = 1.8
MW_LAMBDA = 23.15
MW_COS = math.cos(109.47 / 180.0 * math.pi)


@struct.dataclass
class MonatomicWater:
    """mW water Boltzmann target; ``log p(x) = -beta * U(x)``.

    Attributes:
        n_particles: Number of particles.
        box_length: Cubic box side (minimum-image PBC).
        beta: Inverse temperature (1/kT, in 1/(kcal/mol)).
        min_distance: Two-body squared distance is clipped to ``min_distance**2``
            (removes the r->0 singularity). Default 0.0 (exact). For reverse-KL
            training, where the flow may sample overlapping particles, set this
            > 0 to keep the two-body energy and its gradient finite; the
            three-body norm is floored separately by ``eps`` inside the sqrt.
        linearize_below: If set (Angstrom), the two-body potential below this
            distance is replaced by its tangent line in r,
            ``phi(r_lin) + (r - r_lin) * phi'(r_lin)``, after the ``min_distance``
            clip. This is bgmat's training softening (it uses 1.2 with
            ``min_distance=0.01``); it bounds the steep core that otherwise
            dominates reverse-KL gradients. The three-body term is unchanged.
            Default None (exact).
    """

    n_particles: int = struct.field(pytree_node=False)
    box_length: float
    beta: float = 1.0
    min_distance: float = 0.0
    spatial_dim: int = struct.field(pytree_node=False, default=3)
    linearize_below: Optional[float] = None

    def __post_init__(self):
        if (self.linearize_below is not None and is_concrete(self.linearize_below)
                and not self.linearize_below > 0):
            raise ValueError(
                f"linearize_below must be positive or None, got {self.linearize_below}"
            )

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim

    def _min_image_diff(self, x: Array) -> Array:
        # diff[..., i, j, :] = minimum_image(x_j - x_i)
        diff = x[..., None, :, :] - x[..., :, None, :]
        return diff - self.box_length * jnp.round(diff / self.box_length)

    @staticmethod
    def _pair(r2: Array) -> Array:
        """Unclipped two-body potential (kcal/mol) of the squared distance r2 (Angstrom^2)."""
        red = r2 / MW_SIGMA**2                          # reduced squared distance
        r = jnp.sqrt(red)
        mask = r < MW_REDUCED_CUTOFF
        r = jnp.where(mask, r, 2.0 * MW_REDUCED_CUTOFF)  # safe value (avoid NaN grad)
        term_1 = MW_A * MW_EPSILON * (MW_B / red**2 - 1.0)
        term_2 = jnp.where(mask, jnp.exp(1.0 / (r - MW_REDUCED_CUTOFF)), 0.0)
        return term_1 * term_2

    def _two_body(self, diff: Array) -> Array:
        n = self.n_particles
        r2 = jnp.sum(diff**2, axis=-1)                  # (..., N, N), physical
        r2 = r2 + jnp.eye(n, dtype=r2.dtype)            # diagonal dropped by triu
        r2 = jnp.clip(r2, self.min_distance**2)
        if self.linearize_below is None:
            u = self._pair(r2)
        else:
            # Same recipe as bgmat's PairwisePotentialEnergy._pairwise_potential:
            # value and slope in r (not r**2) at r_lin, applied after the clip.
            lin = jnp.asarray(self.linearize_below, dtype=r2.dtype)
            e0, g0 = jax.value_and_grad(lambda r: self._pair(r**2))(lin)
            u = jnp.where(r2 < lin**2, e0 + (jnp.sqrt(r2) - lin) * g0, self._pair(r2))
        return jnp.sum(jnp.triu(u, k=1), axis=(-2, -1))

    def _three_body(self, dr: Array) -> Array:
        # dr = minimum-image diff / sigma (reduced), shape (..., N, N, 3).
        n = self.n_particles
        # +eps inside sqrt: norm has an infinite gradient at 0, so coincident
        # particles NaN the *gradient* while the forward stays finite (hides
        # until backprop). Negligible to the forward. Cf. nflojax GNN gotcha.
        norms = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)  # (..., N, N)
        eye = jnp.eye(n, dtype=bool)
        keep = (norms < MW_REDUCED_CUTOFF) & (~eye)
        norms_safe = jnp.where(keep, norms, 1e20)
        ne = jnp.where(                                  # radial cutoff factor
            keep, jnp.exp(MW_GAMMA / (norms_safe - MW_REDUCED_CUTOFF)), 0.0
        )
        # cos of angle j-i-k for centre i: dr[i,j].dr[i,k] / (|dr[i,j]| |dr[i,k]|)
        dots = jnp.einsum("...ijm,...ikm->...ijk", dr, dr)
        normprod = norms_safe[..., :, :, None] * norms_safe[..., :, None, :]
        cos_ijk = dots / normprod
        ang = MW_LAMBDA * MW_EPSILON * (MW_COS - cos_ijk) ** 2
        ang = ang * ne[..., :, :, None] * ne[..., :, None, :]
        # sum over distinct neighbour pairs j<k for each centre i, then over i.
        return jnp.sum(jnp.triu(ang, k=1), axis=(-3, -2, -1))

    def __call__(self, x: Array) -> Array:
        """Evaluate unnormalized log-density.

        Args:
            x: Particle positions of shape (..., n_particles, spatial_dim).

        Returns:
            Log-density of shape (...).

        Raises:
            ValueError: If the input does not have trailing shape
                (n_particles, spatial_dim).
        """
        n, d = self.n_particles, self.spatial_dim
        check_event_shape(x, (n, d))
        diff = self._min_image_diff(x)
        u = self._two_body(diff) + self._three_body(diff / MW_SIGMA)
        return -self.beta * u

    def log_normalization(self) -> Array:
        raise NotImplementedError("MonatomicWater normalizing constant is intractable.")
