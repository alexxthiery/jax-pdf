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
        n_neighbours: If set, the three-body sum runs over each particle's
            ``n_neighbours`` nearest neighbours instead of all pairs, which turns
            its ``O(N**3)`` cost into ``O(N * n_neighbours**2)``. It is **exact**
            while every particle within the cutoff (1.8 sigma = 4.31 A) is in the
            list; ``max_neighbours_within_cutoff`` measures that on a
            configuration, and the choice is the caller's. Needed above a few
            hundred particles: the dense tensor asks XLA for 4.3 GB at N=512 with
            eight configurations. The two-body sum stays dense (``O(N**2)``).
            Default None (dense).
    """

    n_particles: int = struct.field(pytree_node=False)
    box_length: float
    beta: float = 1.0
    min_distance: float = 0.0
    spatial_dim: int = struct.field(pytree_node=False, default=3)
    linearize_below: Optional[float] = None
    n_neighbours: Optional[int] = None

    def __post_init__(self):
        if (self.linearize_below is not None and is_concrete(self.linearize_below)
                and not self.linearize_below > 0):
            raise ValueError(
                f"linearize_below must be positive or None, got {self.linearize_below}"
            )
        if self.n_neighbours is not None and not 2 <= self.n_neighbours <= self.n_particles - 1:
            # Two neighbours are the smallest triplet; N-1 is all of them.
            raise ValueError(
                f"n_neighbours must be in [2, n_particles - 1] = "
                f"[2, {self.n_particles - 1}] or None, got {self.n_neighbours}"
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

    def _angular_sum(self, dr: Array, valid: Optional[Array] = None) -> Array:
        """Three-body angular sum over distinct pairs of each centre's neighbours.

        Args:
            dr: Reduced (divided by sigma) minimum-image vectors from each centre
                to its candidate neighbours, shape ``(..., N, M, 3)``. ``M`` is
                ``N`` for the dense sum and ``n_neighbours`` for the list.
            valid: Optional ``(..., N, M)`` mask of candidates that are real
                neighbours; the cutoff is applied on top of it. The dense sum
                passes it to drop the self entries; the list has none.

        Returns:
            The energy in kcal/mol, shape ``dr.shape[:-3]``.
        """
        # +eps inside sqrt: norm has an infinite gradient at 0, so coincident
        # particles NaN the *gradient* while the forward stays finite (hides
        # until backprop). Negligible to the forward. Cf. nflojax GNN gotcha.
        norms = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)  # (..., N, M)
        keep = norms < MW_REDUCED_CUTOFF
        if valid is not None:
            keep = keep & valid
        norms_safe = jnp.where(keep, norms, 1e20)
        ne = jnp.where(                                  # radial cutoff factor
            keep, jnp.exp(MW_GAMMA / (norms_safe - MW_REDUCED_CUTOFF)), 0.0
        )
        # cos of angle j-i-k for centre i: dr[i,j].dr[i,k] / (|dr[i,j]| |dr[i,k]|)
        dots = jnp.einsum("...jm,...km->...jk", dr, dr)
        normprod = norms_safe[..., :, :, None] * norms_safe[..., :, None, :]
        cos_ijk = dots / normprod
        ang = MW_LAMBDA * MW_EPSILON * (MW_COS - cos_ijk) ** 2
        ang = ang * ne[..., :, :, None] * ne[..., :, None, :]
        # sum over distinct neighbour pairs j<k for each centre i, then over i.
        return jnp.sum(jnp.triu(ang, k=1), axis=(-3, -2, -1))

    def _three_body(self, dr: Array) -> Array:
        """Dense three-body sum: every pair of neighbours of every centre.

        ``dr`` is the reduced ``(..., N, N, 3)`` minimum-image difference, whose
        diagonal (the self entries) is masked out here. Cost and memory are
        ``O(N**3)``: 4.3 GB of intermediates for eight configurations at N=512,
        which is why ``n_neighbours`` exists.
        """
        eye = jnp.eye(self.n_particles, dtype=bool)
        return self._angular_sum(dr, ~eye)

    def _three_body_neighbours(self, dr: Array) -> Array:
        """Three-body sum over each centre's ``n_neighbours`` nearest neighbours.

        Exact, not an approximation, whenever every particle inside the cutoff is
        in the list (``max_neighbours_within_cutoff``): the dense sum gives a
        triplet with a neighbour beyond the cutoff exactly zero weight, and the
        radial factor is applied here too, so a listed neighbour beyond the
        cutoff also contributes nothing. Cost and memory are ``O(N k**2)``.

        The selection is by squared distance, so it is exact under the minimum
        image and its gradient is the dense gradient: the excluded neighbours sit
        where the energy is identically flat.
        """
        k = int(self.n_neighbours)
        r2 = jnp.sum(dr**2, axis=-1)                       # (..., N, N), reduced
        # top_k of the negated distances = the k nearest; +inf on the diagonal
        # keeps a centre out of its own list. Written as a where, not as
        # ``r2 + eye * inf``, which is NaN off the diagonal (0 * inf).
        eye = jnp.eye(self.n_particles, dtype=bool)
        _, idx = jax.lax.top_k(-jnp.where(eye, jnp.inf, r2), k)   # (..., N, k)
        sel = jnp.take_along_axis(dr, idx[..., None], axis=-2)   # (..., N, k, 3)
        return self._angular_sum(sel)

    def max_neighbours_within_cutoff(self, x: Array) -> Array:
        """Largest number of particles within the three-body cutoff of any centre.

        The condition under which ``n_neighbours`` is exact: it must be at least
        this number, for every configuration the target is evaluated on. Cheap
        (``O(N**2)``) and meant to be measured once on a sample of the
        configurations of interest, then used to choose ``n_neighbours`` with a
        margin. Returns shape ``x.shape[:-2]`` (integers).
        """
        diff = self._min_image_diff(x)
        r2 = jnp.sum(diff**2, axis=-1)
        cutoff = MW_REDUCED_CUTOFF * MW_SIGMA               # 4.3065 Angstrom
        inside = (r2 < cutoff**2) & (~jnp.eye(self.n_particles, dtype=bool))
        return jnp.max(jnp.sum(inside, axis=-1), axis=-1)

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
        dr = diff / MW_SIGMA
        three = self._three_body(dr) if self.n_neighbours is None \
            else self._three_body_neighbours(dr)
        u = self._two_body(diff) + three
        return -self.beta * u

    def log_normalization(self) -> Array:
        raise NotImplementedError("MonatomicWater normalizing constant is intractable.")
