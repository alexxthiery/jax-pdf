"""Double-well potential over 4 particles in 2D."""

import jax.numpy as jnp
from flax import struct
from jax import Array


@struct.dataclass
class DW4:
    """Double-well potential over 4 particles in 2D (Boltzmann distribution).

    The energy has two terms:

        U(x) = U_DW(x) + U_trap(x)

    Quartic double-well pair potential summed over all 6 unique pairs (i < j):
        U_DW(x) = sum_{i<j} [a*(r_ij - r0)^4 + b*(r_ij - r0)^2 + c]

    where r_ij is the Euclidean distance between particles i and j.

    Harmonic trap centered at the origin:
        U_trap(x) = (trap_scale/2) * sum_i ||x_i||^2

    This breaks translational invariance and makes the distribution
    proper (normalizable). It decomposes as a penalty on both the
    center-of-mass position and the internal spread:
        sum_i ||x_i||^2 = sum_i ||x_i - COM||^2 + N*||COM||^2

    Set trap_scale=0 to disable (distribution becomes improper with
    2 flat directions from translational invariance).

    The log-density is log p(x) = -beta * U(x) + const.

    With the default parameters (a=0.9, b=-4, r0=4), each pair potential
    has two minima at r ~= 2.51 and r ~= 5.49, creating a multimodal
    distribution over particle configurations.

    Attributes:
        a: Quartic coefficient. Default: 0.9.
        b: Quadratic coefficient (negative creates the double well). Default: -4.0.
        c: Constant offset in pair potential. Default: 0.0.
        r0: Distance offset (center of the quartic). Default: 4.0.
        trap_scale: Harmonic trap strength. Default: 1.0.
        beta: Inverse temperature. Default: 1.0.
    """

    a: float = 0.9
    """Quartic coefficient in the pair potential."""

    b: float = -4.0
    """Quadratic coefficient (negative = double well)."""

    c: float = 0.0
    """Constant offset in the pair potential."""

    r0: float = 4.0
    """Distance offset (center of the quartic)."""

    trap_scale: float = 1.0
    """Harmonic trap strength. Set to 0.0 to disable."""

    beta: float = 1.0
    """Inverse temperature."""

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")

    @property
    def dim(self) -> int:
        return 8

    def __call__(self, x: Array) -> Array:
        """Evaluate unnormalized log-density.

        Args:
            x: Particle positions of shape (..., 8). Interpreted as 4
                particles in 2D: x[..., 0:2] is particle 0, etc.

        Returns:
            Log-density of shape (...).
        """
        pos = x.reshape(x.shape[:-1] + (4, 2))

        # Pairwise distances, unique pairs i < j
        diff = pos[..., :, None, :] - pos[..., None, :, :]
        r = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
        i, j = jnp.triu_indices(4, k=1)
        r_pairs = r[..., i, j]
        d = r_pairs - self.r0
        u_dw = jnp.sum(self.a * d**4 + self.b * d**2 + self.c, axis=-1)

        # Harmonic trap centered at the origin
        u_trap = 0.5 * self.trap_scale * jnp.sum(pos**2, axis=(-2, -1))

        return -self.beta * (u_dw + u_trap)

    def log_normalization(self) -> Array:
        """Log normalizing constant (intractable).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "DW4 normalizing constant is intractable."
        )
