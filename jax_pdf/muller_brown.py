"""Muller-Brown potential energy surface as a 2D distribution."""

import jax.numpy as jnp
from flax import struct
from jax import Array

# Standard Muller-Brown coefficients (Muller & Brown, 1979).
_A = jnp.array([-200.0, -100.0, -170.0, 15.0])
_a = jnp.array([-1.0, -1.0, -6.5, 0.7])
_b = jnp.array([0.0, 0.0, 11.0, 0.6])
_c = jnp.array([-10.0, -10.0, -6.5, 0.7])
_x0 = jnp.array([1.0, 0.0, -0.5, -1.0])
_y0 = jnp.array([0.0, 0.5, 1.5, 1.0])


@struct.dataclass
class MullerBrown:
    """Muller-Brown potential energy surface as a 2D log-density.

    The log-density is the negated potential energy:
        log p(x, y) = -beta * U(x, y)

    where U is the standard Muller-Brown potential:
        U(x, y) = sum_{k=1}^{4} A_k * exp(
            a_k * (x - x0_k)^2
            + b_k * (x - x0_k) * (y - y0_k)
            + c_k * (y - y0_k)^2
        )

    The potential has three local minima connected by two saddle points,
    creating a multimodal landscape. Samplers must cross energy barriers
    to transition between modes, making this a challenging test for
    mixing and mode-hopping.

    The three minima are approximately at:
        (-0.558, 1.442), (0.624, 0.028), (-0.050, 0.467)

    Attributes:
        beta: Inverse temperature. Higher values sharpen the modes and
            raise the barriers, making sampling harder. Default: 1.0.
    """

    beta: float = 1.0
    """Inverse temperature. Higher = sharper modes, harder sampling."""

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")

    @property
    def dim(self) -> int:
        return 2

    def __call__(self, x: Array) -> Array:
        """Evaluate unnormalized log-density.

        Args:
            x: Input point(s) of shape (..., 2).

        Returns:
            Unnormalized log-density of shape (...).
        """
        dx = x[..., 0, None] - _x0  # (..., 4)
        dy = x[..., 1, None] - _y0  # (..., 4)
        exponent = _a * dx**2 + _b * dx * dy + _c * dy**2
        potential = jnp.sum(_A * jnp.exp(exponent), axis=-1)
        return -self.beta * potential

    def log_normalization(self) -> Array:
        """Not available: normalizing constant is intractable.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "MullerBrown normalizing constant is intractable."
        )
