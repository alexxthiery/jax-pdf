"""Banana-shaped 2D distribution (Rosenbrock-like)."""

import jax.numpy as jnp
import jax.random as jr
from flax import struct
from jax import Array

from jax_pdf._validation import check_event_shape, is_concrete


@struct.dataclass
class Banana2D:
    """Banana-shaped 2D distribution (Rosenbrock-like).

    The density factorizes as:
        p(x0, x1) = N(x0; 1, 1) * N(x1; x0^2, sigma^2)

    This creates a curved, banana-shaped distribution centered around the
    parabola x1 = x0^2. The marginal in x0 is standard normal shifted to
    mean 1, while x1 follows x0^2 with Gaussian noise.

    The curved geometry forces gradient-based samplers to follow the
    parabola rather than move in straight lines. Smaller sigma creates
    a thinner banana where the sampler needs very small steps to stay
    on the ridge.

    Attributes:
        sigma: Controls the "thickness" of the banana. Smaller values
            (e.g., 0.01) create a thin, tightly curved banana that is
            challenging for MCMC. Larger values (e.g., 1.0) create a
            fatter, easier distribution. Default: 0.1.
    """

    sigma: float = 0.1
    """Std dev of x1 around x0^2. Smaller = thinner banana."""

    def __post_init__(self):
        if is_concrete(self.sigma) and self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    @property
    def dim(self) -> int:
        return 2

    def __call__(self, x: Array) -> Array:
        """Evaluate log probability density.

        Args:
            x: Input point(s) of shape (..., 2).

        Returns:
            Log probability density of shape (...).

        Raises:
            ValueError: If the input does not have trailing shape (2,).
        """
        check_event_shape(x, (self.dim,))
        x0, x1 = x[..., 0], x[..., 1]
        quad = (x0 - 1.0) ** 2 + (x1 - x0**2) ** 2 / self.sigma**2
        # Include both Gaussian constants here: __call__ is normalized, so
        # the shared log_normalization() contract requires log Z = 0.
        log_constant = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(2 * jnp.pi * self.sigma**2)
        return -0.5 * quad + log_constant

    def log_normalization(self) -> Array:
        """Log normalizing constant of the density returned by __call__.

        Returns:
            Scalar 0.0: __call__ already includes the Gaussian constants.
        """
        return jnp.array(0.0)

    def sample(self, key: Array, n: int) -> Array:
        """Draw exact samples from the distribution.

        Args:
            key: JAX PRNG key.
            n: Number of samples to draw.

        Returns:
            Samples of shape (n, 2).
        """
        k1, k2 = jr.split(key)
        x0 = 1.0 + jr.normal(k1, (n,))
        x1 = x0**2 + self.sigma * jr.normal(k2, (n,))
        return jnp.stack([x0, x1], axis=-1)
