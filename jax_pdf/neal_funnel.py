"""Neal's funnel distribution."""

import jax.numpy as jnp
import jax.random as jr
from flax import struct
from jax import Array


@struct.dataclass
class NealFunnel:
    """Neal's funnel distribution in D dimensions.

    The density factorizes as:
        p(x) = N(x0; 0, sigma^2) * prod_{i=1}^{D-1} N(x_i; 0, exp(x0))

    The first coordinate x0 controls the scale of all other coordinates.
    When x0 is large, the remaining coordinates spread out; when x0 is
    small (negative), they concentrate near zero. This creates a funnel
    shape that is notoriously difficult for MCMC. In the narrow neck
    (x0 < 0), step sizes must be tiny; in the wide mouth (x0 > 0),
    they can be large. Fixed step sizes either waste time in the mouth
    or diverge in the neck.

    Attributes:
        dim: Dimensionality of the distribution. Must be >= 2.
        sigma: Std dev of x0 (the "mouth width" of the funnel). Larger
            values create a wider range of scales, making sampling harder.
            Default: 3.0 (standard benchmark setting).
    """

    dim: int = 10
    """Dimensionality of the distribution."""

    sigma: float = 3.0
    """Std dev of x0. Larger = wider funnel, harder sampling."""

    def __post_init__(self):
        if self.dim < 2:
            raise ValueError(f"dim must be >= 2, got {self.dim}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    def __call__(self, x: Array) -> Array:
        """Evaluate log probability density.

        Args:
            x: Input point(s) of shape (..., dim).

        Returns:
            Log probability density of shape (...).
        """
        x0 = x[..., 0]
        x_rest = x[..., 1:]

        # Conditional std for x_i | x0
        std_cond = jnp.exp(x0 / 2.0)

        # log N(x0; 0, sigma^2)
        log_p_x0 = (
            -0.5 * (x0 / self.sigma) ** 2
            - 0.5 * jnp.log(2 * jnp.pi * self.sigma**2)
        )

        # log N(x_i; 0, exp(x0)) for i >= 1
        quad_rest = jnp.sum((x_rest / std_cond[..., None]) ** 2, axis=-1)
        log_p_rest = (
            -0.5 * quad_rest
            - 0.5 * (self.dim - 1) * jnp.log(2 * jnp.pi * std_cond**2)
        )

        return log_p_x0 + log_p_rest

    def log_normalization(self) -> Array:
        """Log normalizing constant (already included in __call__).

        Returns:
            Scalar 0.0 (distribution is normalized by construction).
        """
        return jnp.array(0.0)

    def sample(self, key: Array, n: int) -> Array:
        """Draw exact samples from the distribution.

        Args:
            key: JAX PRNG key.
            n: Number of samples to draw.

        Returns:
            Samples of shape (n, dim).
        """
        k1, k2 = jr.split(key)

        # x0 ~ N(0, sigma^2)
        x0 = self.sigma * jr.normal(k1, (n, 1))

        # x_i | x0 ~ N(0, exp(x0))
        std_cond = jnp.exp(x0 / 2.0)
        x_rest = std_cond * jr.normal(k2, (n, self.dim - 1))

        return jnp.concatenate([x0, x_rest], axis=1)
