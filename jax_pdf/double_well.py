"""Double-well distribution in D dimensions (product of 2D pairs)."""

import jax.numpy as jnp
import jax.random as jr
from flax import struct
from jax import Array

# Precompute 1D grid and weights for the x-marginal
_GRID_SIZE = 1_000_000
_X_GRID = jnp.linspace(-6.0, 6.0, _GRID_SIZE)
_LOG_F = -_X_GRID**4 + 6 * _X_GRID**2 + 0.5 * _X_GRID
_LOG_F_SHIFTED = _LOG_F - jnp.max(_LOG_F)
_WEIGHTS = jnp.exp(_LOG_F_SHIFTED)
_WEIGHTS = _WEIGHTS / jnp.sum(_WEIGHTS)


def _log_z_x() -> float:
    """Log normalizing constant of exp(-x^4 + 6x^2 + 0.5x) via quadrature."""
    z_x = jnp.trapezoid(jnp.exp(_LOG_F_SHIFTED), _X_GRID)
    return jnp.log(z_x) + jnp.max(_LOG_F)


@struct.dataclass
class DoubleWell:
    """Product of D/2 identical 2D double-well distributions.

    Each pair (x_{2i}, x_{2i+1}) has unnormalized density:
        mu(x, y) = exp(-x^4 + 6x^2 + 0.5x - 0.5y^2)

    The quartic -x^4 + 6x^2 creates two modes in the x-coordinate,
    while y is Gaussian. The full distribution in D dimensions has
    2^(D/2) modes, making it increasingly hard to mix as D grows.

    Attributes:
        n_dims: Number of dimensions (must be even, >= 2). Default: 2.
    """

    n_dims: int = 2
    """Number of dimensions (must be even)."""

    def __post_init__(self):
        if self.n_dims < 2:
            raise ValueError(f"n_dims must be >= 2, got {self.n_dims}")
        if self.n_dims % 2 != 0:
            raise ValueError(f"n_dims must be even, got {self.n_dims}")

    @property
    def dim(self) -> int:
        return self.n_dims

    def __call__(self, x: Array) -> Array:
        """Evaluate unnormalized log-density.

        Args:
            x: Input point(s) of shape (..., n_dims).

        Returns:
            Unnormalized log-density of shape (...).
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return jnp.sum(
            -x_even**4 + 6 * x_even**2 + 0.5 * x_even - 0.5 * x_odd**2,
            axis=-1,
        )

    def log_normalization(self) -> Array:
        """Log normalizing constant (numerical quadrature).

        The density factorizes into D/2 independent pairs. Each pair
        has Z_pair = Z_x * Z_y where Z_y = sqrt(2*pi) is exact and
        Z_x = integral of exp(-x^4 + 6x^2 + 0.5x) is computed via
        1D quadrature.

        Returns:
            Scalar log(Z).
        """
        n_pairs = self.n_dims // 2
        log_z_y = 0.5 * jnp.log(2 * jnp.pi)
        return n_pairs * (_log_z_x() + log_z_y)

    def sample(self, key: Array, n: int) -> Array:
        """Draw approximate samples from the distribution.

        Even coordinates are sampled from the 1D double-well marginal
        using jax.random.choice on a fine grid (1M points). Odd
        coordinates are drawn from N(0, 1).

        Args:
            key: JAX PRNG key.
            n: Number of samples to draw.

        Returns:
            Samples of shape (n, n_dims).
        """
        n_pairs = self.n_dims // 2
        k1, k2 = jr.split(key)

        # Even coords: sample grid indices, then look up grid values
        indices = jr.choice(k1, _GRID_SIZE, shape=(n, n_pairs), p=_WEIGHTS)
        x_even = _X_GRID[indices]

        # Odd coords: standard normal
        x_odd = jr.normal(k2, (n, n_pairs))

        # Interleave even and odd coordinates
        samples = jnp.empty((n, self.n_dims))
        samples = samples.at[:, 0::2].set(x_even)
        samples = samples.at[:, 1::2].set(x_odd)
        return samples
