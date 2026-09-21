"""Einstein / harmonic crystal: the one particle target with a tractable Z.

`U(x) = (k/2) * sum_i |x_i - site_i|^2` — independent harmonic wells at fixed
lattice sites. Built to sanity-check Boltzmann-generator pipelines: a flow plus
importance reweighting must recover `log_normalization()` (the exact free
energy). With `box_length` set, displacements use the minimum-image convention
(a periodic/torus harmonic crystal); the analytic `log Z` is the unbounded
value, exact when the wells are narrow relative to the box.
"""

import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_pdf._validation import check_event_shape, is_concrete


@struct.dataclass
class HarmonicCrystal:
    """Harmonic crystal Boltzmann target; `log p(x) = -beta * U(x)`.

    Attributes:
        positions: Well centres, shape `(n_particles, spatial_dim)`.
        spring_constant: Harmonic stiffness `k` (> 0).
        beta: Inverse temperature (> 0).
        box_length: Cubic box side for minimum-image displacements, or `None`
            for plain (unbounded) displacements.
    """

    positions: Array
    spring_constant: float = 1.0
    beta: float = 1.0
    box_length: float = None

    def __post_init__(self):
        # vmap probes the pytree with object placeholders. Actual arrays and
        # tracers still expose shapes, so keep their rank validation active.
        if type(self.positions) is not object and jnp.ndim(self.positions) != 2:
            raise ValueError(
                f"HarmonicCrystal: positions must be (n_particles, spatial_dim), "
                f"got shape {jnp.shape(self.positions)}."
            )
        if is_concrete(self.spring_constant) and self.spring_constant <= 0:
            raise ValueError(
                f"spring_constant must be positive, got {self.spring_constant}."
            )
        if is_concrete(self.beta) and self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}.")

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def spatial_dim(self) -> int:
        return int(self.positions.shape[1])

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim

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
        dx = x - self.positions
        if self.box_length is not None:
            dx = dx - self.box_length * jnp.round(dx / self.box_length)
        u = 0.5 * self.spring_constant * jnp.sum(dx**2, axis=(-2, -1))
        return -self.beta * u

    def log_normalization(self) -> Array:
        """Exact `log Z = (N*d/2) * log(2*pi / (beta*k))` (unbounded reference)."""
        return 0.5 * self.dim * jnp.log(
            2.0 * jnp.pi / (self.beta * self.spring_constant)
        )
