"""Einstein / harmonic crystal: the one particle target with a tractable Z.

`U(x) = (k/2) * sum_i |x_i - site_i|^2` — independent harmonic wells at fixed
lattice sites. Built to sanity-check Boltzmann-generator pipelines: a flow plus
importance reweighting must recover `log_normalization()` (the exact free
energy). With `box_length` set, displacements use the minimum-image convention
(a periodic/torus harmonic crystal); the analytic `log Z` is the unbounded
value, exact when the wells are narrow relative to the box.
"""
import math

import jax.numpy as jnp
from flax import struct
from jax import Array


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
        if jnp.ndim(self.positions) != 2:
            raise ValueError(
                f"HarmonicCrystal: positions must be (n_particles, spatial_dim), "
                f"got shape {jnp.shape(self.positions)}."
            )
        if self.spring_constant <= 0:
            raise ValueError(
                f"spring_constant must be positive, got {self.spring_constant}."
            )
        if self.beta <= 0:
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
        """Unnormalized log-density, `x` shape `(..., n_particles, spatial_dim)`."""
        n, d = self.n_particles, self.spatial_dim
        if x.shape[-2:] != (n, d):
            raise ValueError(
                f"HarmonicCrystal expected x.shape[-2:] == ({n}, {d}), "
                f"got {tuple(x.shape)}."
            )
        dx = x - self.positions
        if self.box_length is not None:
            dx = dx - self.box_length * jnp.round(dx / self.box_length)
        u = 0.5 * self.spring_constant * jnp.sum(dx**2, axis=(-2, -1))
        return -self.beta * u

    def log_normalization(self) -> float:
        """Exact `log Z = (N*d/2) * log(2*pi / (beta*k))` (unbounded reference)."""
        return 0.5 * self.dim * math.log(
            2.0 * math.pi / (self.beta * self.spring_constant)
        )
