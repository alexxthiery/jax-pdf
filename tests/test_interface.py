"""Shared interface tests for all distributions."""

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import (
    Banana2D, DoubleWell, DW4, HarmonicCrystal, LatticePhiFour, LGCP,
    LennardJones, MonatomicWater, MullerBrown, NealFunnel,
    PeriodicLennardJones, PhiFour,
)

# A small crystal, spread out enough that the particle distributions below are
# evaluated away from their singular configurations.
CRYSTAL_POSITIONS = jnp.linspace(0.0, 2.0, 12).reshape(4, 3)

ALL_DISTS = [
    Banana2D(sigma=0.1),
    DoubleWell(n_dims=2),
    DoubleWell(n_dims=10),
    DW4(),
    NealFunnel(dim=5, sigma=3.0),
    LGCP(grid_dim=5),
    LennardJones(n_particles=13),
    LennardJones(n_particles=55),
    MullerBrown(beta=1.0),
    PhiFour(a=0.1, b=0.0, dim_grid=10),
    PhiFour(a=0.1, b=0.0, dim_grid=10, periodic=True),
    PeriodicLennardJones(n_particles=8, box_length=4.0),
    MonatomicWater(n_particles=8, box_length=8.0),
    HarmonicCrystal(positions=CRYSTAL_POSITIONS),
    LatticePhiFour(lattice_shape=(4, 4)),
    LatticePhiFour(u=0.5, a=1.0, kappa=2.0, h=0.1, lattice_shape=(3, 3, 3)),
]

DISTS_WITH_SAMPLE = [
    Banana2D(sigma=0.1),
    DoubleWell(n_dims=4),
    NealFunnel(dim=5, sigma=3.0),
]

DISTS_WITH_LOG_NORM = [
    Banana2D(sigma=0.1),
    DoubleWell(n_dims=2),
    DoubleWell(n_dims=10),
    NealFunnel(dim=5, sigma=3.0),
    HarmonicCrystal(positions=CRYSTAL_POSITIONS),
]


def _test_point(dist):
    """Deterministic non-degenerate test point shaped to the distribution.

    Uses linspace to spread coordinates apart, avoiding singularities in
    potentials with 1/r terms (e.g. LennardJones at r=0). Particle
    distributions (those exposing ``n_particles`` and ``spatial_dim``)
    get the structured shape ``(n_particles, spatial_dim)``; flat
    distributions get ``(dim,)``.
    """
    if hasattr(dist, "n_particles") and hasattr(dist, "spatial_dim"):
        n, d = dist.n_particles, dist.spatial_dim
        return jnp.linspace(0.1, 1.0, n * d).reshape(n, d)
    return jnp.linspace(0.1, 1.0, dist.dim)


@pytest.mark.parametrize("dist", ALL_DISTS, ids=lambda d: type(d).__name__)
class TestInterface:
    """Every distribution must satisfy the core interface."""

    def test_dim_is_int(self, dist):
        assert isinstance(dist.dim, int)
        assert dist.dim > 0

    def test_call_returns_scalar(self, dist):
        x = _test_point(dist)
        lp = dist(x)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_call_batch(self, dist):
        x = _test_point(dist)
        x_batch = jnp.broadcast_to(x, (3,) + x.shape)
        lp = dist(x_batch)
        assert lp.shape == (3,)
        assert jnp.all(jnp.isfinite(lp))

    def test_grad(self, dist):
        x = _test_point(dist)
        g = jax.grad(dist)(x)
        assert g.shape == x.shape
        assert jnp.all(jnp.isfinite(g))

    def test_call_with_the_distribution_as_a_jit_argument(self, dist):
        """A distribution keeps its meaning when it crosses a trace boundary.

        Bug it catches: every field is a pytree child by default, so the
        parameters arrive as tracers, and validation in ``__post_init__`` or a
        Python branch on a flag then raises TracerBoolConversionError.
        Oracle: the same call with the distribution captured by closure.
        """
        x = _test_point(dist)
        traced = jax.jit(lambda d, y: d(y))(dist, x)
        assert jnp.allclose(traced, dist(x), rtol=1e-6)


@pytest.mark.parametrize(
    "dist", DISTS_WITH_LOG_NORM, ids=lambda d: type(d).__name__
)
class TestLogNormalization:
    """Distributions with a computable normalizing constant."""

    def test_log_normalization_scalar(self, dist):
        log_z = dist.log_normalization()
        assert jnp.shape(log_z) == ()
        assert jnp.isfinite(log_z)


@pytest.mark.parametrize("dist", DISTS_WITH_SAMPLE, ids=lambda d: type(d).__name__)
class TestSample:

    def test_sample_shape(self, dist):
        key = jax.random.PRNGKey(0)
        samples = dist.sample(key, 100)
        assert samples.shape == (100, dist.dim)

    def test_sample_finite(self, dist):
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 50)
        assert jnp.all(jnp.isfinite(samples))
