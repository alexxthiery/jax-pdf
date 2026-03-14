"""Shared interface tests for all distributions."""

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import (
    Banana2D, DoubleWell, DW4, LGCP, LennardJones,
    MullerBrown, NealFunnel, PhiFour,
)

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
]


def _test_point(dim):
    """Deterministic non-degenerate test point.

    Uses linspace to spread coordinates apart, avoiding singularities
    in potentials with 1/r terms (e.g. LennardJones at r=0).
    """
    return jnp.linspace(0.1, 1.0, dim)


@pytest.mark.parametrize("dist", ALL_DISTS, ids=lambda d: type(d).__name__)
class TestInterface:
    """Every distribution must satisfy the core interface."""

    def test_dim_is_int(self, dist):
        assert isinstance(dist.dim, int)
        assert dist.dim > 0

    def test_call_returns_scalar(self, dist):
        x = _test_point(dist.dim)
        lp = dist(x)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_call_batch(self, dist):
        x = jnp.broadcast_to(_test_point(dist.dim), (3, dist.dim))
        lp = dist(x)
        assert lp.shape == (3,)
        assert jnp.all(jnp.isfinite(lp))

    def test_grad(self, dist):
        x = _test_point(dist.dim)
        g = jax.grad(dist)(x)
        assert g.shape == (dist.dim,)
        assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize(
    "dist", DISTS_WITH_LOG_NORM, ids=lambda d: type(d).__name__
)
class TestLogNormalization:
    """Distributions with a computable normalizing constant."""

    def test_log_normalization_scalar(self, dist):
        log_z = dist.log_normalization()
        assert log_z.shape == ()
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
