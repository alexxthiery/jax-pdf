"""Shared interface tests for all distributions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import (
    Banana2D, DoubleWell, DW4, HarmonicCrystal, LatticePhiFour, LGCP,
    LennardJones, MonatomicWater, MullerBrown, NealFunnel,
    PeriodicLennardJones, PhiFour,
)

# A small crystal, spread out enough that the particle distributions below are
# evaluated away from their singular configurations.
CRYSTAL_POSITIONS = jnp.linspace(0.0, 2.0, 12).reshape(4, 3)

# One representative per implementation path. Both LGCP coordinate systems and
# PhiFour boundaries have different evaluation paths; changing only a dimension
# does not justify repeating the full interface matrix. Dedicated model tests
# retain DoubleWell's 2D factorization, LJ55 gradients and 3D lattice examples.
ALL_DISTS = [
    Banana2D(sigma=0.1),
    DoubleWell(n_dims=10),
    DW4(),
    NealFunnel(dim=5, sigma=3.0),
    LGCP(grid_dim=5),
    LGCP(grid_dim=5, whitened=True),
    LennardJones(n_particles=13),
    MullerBrown(beta=1.0),
    PhiFour(a=0.1, b=0.0, dim_grid=10),
    PhiFour(a=0.1, b=0.0, dim_grid=10, periodic=True),
    PeriodicLennardJones(n_particles=8, box_length=4.0),
    MonatomicWater(n_particles=8, box_length=8.0),
    HarmonicCrystal(positions=CRYSTAL_POSITIONS),
    LatticePhiFour(lattice_shape=(4, 4)),
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

    @pytest.mark.parametrize("compiled", [False, True], ids=["vmap", "jit-vmap"])
    def test_distribution_as_unmapped_vmap_argument(self, dist, compiled):
        """Pytree structure probing must not validate JAX's object placeholders.

        This differs from a closure over dist: vmap reconstructs the argument
        with sentinel leaves before rebuilding it with actual arrays/tracers.
        Compare distinct points against ordinary evaluation of the same model.
        """
        x = _test_point(dist)
        points = jnp.stack([x, x * 1.1])
        evaluate = jax.vmap(lambda d, y: d(y), in_axes=(None, 0))
        if compiled:
            evaluate = jax.jit(evaluate)
        actual = evaluate(dist, points)
        expected = jnp.stack([dist(point) for point in points])
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-5)

    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
    @pytest.mark.parametrize("invalid", ["long", "layout"])
    def test_rejects_wrong_event_shape(self, dist, compiled, invalid):
        """Reject malformed events before broadcasting or indexing can hide them.

        The layout case preserves the number of values but moves event axes;
        checking total size would incorrectly accept it. Both LGCP forms are
        included because their linear algebra handles broadcasting differently.
        The full rank/empty/singleton/batch boundary matrix belongs to
        test_validation.py; these cases check each distribution uses the guard.
        """
        event_shape = _test_point(dist).shape
        shapes = {
            "long": event_shape[:-1] + (event_shape[-1] + 1,),
            "layout": ((dist.dim,) if len(event_shape) == 2 else (dist.dim, 1)),
        }
        x = jnp.zeros(shapes[invalid])
        call = jax.jit(lambda d, y: d(y)) if compiled else lambda d, y: d(y)

        with pytest.raises(ValueError) as error:
            call(dist, x)

        message = str(error.value)
        assert "expected" in message.lower()
        assert str(event_shape) in message
        assert f"got shape {x.shape}" in message

    @pytest.mark.parametrize("batch_shape", [(2, 3), (0,), (2, 0)])
    def test_batch_axes_are_preserved_under_jit(self, dist, batch_shape):
        """Only event axes are constrained, including for empty batches."""
        x = _test_point(dist)
        batch = jnp.broadcast_to(x, batch_shape + x.shape)

        actual = jax.jit(lambda d, y: d(y))(dist, batch)

        assert actual.shape == batch_shape
        np.testing.assert_allclose(actual, jnp.broadcast_to(dist(x), batch_shape),
                                   rtol=1e-5)

    def test_jitted_vmap_matches_direct_batching(self, dist):
        """Mapping over points must not mistake the batch axis for an event."""
        x = _test_point(dist)
        batch = jnp.stack([x, x * 1.1, x * 0.9])

        actual = jax.jit(jax.vmap(lambda y: dist(y)))(batch)

        np.testing.assert_allclose(actual, dist(batch), rtol=1e-5)


@pytest.mark.parametrize(
    "dist", [d for d in ALL_DISTS if hasattr(d, "n_particles")],
    ids=lambda d: type(d).__name__,
)
def test_particle_count_is_checked_separately(dist):
    """A valid spatial dimension does not excuse a wrong particle count."""
    x = jnp.zeros((dist.n_particles + 1, dist.spatial_dim))
    with pytest.raises(ValueError, match="[Ee]xpected"):
        jax.jit(lambda d, y: d(y))(dist, x)


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
