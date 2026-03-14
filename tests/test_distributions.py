"""Tests for jax-pdf distributions."""

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import Banana2D, LGCP, MullerBrown, NealFunnel


# ---------------------------------------------------------------------------
# Shared interface tests
# ---------------------------------------------------------------------------

ALL_DISTS = [
    Banana2D(sigma=0.1),
    NealFunnel(dim=5, sigma=3.0),
    LGCP(grid_dim=5),
    MullerBrown(beta=1.0),
]

DISTS_WITH_SAMPLE = [
    Banana2D(sigma=0.1),
    NealFunnel(dim=5, sigma=3.0),
]

DISTS_WITH_LOG_NORM = [
    Banana2D(sigma=0.1),
    NealFunnel(dim=5, sigma=3.0),
]


@pytest.mark.parametrize("dist", ALL_DISTS, ids=lambda d: type(d).__name__)
class TestInterface:
    """Every distribution must satisfy the core interface."""

    def test_dim_is_int(self, dist):
        assert isinstance(dist.dim, int)
        assert dist.dim > 0

    def test_call_returns_scalar(self, dist):
        x = jnp.zeros(dist.dim)
        lp = dist(x)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_call_batch(self, dist):
        x = jnp.zeros((3, dist.dim))
        lp = dist(x)
        assert lp.shape == (3,)
        assert jnp.all(jnp.isfinite(lp))

    def test_grad(self, dist):
        x = jnp.zeros(dist.dim)
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


# ---------------------------------------------------------------------------
# Sample tests (Banana2D, NealFunnel only)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------

class TestBanana2D:

    def test_dim_is_2(self):
        assert Banana2D().dim == 2

    def test_sigma_validation(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            Banana2D(sigma=-1.0)

    def test_log_normalization_finite(self):
        b = Banana2D(sigma=0.1)
        assert jnp.isfinite(b.log_normalization())


class TestNealFunnel:

    def test_default_dim(self):
        assert NealFunnel().dim == 10

    def test_dim_validation(self):
        with pytest.raises(ValueError, match="dim must be >= 2"):
            NealFunnel(dim=1)

    def test_sigma_validation(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            NealFunnel(sigma=0.0)

    def test_normalized(self):
        """NealFunnel is normalized, so log_normalization should be 0."""
        f = NealFunnel(dim=3)
        assert jnp.allclose(f.log_normalization(), 0.0)


class TestLGCP:

    def test_dim_equals_grid_squared(self):
        lgcp = LGCP(grid_dim=5)
        assert lgcp.dim == 25

    def test_grid_dim_validation(self):
        with pytest.raises(ValueError, match="grid_dim must be positive"):
            LGCP(grid_dim=0)

    def test_pines_points_shape(self):
        lgcp = LGCP(grid_dim=5)
        pts = lgcp.pines_points
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_whitened_same_dim(self):
        lgcp = LGCP(grid_dim=5, whitened=False)
        lgcp_w = LGCP(grid_dim=5, whitened=True)
        assert lgcp.dim == lgcp_w.dim

    def test_log_normalization_raises(self):
        lgcp = LGCP(grid_dim=5)
        with pytest.raises(NotImplementedError):
            lgcp.log_normalization()

    def test_no_sample_method(self):
        """LGCP is an unnormalized posterior; no exact sampling."""
        lgcp = LGCP(grid_dim=5)
        assert not hasattr(lgcp, "sample")


class TestMullerBrown:

    def test_dim_is_2(self):
        assert MullerBrown().dim == 2

    def test_beta_validation(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            MullerBrown(beta=-1.0)
        with pytest.raises(ValueError, match="beta must be positive"):
            MullerBrown(beta=0.0)

    def test_log_normalization_raises(self):
        mb = MullerBrown()
        with pytest.raises(NotImplementedError):
            mb.log_normalization()

    def test_no_sample_method(self):
        mb = MullerBrown()
        assert not hasattr(mb, "sample")

    def test_minima_higher_than_saddle(self):
        """Log-density at minima should exceed the saddle point value."""
        mb = MullerBrown()
        # Two deepest minima
        minima = jnp.array([[-0.558, 1.442], [0.624, 0.028]])
        # Saddle between them (approximate)
        saddle = jnp.array([[-0.822, 0.624]])
        lp_minima = mb(minima)
        lp_saddle = mb(saddle)
        assert jnp.all(lp_minima > lp_saddle)

    def test_beta_scales_density(self):
        """Higher beta should amplify the log-density magnitude."""
        x = jnp.array([0.624, 0.028])
        lp1 = MullerBrown(beta=1.0)(x)
        lp2 = MullerBrown(beta=2.0)(x)
        assert jnp.allclose(lp2, 2.0 * lp1)
