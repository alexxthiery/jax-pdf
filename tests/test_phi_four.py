"""Tests for PhiFour distribution."""

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import PhiFour


class TestPhiFour:

    def test_default_dim(self):
        assert PhiFour().dim == 100

    def test_a_validation(self):
        with pytest.raises(ValueError, match="a must be positive"):
            PhiFour(a=0.0)
        with pytest.raises(ValueError, match="a must be positive"):
            PhiFour(a=-1.0)

    def test_dim_grid_validation(self):
        with pytest.raises(ValueError, match="dim_grid must be >= 2"):
            PhiFour(dim_grid=1)

    def test_beta_validation(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            PhiFour(beta=0.0)
        with pytest.raises(ValueError, match="beta must be positive"):
            PhiFour(beta=-1.0)

    def test_log_normalization_raises(self):
        with pytest.raises(NotImplementedError):
            PhiFour().log_normalization()

    def test_no_sample_method(self):
        assert not hasattr(PhiFour(), "sample")

    def test_energy_at_zero(self):
        """At x=0, grad_term=0 and V=dim_grid/4, so U=1/(4*a)."""
        dist = PhiFour(a=0.2, dim_grid=20, beta=1.0)
        lp = dist(jnp.zeros(20))
        assert jnp.allclose(lp, -1.0 / (4 * 0.2))

    def test_beta_scales_density(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (10,))
        lp1 = PhiFour(a=0.1, dim_grid=10, beta=1.0)(x)
        lp2 = PhiFour(a=0.1, dim_grid=10, beta=2.0)(x)
        assert jnp.allclose(lp2, 2.0 * lp1)

    def test_z2_symmetry(self):
        """When b=0, distribution is symmetric under phi -> -phi."""
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (10,))
        dist = PhiFour(a=0.1, b=0.0, dim_grid=10)
        assert jnp.allclose(dist(x), dist(-x))

    def test_b_breaks_symmetry(self):
        """Nonzero b breaks Z2 symmetry."""
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (10,))
        dist = PhiFour(a=0.1, b=0.5, dim_grid=10)
        assert not jnp.allclose(dist(x), dist(-x))

    def test_periodic_bc(self):
        """Periodic and Dirichlet BCs give different energies."""
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (10,))
        lp_dir = PhiFour(a=0.1, dim_grid=10, periodic=False)(x)
        lp_per = PhiFour(a=0.1, dim_grid=10, periodic=True)(x)
        assert not jnp.allclose(lp_dir, lp_per)

    def test_periodic_translation_invariance(self):
        """Cyclic shift should preserve energy under periodic BCs."""
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (10,))
        dist = PhiFour(a=0.1, b=0.0, dim_grid=10, periodic=True)
        x_shifted = jnp.roll(x, 3)
        assert jnp.allclose(dist(x), dist(x_shifted))
