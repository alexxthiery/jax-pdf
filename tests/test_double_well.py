"""Tests for DoubleWell distribution."""

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import DoubleWell


class TestDoubleWell:

    def test_default_dim(self):
        assert DoubleWell().dim == 2

    def test_dim_validation_odd(self):
        with pytest.raises(ValueError, match="n_dims must be even"):
            DoubleWell(n_dims=3)

    def test_dim_validation_small(self):
        with pytest.raises(ValueError, match="n_dims must be >= 2"):
            DoubleWell(n_dims=0)

    def test_factorizes(self):
        """Log-density of concatenated pairs equals sum of 2D evaluations."""
        dw2 = DoubleWell(n_dims=2)
        dw4 = DoubleWell(n_dims=4)
        x = jnp.array([1.0, 0.5, -1.0, 0.3])
        lp4 = dw4(x)
        lp_sum = dw2(x[:2]) + dw2(x[2:])
        assert jnp.allclose(lp4, lp_sum)

    def test_log_normalization_scales(self):
        """log Z(4D) should be 2 * log Z(2D)."""
        log_z2 = DoubleWell(n_dims=2).log_normalization()
        log_z4 = DoubleWell(n_dims=4).log_normalization()
        assert jnp.allclose(log_z4, 2.0 * log_z2)

    def test_sample_shape(self):
        key = jax.random.PRNGKey(0)
        dw = DoubleWell(n_dims=4)
        samples = dw.sample(key, 100)
        assert samples.shape == (100, 4)

    def test_sample_within_grid(self):
        """Even coordinates should fall within the grid range."""
        key = jax.random.PRNGKey(1)
        dw = DoubleWell(n_dims=4)
        samples = dw.sample(key, 1000)
        x_even = samples[:, 0::2]
        assert jnp.all(x_even >= -6.0)
        assert jnp.all(x_even <= 6.0)
