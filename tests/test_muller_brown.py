"""Tests for MullerBrown distribution."""

import jax.numpy as jnp
import pytest

from jax_pdf import MullerBrown


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
        minima = jnp.array([[-0.558, 1.442], [0.624, 0.028]])
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
