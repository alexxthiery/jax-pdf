"""Tests for NealFunnel distribution."""

import jax.numpy as jnp
import pytest

from jax_pdf import NealFunnel


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
