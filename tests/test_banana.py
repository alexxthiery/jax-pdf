"""Tests for Banana2D distribution."""

import jax.numpy as jnp
import pytest

from jax_pdf import Banana2D


class TestBanana2D:

    def test_dim_is_2(self):
        assert Banana2D().dim == 2

    def test_sigma_validation(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            Banana2D(sigma=-1.0)

    def test_log_normalization_finite(self):
        b = Banana2D(sigma=0.1)
        assert jnp.isfinite(b.log_normalization())
