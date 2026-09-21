"""Tests for MullerBrown distribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import MullerBrown


class TestMullerBrown:

    @pytest.mark.parametrize("point,exponents,dx,dy", [
        ([0., 0.], [-1., -2.5, -24.5, 0.8], [2., 0., -23., 0.8], [0., 10., 25., -0.8]),
        ([0., 1.], [-11., -2.5, -6., 0.7], [2., 0., -12., 1.4], [-20., -10., 12., 0.6]),
    ])
    def test_value_and_gradient_at_hand_computed_exponents(self, point, exponents, dx, dy):
        """Relative minima and beta scaling also pass if all energies are halved.

        The exponent and derivative tables are hand substitutions into the
        four published Gaussian terms, including their cross terms.
        """
        beta = 0.7
        terms = np.array([-200., -100., -170., 15.]) * np.exp(exponents)
        expected_gradient = -beta * np.array([terms @ dx, terms @ dy])
        value, gradient = jax.jit(jax.value_and_grad(MullerBrown(beta=beta)))(jnp.array(point))
        assert float(value) == pytest.approx(-beta * terms.sum(), rel=2e-5)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-5)

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
