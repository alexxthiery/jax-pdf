"""Tests for NealFunnel distribution."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import NealFunnel


class TestNealFunnel:

    @pytest.mark.parametrize("mouth", [-2.0, 1.5])
    def test_density_and_gradient_match_conditional_gaussians(self, mouth):
        """Protect exp(x0) as the conditional variance, including its log term."""
        sigma = 1.7
        x = jnp.array([mouth, 0.4, -0.8, 1.2])
        rest = np.asarray(x[1:], dtype=np.float64)
        square = float(rest @ rest)
        variance = math.exp(mouth)
        expected = (-0.5 * (mouth / sigma)**2 - math.log(sigma)
                    - 2 * math.log(2 * math.pi) - 1.5 * mouth
                    - 0.5 * square / variance)
        gradient = np.r_[-mouth / sigma**2 - 1.5 + 0.5 * square / variance,
                         -rest / variance]
        value, actual_gradient = jax.jit(jax.value_and_grad(NealFunnel(dim=4, sigma=sigma)))(x)
        assert float(value) == pytest.approx(expected, rel=2e-6)
        np.testing.assert_allclose(actual_gradient, gradient, rtol=2e-5, atol=2e-6)

    def test_sampler_transforms_given_normal_draws(self, monkeypatch):
        """Mouths -2, 0, 2 give conditional standard deviations 1/e, 1, e.

        Controlled innovations expose missing/incorrect conditional scaling
        and reused keys without estimating any moments from random draws.
        """
        draws = [jnp.array([[-1.0], [0.0], [1.0]]),
                 jnp.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 2.0]])]
        keys = []

        def normal(key, shape):
            draw = draws[len(keys)]
            assert shape == draw.shape
            keys.append(np.asarray(key))
            return draw

        monkeypatch.setattr(jax.random, "normal", normal)
        samples = NealFunnel(dim=3, sigma=2.0).sample(jax.random.PRNGKey(7), 3)

        expected = [[-2, 1 / math.e, -2 / math.e], [0, 0.5, 3],
                    [2, -math.e, 2 * math.e]]
        np.testing.assert_allclose(samples, expected, rtol=2e-7)
        assert len(keys) == 2
        assert not np.array_equal(keys[0], keys[1]), "Independent draws need distinct keys"

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
