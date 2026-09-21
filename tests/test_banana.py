"""Tests for Banana2D distribution."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import Banana2D


class TestBanana2D:

    def test_dim_is_2(self):
        assert Banana2D().dim == 2

    def test_sigma_validation(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            Banana2D(sigma=-1.0)

    @pytest.mark.parametrize("sigma", [0.2, 1.5])
    def test_density_and_gradient_at_known_gaussian_residuals(self, sigma):
        """At x=(2, 4+sigma), both standardized residuals equal one.

        A finite value alone cannot detect wrong precision or a missing
        normalizer. Here log p = -1-log(2*pi*sigma), and differentiating the
        two Gaussian factors gives (-1+4/sigma, -1/sigma).
        """
        dist = Banana2D(sigma=sigma)
        x = jnp.array([2.0, 4.0 + sigma])
        value, gradient = jax.jit(jax.value_and_grad(dist))(x)
        assert float(value) == pytest.approx(-1 - math.log(2 * math.pi * sigma),
                                             rel=2e-5, abs=2e-6)
        np.testing.assert_allclose(gradient, [-1 + 4 / sigma, -1 / sigma], rtol=2e-5)

    def test_sampler_transforms_given_normal_draws(self, monkeypatch):
        """Check location, curvature, scale and key separation without sampling.

        Only JAX's random primitive is replaced; the production transform runs
        on three prescribed innovations. We trust JAX to generate normals.
        """
        draws = [jnp.array([-2.0, 0.0, 1.0]), jnp.array([2.0, -1.0, 0.5])]
        keys = []

        def normal(key, shape):
            assert shape == (3,)
            keys.append(np.asarray(key))
            return draws[len(keys) - 1]

        monkeypatch.setattr(jax.random, "normal", normal)
        samples = Banana2D(sigma=0.5).sample(jax.random.PRNGKey(7), 3)

        np.testing.assert_allclose(samples, [[-1, 2], [1, 0.5], [2, 4.25]])
        assert len(keys) == 2
        assert not np.array_equal(keys[0], keys[1]), "Independent draws need distinct keys"
