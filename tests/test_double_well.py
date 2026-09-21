"""Tests for DoubleWell distribution."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import DoubleWell


@pytest.fixture(scope="module")
def quartic_reference():
    """Independent Gauss-Legendre integral; no production grid or weights.

    On [-6,6] the omitted tails are negligible. Doubling 128 to 256 nodes
    changes log Z and the first four moments by less than 1e-9.
    """
    nodes, weights = np.polynomial.legendre.leggauss(256)
    x = 6 * nodes
    mass = 6 * weights * np.exp(-x**4 + 6 * x**2 + 0.5 * x)
    z = mass.sum()
    return math.log(z), [float(mass @ x**p / z) for p in (1, 2, 4)]


class TestDoubleWell:

    def test_ten_dimensions_rejects_two_coordinates(self):
        """Regression: n_dims=10 must not silently evaluate a 2D density."""
        dist = DoubleWell(n_dims=10)
        with pytest.raises(ValueError, match="10"):
            dist(jnp.ones(2))

    def test_jitted_value_gradient_and_hessian_match_analytic_values(self):
        """At five (1, 1/2) pairs, each pair contributes 43/8 to log p.

        The gradient per pair is (17/2, -1/2), and its Hessian is
        diag(0, -1). These independent values also test passing the
        distribution itself through JIT with a shape check in place.
        """
        dist = DoubleWell(n_dims=10)
        x = jnp.tile(jnp.array([1.0, 0.5]), 5)
        call = lambda d, y: d(y)

        value, gradient = jax.jit(jax.value_and_grad(call, argnums=1))(dist, x)
        hessian = jax.jit(jax.hessian(call, argnums=1))(dist, x)

        assert float(value) == pytest.approx(5 * 43 / 8)
        np.testing.assert_allclose(gradient, np.tile([8.5, -0.5], 5))
        np.testing.assert_allclose(hessian, np.diag(np.tile([0.0, -1.0], 5)))

    def test_scan_can_carry_the_distribution(self):
        """The event check must work when scan reconstructs the PyTree."""
        dist = DoubleWell(n_dims=10)
        points = jnp.stack([jnp.zeros(10), jnp.ones(10)])

        def evaluate(d, xs):
            return jax.lax.scan(lambda carry, x: (carry, carry(x)), d, xs)

        final_dist, values = jax.jit(evaluate)(dist, points)

        assert final_dist.dim == 10
        np.testing.assert_allclose(values, [0.0, 25.0])

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

    @pytest.mark.parametrize("dim", [2, 10])
    def test_log_normalization_matches_independent_quadrature(self, dim, quartic_reference):
        """Scaling alone misses a wrong per-pair constant, such as omitting y."""
        log_z_x, _ = quartic_reference
        expected = dim / 2 * (log_z_x + 0.5 * math.log(2 * math.pi))
        assert float(DoubleWell(n_dims=dim).log_normalization()) == pytest.approx(
            expected, abs=1e-5)

    def test_sampler_probabilities_and_coordinate_assembly(self, monkeypatch, quartic_reference):
        """Check the actual choice probabilities, then prescribe three draws.

        Deterministic weighted sums over the documented uniform [-6, 6] grid
        must match independent Gauss-Legendre moments. This catches uniform
        or incorrect weights without Monte Carlo uncertainty. Controlled
        indices/normals check grid lookup, interleaving, and key separation.
        """
        _, moments = quartic_reference
        keys = []
        expected_even = []
        odd = jnp.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 2.0]])

        def choice(key, a, shape, p):
            assert shape == (3, 2)
            assert p is not None, "DoubleWell requires nonuniform sampling probabilities"
            probabilities = np.asarray(p, dtype=np.float64)
            assert probabilities.shape == (a,)
            assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0)
            assert probabilities.sum() == pytest.approx(1.0, abs=2e-6)
            grid = np.linspace(-6.0, 6.0, a)
            actual = [probabilities @ grid**power for power in (1, 2, 4)]
            # Float32 grid/weights dominate quadrature error at this resolution.
            np.testing.assert_allclose(actual, moments, rtol=3e-6, atol=2e-6)
            indices = np.array([[0, a - 1], [a // 4, a // 2], [3 * a // 4, 1]])
            expected_even.append(grid[indices])
            keys.append(np.asarray(key))
            return jnp.asarray(indices)

        def normal(key, shape):
            assert shape == odd.shape
            keys.append(np.asarray(key))
            return odd

        monkeypatch.setattr(jax.random, "choice", choice)
        monkeypatch.setattr(jax.random, "normal", normal)
        samples = DoubleWell(n_dims=4).sample(jax.random.PRNGKey(7), 3)

        assert len(expected_even) == 1
        np.testing.assert_allclose(samples[:, ::2], expected_even[0], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(samples[:, 1::2], odd)
        assert len(keys) == 2
        assert not np.array_equal(keys[0], keys[1]), "Independent draws need distinct keys"

    def test_sample_within_grid(self):
        """Even coordinates should fall within the grid range."""
        key = jax.random.PRNGKey(1)
        dw = DoubleWell(n_dims=4)
        samples = dw.sample(key, 1000)
        x_even = samples[:, 0::2]
        assert jnp.all(x_even >= -6.0)
        assert jnp.all(x_even <= 6.0)
