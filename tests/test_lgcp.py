"""LGCP posterior, coordinate-change and inference tests with independent oracles."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import LGCP


def _reference_model(dist):
    """Rebuild the documented GP and histogram without production utilities."""
    grid = dist.grid_dim
    coordinates = np.array([(i, j) for i in range(grid) for j in range(grid)])
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    covariance = 1.91 * np.exp(-distances / (grid / 33))
    points = np.asarray(dist.pines_points, dtype=np.float64)
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=grid,
                                 range=[[0, 1], [0, 1]])
    return covariance, counts.ravel(), math.log(126) - 1.91 / 2, 1 / grid**2


class TestLGCP:

    @pytest.mark.parametrize("enabled", [False, True], ids=["float32-caller", "float64-caller"])
    @pytest.mark.parametrize("method", ["map_estimate", "laplace_approximation"])
    @pytest.mark.parametrize("initial", ["default", "float32"])
    def test_inference_preserves_precision_setting(self, enabled, method, initial):
        """Inference needs float64 internally, without changing later JAX work.

        A supplied float32 iterate must be promoted too, otherwise it retains
        lower precision despite the float64 context. Zero iterations suffice
        to trace the loop; the stationary-equation test checks convergence.
        """
        previous = jax.config.x64_enabled
        try:
            jax.config.update("jax_enable_x64", enabled)
            dist = LGCP(grid_dim=1)
            x0 = None if initial == "default" else jnp.zeros(1, dtype=jnp.float32)
            result = getattr(dist, method)(x0=x0, max_iter=0)
            assert jax.config.x64_enabled == enabled
            assert jnp.ones(1).dtype == (jnp.float64 if enabled else jnp.float32)
            if method == "map_estimate":
                arrays = [result["x"], result["x_history"], result["loss_history"]]
            else:
                arrays = [result["mu"], result["precision"], result["cov"]]
            for value in arrays:
                assert value.dtype == np.float64
                assert np.all(np.isfinite(np.asarray(value)))
        finally:
            jax.config.update("jax_enable_x64", previous)

    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("method", ["map_estimate", "laplace_approximation"])
    def test_inference_preserves_precision_after_invalid_input(self, enabled, method):
        """An early validation error must also restore the caller's precision."""
        previous = jax.config.x64_enabled
        try:
            jax.config.update("jax_enable_x64", enabled)
            with pytest.raises(ValueError, match="x0 must have shape"):
                getattr(LGCP(grid_dim=1), method)(x0=jnp.zeros(2), max_iter=0)
            assert jax.config.x64_enabled == enabled
        finally:
            jax.config.update("jax_enable_x64", previous)

    @pytest.mark.parametrize("whitened", [False, True])
    def test_posterior_and_gradient_match_gaussian_poisson_model(self, whitened):
        """Finite prior-only outputs miss the entire data likelihood.

        Use NumPy's histogram and dense Gaussian algebra to check the full
        posterior, including bin area, mean, determinant, and coordinate
        Jacobian. No cached counts or covariance from the target are reused.
        """
        dist = LGCP(grid_dim=8, whitened=whitened)
        covariance, counts, mean, area = _reference_model(dist)
        chol = np.linalg.cholesky(covariance)
        x = jnp.linspace(-0.7, 0.8, dist.dim) if whitened else jnp.linspace(2.8, 4.7, dist.dim)
        x64 = np.asarray(x, dtype=np.float64)
        latent = mean + chol @ x64 if whitened else x64
        centered = latent - mean
        precision_times_x = np.linalg.solve(covariance, centered)
        logdet = 2 * np.log(np.diag(chol)).sum()
        expected = (-0.5 * centered @ precision_times_x
                    - 0.5 * (dist.dim * math.log(2 * math.pi) + logdet)
                    + counts @ latent - area * np.exp(latent).sum())
        gradient = counts - area * np.exp(latent) - precision_times_x
        if whitened:
            expected += 0.5 * logdet
            gradient = chol.T @ gradient

        actual, actual_gradient = jax.jit(jax.value_and_grad(dist.__call__))(x)
        assert float(actual) == pytest.approx(expected, rel=3e-6, abs=3e-5)
        np.testing.assert_allclose(actual_gradient, gradient, rtol=3e-5, atol=3e-5)

    @pytest.mark.parametrize("whitened", [False, True])
    def test_hessian_is_prior_precision_plus_poisson_curvature(self, whitened):
        """The Hessian is of negative log p; a sign error reverses curvature."""
        dist = LGCP(grid_dim=3, whitened=whitened)
        covariance, _, mean, area = _reference_model(dist)
        chol = np.linalg.cholesky(covariance)
        x = jnp.linspace(-0.4, 0.5, dist.dim) if whitened else jnp.linspace(3.0, 4.5, dist.dim)
        latent = mean + chol @ np.asarray(x) if whitened else np.asarray(x)
        expected = np.linalg.inv(covariance) + np.diag(area * np.exp(latent))
        if whitened:
            expected = chol.T @ expected @ chol
        np.testing.assert_allclose(dist.hessian_at(x), expected, rtol=3e-5, atol=3e-6)

    @pytest.mark.parametrize("enabled", [False, True])
    def test_laplace_approximation_matches_one_cell_stationary_equation(self, enabled):
        """One cell reduces MAP to a strictly monotone scalar root.

        Bisection of (f-mu)/variance + exp(f) - count gives the mode
        independently of L-BFGS. Its derivative is the Laplace precision.
        Exercise both caller precision settings; inference must keep its
        float64 accuracy even when the caller normally uses float32.
        """
        previous_x64 = jax.config.x64_enabled
        try:
            jax.config.update("jax_enable_x64", enabled)
            dist = LGCP(grid_dim=1)
            count, variance = len(dist.pines_points), 1.91
            mean = math.log(126) - variance / 2
            low, high = 0.0, math.log(count)
            for _ in range(60):
                middle = (low + high) / 2
                if (middle - mean) / variance + math.exp(middle) > count:
                    high = middle
                else:
                    low = middle
            mode = (low + high) / 2
            precision = 1 / variance + math.exp(mode)

            result = dist.laplace_approximation(max_iter=80, tol=1e-7)

            assert jax.config.x64_enabled == enabled
            assert result["optimization"]["converged"]
            np.testing.assert_allclose(result["mu"], [mode], atol=1e-7, rtol=0)
            np.testing.assert_allclose(result["precision"], [[precision]], rtol=1e-7)
            np.testing.assert_allclose(result["cov"], [[1 / precision]], rtol=1e-7)
            history = result["optimization"]
            assert len(history["x_history"]) == history["n_iters"] + 1
            np.testing.assert_allclose(history["x_history"][-1], result["mu"])
        finally:
            jax.config.update("jax_enable_x64", previous_x64)

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
