"""Tests for LGCP distribution."""

import jax.numpy as jnp
import pytest

from jax_pdf import LGCP


class TestLGCP:

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
