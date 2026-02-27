"""Utilities for Log Gaussian Cox Process (LGCP) inference.

This module provides functions for discretized LGCP on a 2D grid:
- Binning spatial points into a histogram
- Computing GP kernel/Gram matrices
- Poisson process likelihood
- Whitening/unwhitening transformations for reparameterization

Reference: DeepMind's annealed importance sampling benchmark (Apache 2.0).
"""

from __future__ import annotations

import itertools
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as slinalg
import numpy as np
from jax import Array


def compute_bin_counts(
    points: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """Bin 2D points into a grid histogram.

    Divides the unit square [0,1]^2 into a num_bins x num_bins grid and counts
    how many points fall into each cell. Used to discretize spatial point
    patterns for LGCP inference.

    Args:
        points: Spatial coordinates of shape (num_points, 2). Each point must
            lie in [0, 1]^2.
        num_bins: Number of bins per dimension. Total cells = num_bins^2.

    Returns:
        Count matrix of shape (num_bins, num_bins) where entry [i, j] is the
        number of points in the cell covering [i/G, (i+1)/G) x [j/G, (j+1)/G),
        where G = num_bins.

    Note:
        Points exactly on the upper boundary (coordinate = 1.0) are assigned
        to the last bin to handle edge cases from data rescaling.
    """
    # Scale points from [0,1] to [0, num_bins] for bin assignment
    scaled_points = points * num_bins
    counts = np.zeros((num_bins, num_bins), dtype=np.float64)

    for point in scaled_points:
        # Floor to get bin indices
        row_idx = int(np.floor(point[0]))
        col_idx = int(np.floor(point[1]))

        # Clamp boundary points (exactly at 1.0) to last bin
        if row_idx == num_bins:
            row_idx = num_bins - 1
        if col_idx == num_bins:
            col_idx = num_bins - 1

        counts[row_idx, col_idx] += 1

    return counts


def make_grid_indices(num_bins: int) -> Array:
    """Generate all 2D grid cell indices as a flat array.

    Creates the Cartesian product of [0, 1, ..., num_bins-1] with itself,
    yielding all (row, col) index pairs for a num_bins x num_bins grid.

    Args:
        num_bins: Number of bins per dimension.

    Returns:
        Array of shape (num_bins^2, 2) containing all grid indices.
        Ordered as: [(0,0), (0,1), ..., (0,G-1), (1,0), ..., (G-1,G-1)].
    """
    indices_1d = jnp.arange(num_bins)
    # Cartesian product via itertools, then stack into array
    grid_pairs = list(itertools.product(indices_1d, indices_1d))
    return jnp.array(grid_pairs)


def compute_gram_matrix(
    kernel_fn: Callable[[Array, Array], Array],
    points: Array,
) -> Array:
    """Compute the Gram (kernel) matrix for a set of points.

    Given a kernel function k and points {x_1, ..., x_n}, computes the
    symmetric positive semi-definite matrix K where K_ij = k(x_i, x_j).

    Args:
        kernel_fn: Kernel function taking two points and returning a scalar.
        points: Array of shape (n, d) containing n points in d dimensions.

    Returns:
        Gram matrix of shape (n, n) where entry [i,j] = kernel_fn(points[i], points[j]).
    """
    # Nested vmap: outer over rows, inner over columns
    return jax.vmap(
        lambda x: jax.vmap(lambda y: kernel_fn(x, y))(points)
    )(points)


def exponential_kernel(
    point_a: Array,
    point_b: Array,
    variance: float,
    num_bins: int,
    length_scale: float,
) -> Array:
    """Exponential (Laplacian/Matern-1/2) covariance kernel.

    Computes the covariance between two grid points using an exponential kernel:

        K(a, b) = variance * exp(-||a - b|| / (num_bins * length_scale))

    This kernel produces rough (non-differentiable) sample paths, appropriate
    for spatial processes with sharp local variations.

    Args:
        point_a: First grid point, shape (2,) with integer indices.
        point_b: Second grid point, shape (2,) with integer indices.
        variance: Marginal variance (sigma^2). Controls overall magnitude.
        num_bins: Grid resolution per dimension. Scales the length scale
            since points are in index space [0, num_bins) not [0, 1).
        length_scale: Correlation length in the original [0, 1] domain.
            Smaller values = faster decay = less smooth.

    Returns:
        Scalar covariance value.

    Note:
        The effective length scale in grid-index space is num_bins * length_scale.
    """
    # Distance in grid-index space, normalized to original [0,1] domain
    distance = jnp.linalg.norm(point_a - point_b, ord=2)
    effective_length_scale = num_bins * length_scale
    normalized_distance = distance / effective_length_scale

    return variance * jnp.exp(-normalized_distance)


def poisson_log_likelihood(
    log_intensity: Array,
    bin_area: float,
    counts: Array,
) -> Array:
    """Compute log-likelihood of counts under a discretized Poisson process.

    Models each grid cell as an independent Poisson:

        n_i | f_i ~ Poisson(bin_area * exp(f_i))

    where f_i is the log-intensity in cell i. The log-likelihood (up to
    constant terms independent of f) is:

        log p(n | f) = sum_i [n_i * f_i - bin_area * exp(f_i)]

    This is the likelihood term in LGCP posterior inference.

    Args:
        log_intensity: Log intensity values f, shape (..., num_cells).
            The actual intensity is exp(f). Supports batch dimensions.
        bin_area: Area of each grid cell (typically 1/num_bins^2 for unit square).
        counts: Observed point counts per cell, shape (num_cells,).
            Non-negative integers.

    Returns:
        Log-likelihood of shape (...). Scalar for single input.

    Note:
        Constant terms (-log(n_i!)) are omitted since they don't affect
        optimization or MCMC acceptance ratios.
    """
    # n_i * f_i: more points in high-intensity regions increases likelihood
    count_term = log_intensity * counts

    # -A * exp(f_i): penalizes high intensity (expected count = A * exp(f))
    intensity_penalty = -bin_area * jnp.exp(log_intensity)

    return jnp.sum(count_term + intensity_penalty, axis=-1)


def whiten_to_latent(
    whitened: Array,
    mean: float,
    cholesky_cov: Array,
) -> Array:
    """Transform whitened variables to latent GP values.

    Implements the reparameterization trick for Gaussian processes:

        f = L @ epsilon + mu

    where epsilon ~ N(0, I) is the whitened variable and L is the Cholesky
    factor of the covariance matrix (L @ L.T = K). This gives f ~ N(mu, K).

    Whitening is useful for MCMC because the prior on epsilon is isotropic
    (no correlations), making sampling geometry easier.

    Args:
        whitened: Whitened (standard normal) variables, shape (..., n).
            Supports batch dimensions.
        mean: Constant mean of the GP (mu_i = mean for all i).
        cholesky_cov: Lower Cholesky factor L of covariance, shape (n, n).
            Must satisfy K = L @ L.T.

    Returns:
        Latent GP values f, shape (..., n).
    """
    # einsum handles arbitrary batch dims: 'ij,...j->...i'
    return jnp.einsum('ij,...j->...i', cholesky_cov, whitened) + mean


def latent_to_whiten(
    latent: Array,
    mean: float,
    cholesky_cov: Array,
) -> Array:
    """Transform latent GP values to whitened variables.

    Inverse of whiten_to_latent. Computes:

        epsilon = L^{-1} @ (f - mu)

    where L is the Cholesky factor of the covariance.

    Args:
        latent: Latent GP values f, shape (..., n).
            Supports batch dimensions.
        mean: Constant mean of the GP.
        cholesky_cov: Lower Cholesky factor L of covariance, shape (n, n).

    Returns:
        Whitened variables epsilon, shape (..., n).
    """
    centered = latent - mean
    n = cholesky_cov.shape[0]

    # Handle batch dims by flattening, solving, then reshaping
    if centered.ndim == 1:
        # Single input: direct solve
        return slinalg.solve_triangular(cholesky_cov, centered, lower=True)
    else:
        # Batch input: flatten batch dims, solve as matrix, reshape back
        batch_shape = centered.shape[:-1]
        centered_flat = centered.reshape(-1, n).T  # (n, batch_size)
        result_flat = slinalg.solve_triangular(cholesky_cov, centered_flat, lower=True)
        return result_flat.T.reshape(*batch_shape, n)

