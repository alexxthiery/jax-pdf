"""Small independent examples for LGCP's binning and linear transforms."""

import jax.numpy as jnp
import numpy as np

from jax_pdf import cox_process_utils as utils


def test_binning_preserves_counts_and_assigns_upper_boundary_to_last_cell():
    """Asymmetric cell counts catch swapped axes as well as lost boundary points."""
    points = np.array([[0, 0], [0.49, 0.49], [0.5, 0], [0.5, 0.5], [1, 1]])
    np.testing.assert_array_equal(utils.compute_bin_counts(points, 2), [[2, 0], [1, 2]])
    np.testing.assert_array_equal(utils.compute_bin_counts(np.empty((0, 2)), 2),
                                  np.zeros((2, 2)))


def test_whitening_matches_hand_solved_nondiagonal_system_with_batch_axes():
    """L=[[2,0],[1,3]], mu=4 maps (1,2) to (6,11), (-1,1) to (2,6).

    A round trip alone would miss using the same wrong transpose both ways;
    check each direction against the independently computed values.
    """
    chol = jnp.array([[2.0, 0.0], [1.0, 3.0]])
    white = jnp.array([[[1.0, 2.0], [-1.0, 1.0]]])
    latent = jnp.array([[[6.0, 11.0], [2.0, 6.0]]])
    np.testing.assert_allclose(utils.whiten_to_latent(white, 4.0, chol), latent)
    np.testing.assert_allclose(utils.latent_to_whiten(latent, 4.0, chol), white)
    np.testing.assert_allclose(utils.latent_to_whiten(latent[0, 0], 4.0, chol), white[0, 0])
