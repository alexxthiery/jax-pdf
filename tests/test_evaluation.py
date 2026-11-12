"""Exact small-array oracles for optional offline sample comparisons."""

import numpy as np
import pytest

from jax_pdf.evaluation import (
    category_probabilities, jensen_shannon_divergence,
    periodic_histogram2d, wasserstein_distance_1d,
)


def test_category_tables_preserve_missing_modes_and_batch_axes():
    labels = np.array([[[0, 1, 1, 3], [2, 2, 0, 2]]])
    np.testing.assert_array_equal(category_probabilities(labels, 4),
                                  [[[.25, .5, 0., .25], [.25, 0., .75, 0.]]])
    np.testing.assert_array_equal(category_probabilities(labels.ravel(), 4),
                                  [.25, .25, .375, .125])


def test_joint_histogram_has_fixed_orientation_and_identifies_seam():
    angles = np.array([[-np.pi, -np.pi], [np.pi, np.pi],
                       [-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2]])
    np.testing.assert_array_equal(periodic_histogram2d(angles, bins=2),
                                  [[.5, .5], [0., 0.]])
    np.testing.assert_array_equal(periodic_histogram2d(angles+2*np.pi, bins=2),
                                  periodic_histogram2d(angles, bins=2))


def test_js_anchors_and_missing_mode_sensitivity():
    p, q = np.array([.25, .75]), np.array([1., 0.])
    expected = .5*(.25*np.log(.25/.625)+.75*np.log(.75/.375)+np.log(1/.625))
    assert jensen_shannon_divergence(p, q) == pytest.approx(expected)
    assert jensen_shannon_divergence(p, p) == 0.
    assert jensen_shannon_divergence([1., 0.], [0., 1.]) == pytest.approx(np.log(2))
    assert jensen_shannon_divergence(p, q) == jensen_shannon_divergence(q, p)


def test_wasserstein_unequal_counts_duplicates_and_units():
    # CDF gaps: 1/2 on [0,1], 1/6 on [1,2], 1/3 on [2,3].
    assert wasserstein_distance_1d([0., 2.], [1., 1., 3.]) == pytest.approx(1.)
    assert wasserstein_distance_1d([2., 2.], [2.]) == 0.
    assert wasserstein_distance_1d([3., 5.], [4., 4., 6.]) == pytest.approx(1.)
    assert wasserstein_distance_1d([0., 4.], [2., 2., 6.]) == pytest.approx(2.)


@pytest.mark.parametrize("p,q", [([.2, .2], [.5, .5]), ([-.1, 1.1], [.5, .5]),
                                  ([np.nan, 1.], [.5, .5]), ([1.], [.5, .5])])
def test_bad_probabilities_raise(p, q):
    with pytest.raises(ValueError):
        jensen_shannon_divergence(p, q)


def test_bad_samples_and_labels_raise_instead_of_being_filtered():
    with pytest.raises(ValueError, match="finite"):
        wasserstein_distance_1d([0., np.nan], [1.])
    with pytest.raises(ValueError, match="nonempty"):
        periodic_histogram2d(np.empty((0, 2)))
    with pytest.raises(ValueError, match="shape"):
        periodic_histogram2d(np.zeros((3, 1)))
    with pytest.raises(ValueError, match="positive integer"):
        periodic_histogram2d(np.zeros((3, 2)), bins=True)
    with pytest.raises(ValueError, match="must lie"):
        category_probabilities(np.array([0, 3]), 3)
    with pytest.raises(TypeError, match="integers"):
        category_probabilities([0., 1.], 2)
