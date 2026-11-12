"""Small NumPy comparisons for saved samples; no sampler or chemistry imports.

These are empirical comparisons, not convergence certificates. Histogram grids
and category definitions must be fixed across the compared experiments.
"""

from numbers import Integral

import numpy as np


def _finite(value, name, ndim=None):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be real numeric values.")
    if value.size == 0 or (ndim is not None and value.ndim != ndim):
        raise ValueError(f"{name} must be nonempty with {ndim if ndim is not None else 'unrestricted'} dimensions; got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite; invalid samples must be reported explicitly.")
    return value.astype(float)


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def category_probabilities(labels, n_categories):
    """Category frequencies along the last (sample) axis.

    Args:
        labels: Integer category labels, shape (..., n_samples), nonempty.
            Include an explicit remainder category in the labeling function.
        n_categories: Positive integer number of categories.

    Returns:
        Probabilities of shape (..., n_categories), including zeros for
        unvisited categories. Samples are not assumed independent.
    """
    n_categories = _count(n_categories, "n_categories")
    labels = np.asarray(labels)
    if labels.ndim < 1 or labels.size == 0:
        raise ValueError("labels must have nonempty shape (..., n_samples).")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must be integers.")
    if np.any((labels < 0) | (labels >= n_categories)):
        raise ValueError(f"labels must lie in [0, {n_categories}).")
    rows = labels.reshape(-1, labels.shape[-1])
    result = np.stack([np.bincount(row, minlength=n_categories)/len(row) for row in rows])
    return result.reshape(labels.shape[:-1]+(n_categories,))


def periodic_histogram2d(angles, bins=32):
    """Probability masses on a fixed square grid of the two-angle torus.

    Args:
        angles: Finite radians of shape (n_samples, 2); values are wrapped to
            [-pi, pi), identifying both sides of the angular seam.
        bins: Positive integer bins per angle. Keep fixed across comparisons.

    Returns:
        Nonnegative probability masses, shape (bins, bins), summing to one.
        Histogram axes retain input angle order.
    """
    bins = _count(bins, "bins")
    angles = _finite(angles, "angles", ndim=2)
    if angles.shape[1] != 2:
        raise ValueError(f"angles must have shape (n_samples, 2), got {angles.shape}.")
    wrapped = (angles+np.pi) % (2*np.pi)-np.pi
    counts, _, _ = np.histogram2d(wrapped[:, 0], wrapped[:, 1], bins=bins,
                                 range=((-np.pi, np.pi), (-np.pi, np.pi)))
    return counts/angles.shape[0]


def jensen_shannon_divergence(p, q):
    """Jensen-Shannon divergence of matching probability-mass arrays, in nats.

    Zero cells contribute zero without pseudocounts. Identical arrays give zero;
    disjoint distributions give log(2). Inputs must already sum to one.
    """
    p, q = _finite(p, "p"), _finite(q, "q")
    if p.shape != q.shape:
        raise ValueError(f"Expected matching probability shapes, got {p.shape} and {q.shape}.")
    for name, value in (("p", p), ("q", q)):
        if np.any(value < 0) or not np.isclose(value.sum(), 1., atol=1e-7, rtol=0):
            raise ValueError(f"{name} must be nonnegative probability masses summing to one.")
    mixture = (p+q)/2
    def relative_entropy(value):
        nonzero = value > 0
        return np.sum(value[nonzero]*np.log(value[nonzero]/mixture[nonzero]))
    return float((relative_entropy(p)+relative_entropy(q))/2)


def wasserstein_distance_1d(samples, reference):
    """Exact empirical 1D Wasserstein-1 distance for equally weighted samples.

    Args:
        samples: Nonempty finite vector, shape (n,).
        reference: Nonempty finite vector, shape (m,); m may differ from n.

    Returns:
        Scalar distance in the input units, integrating the absolute empirical
        CDF difference. This is not an importance-sampling ESS or a full-space
        distributional distance.
    """
    a = np.sort(_finite(samples, "samples", ndim=1))
    b = np.sort(_finite(reference, "reference", ndim=1))
    support = np.unique(np.concatenate([a, b]))
    difference = (np.searchsorted(a, support[:-1], side="right")/len(a)
                  - np.searchsorted(b, support[:-1], side="right")/len(b))
    return float(np.sum(np.abs(difference)*np.diff(support)))
