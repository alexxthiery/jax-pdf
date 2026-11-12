# Comparing saved samples

`jax_pdf.evaluation` contains small NumPy functions for offline comparisons. It
does not import chemistry or any sampler, and is not re-exported from `jax_pdf`.

```python
from jax_pdf.evaluation import (
    category_probabilities, periodic_histogram2d,
    jensen_shannon_divergence, wasserstein_distance_1d,
)

probabilities = category_probabilities(labels, n_categories=4)
p = periodic_histogram2d(sample_angles, bins=32)
q = periodic_histogram2d(reference_angles, bins=32)
js = jensen_shannon_divergence(p, q)
energy_w1 = wasserstein_distance_1d(sample_energies, reference_energies)
```

Labels have shape `(..., n_samples)`; probabilities retain the leading axes.
Include a remainder category and retain zero probabilities for missing modes.
The two-angle histogram takes `(n_samples, 2)` radians, wraps both axes to
`[-pi, pi)` and returns probability masses, not density values. Fix the grid
before comparisons. JS divergence uses natural logarithms, has no pseudocounts,
and ranges from zero to log(2). It measures only the chosen projection.
Wasserstein-1 integrates the empirical CDF difference and permits unequal sample
counts; its result retains the input units.

Inputs must be nonempty and finite. Bad shapes, category labels and probability
masses raise. Invalid generated samples should be counted and investigated,
not silently filtered before scoring. The functions assume unweighted samples;
they do not calculate importance weights or claim independence.

These comparisons do not certify convergence. Inspect rare-region probabilities,
the variation across independently initialized runs and error versus compute.
Energy agreement alone can miss conformational trapping; separate marginals can
miss wrong dependence. A finite reference also has uncertainty: compare independent
reference subsets at comparable budgets where trajectory information permits.
No external reference dataset is installed or downloaded by these functions.
