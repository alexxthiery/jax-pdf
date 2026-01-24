# jax-pdf

Probability density functions for MCMC and variational inference testing.

`jax-pdf` provides a collection of standard benchmark distributions used to test sampling algorithms. Each distribution supports:

- Log probability evaluation with automatic differentiation
- Exact sampling (when analytically possible)
- Batch evaluation via JAX's vectorization

## Installation

```bash
pip install jax-pdf
```

Or install from source:

```bash
pip install git+https://github.com/alexxthiery/jax-pdf.git
```

## Quick Example

```python
import jax
import jax.numpy as jnp
from jax_pdf import Banana2D, NealFunnel

# Create a banana distribution
banana = Banana2D(sigma=0.1)

# Evaluate log probability
x = jnp.array([1.0, 1.0])
log_prob = banana(x)

# Compute gradient
grad = jax.grad(banana)(x)

# Draw samples
key = jax.random.PRNGKey(0)
samples = banana.sample(key, 1000)
```

## Distribution Interface

All distributions follow the same interface:

| Method | Description |
|--------|-------------|
| `__call__(x)` | Log probability density (supports batching) |
| `sample(key, n)` | Draw `n` exact samples |
| `log_normalization()` | Log normalizing constant |
| `dim` | Dimensionality (property) |
