# jax-pdf

Probability density functions for MCMC and variational inference testing.

`jax-pdf` provides a collection of standard benchmark distributions used to test sampling algorithms. Each distribution is callable and returns log probability, with full support for JAX transformations (grad, vmap, jit).

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
