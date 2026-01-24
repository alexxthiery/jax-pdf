# jax-pdf

Probability density functions for MCMC and variational inference testing.

## Installation

```bash
pip install jax-pdf
```

## Quick Example

```python
import jax
from jax_pdf import Banana2D, NealFunnel

banana = Banana2D(sigma=0.1)
key = jax.random.PRNGKey(0)
samples = banana.sample(key, 1000)
```

## Documentation

Full documentation at: https://alexxthiery.github.io/jax-pdf/
