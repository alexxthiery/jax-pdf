# NealFunnel

Neal's funnel distribution, a classic multi-scale MCMC benchmark in $D$ dimensions.

## Mathematical definition

$$
p(x) = \mathcal{N}(x_0; 0, \sigma^2) \prod_{i=1}^{D-1} \mathcal{N}(x_i; 0, e^{x_0})
$$

The first coordinate $x_0$ controls the scale of all others. When $x_0$ is large, the remaining coordinates spread out; when $x_0$ is negative, they concentrate near zero.

## Why it's hard

The funnel requires adapting to vastly different scales. In the narrow neck ($x_0 < 0$), step sizes must be tiny; in the wide mouth ($x_0 > 0$), they can be large. Standard MCMC with fixed step sizes either moves too slowly in the mouth or diverges in the neck. This multi-scale geometry is a fundamental challenge for HMC and related methods.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | `10` | Dimensionality (must be >= 2). |
| `sigma` | `3.0` | Std dev of $x_0$. Larger values create wider scale ranges. |

## Usage

```python
from jax_pdf import NealFunnel
import jax

funnel = NealFunnel(dim=10, sigma=3.0)

# Log probability
x = jax.numpy.zeros(funnel.dim)
log_p = funnel(x)

# Gradient
grad = jax.grad(funnel)(x)

# Exact samples
key = jax.random.PRNGKey(0)
samples = funnel.sample(key, 1000)  # shape (1000, 10)
```

Varying difficulty:

```python
funnel_2d = NealFunnel(dim=2)              # lower dimensional, easier
funnel_narrow = NealFunnel(dim=10, sigma=1.0)  # narrower, easier
```
