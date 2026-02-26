# Banana2D

A banana-shaped 2D distribution (Rosenbrock-like), commonly used as an MCMC benchmark.

## Mathematical definition

$$
p(x_0, x_1) = \mathcal{N}(x_0; 1, 1) \cdot \mathcal{N}(x_1; x_0^2, \sigma^2)
$$

The distribution is centered around the parabola $x_1 = x_0^2$. The marginal in $x_0$ is a standard normal shifted to mean 1, while $x_1$ follows $x_0^2$ with Gaussian noise controlled by $\sigma$.

## Why it's hard

The curved geometry means gradient-based samplers must follow the parabola rather than move in straight lines. Smaller $\sigma$ creates a thinner, more tightly curved banana where the sampler needs very small steps to stay on the ridge. Standard HMC with fixed step sizes wastes many proposals.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma` | `0.1` | Thickness of the banana. Smaller = thinner, harder to sample. |

## Usage

```python
from jax_pdf import Banana2D
import jax

banana = Banana2D(sigma=0.1)

# Log probability
x = jax.numpy.array([1.0, 1.0])
log_p = banana(x)

# Gradient
grad = jax.grad(banana)(x)

# Exact samples
key = jax.random.PRNGKey(0)
samples = banana.sample(key, 1000)  # shape (1000, 2)
```

Varying difficulty:

```python
thin = Banana2D(sigma=0.01)   # harder
fat = Banana2D(sigma=1.0)     # easier
```
