# LGCP (Log Gaussian Cox Process)

A Log Gaussian Cox Process on the Finnish Pines dataset, used as a spatial statistics benchmark. This is an unnormalized posterior density: `log_normalization()` raises `NotImplementedError` and no exact sampler is available.

## Mathematical definition

$$
f \sim \mathcal{GP}(\mu, K), \quad n_i \mid f \sim \text{Poisson}(A \cdot e^{f_i})
$$

where $f$ is the latent log-intensity field on a discretized grid, $K$ is an exponential covariance kernel, and $n_i$ are observed counts per cell.

## Why it's hard

The GP prior induces strong correlations between neighboring grid cells. Standard samplers struggle with this correlated, high-dimensional geometry. The default 40x40 grid gives a 1600-dimensional problem. The whitened parameterization decorrelates the prior, making HMC more efficient, but the posterior correlations remain challenging.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_dim` | `40` | Grid cells per dimension. Total latent dim = `grid_dim`². |
| `whitened` | `False` | If `True`, parameterize in whitened space (easier geometry for HMC). |

## Usage

```python
from jax_pdf import LGCP
import jax

lgcp = LGCP(grid_dim=40)
print(f"Dimension: {lgcp.dim}")  # 1600

# Log probability
x = jax.numpy.zeros(lgcp.dim)
log_prob = lgcp(x)

# Gradient
grad = jax.grad(lgcp)(x)
```

Whitened parameterization (decorrelates the GP prior):

```python
lgcp_white = LGCP(grid_dim=40, whitened=True)
```

## Additional methods

LGCP provides optimization utilities beyond the core interface:

```python
# MAP estimate with optimization trajectory
result = lgcp.map_estimate()
x_map = result["x"]
print(f"Converged in {result['n_iters']} iterations")

# Hessian at any point
H = lgcp.hessian_at(x_map)

# Laplace approximation (Gaussian at MAP)
laplace = lgcp.laplace_approximation()
mu, cov = laplace["mu"], laplace["cov"]
```

## MAP estimate visualization

![MAP estimate of the log-intensity field](../examples/log_gaussian_pine_map_estimate.png)

## References

- Moller, J., Syversveen, A. R., and Waagepetersen, R. P. (1998). Log Gaussian Cox processes. *Scandinavian Journal of Statistics*, 25(3), 451--482. Introduced the LGCP model.
- Girolami, M. and Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. *Journal of the Royal Statistical Society: Series B*, 73(2), 123--214. Popularized this LGCP setup as a standard HMC benchmark.
