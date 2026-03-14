# jax-pdf

Benchmark probability density functions for MCMC and variational inference testing.

Each distribution is callable and returns log probability, with full support for JAX transformations (`grad`, `vmap`, `jit`).

## Installation

```bash
git clone https://github.com/alexxthiery/jax-pdf.git
cd jax-pdf
pip install -e .
```

## Unified interface

All distributions share a common core API:

```python
import jax

dist = SomeDistribution(...)

# Log probability: input (..., dim) -> output (...)
log_p = dist(x)

# Gradient
grad = jax.grad(dist)(x)

# Dimensionality
d = dist.dim

# Log normalizing constant
log_Z = dist.log_normalization()
```

Banana2D, NealFunnel, and DoubleWell support sampling:

```python
samples = dist.sample(jax.random.PRNGKey(0), 1000)  # shape (1000, dim)
```

DW4, LennardJones, LGCP, MullerBrown, and PhiFour are unnormalized: no sampling, and `log_normalization()` raises `NotImplementedError`.

## Distributions

| Distribution | Dim | Description | Docs |
|-------------|-----|-------------|------|
| `Banana2D` | 2 | Banana-shaped (Rosenbrock-like) distribution | [docs/banana.md](docs/banana.md) |
| `NealFunnel` | configurable | Multi-scale funnel distribution | [docs/neal_funnel.md](docs/neal_funnel.md) |
| `LGCP` | grid_dim^2 | Log Gaussian Cox Process on Finnish Pines | [docs/lgcp.md](docs/lgcp.md) |
| `MullerBrown` | 2 | Multimodal potential energy surface | [docs/muller_brown.md](docs/muller_brown.md) |
| `PhiFour` | configurable | 1D lattice field theory with double-well potential | [docs/phi_four.md](docs/phi_four.md) |
| `DoubleWell` | configurable | Product of 2D double-well pairs ($2^{D/2}$ modes) | [docs/double_well.md](docs/double_well.md) |
| `DW4` | 8 | Double-well pair potential over 4 particles in 2D | [docs/dw4.md](docs/dw4.md) |
| `LennardJones` | configurable | Lennard-Jones cluster with harmonic confinement (LJ13, LJ55) | [docs/lennard_jones.md](docs/lennard_jones.md) |

## API reference

Core methods shared by all distributions:

| Method | Signature | Returns |
|--------|-----------|---------|
| `__call__` | `(x: Array) -> Array` | Log probability. Input `(..., dim)`, output `(...)`. |
| `log_normalization` | `() -> float` | Log normalizing constant. Raises `NotImplementedError` if intractable. |
| `dim` | property | Dimensionality (int). |

Banana2D, NealFunnel, and DoubleWell also provide:

| Method | Signature | Returns |
|--------|-----------|---------|
| `sample` | `(key, n: int) -> Array` | `n` exact samples, shape `(n, dim)`. |

LGCP additionally provides `map_estimate()`, `hessian_at(x)`, `laplace_approximation()`, and a `pines_points` property. See [docs/lgcp.md](docs/lgcp.md) for details.
