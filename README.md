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

All distributions share a common core API. Input shape follows a two-tier convention:

- **Generic distributions** (`Banana2D`, `NealFunnel`, `LGCP`, `MullerBrown`, `PhiFour`, `DoubleWell`) take flat input `(..., dim)`.
- **Particle distributions** (`LennardJones`, `PeriodicLennardJones`, `MonatomicWater`, `HarmonicCrystal`, `DW4`) take structured input `(..., n_particles, spatial_dim)` because the particle axis is semantically meaningful and downstream consumers (normalising flows over particle systems) work in that shape natively. `dist.dim` still reports the flat DoF count (`n_particles * spatial_dim`); `n_particles` and `spatial_dim` are exposed as properties.

```python
import jax

dist = SomeDistribution(...)

# Log probability: output is scalar over the event axes.
log_p = dist(x)

# Gradient (grad.shape matches x.shape)
grad = jax.grad(dist)(x)

# Dimensionality (flat DoF count)
d = dist.dim

# Log normalizing constant
log_Z = dist.log_normalization()
```

Banana2D, NealFunnel, and DoubleWell support sampling:

```python
samples = dist.sample(jax.random.PRNGKey(0), 1000)  # shape (1000, dim)
```

DW4, LennardJones, PeriodicLennardJones, MonatomicWater, LGCP, MullerBrown, and PhiFour are unnormalized: no sampling, and `log_normalization()` raises `NotImplementedError`. `HarmonicCrystal` is the exception among the particle targets: its `log_normalization()` is analytic (it exists as a free-energy sanity oracle).

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
| `PeriodicLennardJones` | configurable | Lennard-Jones in a periodic box (minimum-image PBC; soft-core, cutoff, shift) | [docs/periodic_lennard_jones.md](docs/periodic_lennard_jones.md) |
| `MonatomicWater` | configurable | Monatomic (mW) water in a periodic box (Stillinger-Weber 2-body + 3-body tetrahedral-angle term) | [docs/monatomic_water.md](docs/monatomic_water.md) |
| `HarmonicCrystal` | configurable | Einstein/harmonic crystal with analytic free energy (Boltzmann-generator sanity oracle) | [docs/harmonic_crystal.md](docs/harmonic_crystal.md) |

## API reference

Core methods shared by all distributions:

| Method | Signature | Returns |
|--------|-----------|---------|
| `__call__` | `(x: Array) -> Array` | Log probability. Input `(..., dim)` for generic distributions or `(..., n_particles, spatial_dim)` for particle distributions. Output shape is the leading batch shape. |
| `log_normalization` | `() -> float` | Log normalizing constant. Raises `NotImplementedError` if intractable. |
| `dim` | property | Flat DoF count (int). |

Particle distributions (`LennardJones`, `PeriodicLennardJones`, `MonatomicWater`, `HarmonicCrystal`, `DW4`) additionally expose:

| Property | Type | Description |
|----------|------|-------------|
| `n_particles` | int | Number of particles. |
| `spatial_dim` | int | Spatial dimension per particle (2 or 3). |

Banana2D, NealFunnel, and DoubleWell also provide:

| Method | Signature | Returns |
|--------|-----------|---------|
| `sample` | `(key, n: int) -> Array` | `n` exact samples, shape `(n, dim)`. |

LGCP additionally provides `map_estimate()`, `hessian_at(x)`, `laplace_approximation()`, and a `pines_points` property. See [docs/lgcp.md](docs/lgcp.md) for details.
