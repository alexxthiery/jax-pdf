# DW4

A Boltzmann distribution over 4 particles in 2D interacting via a quartic double-well pair potential.

## Mathematical definition

The target density is

$$
p(x) \propto \exp(-\beta \, U(x))
$$

where the energy has two terms:

$$
U(x) = U_{\text{DW}}(x) + U_{\text{trap}}(x)
$$

### Pair potential

Quartic double-well, summed over all 6 unique pairs ($i < j$):

$$
U_{\text{DW}}(x) = \sum_{i < j} \left[a(r_{ij} - r_0)^4 + b(r_{ij} - r_0)^2 + c\right]
$$

where $r_{ij} = \lVert x_i - x_j \rVert$ is the Euclidean distance between particles $i$ and $j$.

With the default parameters ($a = 0.9$, $b = -4$, $r_0 = 4$), each pair potential has two minima at $r \approx 2.51$ and $r \approx 5.49$, creating a multimodal distribution over particle configurations.

### Harmonic trap

The trap breaks translational invariance and makes the distribution proper (normalizable). Two modes are available via `trap_mode`:

**`'individual'` (default)** -- penalizes each particle's distance from the origin:

$$
U_{\text{trap}}(x) = \frac{s}{2} \sum_{i=1}^{N} \lVert x_i \rVert^2
$$

This affects both the center of mass and the internal spread. Used in ML benchmarks (e.g., FAB, normalizing flows).

**`'com'`** -- penalizes only the center of mass:

$$
U_{\text{trap}}(x) = \frac{s}{2} \, N \, \lVert \bar{x} \rVert^2
$$

where $\bar{x} = \frac{1}{N}\sum_i x_i$. The pair potential is unaffected, and the distribution factorizes:

$$
p(x) \propto \underbrace{e^{-\beta U_{\text{DW}}(x_{\text{internal}})}}_{\text{pair potential on internal coords}} \times \underbrace{e^{-\frac{\beta s N}{2} \lVert \bar{x} \rVert^2}}_{\text{Gaussian on COM}}
$$

This means the statistics of relative particle positions are identical to those of the pure (trapless) distribution with COM fixed to zero. Standard in physics/chemistry simulations.

Set `trap_scale=0` to recover the pure pairwise potential (improper, with 2 flat translational directions).

## Why it's hard

Each of the 6 particle pairs independently prefers one of two inter-particle distances, creating combinatorially many local modes separated by energy barriers.

## Symmetries

- **Permutation invariance**: relabeling particles does not change the energy.
- **Translational invariance** (without trap): shifting all particles by the same vector preserves pairwise distances.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a` | `0.9` | Quartic coefficient in pair potential |
| `b` | `-4.0` | Quadratic coefficient (negative = double well) |
| `c` | `0.0` | Constant offset in pair potential |
| `r0` | `4.0` | Distance offset (center of quartic) |
| `trap_scale` | `1.0` | Harmonic trap strength (0 to disable) |
| `trap_mode` | `'individual'` | `'individual'`: per-particle trap; `'com'`: center-of-mass only |
| `beta` | `1.0` | Inverse temperature |

## Usage

Input shape is `(..., 4, 2)` (4 particles, 2D). The `dim` property reports the flat DoF count (8).

```python
from jax_pdf import DW4
import jax

dist = DW4()
print(dist.dim)                                  # 8
print(dist.n_particles, dist.spatial_dim)        # 4 2

# Log-density
x = jax.numpy.ones((4, 2))
log_p = dist(x)

# Gradient
grad = jax.grad(dist)(x)                         # shape (4, 2)

# Batch evaluation
xs = jax.random.normal(jax.random.PRNGKey(0), (100, 4, 2))
log_ps = dist(xs)                                # shape (100,)
```

Varying difficulty:

```python
cold = DW4(beta=5.0)   # sharper modes, harder to mix
hot = DW4(beta=0.5)    # flatter landscape, easier
```

## References

- Midgley, L. I., Stimper, V., Simm, G. N. C., Schoelkopf, B., and Hernandez-Lobato, J. M. (2023). Flow annealed importance sampling bootstrap. *ICLR 2023*.
