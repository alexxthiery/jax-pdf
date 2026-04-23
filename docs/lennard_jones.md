# LennardJones

A Boltzmann distribution over a cluster of particles interacting via the Lennard-Jones 12-6 potential. Standard configurations are LJ13 (13 particles, dim=39) and LJ55 (55 particles, dim=165).

## Mathematical definition

The target density is

$$
p(x) \propto \exp(-\beta \, U(x))
$$

where the energy has two terms:

$$
U(x) = U_{\text{LJ}}(x) + U_{\text{trap}}(x)
$$

### Lennard-Jones pair potential

Summed over unique pairs ($i < j$):

$$
U_{\text{LJ}}(x) = \sum_{i < j} \varepsilon \left[\left(\frac{r_m}{r_{ij}}\right)^{12} - 2\left(\frac{r_m}{r_{ij}}\right)^6\right]
$$

Each pair interaction has minimum $-\varepsilon$ at distance $r = r_m$. The $r^{-12}$ term is short-range repulsion; the $-2r^{-6}$ term is long-range attraction. No cutoff distance is applied (these are isolated clusters, not bulk simulations).

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
p(x) \propto \underbrace{e^{-\beta U_{\text{LJ}}(x_{\text{internal}})}}_{\text{LJ on internal coords}} \times \underbrace{e^{-\frac{\beta s N}{2} \lVert \bar{x} \rVert^2}}_{\text{Gaussian on COM}}
$$

This means the statistics of relative particle positions are identical to those of the pure (trapless) distribution with COM fixed to zero. Standard in physics/chemistry simulations.

Set `trap_scale=0` to recover the pure LJ potential (improper, with `spatial_dim` flat translational directions).

## Symmetries

- **Permutation invariance**: relabeling particles does not change the energy.
- **Translational invariance** (without trap): shifting all particles by the same vector preserves pairwise distances.

## Why it's hard

LJ clusters have a rugged energy landscape with many local minima separated by high barriers. LJ13 has the icosahedral global minimum at $U \approx -44.33$ but hundreds of other local minima. LJ55 is far harder, with an astronomically larger number of local minima. The $r^{-12}$ repulsive core creates steep gradients at short range while the attractive well creates long-range correlations.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_particles` | `13` | Number of particles |
| `spatial_dim` | `3` | Spatial dimension per particle (2D or 3D) |
| `epsilon` | `1.0` | LJ well depth |
| `rm` | `1.0` | Equilibrium distance (potential minimum at $r = r_m$) |
| `trap_scale` | `1.0` | Harmonic trap strength (0 to disable) |
| `trap_mode` | `'individual'` | `'individual'`: per-particle trap; `'com'`: center-of-mass only |
| `beta` | `1.0` | Inverse temperature |

## Usage

Input shape is `(..., n_particles, spatial_dim)`. The `dim` property still reports the flat DoF count for callers that need it.

```python
from jax_pdf import LennardJones
import jax

# LJ13 (default)
lj13 = LennardJones()
print(lj13.dim)                                      # 39
print(lj13.n_particles, lj13.spatial_dim)            # 13 3

# LJ55
lj55 = LennardJones(n_particles=55)
print(lj55.dim)                                      # 165

# Log-density and gradient
x = jax.random.normal(jax.random.PRNGKey(0), (13, 3))
log_p = lj13(x)
grad = jax.grad(lj13)(x)                             # shape (13, 3)

# Batch evaluation
xs = jax.random.normal(jax.random.PRNGKey(1), (100, 13, 3))
log_ps = lj13(xs)                                    # shape (100,)
```

Varying difficulty:

```python
cold = LennardJones(beta=5.0)    # sharper landscape, harder
hot = LennardJones(beta=0.1)     # flatter, easier
no_trap = LennardJones(trap_scale=0.0)  # pure LJ (improper)
```

## References

- Wales, D. J. and Doye, J. P. K. (1997). Global optimization by basin-hopping and the lowest energy structures of Lennard-Jones clusters containing up to 110 atoms. *Journal of Physical Chemistry A*, 101(28), 5111--5116.
- Midgley, L. I., Stimper, V., Simm, G. N. C., Schoelkopf, B., and Hernandez-Lobato, J. M. (2023). Flow annealed importance sampling bootstrap. *ICLR 2023*.
