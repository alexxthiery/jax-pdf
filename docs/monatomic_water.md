# MonatomicWater

A Boltzmann distribution over monatomic water (mW), a single-site coarse-grained water model, in a cubic periodic box. mW reproduces the tetrahedral local order of water without explicit hydrogens by combining a short-range two-body term with a three-body term that penalises deviation from the tetrahedral angle. The periodic box and the Stillinger-Weber functional form make this the canonical periodic Boltzmann-generator target, the regime where a permutation- and translation-equivariant conditioner earns its keep.

## Mathematical definition

The target density is

$$
p(x) \propto \exp(-\beta \, U(x)), \qquad U(x) = U_2(x) + U_3(x).
$$

Distances use the minimum-image convention in a cubic box of side $L$. Write $\tilde{r} = r / \sigma$ for the reduced pair distance and $r_c = 1.8$ for the reduced cutoff.

### Two-body term

Summed over unique pairs $i < j$ with $\tilde{r}_{ij} < r_c$:

$$
U_2(x) = \sum_{i<j} A \varepsilon \left( \frac{B}{\tilde{r}_{ij}^4} - 1 \right) \exp\left( \frac{1}{\tilde{r}_{ij} - r_c} \right).
$$

The exponential cutoff factor and all its derivatives vanish as $\tilde{r} \to r_c^-$, so the energy is smooth with compact support.

**Training softening (optional).** With `min_distance` $= r_{\min}$ and `linearize_below` $= r_\ell$ (both in Angstrom), each pair distance is first clipped to $\max(r, r_{\min})$. The pair potential $\phi$ is then replaced below $r_\ell$ by its tangent line in $r$:

$$
\phi_\ell(r) = \phi(r_\ell) + (r - r_\ell) \phi'(r_\ell) \text{ for } r < r_\ell.
$$

This is bgmat's training energy (`min_distance=0.01`, `linearize_below=1.2`). It removes the steep core that otherwise dominates reverse-KL gradients, is continuous with a continuous first derivative at $r_\ell$, and leaves the three-body term unchanged. Use the exact energy (both options off) for ESS and free-energy estimates.

### Three-body term

For each centre $i$ and each unordered pair of neighbours $j < k$ within the cutoff,

$$
U_3(x) = \sum_i \sum_{j<k} \lambda \varepsilon \, (\cos\theta_{jik} - \cos\theta_0)^2 \, f(\tilde{r}_{ij}) \, f(\tilde{r}_{ik}), \qquad f(\tilde{r}) = \exp\left( \frac{\gamma}{\tilde{r} - r_c} \right),
$$

where $\theta_{jik}$ is the angle at the central particle and $\theta_0 = 109.47^\circ$ is the ideal tetrahedral angle. The term is zero when a triplet sits at the tetrahedral angle and grows quadratically away from it; this is what drives mW into an ice-like (diamond / hexagonal) crystal at low temperature.

## Symmetries

- **Permutation invariance**: relabelling particles does not change the energy.
- **Translational invariance**: the minimum-image energy is invariant under a global shift, and under shifting any particle by a box vector (periodicity).

## Why it's hard

The three-body term couples triplets, so the energy is not a sum of pair interactions: a conditioner must see local angular environments, not just distances. The crystal is periodic, so the flow density must be periodic too (circular couplings). The two-body core is steep, so for reverse-KL training, where the flow may sample overlapping particles, set `min_distance > 0` to bound the core (the three-body norm is floored internally to keep its gradient finite at coincidence).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_particles` | required | Number of particles |
| `box_length` | required | Cubic box side (minimum-image PBC) |
| `beta` | `1.0` | Inverse temperature, $1/kT$ in $1/(\text{kcal/mol})$ |
| `min_distance` | `0.0` | Two-body squared distance clipped to `min_distance**2` (set $> 0$ for training) |
| `linearize_below` | `None` | If set (Angstrom, $> 0$), two-body potential is linear in $r$ below it (training softening) |
| `spatial_dim` | `3` | Spatial dimension per particle |

Fixed mW constants (Molinero and Moore): $A = 7.0496$, $B = 0.6022$, $\gamma = 1.2$, $\varepsilon = 6.189$ kcal/mol, $\sigma = 2.3925$ Angstrom, $\lambda = 23.15$, $r_c = 1.8$.

## Usage

Input shape is `(..., n_particles, 3)`; `log p(x) = -beta * U(x)`.

```python
from jax_pdf import MonatomicWater
import jax

mw = MonatomicWater(n_particles=64, box_length=14.0, beta=0.5)
x = jax.random.uniform(jax.random.PRNGKey(0), (64, 3), maxval=14.0)
log_p = mw(x)                       # scalar
grad = jax.grad(mw)(x)              # (64, 3)

# Batch evaluation
xs = jax.random.uniform(jax.random.PRNGKey(1), (32, 64, 3), maxval=14.0)
log_ps = mw(xs)                     # (32,)

# Training-stable variant (bounded two-body core)
mw_train = MonatomicWater(n_particles=64, box_length=14.0, beta=0.5, min_distance=0.5)

# bgmat's training energy: clip at 0.01 A, linear in r below 1.2 A
mw_bgmat_train = MonatomicWater(n_particles=64, box_length=14.0, beta=0.5,
                                min_distance=0.01, linearize_below=1.2)
```

## References

- Molinero, V. and Moore, E. B. (2009). Water modeled as an intermediate element between carbon and silicon. *The Journal of Physical Chemistry B*, 113(13), 4008--4016. arXiv: 0809.2811.
- Stillinger, F. H. and Weber, T. A. (1985). Computer simulation of local order in condensed phases of silicon. *Physical Review B*, 31(8), 5262--5271.
