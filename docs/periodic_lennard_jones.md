# PeriodicLennardJones

A Boltzmann distribution over Lennard-Jones particles in a cubic periodic box, the bulk-phase counterpart to the free-cluster `LennardJones`. This is the regime used for atomic-solid Boltzmann generators: at liquid/solid density the particles form an FCC crystal, and the periodic box with minimum-image interactions matches the conventions of molecular-dynamics references.

## Mathematical definition

The target density is

$$
p(x) \propto \exp(-\beta \, U(x)), \qquad U(x) = \sum_{i<j} u(r_{ij}),
$$

where $r_{ij}$ is the minimum-image distance and $u$ is a soft-core, truncated, optionally shifted Lennard-Jones pair potential:

$$
g(r) = \left(\frac{r}{\sigma}\right)^6 + \tfrac{1}{2}(1 - \lambda)^2, \qquad u(r) = 4\lambda\varepsilon\left(\frac{1}{g^2} - \frac{1}{g}\right) \quad (r \le r_c),
$$

and $u(r) = 0$ for $r > r_c$. For $\lambda = 1$ this is the standard 12-6 potential $4\varepsilon[(\sigma/r)^{12} - (\sigma/r)^6]$; $\lambda \in (0, 1)$ softens the $r \to 0$ singularity for training stability. With `shift_energy=True` the potential is shifted so $u(r_c) = 0$.

The minimum-image convention is exact only when $r_c \le L/2$; construction does not enforce this, since some callers deliberately use a larger cutoff.

## Symmetries

- **Permutation invariance**: relabelling particles does not change the energy.
- **Periodic translational invariance**: shifting all particles, or any single particle by a box vector, preserves the energy.

## Why it's the solid regime

At the default density $\rho = 1.28$ and $\beta = 0.5$ the system is a Lennard-Jones solid: particles localise near FCC lattice sites and the distribution is sharply peaked, unlike the free cluster's rugged multi-funnel landscape. The challenge for a flow is the periodic topology (the density must be periodic) and the steep repulsive core, not multimodality.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_particles` | `256` | Number of particles |
| `spatial_dim` | `3` | Spatial dimension per particle |
| `box_length` | $\approx 5.848$ | Cubic box side (default is $N = 256$ at $\rho = 1.28$) |
| `epsilon` | `1.0` | LJ well depth |
| `sigma` | `1.0` | LJ length scale |
| `cutoff` | `2.7` | Radial cutoff; pairs beyond it do not interact |
| `beta` | `0.5` | Inverse temperature (bgmat LJ-solid default) |
| `lambda_lj` | `1.0` | Soft-core parameter in $(0, 1]$ (1 = hard LJ) |
| `min_distance` | `0.0` | Squared distance clipped to `min_distance**2` |
| `linearize_below` | `None` | If set, linearise the potential in $r$ below this distance |
| `shift_energy` | `True` | Shift so $u(r_c) = 0$ |

Use `PeriodicLennardJones.from_density(n_particles, density, ...)` to set the box length from a target number density.

## Usage

Input shape is `(..., n_particles, 3)`; `log p(x) = -beta * U(x) + const`.

```python
from jax_pdf import PeriodicLennardJones
import jax

# N=256 LJ solid at density 1.28 (default)
lj = PeriodicLennardJones()
print(lj.dim, lj.box_length)        # 768  5.848...

# Build from a target number density
lj = PeriodicLennardJones.from_density(n_particles=500, density=1.28)

# Training-stable soft-core variant
lj_soft = PeriodicLennardJones(n_particles=256, lambda_lj=0.9, min_distance=0.5)

x = jax.random.uniform(jax.random.PRNGKey(0), (256, 3), maxval=lj.box_length)
log_p = lj(x)
grad = jax.grad(lj)(x)              # (256, 3)
```

## References

- Wirnsberger, P., Papamakarios, G., Ibarz, B., Racaniere, S., Ballard, A. J., Pritzel, A., and Blundell, C. (2022). Normalizing flows for atomic solids. *Machine Learning: Science and Technology*, 3(2), 025009.
