# HarmonicCrystal

An Einstein (harmonic) crystal: independent harmonic wells pinning each particle to a fixed lattice site. Unlike every other particle target here, its normalizing constant is analytic, which is the whole point. It exists as a free-energy sanity oracle for Boltzmann-generator pipelines: a flow plus importance reweighting must recover `log_normalization()`, so a harmonic crystal whose well width matches the flow's Gaussian base gives an end-to-end check against an exact answer.

## Mathematical definition

Each particle $i$ sits in an isotropic harmonic well centred at its lattice site $s_i$:

$$
U(x) = \frac{k}{2} \sum_{i=1}^{N} \lVert x_i - s_i \rVert^2, \qquad p(x) \propto \exp(-\beta U(x)).
$$

The wells are independent and Gaussian, so the partition function factorises over the $Nd$ degrees of freedom:

$$
\log Z = \frac{Nd}{2} \log\!\left( \frac{2\pi}{\beta k} \right).
$$

This is the **unbounded** (whole-space) reference, exact when the wells are narrow relative to the box. With `box_length` set, displacements use the minimum-image convention (a periodic harmonic crystal); the analytic $\log Z$ above still applies in the narrow-well regime, where the wrapped Gaussian is indistinguishable from the unbounded one.

The Boltzmann width of each well is $\sigma_{\text{well}}^2 = 1/(\beta k)$. Matching it to a flow's Gaussian base (set $k = 1/(\beta\,\sigma_{\text{base}}^2)$) makes the importance weights constant, so the free-energy estimate has near-zero variance: the strongest possible pipeline check.

## Symmetries

- The fixed sites $s_i$ deliberately break both permutation and translational invariance (the crystal is pinned). Permutation symmetry holds only if the site set itself is permutation-symmetric.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `positions` | required | Well centres $s_i$, shape `(n_particles, spatial_dim)` |
| `spring_constant` | `1.0` | Harmonic stiffness $k$ ($> 0$) |
| `beta` | `1.0` | Inverse temperature ($> 0$) |
| `box_length` | `None` | Cubic box side for minimum-image displacements, or `None` for unbounded |

## Usage

Input shape is `(..., n_particles, spatial_dim)`; `log p(x) = -beta * U(x)`.

```python
from jax_pdf import HarmonicCrystal
import jax

sites = jax.random.uniform(jax.random.PRNGKey(0), (32, 3), maxval=6.0)
hc = HarmonicCrystal(positions=sites, spring_constant=400.0, beta=1.0)

print(hc(sites))                    # 0.0  (energy minimum at the sites)
print(hc.log_normalization())       # analytic log Z

# Match a flow's Gaussian base of width sigma to get constant IS weights
sigma = 0.05
hc = HarmonicCrystal(positions=sites, spring_constant=1.0 / sigma**2, beta=1.0,
                     box_length=6.0)
```

## References

- Frenkel, D. and Ladd, A. J. C. (1984). New Monte Carlo method to compute the free energy of arbitrary solids. Application to the fcc and hcp phases of hard spheres. *The Journal of Chemical Physics*, 81(7), 3188--3193.
