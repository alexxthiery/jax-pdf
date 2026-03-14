# PhiFour

The phi-four ($\phi^4$) lattice field theory on a 1D lattice, a standard benchmark from statistical physics for MCMC samplers that must handle multimodality and long-range correlations.

## Mathematical definition

The unnormalized log-density is

$$
\log p(\phi) = -\beta \, U(\phi)
$$

where the energy combines nearest-neighbor coupling with a local double-well potential:

$$
U(\phi) = c \sum_{i} \frac{(\phi_{i+1} - \phi_i)^2}{2} + \frac{1}{c} \sum_{i} \left[\frac{(1 - \phi_i^2)^2}{4} + b \, \phi_i\right]
$$

where $c = a \cdot d$ and $d$ is `dim_grid`.

Boundary conditions determine how the field behaves at the lattice edges:

- **Dirichlet** (`periodic=False`): $\phi_0 = \phi_{N+1} = 0$. The field is pinned to zero at both ends.
- **Periodic** (`periodic=True`): $\phi_0 = \phi_N$, $\phi_{N+1} = \phi_1$. The lattice wraps into a ring; no boundary effects.

## Why it's hard

The double-well potential $(1 - \phi^2)^2/4$ creates two energy minima near $\phi = +1$ and $\phi = -1$ at each lattice site.
Nearest-neighbor coupling penalizes spatial variation, so the field tends to be uniformly near $+1$ or $-1$ across all sites.
Samplers must flip the entire field between these two modes, which becomes exponentially harder as `dim_grid` grows.

When $b = 0$, the distribution has exact $\mathbb{Z}_2$ symmetry ($\phi \to -\phi$).
Nonzero $b$ breaks this symmetry, biasing toward one mode.

Difficulty scales with:

- **`dim_grid`**: more sites means a larger barrier between modes
- **`beta`**: higher inverse temperature sharpens modes and raises barriers
- **`a`**: smaller coupling constant strengthens neighbor correlations, increasing the effective barrier

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a` | `0.1` | Coupling constant. Smaller = stronger neighbor correlation. |
| `b` | `0.0` | External field bias. Breaks $\mathbb{Z}_2$ symmetry when nonzero. |
| `dim_grid` | `100` | Number of lattice sites (= distribution dimension). |
| `beta` | `1.0` | Inverse temperature. Higher = sharper modes, harder sampling. |
| `periodic` | `False` | `False` = Dirichlet BCs, `True` = periodic BCs. |

## Usage

```python
from jax_pdf import PhiFour
import jax

dist = PhiFour(a=0.1, dim_grid=100, beta=1.0)

# Log-density (unnormalized)
x = jax.numpy.zeros(100)
log_p = dist(x)

# Gradient
grad = jax.grad(dist)(x)

# Batch evaluation
xs = jax.numpy.zeros((50, 100))
log_ps = dist(xs)  # shape (50,)
```

Varying difficulty:

```python
easy = PhiFour(a=0.1, dim_grid=10, beta=0.5)
hard = PhiFour(a=0.01, dim_grid=200, beta=5.0)
```

Periodic boundary conditions:

```python
dist_pbc = PhiFour(a=0.1, dim_grid=100, periodic=True)
```

Breaking $\mathbb{Z}_2$ symmetry:

```python
biased = PhiFour(a=0.1, b=0.1, dim_grid=100)
```

## Notes

The normalizing constant is intractable.
Calling `log_normalization()` raises `NotImplementedError`.
No exact sampler is available.
