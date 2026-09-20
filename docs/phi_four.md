# PhiFour

The phi-four ($\phi^4$) lattice field theory on a 1D lattice, a standard benchmark from statistical physics for MCMC samplers that must handle multimodality and long-range correlations.

## Mathematical definition

The unnormalized log-density is

$$
\log p(\phi) = -\beta U(\phi)
$$

where the energy combines nearest-neighbor coupling with a local double-well potential:

$$
U(\phi) = c \sum_{i} \frac{(\phi_{i+1} - \phi_i)^2}{2} + \frac{1}{c} \sum_{i} \left[\frac{(1 - \phi_i^2)^2}{4} + b \phi_i\right]
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

Measured along the uniform field $\phi_i \equiv m$, the barrier between a mode and the symmetric point is

$$
\log p(\phi \equiv \pm 1) - \log p(\phi \equiv 0) = \frac{\beta}{4a}
$$

which does not depend on the number of sites: with $c = a d$ the local term is divided by the lattice size, so adding sites refines a fixed continuum field instead of raising the barrier.

Difficulty scales with:

- **`beta`**: the barrier grows in proportion, and the modes sharpen
- **`a`**: the barrier grows as $1/a$, and neighbor correlations strengthen
- **`dim_grid`**: sets the dimension of the sampling problem and its cost, not the height of the barrier

When $b = 0$ the two modes carry exactly equal mass by symmetry, which is an exact check available to any sampler.

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

The normalizing constant is intractable in closed form.
Calling `log_normalization()` raises `NotImplementedError`, and the distribution has no `sample` method.

A chain is exactly solvable numerically, though: [`PhiFourChainOracle`](phi_four_oracle.md) gives $\log Z$, site marginals, two-point functions and exact draws by transfer operator, to a discretization error it reports.

## References

- Albergo, M. S., Kanwar, G., and Shanahan, P. E. (2019). Flow-based generative models for Markov chain Monte Carlo in lattice field theory. *Physical Review D*, 100, 034515. Established $\phi^4$ lattice field theory as a sampling benchmark (2D lattice; the 1D variant here is a direct restriction).
- Montvay, I. and Munster, G. (1994). *Quantum Fields on a Lattice*. Cambridge University Press. Textbook reference for lattice $\phi^4$ field theory.
