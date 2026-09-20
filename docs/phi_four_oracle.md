# PhiFourChainOracle

Reference values for a one-dimensional phi-four chain: $\log Z$, site marginals, two-point functions, and exact draws, with no Monte Carlo anywhere.

A sampler study needs something to be measured against.
On a 2D lattice nothing is available in closed form, so studies fall back on long reference runs, which carry their own error.
In one dimension the coupling is nearest-neighbour, so the chain is a Markov chain in the site index, and discretizing the field on a quadrature grid turns every quantity into linear algebra on a transfer matrix.
That makes a 1D chain the place to develop and validate a method before moving to a lattice where truth is out of reach.

The oracle lives outside the distribution interface on purpose: its answers are controlled quadrature approximations, not closed forms, and `log_normalization` on the distributions keeps its "exact or raise" contract.

## What it solves

For a chain with sites $i = 0, \dots, N-1$,

$$
S(x) = u \sum_i (x_i^2 - a^2)^2 + \frac{\kappa}{2} \sum_{\text{bonds}} (x_i - x_j)^2 - h \sum_i x_i
$$

with either periodic bonds (a ring, $N$ bonds) or Dirichlet ends (the field pinned to zero just outside each end, $N+1$ bonds).
This is [`LatticePhiFour`](lattice_phi_four.md) on a one-axis lattice, and [`PhiFour`](phi_four.md) under the mapping in its documentation; `from_distribution` accepts either.

Writing $L(\phi) = e^{-u(\phi^2-a^2)^2 + h\phi}$ for the local factor and $B(\phi, \phi') = e^{-\frac{\kappa}{2}(\phi-\phi')^2}$ for a bond, the ring's partition function is $Z = \operatorname{tr}(A^N)$ with $A$ the transfer matrix that carries a bond and half of each neighbouring site's factor, and the Dirichlet chain's is a product of messages passed along the chain.
Marginals, correlations and conditionals follow from the same objects.

## What is exact and what is not

The field at a site is represented on `n_grid` points spanning `[-bound, bound]` under the trapezoid rule.
Everything else is linear algebra, so the only error is that discretization, and `discretization_report` measures it rather than assuming it: it recomputes $\log Z$ at twice the resolution and at twice the extent, and reports the density at the boundary relative to its maximum.

For a chain of 32 sites with `n_grid=2001` and `bound=3.0`, those three numbers are $5 \cdot 10^{-10}$, $4 \cdot 10^{-6}$ and $2 \cdot 10^{-6}$.
The tests check the operator against dense quadrature over every configuration of a two- and three-site chain (agreement to $10^{-10}$) and against the closed form of the Gaussian chain that $u = 0$ leaves behind.

The sampler is exact for the discretized chain: sites are drawn one at a time from their exact conditionals, with no Metropolis step and no burn-in, and the value at a site is a grid point jittered inside its cell.
Its law is checked by enumerating every configuration of a small chain, not by comparing histograms.

## API

| Method | Returns |
|--------|---------|
| `log_partition()` | $\log Z$ of the chain |
| `site_marginal(site=0)` | the marginal density of one site, on the grid |
| `mean_field(site=0)` | $E[x_i]$, exactly zero at $h = 0$ |
| `two_point(separation, site=0)` | $E[x_i x_j]$ for sites a given number of bonds apart |
| `sample(rng, n)` | fields of shape `(n, n_sites)`, exact for the discretized chain |
| `discretization_report()` | the measured effect of resolution, extent, and the tail |
| `grid` | the quadrature points |

Construction takes `u`, `a`, `kappa`, `h`, `n_sites`, `periodic`, `n_grid`, `bound`, or use `PhiFourChainOracle.from_distribution(dist, n_grid=..., bound=...)`.
`u = 0` is allowed here, which leaves a Gaussian chain.

## Usage

```python
import numpy as np
from jax_pdf import PhiFour
from jax_pdf.phi_four_oracle import PhiFourChainOracle

dist = PhiFour(a=0.1, b=0.0, dim_grid=32, beta=1.0, periodic=True)
oracle = PhiFourChainOracle.from_distribution(dist, n_grid=2001, bound=3.0)

oracle.log_partition()       # 6.86449448
oracle.mean_field()          # 1.4e-15, and exactly 0 by symmetry
oracle.two_point(1)          # 0.534797
oracle.two_point(8)          # 0.114553
oracle.discretization_report()
```

Exact draws, for a reference dataset or for a forward-KL check of a learned sampler:

```python
oracle = PhiFourChainOracle.from_distribution(dist, n_grid=301, bound=3.0)
x = oracle.sample(np.random.default_rng(0), 4000)   # (4000, 32), a fraction of a second
```

A tilted chain, where the two phases carry unequal mass:

```python
from jax_pdf import LatticePhiFour

tilted = LatticePhiFour(u=0.5, a=1.0, kappa=2.0, h=0.05, lattice_shape=(24,))
PhiFourChainOracle.from_distribution(tilted, n_grid=801).mean_field()   # 0.166
```

## Cost and limits

Building the operator is $O(\texttt{n\_grid}^2)$ in memory.
A ring's $\log Z$ needs the eigenvalues of that matrix, which is $O(\texttt{n\_grid}^3)$; at `n_grid=2001` that is under a second.
A Dirichlet chain needs only matrix-vector products.
Sampling a ring caches the transfer powers, which costs `n_sites * n_grid**2` numbers; the call raises `MemoryError` naming `n_grid` rather than exhausting memory.

One dimension only: a 2D lattice is not a Markov chain in any single index, and nothing here extends to it.
The law of the global magnetization is not provided in closed form; draw from `sample` and estimate it, remembering that this estimate, unlike the rest, carries Monte Carlo error.

## References

- Kramers, H. A. and Wannier, G. H. (1941). Statistics of the two-dimensional ferromagnet. *Physical Review*, 60, 252. The transfer-matrix method.
- Albergo, M. S., Kanwar, G., and Shanahan, P. E. (2019). Flow-based generative models for Markov chain Monte Carlo in lattice field theory. *Physical Review D*, 100, 034515.
