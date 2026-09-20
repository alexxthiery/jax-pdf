# LatticePhiFour

Phi-four ($\phi^4$) field theory on a periodic lattice of any dimension, written in the parameters of the action.
The 2D case is the standard lattice field theory benchmark for flow-based samplers.

## Mathematical definition

The unnormalized log-density is

$$
\log p(x) = -S(x)
$$

with the action

$$
S(x) = u \sum_v (x_v^2 - a^2)^2 + \frac{\kappa}{2} \sum_{\langle v, w \rangle} (x_v - x_w)^2 - h \sum_v x_v
$$

where $v$ runs over the sites of a periodic lattice with sides `lattice_shape`, and $\langle v, w \rangle$ over nearest-neighbour bonds, each counted once.
A field is a flat vector of length $d = \prod_i L_i$, reshaped over the lattice in C order.

The lattice wraps along every axis, so a side of length 2 carries two bonds between its two sites, one each way around the ring.

## Why it's hard

The local term has minima at $x_v = +a$ and $x_v = -a$, and the coupling penalizes variation between neighbours, so the field tends to sit near $+a$ or near $-a$ across the whole lattice.
Moving between those two states means flipping every site, which no local move does.

Along the uniform field $x_v \equiv m$ the coupling vanishes and the action is $V u (m^2 - a^2)^2 - h V m$ over $V$ sites, so at $h = 0$ the barrier between a phase and the symmetric point is

$$
S(x \equiv 0) - S(x \equiv \pm a) = V u a^4
$$

which grows in proportion to the number of sites.
A path through domain walls is usually cheaper than the uniform one: separating the two phases costs roughly $\kappa$ times the area of the wall, which grows like $L^{D-1}$ rather than like $V$.
Either way, difficulty grows with the lattice, unlike `PhiFour`, whose continuum normalization keeps the barrier fixed as sites are added.

At $h = 0$ the action is invariant under $x \to -x$, so the two phases carry exactly equal mass.
That gives an exact check for any sampler, and it also gives a cheap competing move (propose a global sign flip), which is worth remembering when this target is used to compare samplers.
Nonzero $h$ breaks the symmetry and makes the phase masses unequal.

## The parameters are redundant

Writing $x = a y$ turns the action into

$$
S(x; u, a, \kappa, h) = S(y; u a^4, 1, \kappa a^2, h a)
$$

so $a$ is a choice of field units rather than a fourth degree of freedom: every model in this family is one with $a = 1$.
A study that searches for a regime should fix $a = 1$ and vary $u$, $\kappa$ and $h$, and a study that prefers wells at $\pm a$ for readability should remember that $u$ and $\kappa$ carry the corresponding powers of $a$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `u` | `1.0` | Quartic strength. Larger deepens the wells and raises the barrier. |
| `a` | `1.0` | Well location: the local minima are at $\pm a$. |
| `kappa` | `1.0` | Nearest-neighbour coupling. Larger favours smoother fields and raises the cost of a domain wall. |
| `h` | `0.0` | External field. Zero keeps the sign symmetry; nonzero tilts toward one sign. |
| `lattice_shape` | `(16, 16)` | Sides of the periodic lattice. Static: it sets the input dimension $d = \prod_i L_i$. |

The defaults are neutral, not a recommended regime.
A regime with two well separated phases and useful overlap has to be chosen deliberately for the study at hand.

## Usage

```python
from jax_pdf import LatticePhiFour
import jax
import jax.numpy as jnp

dist = LatticePhiFour(u=0.5, a=1.0, kappa=2.0, lattice_shape=(16, 16))
dist.dim                      # 256

x = jnp.zeros(dist.dim)       # a flat field
log_p = dist(x)
grad = jax.grad(dist)(x)

xs = jnp.zeros((32, dist.dim))
log_ps = dist(xs)             # shape (32,)
```

Other lattices:

```python
chain = LatticePhiFour(lattice_shape=(64,))           # 1D, 64 sites
cube = LatticePhiFour(lattice_shape=(8, 8, 8))        # 3D, 512 sites
tilted = LatticePhiFour(h=0.05, lattice_shape=(16, 16))   # unequal phase masses
```

## Relation to `PhiFour`

[`PhiFour`](phi_four.md) is the same family on a 1D chain, with the wells pinned at $\pm 1$ and the quartic tied to the coupling through $c = a_{\text{jp}} d$.
The two agree exactly under

$$
\kappa = \beta c, \qquad u = \frac{\beta}{4c}, \qquad a = 1, \qquad h = -\frac{\beta b}{c}
$$

so `PhiFour(a=0.1, b=0.0, dim_grid=n, beta=1.0, periodic=True)` is `LatticePhiFour` with those values and `lattice_shape=(n,)`.
Use `PhiFour` for the continuum-normalized chain whose difficulty is fixed as sites are added, and `LatticePhiFour` when the action's parameters, or a lattice beyond one dimension, is what matters.

## Notes

The normalizing constant is intractable; `log_normalization()` raises `NotImplementedError`.
No exact sampler is available.

## References

- Albergo, M. S., Kanwar, G., and Shanahan, P. E. (2019). Flow-based generative models for Markov chain Monte Carlo in lattice field theory. *Physical Review D*, 100, 034515.
- Montvay, I. and Munster, G. (1994). *Quantum Fields on a Lattice*. Cambridge University Press.
