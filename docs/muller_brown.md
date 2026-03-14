# MullerBrown

The Muller-Brown potential energy surface as a 2D log-density, a classic benchmark for sampling algorithms that must mix across multiple modes separated by energy barriers.

## Mathematical definition

The unnormalized log-density is the negated potential:

$$
\log p(x, y) = -\beta \sum_{k=1}^{4} A_k \exp\!\bigl(a_k (x - x_k^0)^2 + b_k (x - x_k^0)(y - y_k^0) + c_k (y - y_k^0)^2\bigr)
$$

with standard coefficients from Muller and Brown (1979):

| $k$ | $A_k$ | $a_k$ | $b_k$ | $c_k$ | $x_k^0$ | $y_k^0$ |
|-----|--------|--------|--------|--------|----------|----------|
| 1   | $-200$ | $-1$   | $0$    | $-10$  | $1$      | $0$      |
| 2   | $-100$ | $-1$   | $0$    | $-10$  | $0$      | $0.5$    |
| 3   | $-170$ | $-6.5$ | $11$   | $-6.5$ | $-0.5$   | $1.5$    |
| 4   | $15$   | $0.7$  | $0.6$  | $0.7$  | $-1$     | $1$      |

## Why it's hard

Three local minima at approximately $(-0.558, 1.442)$, $(0.624, 0.028)$, and $(-0.050, 0.467)$ are separated by saddle-point barriers.
Samplers must cross these barriers to visit all modes.
Higher $\beta$ sharpens the wells and raises the barriers, making transitions exponentially rarer.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | `1.0` | Inverse temperature. Higher = sharper modes, harder sampling. |

## Usage

```python
from jax_pdf import MullerBrown
import jax

mb = MullerBrown(beta=1.0)

# Log-density (unnormalized)
x = jax.numpy.array([0.624, 0.028])
log_p = mb(x)

# Gradient
grad = jax.grad(mb)(x)

# Batch evaluation
xs = jax.numpy.zeros((100, 2))
log_ps = mb(xs)  # shape (100,)
```

Varying difficulty:

```python
easy = MullerBrown(beta=0.1)   # flatter landscape
hard = MullerBrown(beta=5.0)   # deep wells, high barriers
```

## Notes

The normalizing constant is intractable.
Calling `log_normalization()` raises `NotImplementedError`.
No exact sampler is available.
