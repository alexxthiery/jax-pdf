# Contributing

## Adding a new distribution

### 1. Create `jax_pdf/<name>.py`

Use `banana.py` as a template. The structure:

```python
"""One-line module docstring."""

import jax.numpy as jnp
from flax import struct
from jax import Array


@struct.dataclass
class MyDist:
    """One-line summary.

    Mathematical definition:
        p(x) = ...

    Intuitive description of the geometry and why it is
    challenging for samplers.

    Attributes:
        param: What it controls. Default: X.
    """

    param: float = 0.1
    """Inline docstring for VS Code hover."""

    def __post_init__(self):
        if self.param <= 0:
            raise ValueError(
                f"param must be positive, got {self.param}"
            )

    @property
    def dim(self) -> int:
        return 2

    def __call__(self, x: Array) -> Array:
        """Evaluate log probability density.

        Args:
            x: Input point(s) of shape (..., dim).

        Returns:
            Log probability density of shape (...).
        """
        ...

    def log_normalization(self) -> Array:
        """Log normalizing constant.

        Returns:
            Scalar log(Z). Return 0.0 if normalized.

        If the normalizing constant is intractable, raise
        NotImplementedError. Never return a partial or
        approximate value.
        """
        ...

    def sample(self, key: Array, n: int) -> Array:
        """Draw exact samples (omit if not available).

        Args:
            key: JAX PRNG key.
            n: Number of samples.

        Returns:
            Samples of shape (n, dim).
        """
        ...
```

Requirements:
- Use `@struct.dataclass` from Flax (not `@dataclass`)
- `__call__` returns log probability, supports batch via `x[..., i]`
- `dim` is a property, not a method
- `log_normalization()` returns 0.0 for normalized distributions; raises `NotImplementedError` if intractable
- `sample()` only if exact sampling is possible; omit otherwise
- `__post_init__` validates parameters with clear error messages
- Google-style docstrings with shape annotations

### 2. Export from `jax_pdf/__init__.py`

```python
from jax_pdf.<name> import MyDist

__all__ = [..., "MyDist"]
```

### 3. Add tests to `tests/test_distributions.py`

Add your distribution to `ALL_DISTS` (and `DISTS_WITH_SAMPLE` if it supports sampling):

```python
ALL_DISTS = [
    ...,
    MyDist(param=0.1),
]
```

This automatically runs all interface tests (dim, call, batch, grad).

If the distribution has a computable normalizing constant, also add it to `DISTS_WITH_LOG_NORM`.

Add a distribution-specific test class for parameter validation and unique behavior:

```python
class TestMyDist:
    def test_dim(self):
        assert MyDist().dim == 2

    def test_param_validation(self):
        with pytest.raises(ValueError, match="param must be positive"):
            MyDist(param=-1.0)
```

### 4. Create `docs/<name>.md`

Follow the pattern in `docs/banana.md`: math definition, why it is hard, parameter table, usage examples.

### 5. Update README.md

Add a row to the distributions table.

## Running tests

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

## Style guide

- Google-style docstrings with shape annotations in Args/Returns
- Comments explain *why*, not *what*
- Error messages: expected vs received, point toward fix
- Keep the library small; every addition must earn its complexity
