# Contributing

## Adding a new distribution

### 1. Create `jax_pdf/<name>.py`

Use `banana.py` as a template. The structure:

```python
"""One-line module docstring."""

import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_pdf._validation import check_event_shape, is_concrete


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
    """Inline docstring for VS Code hover. Numeric: a pytree child."""

    n_sites: int = struct.field(pytree_node=False, default=8)
    """Structural: static, so it can set shapes and drive control flow."""

    def __post_init__(self):
        # A numeric parameter is a tracer when the distribution crosses a jit
        # or vmap boundary, and a tracer cannot be compared.
        if is_concrete(self.param) and self.param <= 0:
            raise ValueError(
                f"param must be positive, got {self.param}"
            )
        if self.n_sites < 1:            # static: concrete everywhere, no guard
            raise ValueError(f"n_sites must be >= 1, got {self.n_sites}")

    @property
    def dim(self) -> int:
        return 2

    def __call__(self, x: Array) -> Array:
        """Evaluate log probability density.

        Args:
            x: Input point(s). Generic distributions take shape
                ``(..., dim)``. Particle distributions take structured
                shape ``(..., n_particles, spatial_dim)`` and also
                expose ``n_particles`` / ``spatial_dim`` properties;
                see `LennardJones` and `DW4` for the template.

        Returns:
            Log probability density of shape (...).

        Raises:
            ValueError: If the input does not have trailing shape (dim,).
        """
        check_event_shape(x, (self.dim,))
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
- Numeric parameters stay pytree children, so they can be swept with `vmap` and differentiated; sizes, flags and modes are `struct.field(pytree_node=False)`, so they stay concrete and may drive Python control flow
- Guard validation of a numeric parameter with `is_concrete`, so the distribution can be passed to a jitted function. Validation of a static field needs no guard
- Start `__call__` with `check_event_shape(x, (self.dim,))` for a generic distribution, or `check_event_shape(x, (self.n_particles, self.spatial_dim))` for a particle distribution. Check trailing axes, not total size or batch axes
- Keep shape checks active when `x` is a tracer: they compare metadata and need no `is_concrete` guard, array operations, or callbacks. Do not coerce, flatten, or reshape malformed events to make the check pass
- `log_normalization()` returns a scalar, either a Python float or a JAX scalar; returns 0.0 for normalized distributions; raises `NotImplementedError` if intractable
- `sample()` only if exact sampling is possible; omit otherwise
- `__post_init__` validates parameters with clear error messages
- Google-style docstrings with shape annotations

### 2. Export from `jax_pdf/__init__.py`

```python
from jax_pdf.<name> import MyDist

__all__ = [..., "MyDist"]
```

### 3. Add tests

Add your distribution to `ALL_DISTS` in `tests/test_interface.py` (and `DISTS_WITH_SAMPLE`, `DISTS_WITH_LOG_NORM` if applicable):

```python
ALL_DISTS = [
    ...,
    MyDist(param=0.1),
]
```

This automatically runs the shared interface tests: dimensions, values, batches (including multiple and empty batch axes), gradients, vectorization, and passing the distribution as a JIT argument.
It also verifies that incorrect event shapes raise a diagnostic `ValueError` in both eager and JIT execution.
Include parameterizations with different input-handling paths, such as both whitened and unwhitened LGCP.

`tests/test_validation.py` checks the helper independently, including that it adds no JAX operations and permits symbolic batch dimensions when JAX export is available.
The `DoubleWell` tests use analytic values, gradients, and Hessians to protect valid evaluations under JIT, and exercise a distribution carried through `lax.scan`.
Keep numeric parameter sweep and differentiation coverage in `tests/test_tracing.py` when adding new parameter behavior.

Then create `tests/test_<name>.py` with distribution-specific tests:

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

## Markdown and math in docs

Documentation must render correctly on GitHub. GitHub uses KaTeX for math, which supports a smaller subset of LaTeX than MathJax.

Rules for math in `.md` files:

- Use `$...$` for inline math and `$$...$$` for display math
- Blank line before and after `$$` blocks
- No `\texttt{}`, `\textrm{}`, or other text-mode commands inside math. Use `\text{}` if you need words in math, or better, put the word outside the math delimiters in backticks
- No escaped underscores (`\_`) inside math. Use raw underscores: `$\phi_i$` not `$\phi\_i$`
- No manual spacing commands (`\,`, `\;`, `\!`, `\quad`). KaTeX handles spacing; just delete them
- Do not mix inline code and math on the same token (e.g., `` `grid_dim`$^2$ ``). Use either all-math (`$d^2$`) or Unicode superscripts (`grid_dim²`)
- Avoid `\mathbb`, `\mathcal` in table cells; they sometimes break. If needed, test on GitHub
- Pipes `|` inside math in tables conflict with table syntax. Use `\lvert`, `\rvert`, `\mid` instead
- Test your doc by previewing on GitHub or with `grip` before merging

Common mistakes:

| Bad | Good | Why |
|-----|------|-----|
| `$c = a \times \texttt{dim\_grid}$` | `$c = a \cdot d$ where $d$ is \`dim_grid\`` | `\texttt` unsupported in KaTeX |
| `` `n`$^2$ `` | `n²` or `$n^2$` | Mixed code+math breaks rendering |
| `$\phi\_i$` | `$\phi_i$` | Escaped underscore breaks math mode |

## Running tests

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

The suite must be self-contained: do not import sibling checkouts, modify `sys.path` to find them, or skip numerical comparisons because an external reference package is missing.
For periodic Lennard-Jones and monatomic water, the test modules contain independent float64 scalar sums of the documented pair and triplet potentials.
They compare batched energies and autodiff gradients against those sums and their finite differences, with analytic configurations to check the references themselves.
These references use explicit loops and independently specified constants, and do not call the production potential helpers.
Historical compatibility references in the distribution documentation do not imply a test dependency on those packages.

For each substantive numerical test, identify the behavior it promises, a
plausible mistake it should catch, and an independent source for its expected
result. Prefer analytic special cases, independent quadrature or scalar
references, and invariants with known values. Eager/JIT agreement and finite
outputs alone cannot establish that a formula is correct. Exercise nondefault
parameters so ignoring a parameter actually changes the expected answer.
For samplers, prefer deterministic checks of transforms using controlled random
innovations, explicit probability weights, quadrature, or exact enumeration.
Avoid Monte Carlo statistical assertions in the default unit suite; if needed,
keep statistical validation in a separate, explicitly run experiment. Small
seeded fixtures and real-RNG shape/support checks are fine when their assertions
do not depend on sampling error. Do not freeze a particular PRNG output stream.

The [test effectiveness audit](docs/test_audit.md) records demonstrated gaps,
the tests that close them, and unresolved API issues. Run the focused mutation
audit after changing these numerical contracts:

```bash
PYTHONPATH=. python tools/audit_test_mutations.py --output /tmp/jax-pdf-mutations.json
```

The runner requires an unmodified passing suite first, then injects selected
faults in memory and reports which tests detect them. Keep the selected faults
aligned with meaningful failure modes as implementations evolve.

## Style guide

- Google-style docstrings with shape annotations in Args/Returns
- Comments explain *why*, not *what*
- Error messages: expected vs received, point toward fix
- Keep the library small; every addition must earn its complexity
