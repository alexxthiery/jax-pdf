# AGENTS.md

Benchmark probability density functions for MCMC and variational inference testing in JAX.

## Quick reference

- **Package:** `jax_pdf` (import name), `jax-pdf` (package name)
- **Python:** >=3.9
- **Core deps:** JAX, Flax (`struct.dataclass`), optax (LGCP optimization)
- **Build:** hatchling
- **License:** MIT

## Commands

```bash
# Install (editable, with dev deps)
pip install -e '.[dev]'

# Run all tests
pytest tests/

# Run a single test class
pytest tests/test_distributions.py::TestBanana2D -v
```

## Architecture

```
jax_pdf/
  __init__.py           # re-exports: Banana2D, NealFunnel, LGCP
  banana.py             # Banana2D distribution
  neal_funnel.py        # NealFunnel distribution
  log_gauss_pines.py    # LGCP distribution
  cox_process_utils.py  # utility functions for LGCP
  finpines.csv          # Finnish pines dataset
tests/
  test_distributions.py # parametrized interface + per-distribution tests
docs/
  banana.md             # per-distribution documentation
  neal_funnel.md
  lgcp.md
```

One file per distribution. `__init__.py` re-exports the public API.

Each distribution is a `@struct.dataclass` (Flax) with a unified interface:

```python
dist = SomeDistribution(param=value)
log_p = dist(x)                    # input (..., dim) -> output (...)
dim = dist.dim                     # int property
log_Z = dist.log_normalization()   # scalar
samples = dist.sample(key, n)      # (n, dim) -- Banana2D, NealFunnel only
```

## Code conventions

- Google-style docstrings with shape annotations in Args/Returns
- Batch dimensions via `x[..., i]` indexing (not explicit reshape)
- `__post_init__` validates parameters with "expected vs received" error messages
- Comments explain *why*, not *what*
- Pure functions, explicit state, no side effects
- Compatible with `jit`, `vmap`, `grad` without surprises

## Boundaries

### Always

- Preserve the unified interface across all distributions
- Add tests to `tests/test_distributions.py` for new distributions
- Update `docs/<name>.md` and the README table when adding distributions
- Use `@struct.dataclass` (Flax), not plain Python dataclasses

### Ask first

- Adding new dependencies
- Changing the public API or method signatures
- Architectural changes (new base classes, mixins, etc.)

### Never

- Break the unified interface (rename `__call__`, change `dim` to a method, etc.)
- Add distribution-specific method names for core functionality
- Use factories (`Dist.create(...)`) instead of direct instantiation
- Remove or modify the Finnish pines dataset (`finpines.csv`)
