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
pytest tests/test_banana.py -v
```

## Architecture

```
jax_pdf/
  __init__.py           # re-exports: Banana2D, DoubleWell, DW4, LGCP, LennardJones, MullerBrown, NealFunnel, PhiFour
  banana.py             # Banana2D distribution
  neal_funnel.py        # NealFunnel distribution
  log_gauss_pines.py    # LGCP distribution
  muller_brown.py       # MullerBrown distribution
  phi_four.py           # PhiFour distribution
  double_well.py        # DoubleWell distribution
  dw4.py                # DW4 distribution (4-particle double-well)
  lennard_jones.py      # LennardJones distribution (LJ13, LJ55)
  cox_process_utils.py  # utility functions for LGCP
  finpines.csv          # Finnish pines dataset
tests/
  test_interface.py     # shared interface tests (parametrized across all dists)
  test_banana.py        # Banana2D-specific tests
  test_double_well.py   # DoubleWell-specific tests
  test_dw4.py           # DW4-specific tests
  test_lennard_jones.py # LennardJones-specific tests
  test_neal_funnel.py   # NealFunnel-specific tests
  test_lgcp.py          # LGCP-specific tests
  test_muller_brown.py  # MullerBrown-specific tests
  test_phi_four.py      # PhiFour-specific tests
docs/
  banana.md             # per-distribution documentation
  neal_funnel.md
  lgcp.md
  muller_brown.md
  phi_four.md
  double_well.md
  dw4.md
  lennard_jones.md
```

One file per distribution. `__init__.py` re-exports the public API.

Each distribution is a `@struct.dataclass` (Flax) with a two-tier interface:

```python
dist = SomeDistribution(param=value)

# Generic distributions (Banana2D, NealFunnel, LGCP, MullerBrown, PhiFour, DoubleWell):
log_p = dist(x)                    # input (..., dim) -> output (...)

# Particle distributions (LennardJones, DW4):
log_p = dist(x)                    # input (..., n_particles, spatial_dim) -> output (...)

# Shared across both tiers:
dim = dist.dim                     # int property: flat DoF count
log_Z = dist.log_normalization()   # scalar; raises NotImplementedError if intractable
samples = dist.sample(key, n)      # (n, dim) -- Banana2D, DoubleWell, NealFunnel

# Particle tier additionally exposes:
n = dist.n_particles               # int
d = dist.spatial_dim               # int
```

Why two tiers: particle distributions have a semantically meaningful particle axis. Downstream consumers (normalising flows over particle systems, equivariant networks) work in `(N, d)` natively, so forcing a reshape at every call site is wasteful and error-prone. The `dim` property is preserved across both tiers so callers that reason about flat DoF count (parameter allocation, MCMC chain length) do not need a conditional.

## Code conventions

- Google-style docstrings with shape annotations in Args/Returns
- Batch dimensions via `x[..., i]` indexing (not explicit reshape)
- `__post_init__` validates parameters with "expected vs received" error messages
- Comments explain *why*, not *what*
- Pure functions, explicit state, no side effects
- Compatible with `jit`, `vmap`, `grad` without surprises

## Markdown and math

Docs must render on GitHub. GitHub uses KaTeX, not full LaTeX.

- `$...$` for inline math, `$$...$$` for display (blank lines around `$$`)
- No `\texttt{}`, `\textrm{}` in math. Use `\text{}` or put words outside math in backticks
- No escaped underscores in math: `$\phi_i$` not `$\phi\_i$`
- No manual spacing (`\,`, `\;`, `\quad`). KaTeX handles spacing; just delete them
- Do not mix inline code and math (e.g., `` `n`$^2$ ``). Use `$n^2$` or Unicode `n²`
- In tables, use `\lvert`, `\rvert`, `\mid` instead of raw `|` inside math

See CONTRIBUTING.md for the full list and examples.

## Boundaries

### Always

- Preserve the unified interface across all distributions
- Add to `tests/test_interface.py` and create `tests/test_<name>.py`
- Update `docs/<name>.md` and the README table when adding distributions
- Use `@struct.dataclass` (Flax), not plain Python dataclasses

### Ask first

- Adding new dependencies
- Changing the public API or method signatures
- Architectural changes (new base classes, mixins, etc.)

### Never

- Rename `__call__`, turn `dim` into a method, or collapse the two tiers into one (do not "fix" particle distributions back to flat input; the structured shape is load-bearing for downstream flow consumers)
- Add distribution-specific method names for core functionality
- Use factories (`Dist.create(...)`) instead of direct instantiation
- Remove or modify the Finnish pines dataset (`finpines.csv`)
