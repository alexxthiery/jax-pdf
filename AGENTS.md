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

# On an NVIDIA GPU: TF32 matmuls cost about 1e-5 relative accuracy, which three
# whitened-LGCP tests detect by comparing across a matmul
JAX_DEFAULT_MATMUL_PRECISION=highest pytest tests/

# Run a single test class
pytest tests/test_banana.py -v
```

## Architecture

```
jax_pdf/
  __init__.py           # re-exports: Banana2D, DoubleWell, DW4, GaussianMixture, HarmonicCrystal,
                        #   LatticePhiFour, LGCP, LennardJones, MonatomicWater, MullerBrown, NealFunnel,
                        #   PeriodicLennardJones, PhiFour, PhiFourChainOracle
  banana.py             # Banana2D distribution
  neal_funnel.py        # NealFunnel distribution
  log_gauss_pines.py    # LGCP distribution
  muller_brown.py       # MullerBrown distribution
  phi_four.py           # PhiFour distribution
  lattice_phi_four.py   # LatticePhiFour distribution (periodic lattice, any dimension)
  phi_four_oracle.py    # PhiFourChainOracle: reference values for a 1D chain by
                        #   transfer operator (NumPy; not a distribution)
  double_well.py        # DoubleWell distribution
  gaussian_mixture.py   # GaussianMixture distribution (per-component full covariance;
                        #   exact density, log Z, moments, responsibilities and sampler)
  dw4.py                # DW4 distribution (4-particle double-well)
  lennard_jones.py      # LennardJones distribution (LJ13, LJ55; free cluster)
  periodic_lennard_jones.py  # PeriodicLennardJones (periodic LJ solid, minimum image)
  monatomic_water.py    # MonatomicWater (mW water, Stillinger-Weber, periodic;
                        #   n_neighbours: exact O(N k^2) three-body sum for large N)
  harmonic_crystal.py   # HarmonicCrystal (Einstein crystal, exact log Z)
  cox_process_utils.py  # utility functions for LGCP
  _particle_utils.py    # shared DW4/free-LJ pair distances and harmonic confinement
  _validation.py        # internal: event shape checks and concrete parameter validation
  finpines.csv          # Finnish pines dataset
tests/
  test_interface.py     # shared interface tests (parametrized across all dists)
  test_tracing.py       # static vs traced fields, sweeps, and validation under tracing
  test_<name>.py        # one file per distribution; periodic LJ and mW include
                        #   independent scalar energy and finite-difference
                        #   gradient references, with no sibling repo imports
docs/
  <name>.md             # per-distribution documentation, one file per distribution
```

One file per distribution. `__init__.py` re-exports the public API.

### Optional chemistry

`jax_pdf/chemistry/alanine.py` contains `AlanineDipeptide` (particle tier) and
`AlanineDipeptideInternal` (generic tier). These are deliberately not re-exported
by the root module: core import must work without jaxmm or OpenMM. The chemistry
extra requires jaxmm >=0.2 and Python >=3.10; float64 is explicit caller setup.
Molecular physics, geometry, support and physical volume factors belong in jaxmm.
The wrappers own Flax/JAX integration, event shapes and benchmark defaults.
Use `transform.pullback(cartesian_target, q, reference=q0)` rather than reimplement
energy, support, reconstruction or Jacobians. Use upstream `map_events`,
`require_float64` and `validate_temperature` rather than copying their logic.
Internal state contains only the shared Cartesian target; its transform is derived.
The physical internal measure includes all radial/angular factors, not the flat
fixed-frame slice determinant. Cartesian sampling fixes COM; evaluation is the
unconfined Boltzmann factor. Never silently recenter inputs or add a trap.

`tools/build_alanine_assets.py` rebuilds local parameter files, independent OpenMM
references and provenance under `jax_pdf/chemistry/data/`. Runtime verifies hashes;
never generate data, run MD or download assets during target construction.
`tests/test_alanine.py` is optional and self-contained once jaxmm is installed.
Run `JAX_ENABLE_X64=1 JAX_PDF_TEST_CHEMISTRY=1 pytest tests/test_interface.py` to
exercise chemistry in the existing shared interface matrix. Use valid molecular
reference points there, not the generic collinear particle fixture.
See `docs/alanine.md` for support, chirality, units, regeneration and migration.

Each distribution is a `@struct.dataclass` (Flax) with a two-tier interface:

```python
dist = SomeDistribution(param=value)

# Generic distributions (Banana2D, NealFunnel, GaussianMixture, LGCP, MullerBrown, PhiFour,
# LatticePhiFour, DoubleWell):
log_p = dist(x)                    # input (..., dim) -> output (...)

# Particle distributions (LennardJones, DW4, PeriodicLennardJones, MonatomicWater, HarmonicCrystal):
log_p = dist(x)                    # input (..., n_particles, spatial_dim) -> output (...)

# Shared across both tiers:
dim = dist.dim                     # int attribute: flat DoF count
log_Z = dist.log_normalization()   # scalar; raises NotImplementedError if intractable
samples = dist.sample(key, n)      # (n, dim) -- Banana2D, DoubleWell, GaussianMixture, NealFunnel

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
- Compatible with `jit`, `vmap`, `grad` without surprises, including passing a distribution itself as an argument to a jitted function
- Numeric parameters are pytree children, so they can be swept with `vmap` and differentiated. Sizes, flags and modes are static (`struct.field(pytree_node=False)`), so they stay concrete and may drive Python control flow
- Validation of a numeric parameter is guarded by `is_concrete` from `jax_pdf._validation`: it runs on concrete values and is skipped for tracers and JAX's plain object placeholders during pytree reconstruction. Shape checks still run on tracers; skip only the object placeholder when validating array fields in `__post_init__`
- Every `__call__` uses `check_event_shape` from `jax_pdf._validation` for its documented trailing axes. Shape metadata remains available under tracing, so this check is never guarded by `is_concrete`; leading batch axes stay unrestricted
- Keep shared interface checks in `test_interface.py`, shape-boundary cases in `test_validation.py`, and numerical oracles/regressions in distribution-specific tests. Avoid duplicate smoke tests and size cross-products without a distinct behavior to protect
- Prefer deterministic sampler tests (controlled innovations, probability weights, quadrature, exact enumeration). Avoid Monte Carlo statistical assertions in the default unit suite; seeded inputs for algebraic or shape/support checks are fine
- Tests must run without sibling repositories or optional external reference implementations. Use analytic or independent scalar numerical oracles; do not modify `sys.path` to import another checkout or skip comparisons when it is absent

## Markdown and math

Docs must render on GitHub. GitHub uses KaTeX, not full LaTeX.

- `$...$` for inline math, `$$...$$` for display (blank lines around `$$`)
- No `\texttt{}`, `\textrm{}` in math. Use `\text{}` or put words outside math in backticks
- No escaped underscores in math: `$\phi_i$` not `$\phi\_i$`
- No manual spacing (`\,`, `\;`, `\quad`). KaTeX handles spacing; just delete them
- Do not mix inline code and math (e.g., `` `n`$^2$ ``). Use `$n^2$` or Unicode `n²`
- In tables, use `\lvert`, `\rvert`, `\mid` instead of raw `|` inside math

See CONTRIBUTING.md for the full list and examples.

## Design principles

The code conventions say how to write a distribution; these say what to build it out of.
The library already follows them, so the named examples are the specification.

- **One stored representation; derive the rest.** A field exists only when nothing else
  determines it. `dim` is a property computed from `n_particles` and `spatial_dim`
  (`LennardJones`), from `lattice_shape` (`LatticePhiFour`) or from `means`
  (`GaussianMixture`), so no two fields can disagree about a size. `NealFunnel` stores
  `dim` because nothing there derives it. `LGCP` is the one distribution that caches
  derived state, because `__post_init__` runs on every pytree reconstruction and cannot
  afford to rebuild a Gram factor; it pays for the cache with a consistency check, since
  `replace(grid_dim=...)` otherwise leaves the cache describing the old grid.
- **One evaluation path per quantity.** `GaussianMixture.sample` is
  `sample_with_component` with the labels dropped, so draws cannot disagree with their
  labels, and a single private helper is the only place weights are normalized. Two paths
  to one number diverge eventually.
- **The parameter name says whether the quantity is squared.** `sigma` is a standard
  deviation (`Banana2D`, `NealFunnel`), `variance` is its square
  (`cox_process_utils.exponential_kernel`), `scale_tril` and `cholesky_cov` are Cholesky
  factors, and `from_covariance` takes second moments. Never leave this to the reader: no
  validation can catch the confusion, since both readings are positive numbers of the same
  shape.
- **A mode flag validates its allowed set.** `trap_mode` (`DW4`, `LennardJones`) is static
  so it may drive Python control flow, and `__post_init__` rejects anything outside
  `individual` and `com`. Without that check a typo silently selects the other branch.
- **An optional numeric field is typed `Optional[...]`.** `None` is an empty subtree to
  JAX, so whether the field is set changes the pytree structure. `MonatomicWater` is the
  pattern.
- **Alternate constructors are welcome; factories are not.** See Never, below.

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
- Use factories (`Dist.create(...)`) instead of direct instantiation. An alternate
  constructor is a different thing and is welcome: a `@classmethod` that computes only
  the declared fields from another parametrization and returns `cls(...)`, adding no
  state, no field it alone can fill, and no branch in evaluation.
  `PeriodicLennardJones.from_density`, `PhiFourChainOracle.from_coefficients` and
  `GaussianMixture.from_covariance` are the examples. What this rules out is a
  `create(mode=...)` that decides which fields exist
- Remove or modify the Finnish pines dataset (`finpines.csv`)
