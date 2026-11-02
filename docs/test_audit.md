# Test effectiveness audit

Audit date: 2026-09-21.

The existing suite missed substantive numerical mistakes: **11 of 16 selected
mutations survived**. The strengthened audit suite detected **all 16** and
passed **724 tests with no skips**. This is evidence for those particular
contracts, not an exhaustive mutation score or proof that the package is
correct. The three API issues discovered by the audit were fixed, bringing
the suite to 775 cases. A subsequent redundancy review reduced it to **519
cases from 201 test functions**, all passing with no skips and warnings treated
as errors. The same 16 selected faults remain detected after the reduction.

The comparison starts from the working tree after event-shape validation and
the removal of external bgmat test imports, with 697 passing tests. It is not a
comparison against an upstream release. The initial audit changed tests and
added a mutation runner. Subsequent commits fix the three API issues described
below.

## What the tests now establish

Shape, finiteness, and comparisons between eager and compiled evaluation are
useful interface checks, but a wrong formula can satisfy all of them. The new
checks use analytic values and derivatives, independent numerical integration,
or deterministic checks of sampler transforms and probability weights.

| Deliberate fault | Before | After | Protecting oracle |
| --- | --- | --- | --- |
| Double Banana quadratic precision | Survived | Detected | Gaussian factorization: absolute log density and gradient at nondefault scales |
| Ignore Banana sampling scale | Survived | Detected | Prescribed normal innovations give hand-computed transformed samples; keys are distinct |
| Use `exp(x0)` for Funnel conditional standard deviation | Survived | Detected | Conditional Gaussian density and gradient at positive and negative mouth coordinates |
| Ignore Funnel sampling scale | Survived | Detected | Prescribed innovations give hand-computed mouth values and conditional scales; keys are distinct |
| Drop the LGCP likelihood | Survived | Detected | Independently reconstructed Gaussian-Poisson posterior, gradient, and curvature |
| Draw DoubleWell quadrature nodes uniformly | Survived | Detected | Actual categorical probabilities give deterministic moments matching independent Gauss-Legendre quadrature |
| Drop DoubleWell Gaussian normalizing constants | Survived | Detected | Absolute quartic integral plus analytic Gaussian factors in dimensions 2 and 10 |
| Halve MullerBrown energy | Survived | Detected | Analytic exponent and derivative tables at two configurations |
| Ignore the DW4 pair offset | Survived | Detected | Six pairs contribute exactly six offsets |
| Ignore LennardJones length scale | Survived | Detected | Two-particle energy and force at a nondefault length scale |
| Omit one PhiFour endpoint bond | Survived | Detected | Uniform field has two open-boundary bonds and no periodic coupling energy |
| Remove DoubleWell shape validation | Detected | Detected | Shared malformed-event tests in eager and JIT execution |
| Double LatticePhiFour coupling | Detected | Detected | Existing analytic lattice and cross-model checks |
| Double HarmonicCrystal energy | Detected | Detected | Existing analytic energy check |
| Remove the periodic Lennard-Jones cutoff shift | Detected | Detected | Independent scalar pair reference and analytic anchors |
| Remove monatomic-water three-body energy | Detected | Detected | Independent scalar triplet reference and analytic anchors |

Additional checks cover known trap energies for both particle confinement
modes, asymmetric histogram bins and their upper boundary, and hand-computed
non-diagonal whitening transformations. A one-cell LGCP example checks the
Laplace mode against an independently bisected scalar score equation and checks
the precision against its analytic derivative.

The existing lattice and PhiFour-oracle tests already provide strong checks:
analytic energies and gradients, boundary counting, dense small-system
quadrature, Gaussian closed forms, and exact sampling-law enumeration. Those
checks were retained.

Sampler tests contain no Monte Carlo statistical assertions. Banana and Funnel
use three prescribed normal innovations and hand-computed transformed samples,
including nondefault scales and distinct PRNG keys. DoubleWell's actual
categorical probabilities are checked by deterministic weighted sums against
independent quadrature, then prescribed indices and normal values verify
coordinate assembly. Only JAX random primitives are stubbed; the production
sampling transforms and probability weights remain under test. These tests
trust JAX's random primitives and do not prescribe a particular PRNG stream.
Small real-RNG shape/support checks remain as integration smoke tests.

The initial audit added six statistical cases (two seeds per sampler, 72,000
samples total). They were subsequently replaced with these three deterministic
cases, reducing the suite from 727 to 724 cases. The affected three sampler
mutations were rechecked after replacement. Numerical tolerances now cover
floating-point and quadrature error, not Monte Carlo uncertainty.

## Removing redundant coverage

The original 775 cases came from 226 functions. Shared interface checks
accounted for 442 cases, including 238 repetitions of malformed-event checks.
The reduced suite removes 25 duplicate smoke-test functions and keeps 14
representative distribution configurations instead of 17. Both LGCP coordinate
systems and both PhiFour boundaries remain because they exercise distinct
paths; additional sizes remain in dedicated numerical tests.

The full shape-boundary matrix stays in `test_validation.py`. Each distribution
still rejects incorrect trailing lengths and layouts in eager and JIT modes,
and particle counts are checked separately. Dedicated tests retain the original
DoubleWell 10-versus-2 regression and the three API regressions below. Analytic
values, gradients, independent scalar references, quadrature, and sampler
transform checks were retained.

The ring-sampler memory guard now uses a five-point grid and a lowered byte
budget instead of a 4,001-point grid. It rejects a 399-byte budget and accepts
400 bytes, the exact size of two 5x5 float64 transfer powers. Removing the guard
or changing its comparison from `>` to `>=` each makes the new test fail.

On the audit machine, the full run decreased from approximately 38 to 27 seconds;
the two memory-boundary cases each took under 5 ms instead of about 3.2 seconds
for the former large fixture. These are local observations, not benchmark
guarantees. The primary gain is fewer repeated contracts to maintain. Mutation
checks are a focused safeguard, not proof of identical coverage for every bug.

## Confirmed API issues and follow-up fixes

The initial audit deliberately reported these implementation issues separately
from its passing tests. Each is now fixed in a separate commit, with regressions
that failed before the corresponding fix. All tests remain deterministic.

### Banana normalizing-constant semantics (fixed)

`Banana2D.__call__` already returned a normalized density, while
`log_normalization()` incorrectly returned its additive Gaussian log constant.
The method now returns zero, following the unified contract in
[CONTRIBUTING](../CONTRIBUTING.md). The Gaussian constants remain in `__call__`,
so density values are unchanged. Regressions at three scales check zero log Z
under JIT and the absolute density after generic normalization. All three
failed before the fix; existing analytic density and gradient tests protect
against accidentally removing the constants from the density itself.

### LGCP optimization changes global JAX precision (fixed)

`LGCP.map_estimate` previously enabled float64 globally, affecting unrelated
computations even when input validation raised. MAP and Laplace now use scoped
float64 contexts and preserve the caller's setting. Supplied float32 initial
iterates are explicitly promoted, and the Laplace Hessian and inverse remain
inside the float64 scope. Cached model arrays retain their construction
accuracy; see [LGCP documentation](lgcp.md).

Regressions check both caller precision settings, default and float32 initial
guesses, validation errors, result dtypes, and subsequent JAX allocations.
The one-cell stationary-equation reference also checks converged Laplace
results with both caller settings. Nine regressions failed before this fix.

### Passing a distribution as an unmapped `vmap` argument (fixed)

JAX reconstructs pytrees with plain object placeholders while determining the
argument structure. Numeric validation previously compared those objects with
numbers, and HarmonicCrystal tried to validate a placeholder's rank. This
caused `vmap(lambda dist, x: dist(x), in_axes=(None, 0))` to fail for several
distributions, with or without an enclosing JIT.

The numeric guard now excludes that exact placeholder type, and HarmonicCrystal
skips the placeholder's shape check. Concrete numeric validation and tracer
shape validation remain active. The original regression exercised
17 distribution configurations in both transformation modes; 24 of the 34
cases failed before the fix. The reduced suite retains 14 representative
configurations in both modes. A separate negative control checks that malformed
HarmonicCrystal positions still fail when passed through JIT.

This handling follows [JAX's pytree initialization guidance](https://docs.jax.dev/en/latest/custom_pytrees.html#custom-pytrees-and-initialization-with-unexpected-values).

## GaussianMixture cases (2026-11-01)

`GaussianMixture` was added after the audit above, with nine cases of its own in
`tools/audit_test_mutations.py`. All nine were detected on the first run, each by
the oracle it was written against.

| Deliberate fault | Protecting oracle |
| --- | --- |
| Drop `log det L_c` from a component | Component log densities against the definition in float64, with unequal correlated covariances |
| Solve with the Cholesky factor transposed (`lower=False`) | The same component densities |
| Remove the event-shape guard | Shared malformed-event tests, eager and JIT |
| Normalize the weights inside `__call__` | One-dimensional quadrature of the density against `log_normalization`, plus the single-component density |
| Form responsibilities without the weights | The posterior over components at unequal weights |
| Treat the isotropic variance as a standard deviation | `from_isotropic_variance` against the matching scaled-identity covariance |
| Apply the sampling scale transposed | Draws whitened back to the innovations of the second key |
| Ignore the weights when drawing components | A zero-weight component never drawn, by the exact support rather than frequencies |
| Drop the between-component term from `covariance` | $\mathbb{E}[xx^{\top}] - \mathbb{E}[x]\mathbb{E}[x]^{\top}$ summed over components by hand |

## LGCP cache cases (2026-11-02)

A design audit of the existing targets found that `LGCP` is the only distribution
that stores derived state, and that neither half of it survives `replace`. Two
cases cover the fixes, both detected on the first run.

| Deliberate fault | Protecting oracle |
| --- | --- |
| Accept cached state built for a different `grid_dim` | `replace(grid_dim=...)` must raise, rather than report the new `dim` while holding the old factor |
| Drop the determinant term from the prior constant | The unwhitened posterior and its gradient against the independently built Gaussian-Poisson model |

The prior constant is no longer cached at all, because it depends on the
`whitened` flag: a cached one left `replace(whitened=True)` short by half the log
determinant of the prior covariance, $2.9120$ at a grid of 3, a silent shift of
the density that no shape check can detect. Densities are bitwise unchanged for
every directly built instance, checked at grids 1, 3, 5 and 8 in both
parameterizations.

Caveat on the baseline, corrected 2026-11-02: three whitened-LGCP cases fail on
an NVIDIA GPU at default matmul precision,
`test_hessian_is_prior_precision_plus_poisson_curvature[True]` and the two
`test_distribution_as_unmapped_vmap_argument[*-LGCP1]` cases, at a relative
difference of about $1.2 \times 10^{-5}$ against an `rtol` of
$3 \times 10^{-6}$. The cause is TF32, not the tolerances and not the code: JAX
runs float32 matmuls in TF32 on NVIDIA hardware by default, worth roughly
$10^{-5}$ relative accuracy, and the whitened LGCP path sits one Cholesky matmul
away from its oracle. All three pass with `JAX_DEFAULT_MATMUL_PRECISION=highest`,
and all three pass on CPU. They were first read as a float32 tolerance problem of
the code, which they are not; either way they reproduce on an unmodified checkout
of the parent commit.

The nine mixture cases were run with those three deselected, since the runner's
baseline gate refuses to interpret mutations while any test fails; that baseline
was green at 297 cases. On CPU no deselection is needed, because the full suite
passes.

With that baseline green and the `water_three_body_missing` spec retargeted at
the line upstream `dc11486` had split, the complete set of 27 cases ran end to
end for the first time on 2026-11-02, on CPU: all 27 detected, none survived.
That run is also the first check of `lgcp_likelihood_missing`, which the baseline
gate had been holding back.

Validated with Python 3.12.14, JAX and jaxlib 0.11.2, Flax 0.12.9, NumPy 2.5.3
and pytest 9.1.1, on CPU and on an NVIDIA RTX PRO 6000 GPU.

## Reproducing the checks

From the repository root, in an environment with the project development
dependencies installed:

```bash
pytest tests/
PYTHONPATH=. python tools/audit_test_mutations.py --output /tmp/jax-pdf-mutations.json
```

The runner first requires a passing unmodified full suite. It then changes one
method at a time in memory, clears JAX caches, runs the relevant distribution
and shared interface/tracing tests, and restores the method in `finally`.
Source files are never modified. The JSON output records failures by test ID;
a surviving mutation makes the command exit nonzero. A source change that
invalidates a mutation's exact replacement causes an explicit error rather
than silently skipping that case.

Individual mutations can be selected after the baseline run:

```bash
PYTHONPATH=. python tools/audit_test_mutations.py \
  --output /tmp/jax-pdf-shape-mutation.json doublewell_shape_check_missing
```

Validation used Python 3.13, JAX 0.7.2, Flax 0.12.0, and NumPy 2.3.3 on CPU.
The focused LGCP tests also passed with warnings treated as errors. Older
supported runtimes and accelerators were not exercised. Invalid parameter
types, nonfinite inputs, and every possible composition of JAX transformations
were not exhaustively tested. These targeted fixes do not establish compatibility
with every possible nesting of JAX transformations.
