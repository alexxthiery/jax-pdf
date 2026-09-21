# Test effectiveness audit

Audit date: 2026-09-21.

The existing suite missed substantive numerical mistakes: **11 of 16 selected
mutations survived**. After strengthening the tests, **all 16 are detected** and
the unmodified suite passes **724 tests with no skips**. This is evidence for
these particular contracts, not an exhaustive mutation score or proof that the
package is correct. Three existing API issues remain unresolved below.

The comparison starts from the working tree after event-shape validation and
the removal of external bgmat test imports, with 697 passing tests. It is not a
comparison against an upstream release. This audit changes tests and adds a
mutation runner; it does not change distribution implementations.

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

## Confirmed unresolved issues

These are existing implementation issues discovered during the audit. They
are not intentionally accepted behavior, and no skipped or expected-failure
tests were added to make them disappear. The passing suite above does not
certify these contracts. The snippets below reproduce the issues separately;
run the precision example in a fresh process.

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

### Passing a distribution as an unmapped `vmap` argument fails

Closure-based `vmap` and passing a distribution directly through `jit` are
covered. The combination below is a distinct path and currently fails during
Flax reconstruction: parameter validation compares an internal `object`
placeholder with an integer.

```python
import jax
import jax.numpy as jnp
from jax_pdf import Banana2D

evaluate = jax.jit(jax.vmap(lambda dist, x: dist(x), in_axes=(None, 0)))
evaluate(Banana2D(), jnp.ones((2, 2)))
# TypeError: '<=' not supported between instances of 'object' and 'int'
```

Follow-up: make numeric parameter validation compatible with pytree
reconstruction, while retaining validation of ordinary user inputs. Add this
specific transformation composition as a regression across distributions.

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
were not exhaustively tested. The three confirmed issues above are concrete
follow-up work before making broader compatibility claims.
