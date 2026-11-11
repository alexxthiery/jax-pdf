# Alanine dipeptide

`jax_pdf.chemistry` provides flexible 22-atom ACE-L-ALA-NME targets in Cartesian
coordinates and in 60 physical internal coordinates.
Energy and geometry are evaluated by the optional `jaxmm` backend, entirely in JAX.
OpenMM is needed only to regenerate the bundled benchmark assets.

## Installation and use

Chemistry requires Python >=3.10 and `jaxmm >=0.2`.
The base library retains its Python >=3.9 requirement and does not import `jaxmm`.
Until the new backend release is published, install the maintained checkout first:

```bash
pip install -e /path/to/jaxmm
pip install -e '.[chemistry,dev]'
```

```python
import jax
jax.config.update("jax_enable_x64", True)

from jax_pdf.chemistry import (
    AlanineDipeptide, AlanineDipeptideInternal, load_alanine_parameters,
)

cartesian = AlanineDipeptide(temperature=300.0)
internal = AlanineDipeptideInternal(cartesian=cartesian)
x = cartesian.reference_positions                       # (22, 3), nm
q = internal.transform.to_internal(x)                    # (60,)
log_cartesian = cartesian(x)
log_internal = internal(q)
gradient = jax.grad(internal)(q)

# Same topology and units, OBC1 implicit solvent instead of vacuum:
implicit = AlanineDipeptide(params=load_alanine_parameters("implicit"))
```

Constructors require float64 mode; coordinate arrays must also have dtype float64.
The library never changes the caller's precision settings.
Construction loads small local assets, without network access, minimization or MD.

## Physical model and measure

The energy includes flexible harmonic bonds and angles, periodic torsions,
and nonbonded Coulomb and Lennard-Jones interactions.
The implicit variant additionally includes OBC1 GBSA.
There are no bond constraints, cutoffs, training-energy clipping, or confinement.
The parameter set is the AMBER topology distributed by `openmmtools` as
`alanine-dipeptide-gbsa`; it is not asserted to be ff99SB-ILDN.
The [asset manifest](../jax_pdf/chemistry/data/alanine_manifest.json) records source
hashes, extraction versions, units, backend source hashes and parameter hashes.

For Cartesian input of shape `(..., 22, 3)`, the target returns

$$
\log \tilde p_X(x) = -U(x)/(k_B T),
$$

where $k_B=0.008314462618$ kJ/(mol K).
The default target restricts the alpha carbon's signed tetrahedron to the L sector.
Use `chirality="unrestricted"` to retain both sectors.
This choice is inherited by the internal target through its `cartesian` field.

Sampling fixes the center of mass: the Cartesian representation stores 66 values
but has 63 intrinsic coordinates after this restriction.
The callable does not recenter its input or enforce a particular COM.
Its translationally unrestricted integral is infinite.
Rotations remain free and have finite group volume.

For internal input of shape `(..., 60)`, pack 21 bond lengths, 20 bond angles, then
19 torsions in construction order:

$$
\log \tilde p_Q(q) = -U(x(q))/(k_B T)
  + 2\sum_{i=0}^{20}\log b_i + \sum_{j=0}^{19}\log\sin a_j.
$$

This is the conformational law obtained from flexible Cartesian equilibrium sampling.
The external translation volume and orientation factor $8\pi^2$ are omitted.
The omitted factors are conventions for the configurational integral, not an
assertion that the Cartesian and internal normalizers are numerically identical.
Both `log_normalization()` methods raise `NotImplementedError`; neither target
provides an exact `sample()` method.

The fixed-frame Cartesian slice is a different measure.
Its determinant omits the additional factor $b_0^2 b_1\sin a_0$ from integrating
rigid orientations.
`jaxmm.log_boltzmann_fixed_frame` requests that slice density explicitly;
the `jax-pdf` internal target always uses the physical conformational measure.
See [Potter and Gilson, Eq. 13](https://doi.org/10.1021/jp0135407).

## Coordinate interface and support

```python
transform = internal.transform
bonds, angles, torsions = transform.unpack(q)
q = transform.pack(bonds, angles, torsions)
valid = transform.in_support(q)
lower, upper = transform.bounds
periodic_indices = transform.periodic_indices
log_volume = transform.log_volume(q)
x_fixed = transform.to_cartesian(q)
```

The internal wrapper delegates to the Cartesian callable through one reusable
upstream operation:

```python
log_p = transform.pullback(cartesian, q, reference=transform.to_internal(x))
```

This retains Cartesian support restrictions and custom density behavior.
The same operation works for other rotation/translation-invariant Cartesian
models, without copying energy evaluation, support masking or Jacobians.
`jaxmm.map_events` supplies batch mapping for scalar-event backends.

The frame is backbone C, CA, N, placed at the origin, on the positive x axis,
and in the xy plane, respectively.
Reconstruction returns original topology atom order.
Converting Cartesian positions to internals discards rigid pose; the reverse
returns a canonical orientation with the same molecular geometry.
Z-matrix torsions 12 and 5 are the negatives of biochemical phi and psi.

| Block | Units | Domain |
| --- | --- | --- |
| Bonds | nm | Positive, no upper bound |
| Angles | radians | Open interval $(0,\pi)$ |
| Torsions | radians | Circle encoded as $[-\pi,\pi)$ |

All entries must be finite and construction reference triples noncollinear.
The internal density returns `-inf` outside support, with zero rejection gradients.
Safe inputs are substituted before evaluating energies or logarithms.
Negative bonds are never clipped into valid configurations.
Torsion values outside the principal interval are rejected rather than assigning
infinitely many copies of the same molecular configuration probability mass.

Malformed event shapes, non-float64 inputs and invalid constructor parameters raise.
Geometry conversions also reject concrete invalid coordinates; `validate(q)` is
an explicit setup-time check, and `in_support(q)` is the compiled predicate.
Shape/dtype checks run during JIT tracing; unknown numeric values cannot raise
Python validation errors inside compiled code.
Cartesian nonfinite values raise eagerly and are rejected under JIT.
Physical energy singularities at overlapping atoms are not softened.

Whitening is optional and belongs to `jaxmm` (`internal_whitener`,
`whiten_internals`, `unwhiten_internals`).
It is an affine rescaling, does not enforce support, and has its own constant
log determinant that must be added when changing density coordinates.
No whitening or data-dependent coordinate fitting occurs in these targets.

## Verification and regeneration

```bash
# Self-contained chemistry checks, without OpenMM:
pytest tests/test_alanine.py -q

# Include chemistry instances in the existing shared interface matrix:
JAX_ENABLE_X64=1 JAX_PDF_TEST_CHEMISTRY=1 pytest tests/test_interface.py -q

# Setup environment with jaxmm, OpenMM and openmmtools:
python tools/build_alanine_assets.py
```

The generator freezes both parameter sets and independent OpenMM Reference
energies/forces at three deterministic configurations.
It runs no trajectory simulation and writes a provenance manifest.
Tests compare both solvents with these references, check rigid-motion invariance,
the internal measure, chirality, support rejection, temperature derivatives,
asset corruption and the optional import boundary.
The upstream full Cartesian Jacobian test independently anchors the physical
volume factor, including the initial frame atoms.
