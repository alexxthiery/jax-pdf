"""Optional chemistry integration, with frozen independent OpenMM references."""

import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jaxmm", reason="Optional chemistry backend is not installed")
from jax_pdf.chemistry import (
    AlanineDipeptide, AlanineDipeptideInternal, load_alanine_parameters,
)
from jax_pdf.chemistry import alanine

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64


@pytest.fixture(autouse=True)
def precision():
    with enable_x64():
        yield


@pytest.fixture
def references():
    with np.load(alanine._asset("reference"), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


@pytest.mark.parametrize("solvent", ["vacuum", "implicit"])
def test_energy_and_force_against_openmm_assets(solvent, references):
    """Catch wrong solvent, units, parameters, scaling or force sign.

    Oracle: Reference-platform OpenMM evaluations, stored at three distinct
    structures. They do not depend on production energy or parsing helpers.
    Existing jaxmm/OpenMM discrepancies are <1e-5 kJ/mol in energy and 0.007
    kJ/mol/nm in force; absolute tolerances bound those documented differences.
    """
    dist = AlanineDipeptide(params=load_alanine_parameters(solvent), temperature=427.)
    x = jnp.asarray(references["positions"])
    energy = -dist(x)*0.008314462618*427.
    forces = jax.vmap(jax.grad(dist))(x)*0.008314462618*427.
    np.testing.assert_allclose(energy, references[f"{solvent}_energies"], rtol=0, atol=2e-5)
    np.testing.assert_allclose(forces, references[f"{solvent}_forces"], rtol=0, atol=0.01)


def test_internal_density_matches_cartesian_conformational_law(references):
    """All 21 radial and 20 angular factors, including the frame, are needed."""
    cartesian = AlanineDipeptide(temperature=381.)
    internal = AlanineDipeptideInternal(cartesian)
    x = jnp.asarray(references["positions"])
    q = internal.transform.to_internal(x)
    bonds, angles, _ = internal.transform.unpack(q)
    # The full-volume identity is independently proved upstream using a square
    # Cartesian Jacobian including pose. This integration check pins the choice
    # of that measure and its sign in the wrapper, rather than the slice measure.
    expected = cartesian(x) + 2*jnp.log(bonds).sum(-1) + jnp.log(jnp.sin(angles)).sum(-1)
    np.testing.assert_allclose(jax.jit(lambda d,y:d(y))(internal,q), expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(jax.grad(lambda t: internal.replace(
        cartesian=cartesian.replace(temperature=t))(q[0]))(381.),
        -cartesian(x[0])/381., rtol=1e-10)


@pytest.mark.parametrize("index,value", [(0, 0.), (5, -.1), (21, 0.), (22, np.pi),
                                         (41, np.pi), (43, np.nan), (0, np.inf)])
def test_rejected_events_have_zero_gradient(index, value):
    internal = AlanineDipeptideInternal()
    q = internal.transform.to_internal(internal.cartesian.reference_positions)
    bad = q.at[index].set(value)
    value, gradient = jax.jit(jax.value_and_grad(internal))(bad)
    assert value == -jnp.inf
    np.testing.assert_array_equal(gradient, jnp.zeros(60))


def test_chirality_is_the_same_support_in_both_representations():
    cartesian = AlanineDipeptide()
    internal = AlanineDipeptideInternal(cartesian)
    x = cartesian.reference_positions
    mirror = x*jnp.array([1., 1., -1.])
    q = internal.transform.to_internal(mirror)
    assert cartesian(mirror) == -jnp.inf
    assert internal(q) == -jnp.inf
    np.testing.assert_array_equal(jax.grad(internal)(q), jnp.zeros_like(q))
    free = cartesian.replace(chirality="unrestricted")
    # The actual AMBER phase constants need not be exactly reflection symmetric.
    # Removing the support restriction must admit the mirrored event unchanged.
    assert jnp.isfinite(free(mirror))
    np.testing.assert_allclose(AlanineDipeptideInternal(free)(q),
                               free(mirror)+internal.transform.log_volume(q), atol=1e-8, rtol=0)
    with pytest.raises(ValueError, match="chirality"):
        cartesian.replace(chirality="left")


def test_rigid_motion_does_not_change_energy():
    cartesian = AlanineDipeptide()
    x = cartesian.reference_positions
    rotated = x[:, jnp.array([1, 2, 0])] + jnp.array([.7, -.3, .5])
    np.testing.assert_allclose(cartesian(rotated), cartesian(x), atol=1e-8, rtol=0)


def test_jitted_temperature_sweep_with_captured_reference():
    """Captured concrete arrays can produce traced validation predicates."""
    dist = AlanineDipeptide()
    x = dist.reference_positions
    temperatures = jnp.array([270., 350., 440.])
    call = jax.jit(jax.vmap(lambda t: dist.replace(temperature=t)(x)))
    np.testing.assert_allclose(call(temperatures), dist(x)*300./temperatures, rtol=1e-10)


def test_internal_target_uses_the_cartesian_callable():
    """Coordinate conversion must retain custom Cartesian target behavior.

    Previously the wrapper bypassed __call__ and evaluated force-field fields
    directly. A simple analytic density offset exposes that loss of semantics.
    """
    from flax import struct

    @struct.dataclass
    class OffsetTarget(AlanineDipeptide):
        def __call__(self, x):
            return super().__call__(x)+2.3

    plain = AlanineDipeptide()
    shifted = OffsetTarget()
    q = AlanineDipeptideInternal(plain).transform.to_internal(plain.reference_positions)
    actual = (AlanineDipeptideInternal(shifted)(q)
              - AlanineDipeptideInternal(plain)(q))
    np.testing.assert_allclose(actual, 2.3, atol=1e-12)


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, jnp.ones(1)])
def test_temperature_misuse_is_diagnostic(bad):
    with pytest.raises(ValueError, match="temperature must be"):
        AlanineDipeptide(temperature=bad)


def test_configuration_and_input_misuse():
    from dataclasses import replace
    dist = AlanineDipeptide()
    for call, error, match in [
        (lambda: load_alanine_parameters("water"), ValueError, "solvent"),
        (lambda: AlanineDipeptide(params=None), TypeError, "ForceFieldParams"),
        (lambda: AlanineDipeptide(params=replace(dist.params, n_atoms=23)), ValueError, "22 alanine atoms"),
        (lambda: AlanineDipeptideInternal(cartesian=None), TypeError, "AlanineDipeptide instance"),
        (lambda: dist(dist.reference_positions.astype(jnp.float32)), TypeError, "float64"),
        (lambda: dist(dist.reference_positions.at[0, 0].set(jnp.nan)), ValueError, "finite"),
        (lambda: dist(jnp.zeros((66,))), ValueError, "Expected"),
    ]:
        with pytest.raises(error, match=match):
            call()
    with pytest.raises(RuntimeError, match="float64"):
        with jax.disable_jit(), enable_x64(False):
            AlanineDipeptide()


def test_asset_corruption_is_detected(tmp_path, monkeypatch):
    """A bad parameter asset must not silently redefine the benchmark."""
    manifest = json.loads((alanine._DATA/"alanine_manifest.json").read_text())
    (tmp_path/"alanine_manifest.json").write_text(json.dumps(manifest))
    (tmp_path/manifest["parameters"]["vacuum"]["file"]).write_bytes(b"corrupt")
    monkeypatch.setattr(alanine, "_DATA", tmp_path)
    with pytest.raises(ValueError, match="expected SHA256"):
        load_alanine_parameters()


def test_core_import_does_not_load_optional_backend():
    """Core import and evaluation work even when importing jaxmm is blocked."""
    code = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname == 'jaxmm' or fullname.startswith('openmm'):
            raise ModuleNotFoundError('blocked optional backend', name=fullname)
sys.meta_path.insert(0, Block())
import jax_pdf
assert 'jaxmm' not in sys.modules
assert jax_pdf.Banana2D().dim == 2
try:
    import jax_pdf.chemistry
except ModuleNotFoundError as error:
    assert 'jax-pdf[chemistry]' in str(error)
else:
    raise AssertionError('chemistry import unexpectedly succeeded')
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
