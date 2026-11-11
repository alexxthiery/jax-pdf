"""Flexible alanine dipeptide in Cartesian and physical internal coordinates."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmm
import numpy as np
from flax import struct
from jax import Array

from jax_pdf._validation import check_event_shape, is_concrete


_DATA = Path(__file__).with_name("data")


def _asset(kind):
    """Validate provenance schema and asset hash before exposing numeric data."""
    manifest = json.loads((_DATA/"alanine_manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("n_atoms") != 22:
        raise ValueError("Expected alanine asset schema_version=1 and n_atoms=22; regenerate assets.")
    record = manifest["reference"] if kind == "reference" else manifest["parameters"][kind]
    path = _DATA/record["file"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(f"Alanine asset {path.name}: expected SHA256 {record['sha256']}, got {actual}; regenerate assets.")
    return path


def load_alanine_parameters(solvent="vacuum") -> jaxmm.ForceFieldParams:
    """Load frozen flexible ALDP parameters, without OpenMM or network access.

    Args:
        solvent: ``'vacuum'`` or ``'implicit'`` (OBC1). Units: nm, kJ/mol.

    Returns:
        ForceFieldParams for the bundled 22-atom openmmtools topology.
    """
    if solvent not in ("vacuum", "implicit"):
        raise ValueError(f"Expected solvent 'vacuum' or 'implicit', got {solvent!r}.")
    jaxmm.require_float64()
    return jaxmm.load_params(str(_asset(solvent)))


def _reference_positions():
    with np.load(_asset("reference"), allow_pickle=False) as data:
        return jnp.asarray(data["positions"][0])


@struct.dataclass
class AlanineDipeptide:
    """Flexible ACE-L-ALA-NME Boltzmann factor ``-U/(kB*T)``.

    Input positions are in nm. The factor is translation invariant: sampling
    fixes the center of mass; this is not a proper density on unrestricted
    R**66. Rotations remain free. No confinement, clipping or bond constraints
    are added. The default restricts the signed CA tetrahedron to the L sector.

    Attributes:
        params: Molecular force field. Default: bundled vacuum parameters.
        temperature: Positive finite temperature in Kelvin. Default: 300.
        chirality: ``'L'`` or ``'unrestricted'``. Default: ``'L'``.
        reference_positions: Valid fallback structure, shape (22, 3), in nm.
            Independent of the force-field parameters; never used on valid events.
    """

    params: jaxmm.ForceFieldParams = struct.field(default_factory=load_alanine_parameters)
    temperature: float = 300.0
    chirality: str = struct.field(pytree_node=False, default="L")
    reference_positions: Array = struct.field(default_factory=_reference_positions)

    def __post_init__(self):
        jaxmm.require_float64()
        if self.chirality not in ("L", "unrestricted"):
            raise ValueError(f"Expected chirality 'L' or 'unrestricted', got {self.chirality!r}.")
        jaxmm.validate_temperature(self.temperature)
        if type(self.params) is not object:
            if not isinstance(self.params, jaxmm.ForceFieldParams):
                raise TypeError("params must be jaxmm.ForceFieldParams; use load_alanine_parameters.")
            if self.params.n_atoms != 22:
                raise ValueError(f"Expected 22 alanine atoms, got {self.params.n_atoms}.")
            if self.params.box is not None or self.params.restraints is not None:
                raise ValueError("Alanine benchmark expects no periodic box or external position restraints.")
            for leaf in jax.tree_util.tree_leaves(self.params):
                if is_concrete(leaf) and not bool(np.all(np.isfinite(np.asarray(leaf)))):
                    raise ValueError("Force-field parameters must be finite.")
        if type(self.reference_positions) is not object:
            check_event_shape(self.reference_positions, (22, 3))
            if self.reference_positions.shape != (22, 3):
                raise ValueError("reference_positions must have exactly shape (22, 3).")
            jaxmm.require_float64(self.reference_positions)
            if is_concrete(self.reference_positions):
                transform = jaxmm.InternalCoordinates(jaxmm.aldp_zmatrix())
                transform.to_internal(self.reference_positions)
                sector = jaxmm.aldp_in_l_sector(self.reference_positions)
                if is_concrete(sector) and not bool(sector):
                    raise ValueError("reference_positions must be in the L-alanine sector.")

    @property
    def n_particles(self) -> int:
        return 22

    @property
    def spatial_dim(self) -> int:
        return 3

    @property
    def dim(self) -> int:
        """Flat storage dimension; fixing COM leaves 63 intrinsic Cartesian DoF."""
        return self.n_particles*self.spatial_dim

    def __call__(self, x: Array) -> Array:
        """Log Boltzmann factor: positions ``(..., 22, 3)`` -> shape ``(...)``.

        Args:
            x: Float64 atom positions in nm, shape (..., 22, 3).

        Returns:
            Log Boltzmann factor of shape (...), or -inf outside the L sector.

        Raises:
            ValueError: On incorrect event shape or concrete nonfinite positions.
            TypeError: On coordinates other than float64.
        """
        check_event_shape(x, (self.n_particles, self.spatial_dim))
        jaxmm.require_float64(x)
        if is_concrete(x) and not bool(np.all(np.isfinite(np.asarray(x)))):
            raise ValueError("Cartesian positions must be finite.")
        valid = jnp.all(jnp.isfinite(x), axis=(-2, -1))
        if self.chirality == "L":
            valid = valid & jaxmm.aldp_in_l_sector(x)
        safe = jnp.where(valid[..., None, None], x, self.reference_positions)
        values = jaxmm.map_events(lambda y: jaxmm.log_boltzmann(y, self.params, self.temperature), safe, 2)
        return jnp.where(valid, values, -jnp.inf)

    def log_normalization(self) -> Array:
        """Unavailable; the unrestricted translational integral also diverges."""
        raise NotImplementedError("Alanine normalization is unknown on the fixed-COM space; unrestricted translations give an infinite integral.")


@struct.dataclass
class AlanineDipeptideInternal:
    """Physical 60D conformational density of a shared Cartesian ALDP target.

    ``q = [21 bonds, 20 angles, 19 torsions]`` in nm/radians. Both targets
    share parameters, temperature and chirality. Rigid orientation is integrated
    out, with constant 8*pi**2 and translation volume omitted. No whitening.

    Attributes:
        cartesian: Shared Cartesian target. Default: vacuum, 300 K, L sector.
    """

    cartesian: AlanineDipeptide = struct.field(default_factory=AlanineDipeptide)

    def __post_init__(self):
        if type(self.cartesian) is not object and not isinstance(self.cartesian, AlanineDipeptide):
            raise TypeError("cartesian must be an AlanineDipeptide instance.")

    @property
    def transform(self) -> jaxmm.InternalCoordinates:
        """Geometry, physical volume, support, bounds and periodic metadata."""
        return jaxmm.InternalCoordinates(jaxmm.aldp_zmatrix())

    @property
    def dim(self) -> int:
        return self.transform.dim

    def __call__(self, q: Array) -> Array:
        """Physical log density: internals ``(..., 60)`` -> shape ``(...)``.

        Invalid support events return -inf with zero rejection gradients.
        Malformed shapes and dtypes raise, including during JIT tracing.

        Args:
            q: Float64 packed internals in nm/radians, shape (..., 60).

        Returns:
            Physical internal log density of shape (...).
        """
        check_event_shape(q, (self.dim,))
        jaxmm.require_float64(q)
        transform = self.transform
        return transform.pullback(
            self.cartesian, q,
            reference=transform.to_internal(self.cartesian.reference_positions))

    def log_normalization(self) -> Array:
        """Unknown configurational integral, in the documented shape measure."""
        raise NotImplementedError("Alanine internal normalizing constant is intractable.")
