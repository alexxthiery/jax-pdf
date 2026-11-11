"""Rebuild frozen alanine benchmark assets and independent OpenMM references.

Run with jaxmm >=0.2, OpenMM and openmmtools installed. This is setup tooling;
normal chemistry target construction never imports OpenMM or runs simulations.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import jaxmm
    import openmm
    from openmm import unit
    from openmmtools import testsystems

    output.mkdir(parents=True, exist_ok=True)
    frames = None
    references = {}
    parameters = {}
    for solvent, system_type in (("vacuum", testsystems.AlanineDipeptideVacuum),
                                  ("implicit", testsystems.AlanineDipeptideImplicit)):
        molecule = system_type(constraints=None)
        params = jaxmm.extract_params(molecule.system)
        path = output / f"alanine_{solvent}.npz"
        jaxmm.save_params(params, str(path))
        parameters[solvent] = {"file": path.name, "sha256": digest(path),
                               "system": system_type.__name__}
        if frames is None:
            x = np.asarray(molecule.positions.value_in_unit(unit.nanometer), dtype=np.float64)
            # Deterministic, nonsymmetric configurations; no MD or fitted statistics.
            delta = 0.001*np.sin(np.arange(x.size).reshape(x.shape)*1.7 + 0.3)
            frames = np.stack((x, x+delta, x-1.3*delta))
            if not bool(jnp.all(jaxmm.aldp_in_l_sector(jnp.asarray(frames)))):
                raise ValueError("Reference configurations must be in the L-alanine sector.")
            references["positions"] = frames
        context = openmm.Context(molecule.system, openmm.VerletIntegrator(0.001),
                                 openmm.Platform.getPlatformByName("Reference"))
        energies, forces = [], []
        for x in frames:
            context.setPositions(x*unit.nanometer)
            state = context.getState(getEnergy=True, getForces=True)
            energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            forces.append(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole/unit.nanometer))
        references[f"{solvent}_energies"] = np.asarray(energies)
        references[f"{solvent}_forces"] = np.asarray(forces)
        del context
    reference_path = output / "alanine_reference.npz"
    np.savez(reference_path, **references)
    source_dir = Path(testsystems.get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop")).parent
    source_hashes = {name: digest(source_dir/name) for name in
                     ("alanine-dipeptide.prmtop", "alanine-dipeptide.crd")}
    backend_dir = Path(jaxmm.__file__).parent
    manifest = {
        "schema_version": 1,
        "molecule": "ACE-L-ALA-NME",
        "n_atoms": 22,
        "atom_order": "openmmtools alanine-dipeptide-gbsa topology",
        "atoms": [{"index": atom.index, "name": atom.name,
                   "residue": atom.residue.name, "residue_index": atom.residue.index}
                  for atom in molecule.topology.atoms()],
        "units": {"positions": "nm", "energy": "kJ/mol", "angles": "radians"},
        "constraints": None,
        "force_field": "AMBER parameters in the hashed openmmtools prmtop; not relabeled as ff99SB-ILDN",
        "implicit_solvent": "OBC1",
        "source_files_sha256": source_hashes,
        "parameters": parameters,
        "reference": {"file": reference_path.name, "sha256": digest(reference_path),
                      "generator": "OpenMM Reference platform, original and deterministic perturbed coordinates"},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("jax", "numpy", "openmm", "openmmtools")},
        "jaxmm_version": jaxmm.__version__,
        "backend_source_sha256": {name: digest(backend_dir/name) for name in
                                  ("energy.py", "extract.py", "coordinates.py", "internal.py", "utils.py", "batching.py", "_validation.py")},
    }
    (output/"alanine_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Wrote alanine assets and provenance to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1]/"jax_pdf"/"chemistry"/"data")
    build(parser.parse_args().output)
