"""Evaluate the optional alanine target; no OpenMM or trajectory required."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvent", choices=("vacuum", "implicit"), default="vacuum")
    parser.add_argument("--temperature", type=float, default=300.)
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    from jax_pdf.chemistry import (
        AlanineDipeptide, AlanineDipeptideInternal, load_alanine_parameters,
    )
    cartesian = AlanineDipeptide(params=load_alanine_parameters(args.solvent),
                                 temperature=args.temperature)
    internal = AlanineDipeptideInternal(cartesian)
    x = cartesian.reference_positions
    q = internal.transform.to_internal(x)
    print(json.dumps({"solvent": args.solvent, "temperature_kelvin": args.temperature,
                      "cartesian_log_factor": float(cartesian(x)),
                      "internal_log_density": float(internal(q)),
                      "physical_log_volume": float(internal.transform.log_volume(q)),
                      "internal_dim": internal.dim}, indent=2))


if __name__ == "__main__":
    main()
