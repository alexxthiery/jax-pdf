"""Optional molecular benchmarks; install jax-pdf[chemistry] before importing.

OpenMM is only needed by the asset regeneration tool, never by these targets.
"""

try:
    import jaxmm as _jaxmm
except ModuleNotFoundError as error:
    if error.name != "jaxmm":
        raise
    raise ModuleNotFoundError(
        "Molecular targets require jaxmm >= 0.2. Install jax-pdf[chemistry] "
        "(Python >= 3.10), or install the maintained jaxmm checkout."
    ) from error

if any(not hasattr(_jaxmm, name) for name in
       ("InternalCoordinates", "log_boltzmann_fixed_frame", "aldp_in_l_sector",
        "map_events", "require_float64", "validate_temperature")):
    raise ImportError("Molecular targets require the current jaxmm >= 0.2 API; upgrade jaxmm.")

if not hasattr(_jaxmm.InternalCoordinates, "pullback"):
    raise ImportError("Molecular targets require the current jaxmm coordinate pullback API; upgrade jaxmm.")

from jax_pdf.chemistry.alanine import (
    AlanineDipeptide, AlanineDipeptideInternal, load_alanine_parameters,
)

__all__ = ["AlanineDipeptide", "AlanineDipeptideInternal", "load_alanine_parameters"]
