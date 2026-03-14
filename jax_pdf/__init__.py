"""jax-pdf: Probability density functions for MCMC/VI testing."""

from jax_pdf.banana import Banana2D
from jax_pdf.double_well import DoubleWell
from jax_pdf.dw4 import DW4
from jax_pdf.lennard_jones import LennardJones
from jax_pdf.log_gauss_pines import LGCP
from jax_pdf.muller_brown import MullerBrown
from jax_pdf.neal_funnel import NealFunnel
from jax_pdf.phi_four import PhiFour

__version__ = "0.1.0"
__all__ = [
    "Banana2D",
    "DoubleWell",
    "DW4",
    "LGCP",
    "LennardJones",
    "MullerBrown",
    "NealFunnel",
    "PhiFour",
]
