"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP (Base High-Frequency Field) package.

This package implements the central framework of the 7D theory where all
observed "modes" are envelope modulations and beatings of the Base
High-Frequency Field (BVP).

Physical Meaning:
    BVP serves as the central backbone of the entire system, where all
    observed particles and fields are manifestations of envelope modulations
    and beatings of the high-frequency carrier field.

Mathematical Foundation:
    BVP implements the envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.
"""

from .bvp_core import BVPCore
from .bvp_envelope_solver import BVPEnvelopeSolver
from .bvp_impedance_calculator import BVPImpedanceCalculator
from .bvp_interface import BVPInterface
from .bvp_constants import BVPConstants
from .quench_detector import QuenchDetector
from .phase_vector import PhaseVector
from .bvp_postulates import BVPPostulates

__all__ = [
    "BVPCore",
    "BVPEnvelopeSolver",
    "BVPImpedanceCalculator",
    "BVPInterface",
    "BVPConstants",
    "QuenchDetector",
    "PhaseVector",
    "BVPPostulates",
]
