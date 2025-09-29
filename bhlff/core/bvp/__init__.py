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
    BVP implements the 7D envelope equation in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.
    The envelope a(x,φ,t) is a vector of three U(1) phase components Θ_a (a=1..3).
"""

from .bvp_core import BVPCore
from .bvp_envelope_solver import BVPEnvelopeSolver
from .bvp_impedance_calculator import BVPImpedanceCalculator
from .bvp_interface import BVPInterface
from .bvp_constants import BVPConstants
from .quench_detector import QuenchDetector
from .phase_vector import PhaseVector
from .bvp_postulates import BVPPostulates
from .bvp_level_integration import BVPLevelIntegration

__all__ = [
    "BVPCore",
    "BVPEnvelopeSolver",
    "BVPImpedanceCalculator",
    "BVPInterface",
    "BVPConstants",
    "QuenchDetector",
    "PhaseVector",
    "BVPPostulates",
    "BVPLevelIntegration",
]
