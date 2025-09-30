"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power Balance postulate package.

This package implements the Power Balance postulate for the BVP framework,
validating that the BVP flux at the outer boundary equals the sum of core
energy growth, radiation losses, and reflection.

Physical Meaning:
    The Power Balance postulate ensures energy conservation by requiring that
    the BVP flux at the outer boundary equals the sum of growth of static
    core energy, EM/weak radiation/losses, and reflection. This is controlled
    by an integral identity.

Mathematical Foundation:
    Validates power balance by computing energy fluxes and ensuring
    conservation through the integral identity. The balance should be
    satisfied within a specified tolerance.

Example:
    >>> from bhlff.core.bvp.postulates.power_balance import BVPPostulate9_PowerBalance
    >>> postulate = BVPPostulate9_PowerBalance(domain_7d, config)
    >>> results = postulate.apply(envelope_7d)
    >>> print(f"Power balance satisfied: {results['postulate_satisfied']}")
"""

from .power_balance_postulate import BVPPostulate9_PowerBalance
from .flux_computer import FluxComputer
from .energy_computer import EnergyComputer
from .boundary_analyzer import BoundaryAnalyzer

__all__ = [
    "BVPPostulate9_PowerBalance",
    "FluxComputer",
    "EnergyComputer",
    "BoundaryAnalyzer",
]
