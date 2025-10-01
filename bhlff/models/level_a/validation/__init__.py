"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level A validation package for BVP framework compliance.

This package implements validation operations for the BVP framework,
ensuring that all components work correctly according to the 7D theory.
"""

from .validation import LevelAValidator
from .convergence_analysis import ConvergenceAnalysis
from .energy_analysis import EnergyAnalysis

__all__ = [
    'LevelAValidator',
    'ConvergenceAnalysis',
    'EnergyAnalysis'
]
