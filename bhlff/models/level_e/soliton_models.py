"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Soliton models for Level E experiments in 7D phase field theory.

This module provides the main interface for soliton models, importing
from specialized modules for core functionality and specific implementations.

Theoretical Background:
    Solitons are stable localized field configurations that minimize
    the energy functional while preserving topological charge. In the
    7D theory, they represent baryons and other particle-like structures
    through SU(3) field configurations with non-trivial winding numbers.

Mathematical Foundation:
    Implements SU(3) field configuration U(x,φ,t) with topological
    charge B = (1/24π²)∫ε^μνρσTr(L_ν L_ρ L_σ) and WZW term for
    baryon number conservation.

Example:
    >>> soliton = BaryonSoliton(domain, physics_params)
    >>> solution = soliton.find_soliton_solution(initial_guess)
"""

# Import core soliton functionality
from .soliton_core import SolitonModel
from .soliton_optimization import ConvergenceError

# Import specific soliton implementations
from .soliton_implementations import BaryonSoliton, SkyrmionSoliton

# Import specialized modules
from .soliton_energy import SolitonEnergyCalculator
from .soliton_stability import SolitonStabilityAnalyzer
from .soliton_optimization import SolitonOptimizer

# Re-export for backward compatibility
__all__ = [
    'SolitonModel',
    'BaryonSoliton', 
    'SkyrmionSoliton',
    'ConvergenceError',
    'SolitonEnergyCalculator',
    'SolitonStabilityAnalyzer',
    'SolitonOptimizer'
]