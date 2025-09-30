"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level F: Collective effects.

This module implements Level F functionality for multi-particle systems,
collective modes, phase transitions, and nonlinear effects in the 7D phase field theory.
"""

from .multi_particle import MultiParticleAnalyzer
from .collective import CollectiveModeAnalyzer
from .transitions import PhaseTransitionAnalyzer
from .nonlinear import NonlinearEffectAnalyzer

__all__ = [
    "MultiParticleAnalyzer",
    "CollectiveModeAnalyzer",
    "PhaseTransitionAnalyzer",
    "NonlinearEffectAnalyzer",
]
