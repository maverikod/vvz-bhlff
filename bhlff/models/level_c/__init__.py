"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level C: Boundaries and Cells module for BVP framework.

This module implements Level C analysis focusing on boundaries and cells
in the 7D phase field theory, including boundary effects, resonators,
memory systems, and mode beating analysis.

Physical Meaning:
    Level C analyzes the boundary effects and cellular structures that
    emerge in the 7D phase field, including:
    - Boundary effects and their influence on field dynamics
    - Resonator structures and their frequency characteristics
    - Memory systems and information storage mechanisms
    - Mode beating and interference patterns

Mathematical Foundation:
    Implements analysis of:
    - Boundary conditions and their effects on field evolution
    - Resonator equations and frequency response
    - Memory kernel analysis and information retention
    - Mode coupling and beating frequency analysis

Example:
    >>> from bhlff.models.level_c import LevelCAnalyzer
    >>> analyzer = LevelCAnalyzer(bvp_core)
    >>> results = analyzer.analyze_boundaries_and_cells(envelope)
"""

from .boundaries import BoundaryAnalyzer
from .resonators import ResonatorAnalyzer
from .memory import MemoryAnalyzer
from .beating import BeatingAnalyzer

__all__ = [
    "BoundaryAnalyzer",
    "ResonatorAnalyzer", 
    "MemoryAnalyzer",
    "BeatingAnalyzer"
]
