"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Phase field components for BHLFF.

This module provides phase field implementation, topology analysis,
winding calculations, and defect analysis for the 7D phase field theory.
"""

from .phase_field import PhaseField
from .topology import TopologyAnalyzer
from .winding import WindingCalculator
from .defects import DefectAnalyzer

__all__ = ["PhaseField", "TopologyAnalyzer", "WindingCalculator", "DefectAnalyzer"]
