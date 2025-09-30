"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level E: Solitons and defects.

This module implements Level E functionality for soliton analysis,
defect dynamics, interactions, and formation in the 7D phase field theory.
"""

from .solitons import SolitonAnalyzer
from .dynamics import DefectDynamicsAnalyzer
from .interactions import DefectInteractionAnalyzer
from .formation import DefectFormationAnalyzer

__all__ = [
    "SolitonAnalyzer",
    "DefectDynamicsAnalyzer",
    "DefectInteractionAnalyzer",
    "DefectFormationAnalyzer",
]
