"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Nonlinear effects package.

This package contains modules for nonlinear effects analysis
for Level F models in 7D phase field theory.

Physical Meaning:
    Implements nonlinear effects analysis including basic effects,
    soliton analysis, and mode analysis.

Example:
    >>> from bhlff.models.level_f.nonlinear import NonlinearEffects
    >>> effects = NonlinearEffects(system, nonlinear_params)
    >>> effects.add_nonlinear_interactions(nonlinear_params)
"""

from .basic_effects import BasicNonlinearEffects
from .soliton_analysis import SingleSolitonSolver, MultiSolitonSolutions
from .mode_analysis import NonlinearModeAnalyzer
