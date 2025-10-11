"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for NonlinearEffects class in Level F models.

This module provides a facade for the nonlinear effects tests
for Level F models, ensuring proper functionality
of nonlinear interactions, soliton solutions, and stability analysis.

Physical Meaning:
    Tests verify that nonlinear effects are correctly
    implemented in multi-particle systems, including
    nonlinear modes, solitons, and stability analysis.

Example:
    >>> pytest tests/unit/test_level_f/test_nonlinear.py
"""

# Import all test classes from the nonlinear package
from .nonlinear.test_nonlinear_initialization import TestNonlinearInitialization
from .nonlinear.test_soliton_analysis import TestSolitonAnalysis
from .nonlinear.test_nonlinear_types import TestNonlinearTypes

# Re-export all test classes for backward compatibility
__all__ = [
    'TestNonlinearInitialization',
    'TestSolitonAnalysis',
    'TestNonlinearTypes'
]