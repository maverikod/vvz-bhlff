"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple tests for Level E experiments.

This module provides a facade for the Level E experiments tests
for 7D phase field theory, ensuring proper functionality
of sensitivity analysis, robustness testing, and soliton/defect models.

Physical Meaning:
    Tests the fundamental functionality of Level E experiments,
    ensuring that sensitivity analysis, robustness testing, and
    soliton/defect models work correctly.

Example:
    >>> pytest tests/unit/test_level_e/test_level_e_simple.py
"""

# Import all test classes from the Level E packages
from .sensitivity.test_sensitivity_analyzer import TestSensitivityAnalyzer
from .robustness.test_robustness_tester import TestRobustnessTester
from .discretization.test_discretization_analyzer import TestDiscretizationAnalyzer
from .failure.test_failure_detector import TestFailureDetector
from .phase_mapping.test_phase_mapper import TestPhaseMapper
from .performance.test_performance_analyzer import TestPerformanceAnalyzer
from .solitons.test_soliton_models import TestSolitonModels
from .defects.test_defect_models import TestDefectModels
from .experiments.test_level_e_experiments import TestLevelEExperiments
from .integration.test_integration import TestIntegration

# Re-export all test classes for backward compatibility
__all__ = [
    'TestSensitivityAnalyzer',
    'TestRobustnessTester',
    'TestDiscretizationAnalyzer',
    'TestFailureDetector',
    'TestPhaseMapper',
    'TestPerformanceAnalyzer',
    'TestSolitonModels',
    'TestDefectModels',
    'TestLevelEExperiments',
    'TestIntegration'
]