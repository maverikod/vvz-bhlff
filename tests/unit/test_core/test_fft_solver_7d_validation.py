"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Facade for 7D FFT Solver validation tests.

This module provides a unified interface to all 7D FFT solver validation tests,
importing from the modular test structure for better maintainability.
"""

# Import all test classes from the modular structure
from tests.unit.test_core.fft_solver_7d_validation.test_basic_validation import TestBasicValidation
from tests.unit.test_core.fft_solver_7d_validation.test_numerical_validation import TestNumericalValidation
from tests.unit.test_core.fft_solver_7d_validation.test_boundary_cases import TestBoundaryCases

# Re-export all test classes for pytest discovery
__all__ = [
    'TestBasicValidation',
    'TestNumericalValidation', 
    'TestBoundaryCases'
]

# Legacy class name for backward compatibility - inherits from basic validation
class TestFFTSolver7DValidation(TestBasicValidation):
    """
    Legacy validation tests for 7D FFT Solver.
    
    Physical Meaning:
        Maintains backward compatibility while using the modular test structure.
    """
    pass
