"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Facade for nonlinear coefficients physics tests.

This module provides a unified interface to all nonlinear coefficients tests,
importing from the modular test structure for better maintainability.
"""

# Import all test classes from the modular structure
from .nonlinear_coefficients.test_basic_coefficients import TestBasicCoefficients
from .nonlinear_coefficients.test_advanced_coefficients import TestAdvancedCoefficients

# Re-export all test classes for pytest discovery
__all__ = [
    'TestBasicCoefficients',
    'TestAdvancedCoefficients'
]

# Legacy class name for backward compatibility
class TestNonlinearCoefficientsPhysics(TestBasicCoefficients):
    """
    Legacy nonlinear coefficients physics tests.
    
    Physical Meaning:
        Maintains backward compatibility while using the modular test structure.
    """
    pass