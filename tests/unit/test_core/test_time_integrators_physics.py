"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Facade for time integrators physics tests.

This module provides a unified interface to all time integrators tests,
importing from the modular test structure for better maintainability.
"""

# Import all test classes from the modular structure
from tests.unit.test_core.time_integrators.test_basic_integrators import (
    TestBasicIntegrators,
)
from tests.unit.test_core.time_integrators.test_advanced_integrators import (
    TestAdvancedIntegrators,
)

# Re-export all test classes for pytest discovery
__all__ = ["TestBasicIntegrators", "TestAdvancedIntegrators"]


# Legacy class removed - no longer needed
