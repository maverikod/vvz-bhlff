"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Facade for frequency-dependent properties physics tests.

This module provides a unified interface to all frequency-dependent properties tests,
importing from the modular test structure for better maintainability.
"""

# Import all test classes from the modular structure
from tests.unit.test_core.frequency_dependent_properties.test_basic_properties import (
    TestBasicProperties,
)
from tests.unit.test_core.frequency_dependent_properties.test_advanced_properties import (
    TestAdvancedProperties,
)

# Re-export all test classes for pytest discovery
__all__ = ["TestBasicProperties", "TestAdvancedProperties"]


# Legacy class removed - no longer needed
