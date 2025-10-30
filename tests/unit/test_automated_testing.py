"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for automated testing system.

This module provides a facade for the automated testing unit tests
for 7D phase field theory experiments, ensuring proper functionality
of physics validation, test scheduling, and quality monitoring.

Physical Meaning:
    Tests validate that the automated testing system correctly
    implements physics-first prioritization and maintains
    adherence to 7D theory principles during test execution.

Example:
    >>> pytest tests/unit/test_automated_testing.py -v
"""

# Import all test classes from the automated_testing package
from .automated_testing.test_automated_testing_system import TestAutomatedTestingSystem
from .automated_testing.test_quality_monitor import TestQualityMonitor
from .automated_testing.test_automated_reporting import TestAutomatedReportingSystem
from .automated_testing.test_integration import TestIntegration

# Re-export all test classes for backward compatibility
__all__ = [
    "TestAutomatedTestingSystem",
    "TestQualityMonitor",
    "TestAutomatedReportingSystem",
    "TestIntegration",
]
