"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Integration tests for automated testing system.

This module provides a facade for the automated testing integration tests
for 7D phase field theory experiments, ensuring end-to-end functionality
of physics validation, quality monitoring, and automated reporting.

Physical Meaning:
    Tests validate that the complete automated testing system
    correctly implements physics-first prioritization, quality
    monitoring, and automated reporting for 7D theory validation.

Example:
    >>> pytest tests/integration/test_automated_testing_integration.py -v
"""

# Import all test classes from the automated_testing package
from .automated_testing.test_basic_integration import TestBasicIntegration
from .automated_testing.test_physics_validation_integration import (
    TestPhysicsValidationIntegration,
)
from .automated_testing.test_quality_monitoring_integration import (
    TestQualityMonitoringIntegration,
)
from .automated_testing.test_automated_reporting_integration import (
    TestAutomatedReportingIntegration,
)
from .automated_testing.test_error_handling_integration import (
    TestErrorHandlingIntegration,
)
from .automated_testing.test_performance_integration import TestPerformanceIntegration

# Re-export all test classes for backward compatibility
__all__ = [
    "TestBasicIntegration",
    "TestPhysicsValidationIntegration",
    "TestQualityMonitoringIntegration",
    "TestAutomatedReportingIntegration",
    "TestErrorHandlingIntegration",
    "TestPerformanceIntegration",
]
