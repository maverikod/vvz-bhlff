"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Automated testing integration tests package.

This package contains integration tests for the automated testing system
for 7D phase field theory experiments.
"""

from .test_basic_integration import TestBasicIntegration
from .test_physics_validation_integration import TestPhysicsValidationIntegration
from .test_quality_monitoring_integration import TestQualityMonitoringIntegration
from .test_automated_reporting_integration import TestAutomatedReportingIntegration
from .test_error_handling_integration import TestErrorHandlingIntegration
from .test_performance_integration import TestPerformanceIntegration
