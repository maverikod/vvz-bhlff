"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic integration tests for automated testing system.

This module tests the basic integration functionality of the automated
testing system for 7D phase field theory experiments.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from bhlff.testing.automated_testing import (
    AutomatedTestingSystem,
    PhysicsValidator,
    TestScheduler,
    ResourceManager,
    TestResult,
    TestStatus,
    LevelTestResults,
    TestResults,
    TestPriority,
)
from bhlff.testing.quality_monitor import (
    QualityMonitor,
    PhysicsConstraints,
    QualityMetrics,
    QualityStatus,
    DegradationReport,
    QualityAlert,
    AlertSeverity,
)
from bhlff.testing.automated_reporting import (
    AutomatedReportingSystem,
    PhysicsInterpreter,
    TemplateEngine,
    DataAggregator,
)


class TestBasicIntegration:
    """
    Basic integration tests for automated testing system.

    Physical Meaning:
        Tests ensure the basic integration functionality
        works correctly for 7D theory validation.
    """

    def setup_method(self):
        """Setup integration test fixtures."""
        self.config_path = self._create_integration_config()
        self.reporting_config_path = self._create_reporting_config()

        # Initialize physics validator
        self.physics_validator = PhysicsValidator(
            {
                "energy_conservation": {"max_relative_error": 1e-6},
                "virial_conditions": {"max_relative_error": 1e-6},
                "topological_charge": {"max_relative_error": 1e-8},
                "passivity": {"tolerance": 1e-12},
            }
        )

        # Initialize testing system
        self.testing_system = AutomatedTestingSystem(
            self.config_path, self.physics_validator
        )

        # Initialize quality monitor
        self.baseline_metrics = {
            "energy_conservation": 0.95,
            "virial_conditions": 0.90,
            "topological_charge": 0.98,
            "passivity": 0.99,
        }

        self.physics_constraints = PhysicsConstraints(
            {
                "energy_conservation": {"max_relative_error": 1e-6},
                "virial_conditions": {"max_relative_error": 1e-6},
                "topological_charge": {"max_relative_error": 1e-8},
                "passivity": {"tolerance": 1e-12},
            }
        )

        self.quality_monitor = QualityMonitor(
            self.baseline_metrics, self.physics_constraints
        )

        # Initialize reporting system
        self.reporting_system = AutomatedReportingSystem(self.reporting_config_path)

    def teardown_method(self):
        """Cleanup integration test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.reporting_config_path):
            os.remove(self.reporting_config_path)

    def _create_integration_config(self) -> str:
        """Create integration test configuration."""
        config = {
            "domain": {"L": 1.0, "N": 64, "dimensions": 7},
            "physics": {"mu": 1.0, "beta": 1.5, "lambda": 0.1, "nu": 1.0},
            "solver": {
                "precision": "float64",
                "fft_plan": "MEASURE",
                "tolerance": 1e-12,
            },
            "testing": {"max_tests": 10, "timeout": 300, "parallel": False},
            "levels": {
                "A": {"enabled": True, "priority": "high"},
                "B": {"enabled": True, "priority": "high"},
                "C": {"enabled": True, "priority": "medium"},
                "D": {"enabled": True, "priority": "medium"},
                "E": {"enabled": True, "priority": "low"},
                "F": {"enabled": True, "priority": "low"},
                "G": {"enabled": True, "priority": "low"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f, indent=2)
            return f.name

    def _create_reporting_config(self) -> str:
        """Create reporting test configuration."""
        config = {
            "reporting": {
                "enabled": True,
                "format": "html",
                "template": "physics_report.html",
            },
            "distribution": {
                "email": {
                    "enabled": False,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "test@example.com",
                    "password": "test_password",
                }
            },
            "physics": {
                "interpretation": {
                    "energy_conservation": "Energy conservation validation",
                    "virial_conditions": "Virial conditions validation",
                    "topological_charge": "Topological charge validation",
                    "passivity": "Passivity validation",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f, indent=2)
            return f.name

    def test_complete_automated_testing_workflow(self):
        """
        Test complete automated testing workflow.

        Physical Meaning:
            Tests the complete workflow from test execution
            through quality monitoring to automated reporting.
        """
        # Mock test execution
        with patch.object(self.testing_system, "execute_tests") as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()

            # Execute tests
            results = self.testing_system.execute_tests()

            # Verify test execution
            assert results is not None
            assert len(results.level_results) > 0

            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            assert quality_metrics is not None
            assert quality_metrics.overall_score > 0

            # Test reporting
            report = self.reporting_system.generate_report(results, quality_metrics)
            assert report is not None
            assert "html" in report.lower() or "json" in report.lower()

    def _create_mock_test_results(self) -> TestResults:
        """Create mock test results for integration testing."""
        # Create mock test results for each level
        level_results = {}

        for level in ["A", "B", "C", "D", "E", "F", "G"]:
            # Create mock test result
            test_result = TestResult(
                test_name=f"test_{level}_physics",
                status=TestStatus.PASSED,
                execution_time=1.0,
                physics_validation={
                    "energy_conservation": {"relative_error": 1e-7, "valid": True},
                    "virial_conditions": {"relative_error": 1e-7, "valid": True},
                    "topological_charge": {"relative_error": 1e-9, "valid": True},
                    "passivity": {"min_real_part": 1e-13, "valid": True},
                },
                numerical_metrics={
                    "convergence_rate": 0.95,
                    "accuracy": 0.98,
                    "stability": 0.97,
                },
                metadata={"level": level, "timestamp": datetime.now().isoformat()},
            )

            # Create level test results
            level_test_results = LevelTestResults(
                level=level,
                test_results=[test_result],
                total_tests=1,
                passed_tests=1,
                failed_tests=0,
                execution_time=1.0,
            )

            level_results[level] = level_test_results

        # Create overall test results
        return TestResults(
            level_results=level_results,
            total_tests=7,
            passed_tests=7,
            failed_tests=0,
            total_execution_time=7.0,
            timestamp=datetime.now(),
        )
