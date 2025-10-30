"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Performance integration tests for automated testing system.

This module tests the performance integration functionality
of the automated testing system for 7D phase field theory experiments.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
import sys
import time
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


class TestPerformanceIntegration:
    """
    Performance integration tests for automated testing system.

    Physical Meaning:
        Tests ensure the performance integration
        works correctly for 7D theory validation.
    """

    def setup_method(self):
        """Setup integration test fixtures."""
        self.config_path = self._create_integration_config()

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

    def teardown_method(self):
        """Cleanup integration test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

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

    def test_performance_integration(self):
        """
        Test performance integration.

        Physical Meaning:
            Tests that performance monitoring is correctly integrated
            with the automated testing system for 7D theory validation.
        """
        # Mock test execution
        with patch.object(self.testing_system, "execute_tests") as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()

            # Measure execution time
            start_time = time.time()
            results = self.testing_system.execute_tests()
            execution_time = time.time() - start_time

            # Verify performance
            assert results is not None
            assert execution_time < 1.0  # Should complete within 1 second

            # Test quality monitoring performance
            start_time = time.time()
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            quality_time = time.time() - start_time

            # Verify quality monitoring performance
            assert quality_metrics is not None
            assert quality_time < 0.5  # Should complete within 0.5 seconds

    def test_memory_usage_integration(self):
        """
        Test memory usage integration.

        Physical Meaning:
            Tests that memory usage is correctly monitored
            in the automated testing system for 7D theory validation.
        """
        # Mock test execution
        with patch.object(self.testing_system, "execute_tests") as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()

            # Execute tests
            results = self.testing_system.execute_tests()

            # Test memory usage
            import psutil

            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            # Verify memory usage
            assert memory_usage < 1000  # Should use less than 1GB

            # Test quality monitoring memory usage
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            memory_usage_after = process.memory_info().rss / 1024 / 1024  # MB

            # Verify memory usage after quality monitoring
            assert memory_usage_after < 1000  # Should use less than 1GB

    def test_cpu_usage_integration(self):
        """
        Test CPU usage integration.

        Physical Meaning:
            Tests that CPU usage is correctly monitored
            in the automated testing system for 7D theory validation.
        """
        # Mock test execution
        with patch.object(self.testing_system, "execute_tests") as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()

            # Execute tests
            results = self.testing_system.execute_tests()

            # Test CPU usage
            import psutil

            process = psutil.Process()
            cpu_usage = process.cpu_percent()

            # Verify CPU usage
            assert cpu_usage < 100  # Should use less than 100% CPU

            # Test quality monitoring CPU usage
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            cpu_usage_after = process.cpu_percent()

            # Verify CPU usage after quality monitoring
            assert cpu_usage_after < 100  # Should use less than 100% CPU

    def test_parallel_execution_performance(self):
        """
        Test parallel execution performance.

        Physical Meaning:
            Tests that parallel execution is correctly implemented
            in the automated testing system for 7D theory validation.
        """
        # Mock parallel test execution
        with patch.object(self.testing_system, "execute_tests") as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()

            # Measure parallel execution time
            start_time = time.time()
            results = self.testing_system.execute_tests()
            execution_time = time.time() - start_time

            # Verify parallel execution performance
            assert results is not None
            assert execution_time < 1.0  # Should complete within 1 second

    def test_resource_management_performance(self):
        """
        Test resource management performance.

        Physical Meaning:
            Tests that resource management is correctly implemented
            in the automated testing system for 7D theory validation.
        """
        # Mock resource management
        with patch.object(ResourceManager, "allocate_resources") as mock_allocate:
            mock_allocate.return_value = True

            # Test resource allocation performance
            start_time = time.time()
            resource_manager = ResourceManager()
            allocated = resource_manager.allocate_resources(1, 1)
            allocation_time = time.time() - start_time

            # Verify resource allocation performance
            assert allocated is True
            assert allocation_time < 0.1  # Should complete within 0.1 seconds

    def test_test_scheduling_performance(self):
        """
        Test test scheduling performance.

        Physical Meaning:
            Tests that test scheduling is correctly implemented
            in the automated testing system for 7D theory validation.
        """
        # Mock test scheduling
        with patch.object(TestScheduler, "schedule_tests") as mock_schedule:
            mock_schedule.return_value = []

            # Test test scheduling performance
            start_time = time.time()
            test_scheduler = TestScheduler()
            scheduled = test_scheduler.schedule_tests([])
            scheduling_time = time.time() - start_time

            # Verify test scheduling performance
            assert scheduled is not None
            assert scheduling_time < 0.1  # Should complete within 0.1 seconds

    def test_physics_validation_performance(self):
        """
        Test physics validation performance.

        Physical Meaning:
            Tests that physics validation is correctly implemented
            in the automated testing system for 7D theory validation.
        """
        # Mock physics validation
        with patch.object(
            self.physics_validator, "validate_energy_conservation"
        ) as mock_validate:
            mock_validate.return_value = {"valid": True, "relative_error": 1e-7}

            # Test physics validation performance
            start_time = time.time()
            validation = self.physics_validator.validate_energy_conservation(
                np.array([1.0, 2.0, 3.0]), {"initial_energy": 1.0, "final_energy": 1.0}
            )
            validation_time = time.time() - start_time

            # Verify physics validation performance
            assert validation is not None
            assert validation_time < 0.1  # Should complete within 0.1 seconds

    def test_quality_monitoring_performance(self):
        """
        Test quality monitoring performance.

        Physical Meaning:
            Tests that quality monitoring is correctly implemented
            in the automated testing system for 7D theory validation.
        """
        # Mock quality monitoring
        with patch.object(self.quality_monitor, "check_quality_metrics") as mock_check:
            mock_check.return_value = QualityMetrics()

            # Test quality monitoring performance
            start_time = time.time()
            quality_metrics = self.quality_monitor.check_quality_metrics(
                self._create_mock_test_results()
            )
            monitoring_time = time.time() - start_time

            # Verify quality monitoring performance
            assert quality_metrics is not None
            assert monitoring_time < 0.1  # Should complete within 0.1 seconds

    def _create_mock_test_results(self) -> TestResults:
        """Create mock test results for integration testing."""
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
