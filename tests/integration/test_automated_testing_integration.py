"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Integration tests for automated testing system.

This module tests the complete automated testing system integration
for 7D phase field theory experiments, ensuring end-to-end functionality
of physics validation, quality monitoring, and automated reporting.

Physical Meaning:
    Tests validate that the complete automated testing system
    correctly implements physics-first prioritization, quality
    monitoring, and automated reporting for 7D theory validation.

Example:
    >>> pytest tests/integration/test_automated_testing_integration.py -v
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
project_root = Path(__file__).parent.parent.parent
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
    DailyReport,
    WeeklyReport,
    MonthlyReport,
    TemplateEngine,
    DataAggregator,
)


class TestAutomatedTestingIntegration:
    """
    Integration tests for complete automated testing system.

    Physical Meaning:
        Tests ensure the complete automated testing system
        works together correctly for 7D theory validation
        with physics-first prioritization and quality monitoring.
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
        self.reporting_config = self._load_reporting_config()
        self.physics_interpreter = PhysicsInterpreter(
            {
                "energy_conservation": {
                    "description": "Energy conservation validation in 7D phase field theory",
                    "mathematical_expression": "|dE/dt| < ε_energy",
                    "physical_meaning": "Fundamental conservation law for phase field dynamics",
                },
                "virial_conditions": {
                    "description": "Virial condition validation for energy balance",
                    "mathematical_expression": "|dE/dλ|λ=1| < ε_virial",
                    "physical_meaning": "Ensures proper energy distribution in phase fields",
                },
                "topological_charge": {
                    "description": "Topological charge conservation validation",
                    "mathematical_expression": "|dB/dt| < ε_topology",
                    "physical_meaning": "Essential for particle stability and charge quantization",
                },
                "passivity": {
                    "description": "Passivity condition validation",
                    "mathematical_expression": "Re Y(ω) ≥ 0 for all ω",
                    "physical_meaning": "Ensures physical realizability of phase field responses",
                },
            }
        )

        self.reporting_system = AutomatedReportingSystem(
            self.reporting_config, self.physics_interpreter
        )

    def teardown_method(self):
        """Cleanup integration test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.reporting_config_path):
            os.remove(self.reporting_config_path)

    def _create_integration_config(self):
        """Create integration test configuration."""
        config = {
            "scheduling": {
                "critical_physics_validation": {
                    "frequency": "daily",
                    "time": "00:00",
                    "priority": "critical",
                    "physics_checks": [
                        "energy_conservation",
                        "topological_charge",
                        "virial_conditions",
                    ],
                },
                "level_a_validation": {
                    "frequency": "daily",
                    "time": "01:00",
                    "priority": "high",
                    "dependencies": [],
                    "physics_checks": [
                        "solver_accuracy",
                        "energy_balance",
                        "passivity",
                    ],
                },
                "level_b_validation": {
                    "frequency": "daily",
                    "time": "02:00",
                    "priority": "high",
                    "dependencies": ["level_a_validation"],
                    "physics_checks": [
                        "power_law_tail",
                        "topological_charge",
                        "zone_separation",
                    ],
                },
            },
            "parallel_execution": {
                "max_workers": 2,
                "timeout": 300,
                "resource_limits": {
                    "max_memory_per_worker": "1GB",
                    "max_cpu_per_worker": 50,
                    "max_disk_io_per_worker": "50MB/s",
                },
                "physics_validation": {
                    "real_time_validation": True,
                    "validation_frequency": 10,
                    "energy_conservation_tolerance": 1e-6,
                    "topological_charge_tolerance": 1e-8,
                    "virial_condition_tolerance": 1e-6,
                },
            },
            "monitoring": {
                "performance_thresholds": {
                    "max_execution_time": 300,
                    "max_memory_usage": "2GB",
                    "max_cpu_usage": 80,
                    "fft_scaling_efficiency": 0.8,
                    "memory_scaling_efficiency": 0.7,
                    "physics_validation_overhead": 0.1,
                },
                "quality_thresholds": {
                    "min_success_rate": 0.95,
                    "max_accuracy_degradation": 0.05,
                    "energy_conservation_tolerance": 1e-6,
                    "virial_condition_tolerance": 1e-6,
                    "topological_charge_tolerance": 1e-8,
                    "passivity_violation_tolerance": 0.0,
                    "power_law_accuracy_tolerance": 0.03,
                    "spectral_peak_accuracy_tolerance": 0.05,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            return f.name

    def _create_reporting_config(self):
        """Create reporting configuration."""
        config = {
            "report_types": {
                "daily": {
                    "enabled": True,
                    "recipients": {
                        "physicists": ["physics-team@example.com"],
                        "developers": ["dev-team@example.com"],
                        "management": ["management@example.com"],
                    },
                    "format": ["html", "json"],
                    "physics_highlights": True,
                    "technical_details": True,
                    "executive_summary": True,
                }
            },
            "distribution": {
                "email": {
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "use_tls": False,
                    "username": "test@example.com",
                    "password": "test_password",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            return f.name

    def _load_reporting_config(self):
        """Load reporting configuration."""
        with open(self.reporting_config_path, "r") as f:
            return json.load(f)

    def test_complete_automated_testing_workflow(self):
        """
        Test complete automated testing workflow.

        Physical Meaning:
            Verifies that the complete automated testing workflow
            correctly executes with physics validation, quality
            monitoring, and automated reporting.
        """
        # Mock test execution to avoid actual test runs
        with patch.object(self.testing_system, "run_level_tests") as mock_run_level:
            # Create mock level results
            mock_level_results = LevelTestResults(level="A")
            mock_level_results.add_test_result(
                TestResult(
                    test_id="integration_test_1",
                    test_name="Energy Conservation Test",
                    level="A",
                    status=TestStatus.PASSED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    physics_validation={
                        "compliance_score": 0.95,
                        "energy_conservation": {"relative_error": 1e-8},
                        "virial_conditions": {"relative_error": 1e-8},
                        "topological_charge": {"relative_error": 1e-10},
                        "passivity": {"min_real_part": 1e-12},
                    },
                    numerical_metrics={
                        "convergence_rate": 0.95,
                        "accuracy": 0.98,
                        "stability": 0.97,
                    },
                )
            )

            mock_level_results.add_test_result(
                TestResult(
                    test_id="integration_test_2",
                    test_name="Virial Conditions Test",
                    level="A",
                    status=TestStatus.PASSED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    physics_validation={
                        "compliance_score": 0.92,
                        "energy_conservation": {"relative_error": 1e-7},
                        "virial_conditions": {"relative_error": 1e-7},
                        "topological_charge": {"relative_error": 1e-9},
                        "passivity": {"min_real_part": 1e-11},
                    },
                    numerical_metrics={
                        "convergence_rate": 0.92,
                        "accuracy": 0.95,
                        "stability": 0.94,
                    },
                )
            )

            mock_run_level.return_value = mock_level_results

            # Run automated testing
            test_results = self.testing_system.run_all_tests(
                levels=["A"], priority="physics"
            )

            # Verify test results
            assert isinstance(test_results, TestResults)
            assert "A" in test_results.level_results
            assert test_results.level_results["A"].total_tests == 2
            assert test_results.level_results["A"].passed_tests == 2
            assert test_results.level_results["A"].get_success_rate() == 1.0

            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(test_results)
            assert isinstance(quality_metrics, QualityMetrics)
            assert quality_metrics.energy_conservation > 0.9
            assert quality_metrics.virial_conditions > 0.9
            assert quality_metrics.topological_charge > 0.9
            assert quality_metrics.passivity > 0.9
            assert quality_metrics.overall_score > 0.9
            assert quality_metrics.status in [
                QualityStatus.EXCELLENT,
                QualityStatus.GOOD,
            ]

            # Test automated reporting
            daily_report = self.reporting_system.generate_daily_report(test_results)
            assert isinstance(daily_report, DailyReport)
            assert daily_report.date is not None
            assert "A" in daily_report.level_analysis
            assert daily_report.physics_summary is not None

    def test_physics_validation_integration(self):
        """
        Test physics validation integration.

        Physical Meaning:
            Verifies that physics validation correctly
            identifies violations of 7D theory principles
            and maintains conservation laws.
        """
        # Test with valid physics
        valid_test_result = TestResult(
            test_id="physics_test_valid",
            test_name="Valid Physics Test",
            level="A",
            status=TestStatus.PASSED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            physics_validation={
                "energy_conservation": {"relative_error": 1e-8},
                "virial_conditions": {"relative_error": 1e-8},
                "topological_charge": {"relative_error": 1e-10},
                "passivity": {"min_real_part": 1e-12},
            },
        )

        validation_result = self.physics_validator.validate_result(valid_test_result)
        assert validation_result["is_valid"] is True
        assert validation_result["compliance_score"] > 0.9

        # Test with invalid physics
        invalid_test_result = TestResult(
            test_id="physics_test_invalid",
            test_name="Invalid Physics Test",
            level="A",
            status=TestStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            physics_validation={
                "energy_conservation": {"relative_error": 1e-4},  # Too large
                "virial_conditions": {"relative_error": 1e-4},  # Too large
                "topological_charge": {"relative_error": 1e-6},  # Too large
                "passivity": {"min_real_part": -1e-6},  # Negative
            },
        )

        validation_result = self.physics_validator.validate_result(invalid_test_result)
        assert validation_result["is_valid"] is False
        assert len(validation_result["violations"]) > 0

        # Check specific violations
        violation_types = [v["constraint"] for v in validation_result["violations"]]
        assert "energy_conservation" in violation_types
        assert "virial_conditions" in violation_types
        assert "topological_charge" in violation_types
        assert "passivity" in violation_types

    def test_quality_monitoring_integration(self):
        """
        Test quality monitoring integration.

        Physical Meaning:
            Verifies that quality monitoring correctly
            tracks physics validation metrics and detects
            degradation patterns.
        """
        # Create test results with good quality
        test_results = TestResults(start_time=datetime.now())
        test_results.end_time = datetime.now()

        level_results = LevelTestResults(level="A")
        level_results.add_test_result(
            TestResult(
                test_id="quality_test_1",
                test_name="Quality Test 1",
                level="A",
                status=TestStatus.PASSED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                physics_validation={
                    "compliance_score": 0.95,
                    "energy_conservation": {"relative_error": 1e-8},
                    "virial_conditions": {"relative_error": 1e-8},
                    "topological_charge": {"relative_error": 1e-10},
                    "passivity": {"min_real_part": 1e-12},
                },
                numerical_metrics={
                    "convergence_rate": 0.95,
                    "accuracy": 0.98,
                    "stability": 0.97,
                    "peak_accuracy": 0.96,
                    "quality_factor": 0.94,
                    "abcd_accuracy": 0.93,
                },
            )
        )

        test_results.add_level_results("A", level_results)

        # Test quality monitoring
        quality_metrics = self.quality_monitor.check_quality_metrics(test_results)
        assert isinstance(quality_metrics, QualityMetrics)
        assert quality_metrics.energy_conservation > 0.9
        assert quality_metrics.virial_conditions > 0.9
        assert quality_metrics.topological_charge > 0.9
        assert quality_metrics.passivity > 0.9
        assert quality_metrics.overall_score > 0.9
        assert quality_metrics.status in [QualityStatus.EXCELLENT, QualityStatus.GOOD]

        # Test degradation detection
        current_metrics = {
            "energy_conservation": 0.80,  # Degraded
            "virial_conditions": 0.85,  # Degraded
            "topological_charge": 0.90,  # Slightly degraded
            "passivity": 0.95,  # Good
        }

        historical_metrics = [
            {
                "energy_conservation": 0.95,
                "virial_conditions": 0.90,
                "topological_charge": 0.98,
                "passivity": 0.99,
            },
            {
                "energy_conservation": 0.93,
                "virial_conditions": 0.89,
                "topological_charge": 0.97,
                "passivity": 0.98,
            },
            {
                "energy_conservation": 0.90,
                "virial_conditions": 0.87,
                "topological_charge": 0.95,
                "passivity": 0.97,
            },
        ]

        degradation_report = self.quality_monitor.detect_quality_degradation(
            current_metrics, historical_metrics
        )
        assert isinstance(degradation_report, DegradationReport)
        assert "energy_conservation" in degradation_report.physics_degradation
        assert "virial_conditions" in degradation_report.physics_degradation

        # Test alert generation
        alerts = self.quality_monitor.generate_quality_alerts(degradation_report)
        assert len(alerts) > 0
        assert any(
            alert.alert_type == "energy_conservation_violation" for alert in alerts
        )
        assert any(alert.severity == AlertSeverity.HIGH for alert in alerts)

    def test_automated_reporting_integration(self):
        """
        Test automated reporting integration.

        Physical Meaning:
            Verifies that automated reporting correctly
            generates physics-aware reports with appropriate
            interpretation and context.
        """
        # Create test results for reporting
        test_results = TestResults(start_time=datetime.now())
        test_results.end_time = datetime.now()

        level_results = LevelTestResults(level="A")
        level_results.add_test_result(
            TestResult(
                test_id="reporting_test_1",
                test_name="Reporting Test 1",
                level="A",
                status=TestStatus.PASSED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                physics_validation={
                    "compliance_score": 0.95,
                    "energy_conservation": {"relative_error": 1e-8},
                    "virial_conditions": {"relative_error": 1e-8},
                    "topological_charge": {"relative_error": 1e-10},
                    "passivity": {"min_real_part": 1e-12},
                },
                numerical_metrics={
                    "convergence_rate": 0.95,
                    "accuracy": 0.98,
                    "stability": 0.97,
                },
            )
        )

        test_results.add_level_results("A", level_results)

        # Test daily report generation
        daily_report = self.reporting_system.generate_daily_report(test_results)
        assert isinstance(daily_report, DailyReport)
        assert daily_report.date is not None
        assert "A" in daily_report.level_analysis
        assert daily_report.physics_summary is not None
        assert daily_report.quality_summary is not None
        assert daily_report.performance_summary is not None
        assert daily_report.validation_status is not None

        # Test weekly report generation
        weekly_results = {
            "start_date": datetime.now() - timedelta(days=7),
            "end_date": datetime.now(),
            "energy_conservation_data": [0.95, 0.96, 0.94, 0.97, 0.95, 0.96, 0.94],
            "virial_conditions_data": [0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.89],
            "topological_charge_data": [0.98, 0.99, 0.97, 0.99, 0.98, 0.99, 0.97],
            "passivity_data": [0.99, 0.99, 0.98, 0.99, 0.99, 0.99, 0.98],
        }

        weekly_report = self.reporting_system.generate_weekly_report(weekly_results)
        assert isinstance(weekly_report, WeeklyReport)
        assert weekly_report.week_start is not None
        assert weekly_report.week_end is not None
        assert weekly_report.physics_trends is not None
        assert weekly_report.convergence_analysis is not None
        assert weekly_report.quality_evolution is not None
        assert weekly_report.performance_trends is not None
        assert weekly_report.recommendations is not None

        # Test monthly report generation
        monthly_results = {
            "start_date": datetime.now() - timedelta(days=30),
            "end_date": datetime.now(),
            "energy_conservation_data": [0.95] * 30,
            "virial_conditions_data": [0.90] * 30,
            "topological_charge_data": [0.98] * 30,
            "passivity_data": [0.99] * 30,
        }

        monthly_report = self.reporting_system.generate_monthly_report(monthly_results)
        assert isinstance(monthly_report, MonthlyReport)
        assert monthly_report.month_start is not None
        assert monthly_report.month_end is not None
        assert monthly_report.physics_validation is not None
        assert monthly_report.prediction_comparison is not None
        assert monthly_report.long_term_trends is not None
        assert monthly_report.progress_assessment is not None
        assert monthly_report.future_recommendations is not None

    def test_physics_interpreter_integration(self):
        """
        Test physics interpreter integration.

        Physical Meaning:
            Verifies that physics interpreter correctly
            provides physical interpretation of experimental
            results in the context of 7D theory.
        """
        # Create test results for interpretation
        test_results = TestResults(start_time=datetime.now())
        test_results.end_time = datetime.now()

        level_results = LevelTestResults(level="A")
        level_results.add_test_result(
            TestResult(
                test_id="interpretation_test_1",
                test_name="Interpretation Test 1",
                level="A",
                status=TestStatus.PASSED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                physics_validation={
                    "compliance_score": 0.95,
                    "energy_conservation": {"relative_error": 1e-8},
                    "virial_conditions": {"relative_error": 1e-8},
                    "topological_charge": {"relative_error": 1e-10},
                    "passivity": {"min_real_part": 1e-12},
                },
            )
        )

        test_results.add_level_results("A", level_results)

        # Test daily physics summary
        physics_summary = self.physics_interpreter.summarize_daily_physics(test_results)
        assert isinstance(physics_summary, dict)
        assert "overall_physics_status" in physics_summary
        assert "energy_conservation_status" in physics_summary
        assert "virial_conditions_status" in physics_summary
        assert "topological_charge_status" in physics_summary
        assert "passivity_status" in physics_summary
        assert "key_insights" in physics_summary
        assert "theoretical_agreement" in physics_summary

        # Test weekly trends analysis
        weekly_results = {
            "energy_conservation_data": [0.95, 0.96, 0.94, 0.97, 0.95],
            "virial_conditions_data": [0.90, 0.91, 0.89, 0.92, 0.90],
            "topological_charge_data": [0.98, 0.99, 0.97, 0.99, 0.98],
            "passivity_data": [0.99, 0.99, 0.98, 0.99, 0.99],
        }

        physics_trends = self.physics_interpreter.analyze_weekly_trends(weekly_results)
        assert isinstance(physics_trends, dict)
        assert "energy_conservation_trend" in physics_trends
        assert "virial_conditions_trend" in physics_trends
        assert "topological_charge_trend" in physics_trends
        assert "passivity_trend" in physics_trends
        assert "overall_trend" in physics_trends
        assert "trend_analysis" in physics_trends

        # Test comprehensive validation
        monthly_results = {
            "energy_conservation_data": [0.95] * 30,
            "virial_conditions_data": [0.90] * 30,
            "topological_charge_data": [0.98] * 30,
            "passivity_data": [0.99] * 30,
        }

        comprehensive_validation = self.physics_interpreter.comprehensive_validation(
            monthly_results
        )
        assert isinstance(comprehensive_validation, dict)
        assert "overall_validation_status" in comprehensive_validation
        assert "principle_validations" in comprehensive_validation
        assert "theoretical_agreement" in comprehensive_validation
        assert "physics_insights" in comprehensive_validation

    def test_error_handling_integration(self):
        """
        Test error handling integration.

        Physical Meaning:
            Verifies that the system correctly handles
            errors and maintains physics validation
            integrity during failures.
        """
        # Test with invalid configuration
        invalid_config_path = self._create_invalid_config()

        try:
            # This should handle invalid configuration gracefully
            testing_system = AutomatedTestingSystem(
                invalid_config_path, self.physics_validator
            )
            assert testing_system.config is not None  # Should use defaults

        finally:
            if os.path.exists(invalid_config_path):
                os.remove(invalid_config_path)

        # Test with physics violations
        invalid_test_result = TestResult(
            test_id="error_test",
            test_name="Error Test",
            level="A",
            status=TestStatus.ERROR,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error_message="Physics violation detected",
            physics_validation={
                "compliance_score": 0.0,
                "violations": [
                    {"constraint": "energy_conservation", "severity": "critical"},
                    {"constraint": "virial_conditions", "severity": "high"},
                ],
            },
        )

        validation_result = self.physics_validator.validate_result(invalid_test_result)
        assert validation_result["is_valid"] is False
        assert validation_result["compliance_score"] < 0.5
        assert len(validation_result["violations"]) > 0

    def _create_invalid_config(self):
        """Create invalid configuration for error testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": "json"')  # Invalid JSON
            return f.name

    def test_performance_integration(self):
        """
        Test performance integration.

        Physical Meaning:
            Verifies that the system maintains good
            performance while ensuring physics validation
            accuracy for 7D computations.
        """
        # Test resource management
        resource_manager = ResourceManager(
            max_workers=2, memory_limit="1GB", cpu_limit=50.0
        )
        assert resource_manager.max_workers == 2
        assert resource_manager.memory_limit > 0
        assert resource_manager.cpu_limit == 50.0

        # Test with resource context
        with resource_manager.get_execution_context() as context:
            assert context is not None

        # Test test scheduler performance
        scheduler = TestScheduler()

        # Add multiple tests
        for i in range(10):
            scheduler.add_test(
                f"test_{i}",
                "A",
                TestPriority.MEDIUM,
                [],
                ["energy_conservation", "virial_conditions"],
            )

        execution_order = scheduler.get_execution_order()
        assert len(execution_order) == 10
        assert all(f"test_{i}" in execution_order for i in range(10))

        # Test data aggregation performance
        data_aggregator = DataAggregator()

        # Create multiple test results
        test_results_list = []
        for i in range(5):
            test_results = TestResults(start_time=datetime.now())
            test_results.end_time = datetime.now()

            level_results = LevelTestResults(level="A")
            level_results.add_test_result(
                TestResult(
                    test_id=f"perf_test_{i}",
                    test_name=f"Performance Test {i}",
                    level="A",
                    status=TestStatus.PASSED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    physics_validation={"compliance_score": 0.95},
                )
            )

            test_results.add_level_results("A", level_results)
            test_results_list.append(test_results)

        # Test aggregation
        daily_data = data_aggregator.aggregate_daily_data(test_results_list[0])
        assert isinstance(daily_data, dict)
        assert "total_tests" in daily_data
        assert "physics_metrics" in daily_data

        weekly_data = data_aggregator.aggregate_weekly_data(test_results_list)
        assert isinstance(weekly_data, dict)
        assert "total_days" in weekly_data
        assert "overall_success_rate" in weekly_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
