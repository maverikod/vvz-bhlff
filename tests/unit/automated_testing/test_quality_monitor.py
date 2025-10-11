"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for quality monitor functionality.

This module tests the quality monitoring functionality
for 7D phase field theory experiments, ensuring proper
quality metrics tracking and degradation detection.

Physical Meaning:
    Tests validate that quality monitoring correctly
    tracks quality metrics and detects degradation
    for 7D theory validation.

Example:
    >>> pytest tests/unit/automated_testing/test_quality_monitor.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

from bhlff.testing.automated_testing import (
    TestResult,
    TestStatus,
    LevelTestResults,
    TestResults,
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


class TestQualityMonitor:
    """
    Unit tests for quality monitor.
    
    Physical Meaning:
        Tests ensure the quality monitor correctly
        tracks quality metrics for 7D theory validation.
    """
    
    def setup_method(self):
        """Setup test fixtures."""
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
    
    def test_quality_monitor_initialization(self):
        """
        Test quality monitor initialization.
        
        Physical Meaning:
            Tests that the quality monitor is correctly
            initialized with proper baseline metrics.
        """
        assert self.quality_monitor is not None
        assert self.quality_monitor.baseline_metrics is not None
        assert self.quality_monitor.physics_constraints is not None
        assert self.quality_monitor.metric_history is not None
    
    def test_quality_metrics_creation(self):
        """
        Test quality metrics creation.
        
        Physical Meaning:
            Tests that quality metrics are correctly created
            for 7D theory validation.
        """
        # Create quality metrics
        quality_metrics = QualityMetrics(
            energy_conservation=0.95,
            virial_conditions=0.90,
            topological_charge=0.98,
            passivity=0.99,
            overall_score=0.95,
            status=QualityStatus.EXCELLENT
        )
        
        # Verify quality metrics
        assert quality_metrics.energy_conservation == 0.95
        assert quality_metrics.virial_conditions == 0.90
        assert quality_metrics.topological_charge == 0.98
        assert quality_metrics.passivity == 0.99
        assert quality_metrics.overall_score == 0.95
        assert quality_metrics.status == QualityStatus.EXCELLENT
    
    def test_physics_constraints_validation(self):
        """
        Test physics constraints validation.
        
        Physical Meaning:
            Tests that physics constraints are correctly validated
            for 7D theory validation.
        """
        # Test physics constraints
        constraints = PhysicsConstraints(
            {
                "energy_conservation": {"max_relative_error": 1e-6},
                "virial_conditions": {"max_relative_error": 1e-6},
                "topological_charge": {"max_relative_error": 1e-8},
                "passivity": {"tolerance": 1e-12},
            }
        )
        
        # Verify constraints
        assert constraints.energy_conservation is not None
        assert constraints.virial_conditions is not None
        assert constraints.topological_charge is not None
        assert constraints.passivity is not None
        
        # Check constraint values
        assert constraints.energy_conservation["max_relative_error"] == 1e-6
        assert constraints.virial_conditions["max_relative_error"] == 1e-6
        assert constraints.topological_charge["max_relative_error"] == 1e-8
        assert constraints.passivity["tolerance"] == 1e-12
    
    def test_degradation_report_creation(self):
        """
        Test degradation report creation.
        
        Physical Meaning:
            Tests that degradation reports are correctly created
            for 7D theory validation quality monitoring.
        """
        # Create degradation report
        degradation_report = DegradationReport(
            overall_severity=AlertSeverity.MEDIUM,
            physics_degradation={
                "energy_conservation": {"severity": AlertSeverity.MEDIUM, "degradation": 0.1},
                "virial_conditions": {"severity": AlertSeverity.LOW, "degradation": 0.05}
            },
            numerical_degradation={
                "convergence_rate": {"severity": AlertSeverity.LOW, "degradation": 0.05}
            },
            spectral_degradation={
                "accuracy": {"severity": AlertSeverity.MEDIUM, "degradation": 0.1}
            },
            convergence_degradation={
                "stability": {"severity": AlertSeverity.LOW, "degradation": 0.05}
            }
        )
        
        # Verify degradation report
        assert degradation_report.overall_severity == AlertSeverity.MEDIUM
        assert degradation_report.physics_degradation is not None
        assert degradation_report.numerical_degradation is not None
        assert degradation_report.spectral_degradation is not None
        assert degradation_report.convergence_degradation is not None
        
        # Check degradation details
        assert "energy_conservation" in degradation_report.physics_degradation
        assert "virial_conditions" in degradation_report.physics_degradation
        assert "convergence_rate" in degradation_report.numerical_degradation
        assert "accuracy" in degradation_report.spectral_degradation
        assert "stability" in degradation_report.convergence_degradation
    
    def test_quality_alert_creation(self):
        """
        Test quality alert creation.
        
        Physical Meaning:
            Tests that quality alerts are correctly created
            for 7D theory validation quality monitoring.
        """
        # Create quality alert
        quality_alert = QualityAlert(
            alert_type="energy_conservation_degradation",
            severity=AlertSeverity.MEDIUM,
            timestamp=datetime.now(),
            metric_name="energy_conservation",
            current_value=0.85,
            baseline_value=0.95,
            threshold=0.90,
            physical_interpretation="Energy conservation degradation detected",
            recommended_actions=["Check solver parameters", "Verify boundary conditions"],
            theoretical_context="Energy conservation is fundamental to 7D theory",
            mathematical_expression="E_total = E_kinetic + E_potential"
        )
        
        # Verify quality alert
        assert quality_alert.alert_type == "energy_conservation_degradation"
        assert quality_alert.severity == AlertSeverity.MEDIUM
        assert quality_alert.timestamp is not None
        assert quality_alert.metric_name == "energy_conservation"
        assert quality_alert.current_value == 0.85
        assert quality_alert.baseline_value == 0.95
        assert quality_alert.threshold == 0.90
        assert quality_alert.physical_interpretation is not None
        assert quality_alert.recommended_actions is not None
        assert quality_alert.theoretical_context is not None
        assert quality_alert.mathematical_expression is not None
    
    def test_quality_monitoring_integration(self):
        """
        Test quality monitoring integration.
        
        Physical Meaning:
            Tests that quality monitoring is correctly
            integrated with the automated testing system.
        """
        # Create mock test results
        test_results = self._create_mock_test_results()
        
        # Test quality monitoring
        quality_metrics = self.quality_monitor.check_quality_metrics(test_results)
        
        # Verify quality metrics
        assert quality_metrics is not None
        assert quality_metrics.overall_score > 0
        assert quality_metrics.status in [QualityStatus.EXCELLENT, QualityStatus.GOOD, QualityStatus.ACCEPTABLE]
        
        # Verify individual metrics
        assert quality_metrics.energy_conservation > 0
        assert quality_metrics.virial_conditions > 0
        assert quality_metrics.topological_charge > 0
        assert quality_metrics.passivity > 0
        assert quality_metrics.convergence_rate > 0
        assert quality_metrics.accuracy > 0
        assert quality_metrics.stability > 0
    
    def test_quality_degradation_detection(self):
        """
        Test quality degradation detection.
        
        Physical Meaning:
            Tests that quality degradation is correctly detected
            in the automated testing system.
        """
        # Create mock test results with degraded quality
        test_results = self._create_mock_test_results_with_degradation()
        
        # Test quality monitoring
        quality_metrics = self.quality_monitor.check_quality_metrics(test_results)
        
        # Verify degraded quality
        assert quality_metrics is not None
        assert quality_metrics.overall_score < 0.8  # Degraded quality
        assert quality_metrics.status in [QualityStatus.DEGRADED, QualityStatus.CRITICAL]
        
        # Test degradation detection
        current_metrics = {
            "energy_conservation": quality_metrics.energy_conservation,
            "virial_conditions": quality_metrics.virial_conditions,
            "topological_charge": quality_metrics.topological_charge,
            "passivity": quality_metrics.passivity,
        }
        
        historical_metrics = [
            {"energy_conservation": 0.95, "virial_conditions": 0.90, "topological_charge": 0.98, "passivity": 0.99},
            {"energy_conservation": 0.94, "virial_conditions": 0.89, "topological_charge": 0.97, "passivity": 0.98},
            {"energy_conservation": 0.93, "virial_conditions": 0.88, "topological_charge": 0.96, "passivity": 0.97},
        ]
        
        degradation_report = self.quality_monitor.detect_quality_degradation(
            current_metrics, historical_metrics
        )
        
        # Verify degradation detection
        assert degradation_report is not None
        assert degradation_report.overall_severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        
        # Check for degradation in specific areas
        assert len(degradation_report.physics_degradation) > 0
        assert len(degradation_report.numerical_degradation) > 0
        assert len(degradation_report.spectral_degradation) > 0
        assert len(degradation_report.convergence_degradation) > 0
    
    def test_quality_alerts_generation(self):
        """
        Test quality alerts generation.
        
        Physical Meaning:
            Tests that quality alerts are correctly generated
            for quality degradation in the automated testing system.
        """
        # Create mock test results with degraded quality
        test_results = self._create_mock_test_results_with_degradation()
        
        # Test quality monitoring
        quality_metrics = self.quality_monitor.check_quality_metrics(test_results)
        
        # Test degradation detection
        current_metrics = {
            "energy_conservation": quality_metrics.energy_conservation,
            "virial_conditions": quality_metrics.virial_conditions,
            "topological_charge": quality_metrics.topological_charge,
            "passivity": quality_metrics.passivity,
        }
        
        historical_metrics = [
            {"energy_conservation": 0.95, "virial_conditions": 0.90, "topological_charge": 0.98, "passivity": 0.99},
            {"energy_conservation": 0.94, "virial_conditions": 0.89, "topological_charge": 0.97, "passivity": 0.98},
            {"energy_conservation": 0.93, "virial_conditions": 0.88, "topological_charge": 0.96, "passivity": 0.97},
        ]
        
        degradation_report = self.quality_monitor.detect_quality_degradation(
            current_metrics, historical_metrics
        )
        
        # Test alert generation
        alerts = self.quality_monitor.generate_quality_alerts(degradation_report)
        
        # Verify alerts
        assert alerts is not None
        assert len(alerts) > 0
        
        # Check alert properties
        for alert in alerts:
            assert isinstance(alert, QualityAlert)
            assert alert.alert_type is not None
            assert alert.severity in [AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            assert alert.timestamp is not None
            assert alert.metric_name is not None
            assert alert.current_value is not None
            assert alert.baseline_value is not None
            assert alert.threshold is not None
            assert alert.physical_interpretation is not None
            assert alert.recommended_actions is not None
            assert alert.theoretical_context is not None
            assert alert.mathematical_expression is not None
    
    def _create_mock_test_results(self) -> TestResults:
        """Create mock test results for testing."""
        level_results = {}
        
        for level in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            # Create mock test result
            test_result = TestResult(
                test_name=f"test_{level}_physics",
                status=TestStatus.PASSED,
                execution_time=1.0,
                physics_validation={
                    "energy_conservation": {"relative_error": 1e-7, "valid": True},
                    "virial_conditions": {"relative_error": 1e-7, "valid": True},
                    "topological_charge": {"relative_error": 1e-9, "valid": True},
                    "passivity": {"min_real_part": 1e-13, "valid": True}
                },
                numerical_metrics={
                    "convergence_rate": 0.95,
                    "accuracy": 0.98,
                    "stability": 0.97
                },
                metadata={
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Create level test results
            level_test_results = LevelTestResults(
                level=level,
                test_results=[test_result],
                total_tests=1,
                passed_tests=1,
                failed_tests=0,
                execution_time=1.0
            )
            
            level_results[level] = level_test_results
        
        # Create overall test results
        return TestResults(
            level_results=level_results,
            total_tests=7,
            passed_tests=7,
            failed_tests=0,
            total_execution_time=7.0,
            timestamp=datetime.now()
        )
    
    def _create_mock_test_results_with_degradation(self) -> TestResults:
        """Create mock test results with quality degradation."""
        level_results = {}
        
        for level in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            # Create mock test result with degraded quality
            test_result = TestResult(
                test_name=f"test_{level}_physics",
                status=TestStatus.PASSED,
                execution_time=1.0,
                physics_validation={
                    "energy_conservation": {"relative_error": 1e-5, "valid": True},
                    "virial_conditions": {"relative_error": 1e-5, "valid": True},
                    "topological_charge": {"relative_error": 1e-7, "valid": True},
                    "passivity": {"min_real_part": 1e-11, "valid": True}
                },
                numerical_metrics={
                    "convergence_rate": 0.85,
                    "accuracy": 0.88,
                    "stability": 0.87
                },
                metadata={
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Create level test results
            level_test_results = LevelTestResults(
                level=level,
                test_results=[test_result],
                total_tests=1,
                passed_tests=1,
                failed_tests=0,
                execution_time=1.0
            )
            
            level_results[level] = level_test_results
        
        # Create overall test results
        return TestResults(
            level_results=level_results,
            total_tests=7,
            passed_tests=7,
            failed_tests=0,
            total_execution_time=7.0,
            timestamp=datetime.now()
        )
