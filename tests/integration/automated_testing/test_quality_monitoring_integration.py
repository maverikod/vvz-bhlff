"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quality monitoring integration tests for automated testing system.

This module tests the quality monitoring integration functionality
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


class TestQualityMonitoringIntegration:
    """
    Quality monitoring integration tests for automated testing system.
    
    Physical Meaning:
        Tests ensure the quality monitoring integration
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
            "domain": {
                "L": 1.0,
                "N": 64,
                "dimensions": 7
            },
            "physics": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda": 0.1,
                "nu": 1.0
            },
            "solver": {
                "precision": "float64",
                "fft_plan": "MEASURE",
                "tolerance": 1e-12
            },
            "testing": {
                "max_tests": 10,
                "timeout": 300,
                "parallel": False
            },
            "levels": {
                "A": {"enabled": True, "priority": "high"},
                "B": {"enabled": True, "priority": "high"},
                "C": {"enabled": True, "priority": "medium"},
                "D": {"enabled": True, "priority": "medium"},
                "E": {"enabled": True, "priority": "low"},
                "F": {"enabled": True, "priority": "low"},
                "G": {"enabled": True, "priority": "low"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            return f.name
    
    def test_quality_monitoring_integration(self):
        """
        Test quality monitoring integration.
        
        Physical Meaning:
            Tests that quality monitoring is correctly integrated
            with the automated testing system for 7D theory validation.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            
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
            and reported in the automated testing system.
        """
        # Mock test execution with degraded quality
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results_with_degradation()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            
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
        # Mock test execution with degraded quality
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results_with_degradation()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            
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
        """Create mock test results for integration testing."""
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
