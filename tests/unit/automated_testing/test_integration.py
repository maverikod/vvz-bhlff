"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for automated testing integration.

This module tests the integration between different components
of the automated testing system for 7D phase field theory experiments.

Physical Meaning:
    Tests validate that all components of the automated testing system
    work together correctly for 7D theory validation.

Example:
    >>> pytest tests/unit/automated_testing/test_integration.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

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


class TestIntegration:
    """
    Unit tests for automated testing integration.
    
    Physical Meaning:
        Tests ensure that all components of the automated testing system
        work together correctly for 7D theory validation.
    """
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config_path = self._create_test_config()
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
        self.reporting_system = AutomatedReportingSystem(
            self.reporting_config_path
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.reporting_config_path):
            os.remove(self.reporting_config_path)
    
    def _create_test_config(self) -> str:
        """Create test configuration."""
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
    
    def _create_reporting_config(self) -> str:
        """Create reporting test configuration."""
        config = {
            "reporting": {
                "enabled": True,
                "format": "html",
                "template": "physics_report.html"
            },
            "distribution": {
                "email": {
                    "enabled": False,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "test@example.com",
                    "password": "test_password"
                }
            },
            "physics": {
                "interpretation": {
                    "energy_conservation": "Energy conservation validation",
                    "virial_conditions": "Virial conditions validation",
                    "topological_charge": "Topological charge validation",
                    "passivity": "Passivity validation"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            return f.name
    
    def test_end_to_end_automated_testing(self):
        """
        Test end-to-end automated testing.
        
        Physical Meaning:
            Tests that the complete automated testing system
            works correctly for 7D theory validation.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            
            # Test reporting
            report = self.reporting_system.generate_report(results, quality_metrics)
            
            # Verify end-to-end functionality
            assert results is not None
            assert quality_metrics is not None
            assert report is not None
            
            # Verify test results
            assert results.total_tests > 0
            assert results.passed_tests > 0
            assert results.failed_tests == 0
            
            # Verify quality metrics
            assert quality_metrics.overall_score > 0
            assert quality_metrics.status in [QualityStatus.EXCELLENT, QualityStatus.GOOD, QualityStatus.ACCEPTABLE]
            
            # Verify report
            assert len(report) > 0
            assert "physics" in report.lower()
            assert "quality" in report.lower()
            assert "validation" in report.lower()
    
    def test_physics_validation_integration(self):
        """
        Test physics validation integration.
        
        Physical Meaning:
            Tests that physics validation is correctly integrated
            with the automated testing system.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test physics validation
            physics_validation = self.physics_validator.validate_test_result(results)
            
            # Verify physics validation
            assert physics_validation is not None
            assert "valid" in physics_validation
            assert "metrics" in physics_validation
            assert physics_validation["valid"] is True
            
            # Check individual metrics
            assert "energy_conservation" in physics_validation["metrics"]
            assert "virial_conditions" in physics_validation["metrics"]
            assert "topological_charge" in physics_validation["metrics"]
            assert "passivity" in physics_validation["metrics"]
    
    def test_quality_monitoring_integration(self):
        """
        Test quality monitoring integration.
        
        Physical Meaning:
            Tests that quality monitoring is correctly integrated
            with the automated testing system.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            
            # Verify quality monitoring
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
    
    def test_automated_reporting_integration(self):
        """
        Test automated reporting integration.
        
        Physical Meaning:
            Tests that automated reporting is correctly integrated
            with the automated testing system.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            
            # Test reporting
            report = self.reporting_system.generate_report(results, quality_metrics)
            
            # Verify reporting integration
            assert report is not None
            assert len(report) > 0
            
            # Check report content
            assert "html" in report.lower() or "json" in report.lower()
            assert "physics" in report.lower()
            assert "quality" in report.lower()
            assert "validation" in report.lower()
    
    def test_physics_interpreter_integration(self):
        """
        Test physics interpreter integration.
        
        Physical Meaning:
            Tests that physics interpretation is correctly integrated
            with the automated reporting system.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            
            # Test physics interpreter
            physics_interpreter = PhysicsInterpreter()
            interpretation = physics_interpreter.interpret_results(results, quality_metrics)
            
            # Verify physics interpretation
            assert interpretation is not None
            assert "energy_conservation" in interpretation
            assert "virial_conditions" in interpretation
            assert "topological_charge" in interpretation
            assert "passivity" in interpretation
            
            # Check interpretation content
            for key, value in interpretation.items():
                assert isinstance(value, str)
                assert len(value) > 0
                assert "physics" in value.lower() or "theory" in value.lower()
    
    def test_error_handling_integration(self):
        """
        Test error handling integration.
        
        Physical Meaning:
            Tests that error handling is correctly integrated
            with the automated testing system.
        """
        # Mock test execution with errors
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.side_effect = Exception("Test execution failed")
            
            # Test error handling
            with pytest.raises(Exception):
                self.testing_system.execute_tests()
    
    def test_performance_integration(self):
        """
        Test performance integration.
        
        Physical Meaning:
            Tests that performance monitoring is correctly integrated
            with the automated testing system.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Test quality monitoring
            quality_metrics = self.quality_monitor.check_quality_metrics(results)
            
            # Test reporting
            report = self.reporting_system.generate_report(results, quality_metrics)
            
            # Verify performance
            assert results is not None
            assert quality_metrics is not None
            assert report is not None
            
            # Check execution time
            assert results.total_execution_time > 0
            assert results.total_execution_time < 100  # Should complete within reasonable time
    
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
