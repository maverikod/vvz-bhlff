"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for automated reporting functionality.

This module tests the automated reporting functionality
for 7D phase field theory experiments, ensuring proper
report generation and distribution.

Physical Meaning:
    Tests validate that automated reporting correctly
    generates and distributes reports for 7D theory validation.

Example:
    >>> pytest tests/unit/automated_testing/test_automated_reporting.py -v
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
    QualityMetrics,
    QualityStatus,
)
from bhlff.testing.automated_reporting import (
    AutomatedReportingSystem,
    PhysicsInterpreter,
    TemplateEngine,
    DataAggregator,
)


class TestAutomatedReportingSystem:
    """
    Unit tests for automated reporting system.
    
    Physical Meaning:
        Tests ensure the automated reporting system correctly
        generates reports for 7D theory validation.
    """
    
    def setup_method(self):
        """Setup test fixtures."""
        self.reporting_config_path = self._create_reporting_config()
        
        # Initialize reporting system
        self.reporting_system = AutomatedReportingSystem(
            self.reporting_config_path
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if os.path.exists(self.reporting_config_path):
            os.remove(self.reporting_config_path)
    
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
    
    def test_reporting_system_initialization(self):
        """
        Test reporting system initialization.
        
        Physical Meaning:
            Tests that the automated reporting system is correctly
            initialized with proper configuration.
        """
        assert self.reporting_system is not None
        assert self.reporting_system.config is not None
        assert self.reporting_system.report_output_dir is not None
    
    def test_daily_report_creation(self):
        """
        Test daily report creation.
        
        Physical Meaning:
            Tests that daily reports are correctly created
            for 7D theory validation.
        """
        # Create mock test results
        test_results = self._create_mock_test_results()
        quality_metrics = self._create_mock_quality_metrics()
        
        # Test daily report creation
        report = self.reporting_system.generate_daily_report(test_results, quality_metrics)
        
        # Verify report
        assert report is not None
        assert len(report) > 0
        
        # Check report content
        assert "html" in report.lower() or "json" in report.lower()
        assert "physics" in report.lower()
        assert "quality" in report.lower()
        assert "validation" in report.lower()
    
    def test_weekly_report_creation(self):
        """
        Test weekly report creation.
        
        Physical Meaning:
            Tests that weekly reports are correctly created
            for 7D theory validation.
        """
        # Create mock test results
        test_results = self._create_mock_test_results()
        quality_metrics = self._create_mock_quality_metrics()
        
        # Test weekly report creation
        report = self.reporting_system.generate_weekly_report(test_results, quality_metrics)
        
        # Verify report
        assert report is not None
        assert len(report) > 0
        
        # Check report content
        assert "html" in report.lower() or "json" in report.lower()
        assert "physics" in report.lower()
        assert "quality" in report.lower()
        assert "validation" in report.lower()
    
    def test_monthly_report_creation(self):
        """
        Test monthly report creation.
        
        Physical Meaning:
            Tests that monthly reports are correctly created
            for 7D theory validation.
        """
        # Create mock test results
        test_results = self._create_mock_test_results()
        quality_metrics = self._create_mock_quality_metrics()
        
        # Test monthly report creation
        report = self.reporting_system.generate_monthly_report(test_results, quality_metrics)
        
        # Verify report
        assert report is not None
        assert len(report) > 0
        
        # Check report content
        assert "html" in report.lower() or "json" in report.lower()
        assert "physics" in report.lower()
        assert "quality" in report.lower()
        assert "validation" in report.lower()
    
    def test_physics_interpreter_initialization(self):
        """
        Test physics interpreter initialization.
        
        Physical Meaning:
            Tests that the physics interpreter is correctly
            initialized for 7D theory validation.
        """
        # Initialize physics interpreter
        physics_interpreter = PhysicsInterpreter()
        
        # Verify initialization
        assert physics_interpreter is not None
    
    def test_template_engine_initialization(self):
        """
        Test template engine initialization.
        
        Physical Meaning:
            Tests that the template engine is correctly
            initialized for 7D theory validation.
        """
        # Initialize template engine
        template_engine = TemplateEngine()
        
        # Verify initialization
        assert template_engine is not None
    
    def test_data_aggregator_initialization(self):
        """
        Test data aggregator initialization.
        
        Physical Meaning:
            Tests that the data aggregator is correctly
            initialized for 7D theory validation.
        """
        # Initialize data aggregator
        data_aggregator = DataAggregator()
        
        # Verify initialization
        assert data_aggregator is not None
    
    def test_physics_interpreter_integration(self):
        """
        Test physics interpreter integration.
        
        Physical Meaning:
            Tests that physics interpretation is correctly
            integrated with the automated reporting system.
        """
        # Create mock test results
        test_results = self._create_mock_test_results()
        quality_metrics = self._create_mock_quality_metrics()
        
        # Test physics interpreter
        physics_interpreter = PhysicsInterpreter()
        interpretation = physics_interpreter.interpret_results(test_results, quality_metrics)
        
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
    
    def test_template_engine_integration(self):
        """
        Test template engine integration.
        
        Physical Meaning:
            Tests that template engine is correctly
            integrated with the automated reporting system.
        """
        # Create mock test results
        test_results = self._create_mock_test_results()
        quality_metrics = self._create_mock_quality_metrics()
        
        # Test template engine
        template_engine = TemplateEngine()
        template = template_engine.render_report(test_results, quality_metrics)
        
        # Verify template rendering
        assert template is not None
        assert len(template) > 0
        
        # Check template content
        assert "html" in template.lower() or "json" in template.lower()
        assert "physics" in template.lower()
        assert "quality" in template.lower()
        assert "validation" in template.lower()
    
    def test_data_aggregator_integration(self):
        """
        Test data aggregator integration.
        
        Physical Meaning:
            Tests that data aggregation is correctly
            integrated with the automated reporting system.
        """
        # Create mock test results
        test_results = self._create_mock_test_results()
        quality_metrics = self._create_mock_quality_metrics()
        
        # Test data aggregator
        data_aggregator = DataAggregator()
        aggregated_data = data_aggregator.aggregate_results(test_results, quality_metrics)
        
        # Verify data aggregation
        assert aggregated_data is not None
        assert "test_results" in aggregated_data
        assert "quality_metrics" in aggregated_data
        assert "summary" in aggregated_data
        
        # Check aggregated data content
        assert aggregated_data["test_results"] is not None
        assert aggregated_data["quality_metrics"] is not None
        assert aggregated_data["summary"] is not None
    
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
    
    def _create_mock_quality_metrics(self) -> QualityMetrics:
        """Create mock quality metrics for testing."""
        return QualityMetrics(
            energy_conservation=0.95,
            virial_conditions=0.90,
            topological_charge=0.98,
            passivity=0.99,
            overall_score=0.95,
            status=QualityStatus.EXCELLENT
        )
