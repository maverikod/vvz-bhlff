"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for automated testing system core functionality.

This module tests the core automated testing system functionality
for 7D phase field theory experiments, ensuring proper functionality
of physics validation, test scheduling, and resource management.

Physical Meaning:
    Tests validate that the automated testing system correctly
    implements physics-first prioritization and maintains
    adherence to 7D theory principles during test execution.

Example:
    >>> pytest tests/unit/automated_testing/test_automated_testing_system.py -v
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


class TestAutomatedTestingSystem:
    """
    Unit tests for automated testing system.
    
    Physical Meaning:
        Tests ensure the automated testing system correctly
        implements physics-first prioritization for 7D theory validation.
    """
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config_path = self._create_test_config()
        
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
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
    
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
    
    def test_automated_testing_system_initialization(self):
        """
        Test automated testing system initialization.
        
        Physical Meaning:
            Tests that the automated testing system is correctly
            initialized with proper physics validation configuration.
        """
        assert self.testing_system is not None
        assert self.testing_system.physics_validator is not None
        assert self.testing_system.config is not None
        assert self.testing_system.test_results is not None
    
    def test_run_all_tests_physics_priority(self):
        """
        Test running all tests with physics priority.
        
        Physical Meaning:
            Tests that all tests are executed with physics-first
            prioritization for 7D theory validation.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Run all tests
            results = self.testing_system.run_all_tests()
            
            # Verify results
            assert results is not None
            assert results.total_tests > 0
            assert results.passed_tests > 0
            assert results.failed_tests == 0
    
    def test_run_level_tests_physics_validation(self):
        """
        Test running level tests with physics validation.
        
        Physical Meaning:
            Tests that level tests are executed with proper
            physics validation for 7D theory validation.
        """
        # Mock test execution
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results()
            
            # Run level tests
            results = self.testing_system.run_level_tests(['A', 'B', 'C'])
            
            # Verify results
            assert results is not None
            assert len(results.level_results) == 3
            assert 'A' in results.level_results
            assert 'B' in results.level_results
            assert 'C' in results.level_results
    
    def test_physics_validator_energy_conservation(self):
        """
        Test physics validator energy conservation.
        
        Physical Meaning:
            Tests that energy conservation is correctly validated
            for 7D theory physics validation.
        """
        # Test data
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        initial_energy = 1.0
        final_energy = 1.0
        
        # Test energy conservation validation
        result = self.physics_validator.validate_energy_conservation(
            field, {"initial_energy": initial_energy, "final_energy": final_energy}
        )
        
        # Verify result
        assert result is not None
        assert "valid" in result
        assert "relative_error" in result
        assert result["valid"] is True
        assert result["relative_error"] < 1e-6
    
    def test_physics_validator_virial_conditions(self):
        """
        Test physics validator virial conditions.
        
        Physical Meaning:
            Tests that virial conditions are correctly validated
            for 7D theory physics validation.
        """
        # Test data
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        virial_data = {"kinetic_energy": 1.0, "potential_energy": 1.0}
        
        # Test virial conditions validation
        result = self.physics_validator.validate_virial_conditions(
            field, virial_data
        )
        
        # Verify result
        assert result is not None
        assert "valid" in result
        assert "relative_error" in result
        assert result["valid"] is True
        assert result["relative_error"] < 1e-6
    
    def test_physics_validator_topological_charge(self):
        """
        Test physics validator topological charge.
        
        Physical Meaning:
            Tests that topological charge is correctly validated
            for 7D theory physics validation.
        """
        # Test data
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        charge_data = {"expected_charge": 1.0, "computed_charge": 1.0}
        
        # Test topological charge validation
        result = self.physics_validator.validate_topological_charge(
            field, charge_data
        )
        
        # Verify result
        assert result is not None
        assert "valid" in result
        assert "relative_error" in result
        assert result["valid"] is True
        assert result["relative_error"] < 1e-8
    
    def test_physics_validator_passivity(self):
        """
        Test physics validator passivity.
        
        Physical Meaning:
            Tests that passivity is correctly validated
            for 7D theory physics validation.
        """
        # Test data
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        passivity_data = {"min_real_part": 1e-13}
        
        # Test passivity validation
        result = self.physics_validator.validate_passivity(
            field, passivity_data
        )
        
        # Verify result
        assert result is not None
        assert "valid" in result
        assert "min_real_part" in result
        assert result["valid"] is True
        assert result["min_real_part"] > 1e-13
    
    def test_test_scheduler_physics_priority(self):
        """
        Test test scheduler physics priority.
        
        Physical Meaning:
            Tests that test scheduler correctly implements
            physics-first prioritization for 7D theory validation.
        """
        # Initialize test scheduler
        test_scheduler = TestScheduler()
        
        # Test data
        tests = [
            {"name": "test_A", "level": "A", "priority": "high"},
            {"name": "test_B", "level": "B", "priority": "high"},
            {"name": "test_C", "level": "C", "priority": "medium"},
        ]
        
        # Test scheduling
        scheduled_tests = test_scheduler.schedule_tests(tests)
        
        # Verify scheduling
        assert scheduled_tests is not None
        assert len(scheduled_tests) == 3
        
        # Check priority order
        assert scheduled_tests[0]["priority"] == "high"
        assert scheduled_tests[1]["priority"] == "high"
        assert scheduled_tests[2]["priority"] == "medium"
    
    def test_resource_manager_initialization(self):
        """
        Test resource manager initialization.
        
        Physical Meaning:
            Tests that resource manager is correctly initialized
            for 7D theory validation resource management.
        """
        # Initialize resource manager
        resource_manager = ResourceManager()
        
        # Test resource allocation
        allocated = resource_manager.allocate_resources(1, 1)
        
        # Verify allocation
        assert allocated is True
    
    def test_level_test_results_aggregation(self):
        """
        Test level test results aggregation.
        
        Physical Meaning:
            Tests that level test results are correctly aggregated
            for 7D theory validation.
        """
        # Create mock test results
        test_results = [
            TestResult(
                test_name="test_A_physics",
                status=TestStatus.PASSED,
                execution_time=1.0,
                physics_validation={"energy_conservation": {"valid": True}},
                numerical_metrics={"convergence_rate": 0.95},
                metadata={"level": "A"}
            ),
            TestResult(
                test_name="test_B_physics",
                status=TestStatus.PASSED,
                execution_time=1.0,
                physics_validation={"energy_conservation": {"valid": True}},
                numerical_metrics={"convergence_rate": 0.95},
                metadata={"level": "B"}
            )
        ]
        
        # Create level test results
        level_results = LevelTestResults(
            level="A",
            test_results=test_results,
            total_tests=2,
            passed_tests=2,
            failed_tests=0,
            execution_time=2.0
        )
        
        # Verify level test results
        assert level_results.level == "A"
        assert level_results.total_tests == 2
        assert level_results.passed_tests == 2
        assert level_results.failed_tests == 0
        assert level_results.execution_time == 2.0
    
    def test_level_test_results_critical_failures(self):
        """
        Test level test results critical failures.
        
        Physical Meaning:
            Tests that critical failures are correctly identified
            in level test results for 7D theory validation.
        """
        # Create mock test results with failures
        test_results = [
            TestResult(
                test_name="test_A_physics",
                status=TestStatus.FAILED,
                execution_time=1.0,
                physics_validation={"energy_conservation": {"valid": False}},
                numerical_metrics={"convergence_rate": 0.85},
                metadata={"level": "A"}
            ),
            TestResult(
                test_name="test_B_physics",
                status=TestStatus.PASSED,
                execution_time=1.0,
                physics_validation={"energy_conservation": {"valid": True}},
                numerical_metrics={"convergence_rate": 0.95},
                metadata={"level": "B"}
            )
        ]
        
        # Create level test results
        level_results = LevelTestResults(
            level="A",
            test_results=test_results,
            total_tests=2,
            passed_tests=1,
            failed_tests=1,
            execution_time=2.0
        )
        
        # Verify level test results
        assert level_results.level == "A"
        assert level_results.total_tests == 2
        assert level_results.passed_tests == 1
        assert level_results.failed_tests == 1
        assert level_results.execution_time == 2.0
        
        # Check for critical failures
        critical_failures = [r for r in test_results if r.status == TestStatus.FAILED]
        assert len(critical_failures) == 1
        assert critical_failures[0].test_name == "test_A_physics"
    
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
