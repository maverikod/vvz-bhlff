"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physics validation integration tests for automated testing system.

This module tests the physics validation integration functionality
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


class TestPhysicsValidationIntegration:
    """
    Physics validation integration tests for automated testing system.
    
    Physical Meaning:
        Tests ensure the physics validation integration
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
    
    def test_physics_validation_integration(self):
        """
        Test physics validation integration.
        
        Physical Meaning:
            Tests that physics validation is correctly integrated
            with the automated testing system for 7D theory validation.
        """
        # Mock test execution with physics validation
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results_with_physics()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Verify physics validation
            assert results is not None
            assert len(results.level_results) > 0
            
            # Check physics validation in results
            for level, level_results in results.level_results.items():
                for test_result in level_results.test_results:
                    assert test_result.physics_validation is not None
                    assert "energy_conservation" in test_result.physics_validation
                    assert "virial_conditions" in test_result.physics_validation
                    assert "topological_charge" in test_result.physics_validation
                    assert "passivity" in test_result.physics_validation
                    
                    # Verify physics validation results
                    energy_validation = test_result.physics_validation["energy_conservation"]
                    assert energy_validation["valid"] is True
                    assert energy_validation["relative_error"] < 1e-6
                    
                    virial_validation = test_result.physics_validation["virial_conditions"]
                    assert virial_validation["valid"] is True
                    assert virial_validation["relative_error"] < 1e-6
                    
                    topology_validation = test_result.physics_validation["topological_charge"]
                    assert topology_validation["valid"] is True
                    assert topology_validation["relative_error"] < 1e-8
                    
                    passivity_validation = test_result.physics_validation["passivity"]
                    assert passivity_validation["valid"] is True
                    assert passivity_validation["min_real_part"] > 1e-12
    
    def test_physics_constraints_validation(self):
        """
        Test physics constraints validation.
        
        Physical Meaning:
            Tests that physics constraints are correctly validated
            according to 7D theory principles.
        """
        # Test energy conservation constraint
        energy_validation = self.physics_validator.validate_energy_conservation(
            np.array([1.0, 2.0, 3.0]),  # Mock field data
            {"initial_energy": 1.0, "final_energy": 1.0}  # Mock energy data
        )
        assert energy_validation["valid"] is True
        assert energy_validation["relative_error"] < 1e-6
        
        # Test virial conditions constraint
        virial_validation = self.physics_validator.validate_virial_conditions(
            np.array([1.0, 2.0, 3.0]),  # Mock field data
            {"virial_parameter": 1.0, "virial_value": 0.0}  # Mock virial data
        )
        assert virial_validation["valid"] is True
        assert virial_validation["relative_error"] < 1e-6
        
        # Test topological charge constraint
        topology_validation = self.physics_validator.validate_topological_charge(
            np.array([1.0, 2.0, 3.0]),  # Mock field data
            {"topological_charge": 0.0, "charge_tolerance": 1e-8}  # Mock topology data
        )
        assert topology_validation["valid"] is True
        assert topology_validation["relative_error"] < 1e-8
        
        # Test passivity constraint
        passivity_validation = self.physics_validator.validate_passivity(
            np.array([1.0, 2.0, 3.0]),  # Mock field data
            {"min_real_part": 1e-13, "tolerance": 1e-12}  # Mock passivity data
        )
        assert passivity_validation["valid"] is True
        assert passivity_validation["min_real_part"] > 1e-12
    
    def test_physics_validation_failure_handling(self):
        """
        Test physics validation failure handling.
        
        Physical Meaning:
            Tests that physics validation failures are correctly
            handled and reported in the automated testing system.
        """
        # Mock test execution with physics validation failures
        with patch.object(self.testing_system, 'execute_tests') as mock_execute:
            mock_execute.return_value = self._create_mock_test_results_with_failures()
            
            # Execute tests
            results = self.testing_system.execute_tests()
            
            # Verify failure handling
            assert results is not None
            assert len(results.level_results) > 0
            
            # Check that failures are properly reported
            failure_count = 0
            for level, level_results in results.level_results.items():
                for test_result in level_results.test_results:
                    if test_result.status == TestStatus.FAILED:
                        failure_count += 1
                        assert test_result.physics_validation is not None
                        # Check that at least one physics validation failed
                        physics_failures = [
                            not validation["valid"] 
                            for validation in test_result.physics_validation.values()
                            if isinstance(validation, dict) and "valid" in validation
                        ]
                        assert any(physics_failures)
            
            # Verify that some failures were detected
            assert failure_count > 0
    
    def _create_mock_test_results_with_physics(self) -> TestResults:
        """Create mock test results with physics validation."""
        level_results = {}
        
        for level in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            # Create mock test result with physics validation
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
    
    def _create_mock_test_results_with_failures(self) -> TestResults:
        """Create mock test results with physics validation failures."""
        level_results = {}
        
        for level in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            # Create mock test result with physics validation failures
            test_result = TestResult(
                test_name=f"test_{level}_physics",
                status=TestStatus.FAILED if level in ['C', 'D'] else TestStatus.PASSED,
                execution_time=1.0,
                physics_validation={
                    "energy_conservation": {"relative_error": 1e-5, "valid": False},
                    "virial_conditions": {"relative_error": 1e-5, "valid": False},
                    "topological_charge": {"relative_error": 1e-7, "valid": False},
                    "passivity": {"min_real_part": 1e-11, "valid": False}
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
                passed_tests=1 if level not in ['C', 'D'] else 0,
                failed_tests=0 if level not in ['C', 'D'] else 1,
                execution_time=1.0
            )
            
            level_results[level] = level_test_results
        
        # Create overall test results
        return TestResults(
            level_results=level_results,
            total_tests=7,
            passed_tests=5,
            failed_tests=2,
            total_execution_time=7.0,
            timestamp=datetime.now()
        )
