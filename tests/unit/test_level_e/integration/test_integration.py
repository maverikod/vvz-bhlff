"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for Level E integration functionality.

This module tests the integration between different components
of Level E experiments in 7D phase field theory.

Physical Meaning:
    Tests ensure that all components of Level E experiments
    work together correctly for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/integration/test_integration.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import (
    SensitivityAnalyzer,
    RobustnessTester,
    DiscretizationAnalyzer,
    FailureDetector,
    PhaseMapper,
    PerformanceAnalyzer,
    SolitonModel,
    BaryonSoliton,
    SkyrmionSoliton,
    DefectModel,
    VortexDefect,
    MultiDefectSystem,
    LevelEExperiments,
)


class TestIntegration:
    """Test Level E integration functionality."""

    def test_sensitivity_robustness_integration(self):
        """Test sensitivity-robustness integration."""
        # Initialize components
        sensitivity_analyzer = SensitivityAnalyzer(
            {"beta": (0.6, 1.4), "mu": (0.5, 1.5), "eta": (0.0, 0.3)}
        )
        robustness_tester = RobustnessTester(
            {"noise_level": 0.1, "perturbation_range": 0.05, "test_iterations": 100}
        )

        # Mock test data
        test_data = {
            "field_data": np.random.rand(64, 64, 64),
            "parameter_ranges": {
                "beta": (0.6, 1.4),
                "mu": (0.5, 1.5),
                "eta": (0.0, 0.3),
            },
        }

        # Test sensitivity analysis
        sensitivity_results = sensitivity_analyzer.analyze_parameter_sensitivity(
            test_data
        )

        # Test robustness analysis
        robustness_results = robustness_tester.test_noise_robustness(test_data)

        # Verify integration
        assert sensitivity_results is not None
        assert robustness_results is not None

        # Check sensitivity results
        assert "sensitivity_indices" in sensitivity_results
        assert "parameter_importance" in sensitivity_results

        # Check robustness results
        assert "noise_tolerance" in robustness_results
        assert "stability_metrics" in robustness_results

    def test_discretization_failure_integration(self):
        """Test discretization-failure integration."""
        # Initialize components
        discretization_analyzer = DiscretizationAnalyzer(
            [32, 64, 128, 256], [0.001, 0.01, 0.1, 1.0]
        )
        failure_detector = FailureDetector(
            {
                "energy_conservation": 1e-6,
                "momentum_conservation": 1e-6,
                "topological_charge": 1e-8,
            }
        )

        # Mock test data
        test_data = {
            "field_data": {
                32: np.random.rand(32, 32, 32),
                64: np.random.rand(64, 64, 64),
            },
            "convergence_history": np.random.rand(100),
        }

        # Test discretization analysis
        discretization_results = discretization_analyzer.analyze_grid_convergence(
            test_data
        )

        # Test failure detection
        failure_results = failure_detector.detect_convergence_failure(test_data)

        # Verify integration
        assert discretization_results is not None
        assert failure_results is not None

        # Check discretization results
        assert "convergence_rate" in discretization_results
        assert "error_analysis" in discretization_results

        # Check failure results
        assert "failure_detected" in failure_results
        assert "failure_severity" in failure_results

    def test_phase_mapping_performance_integration(self):
        """Test phase mapping-performance integration."""
        # Initialize components
        phase_mapper = PhaseMapper(
            {
                "phase_resolution": 0.01,
                "mapping_tolerance": 1e-6,
                "max_iterations": 1000,
            }
        )
        performance_analyzer = PerformanceAnalyzer(
            {
                "execution_time": True,
                "memory_usage": True,
                "cpu_usage": True,
                "gpu_usage": True,
            }
        )

        # Mock test data
        test_data = {
            "field_data": np.random.rand(64, 64, 64),
            "phase_reference": np.random.rand(64, 64, 64),
            "execution_times": [1.0, 1.1, 1.2, 1.3, 1.4],
        }

        # Test phase mapping
        phase_results = phase_mapper.map_phase_structure(test_data)

        # Test performance analysis
        performance_results = performance_analyzer.analyze_execution_time(test_data)

        # Verify integration
        assert phase_results is not None
        assert performance_results is not None

        # Check phase results
        assert "phase_map" in phase_results
        assert "mapping_accuracy" in phase_results

        # Check performance results
        assert "execution_statistics" in performance_results
        assert "performance_trends" in performance_results

    def test_soliton_defect_integration(self):
        """Test soliton-defect integration."""
        # Initialize components
        soliton_model = SolitonModel(
            {"mass": 1.0, "charge": 1.0, "radius": 1.0, "velocity": 0.1}
        )
        defect_model = DefectModel(
            {"defect_type": "vortex", "charge": 1.0, "radius": 1.0, "strength": 1.0}
        )

        # Mock test data
        test_data = {
            "soliton_data": {
                "position": np.array([0.0, 0.0, 0.0]),
                "velocity": np.array([0.1, 0.0, 0.0]),
            },
            "defect_data": {
                "position": np.array([2.0, 0.0, 0.0]),
                "velocity": np.array([-0.1, 0.0, 0.0]),
            },
        }

        # Test soliton creation
        soliton_results = soliton_model.create_soliton(test_data["soliton_data"])

        # Test defect creation
        defect_results = defect_model.create_defect(test_data["defect_data"])

        # Verify integration
        assert soliton_results is not None
        assert defect_results is not None

        # Check soliton results
        assert "soliton_field" in soliton_results
        assert "soliton_properties" in soliton_results

        # Check defect results
        assert "defect_field" in defect_results
        assert "defect_properties" in defect_results

    def test_baryon_skyrmion_integration(self):
        """Test baryon-skyrmion integration."""
        # Initialize components
        baryon_model = BaryonSoliton(
            {"mass": 1.0, "charge": 1.0, "radius": 1.0, "velocity": 0.1}
        )
        skyrmion_model = SkyrmionSoliton(
            {"mass": 1.0, "charge": 1.0, "radius": 1.0, "velocity": 0.1}
        )

        # Mock test data
        test_data = {
            "baryon_data": {
                "position": np.array([0.0, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0]),
            },
            "skyrmion_data": {
                "position": np.array([2.0, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0]),
            },
        }

        # Test baryon creation
        baryon_results = baryon_model.create_baryon_soliton(test_data["baryon_data"])

        # Test skyrmion creation
        skyrmion_results = skyrmion_model.create_skyrmion_soliton(
            test_data["skyrmion_data"]
        )

        # Verify integration
        assert baryon_results is not None
        assert skyrmion_results is not None

        # Check baryon results
        assert "baryon_field" in baryon_results
        assert "baryon_properties" in baryon_results

        # Check skyrmion results
        assert "skyrmion_field" in skyrmion_results
        assert "skyrmion_properties" in skyrmion_results

    def test_vortex_multi_defect_integration(self):
        """Test vortex-multi-defect integration."""
        # Initialize components
        vortex_model = VortexDefect(
            {"defect_type": "vortex", "charge": 1.0, "radius": 1.0, "strength": 1.0}
        )
        multi_defect_model = MultiDefectSystem(
            {"defect_type": "vortex", "charge": 1.0, "radius": 1.0, "strength": 1.0}
        )

        # Mock test data
        test_data = {
            "vortex_data": {
                "position": np.array([0.0, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0]),
            },
            "multi_defect_data": {
                "defect_positions": [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([2.0, 0.0, 0.0]),
                    np.array([0.0, 2.0, 0.0]),
                ],
                "defect_charges": [1.0, -1.0, 1.0],
            },
        }

        # Test vortex creation
        vortex_results = vortex_model.create_vortex_defect(test_data["vortex_data"])

        # Test multi-defect creation
        multi_defect_results = multi_defect_model.create_multi_defect_system(
            test_data["multi_defect_data"]
        )

        # Verify integration
        assert vortex_results is not None
        assert multi_defect_results is not None

        # Check vortex results
        assert "vortex_field" in vortex_results
        assert "vortex_properties" in vortex_results

        # Check multi-defect results
        assert "defect_system_field" in multi_defect_results
        assert "defect_system_properties" in multi_defect_results

    def test_experiment_validation_integration(self):
        """Test experiment validation integration."""
        # Initialize components
        experiments = LevelEExperiments(
            {
                "experiment_type": "soliton_formation",
                "duration": 10.0,
                "time_step": 0.01,
                "grid_size": 64,
            }
        )
        failure_detector = FailureDetector(
            {
                "energy_conservation": 1e-6,
                "momentum_conservation": 1e-6,
                "topological_charge": 1e-8,
            }
        )

        # Mock test data
        test_data = {
            "experiment_data": {
                "initial_field": np.random.rand(64, 64, 64),
                "formation_parameters": {"mass": 1.0, "charge": 1.0, "radius": 1.0},
            },
            "validation_data": {
                "experiment_results": {
                    "final_field": np.random.rand(64, 64, 64),
                    "formation_time": 5.0,
                },
                "validation_criteria": {
                    "energy_conservation": 1e-6,
                    "momentum_conservation": 1e-6,
                    "topological_charge": 1e-8,
                },
            },
        }

        # Test experiment execution
        experiment_results = experiments.run_soliton_formation_experiment(
            test_data["experiment_data"]
        )

        # Test experiment validation
        validation_results = experiments.validate_experiment(
            test_data["validation_data"]
        )

        # Verify integration
        assert experiment_results is not None
        assert validation_results is not None

        # Check experiment results
        assert "experiment_results" in experiment_results
        assert "formation_analysis" in experiment_results

        # Check validation results
        assert "validation_results" in validation_results
        assert "validation_metrics" in validation_results

    def test_end_to_end_integration(self):
        """Test end-to-end integration."""
        # Initialize all components
        sensitivity_analyzer = SensitivityAnalyzer(
            {"beta": (0.6, 1.4), "mu": (0.5, 1.5), "eta": (0.0, 0.3)}
        )
        robustness_tester = RobustnessTester(
            {"noise_level": 0.1, "perturbation_range": 0.05, "test_iterations": 100}
        )
        discretization_analyzer = DiscretizationAnalyzer(
            [32, 64, 128, 256], [0.001, 0.01, 0.1, 1.0]
        )
        failure_detector = FailureDetector(
            {
                "energy_conservation": 1e-6,
                "momentum_conservation": 1e-6,
                "topological_charge": 1e-8,
            }
        )
        phase_mapper = PhaseMapper(
            {
                "phase_resolution": 0.01,
                "mapping_tolerance": 1e-6,
                "max_iterations": 1000,
            }
        )
        performance_analyzer = PerformanceAnalyzer(
            {
                "execution_time": True,
                "memory_usage": True,
                "cpu_usage": True,
                "gpu_usage": True,
            }
        )
        soliton_model = SolitonModel(
            {"mass": 1.0, "charge": 1.0, "radius": 1.0, "velocity": 0.1}
        )
        defect_model = DefectModel(
            {"defect_type": "vortex", "charge": 1.0, "radius": 1.0, "strength": 1.0}
        )
        experiments = LevelEExperiments(
            {
                "experiment_type": "soliton_formation",
                "duration": 10.0,
                "time_step": 0.01,
                "grid_size": 64,
            }
        )

        # Mock test data
        test_data = {
            "field_data": np.random.rand(64, 64, 64),
            "parameter_ranges": {
                "beta": (0.6, 1.4),
                "mu": (0.5, 1.5),
                "eta": (0.0, 0.3),
            },
            "experiment_data": {
                "initial_field": np.random.rand(64, 64, 64),
                "formation_parameters": {"mass": 1.0, "charge": 1.0, "radius": 1.0},
            },
        }

        # Test all components
        sensitivity_results = sensitivity_analyzer.analyze_parameter_sensitivity(
            test_data
        )
        robustness_results = robustness_tester.test_noise_robustness(test_data)
        discretization_results = discretization_analyzer.analyze_grid_convergence(
            test_data
        )
        failure_results = failure_detector.detect_energy_conservation_failure(test_data)
        phase_results = phase_mapper.map_phase_structure(test_data)
        performance_results = performance_analyzer.analyze_execution_time(test_data)
        soliton_results = soliton_model.create_soliton(test_data["experiment_data"])
        defect_results = defect_model.create_defect(test_data["experiment_data"])
        experiment_results = experiments.run_soliton_formation_experiment(
            test_data["experiment_data"]
        )

        # Verify all results
        assert sensitivity_results is not None
        assert robustness_results is not None
        assert discretization_results is not None
        assert failure_results is not None
        assert phase_results is not None
        assert performance_results is not None
        assert soliton_results is not None
        assert defect_results is not None
        assert experiment_results is not None

        # Check all result structures
        assert "sensitivity_indices" in sensitivity_results
        assert "noise_tolerance" in robustness_results
        assert "convergence_rate" in discretization_results
        assert "failure_detected" in failure_results
        assert "phase_map" in phase_results
        assert "execution_statistics" in performance_results
        assert "soliton_field" in soliton_results
        assert "defect_field" in defect_results
        assert "experiment_results" in experiment_results
