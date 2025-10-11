"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for Level E experiments functionality.

This module tests the Level E experiments functionality
for 7D phase field theory.

Physical Meaning:
    Tests ensure that Level E experiments correctly
    implement experimental protocols for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/experiments/test_level_e_experiments.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import LevelEExperiments


class TestLevelEExperiments:
    """Test Level E experiments functionality."""

    def test_initialization(self):
        """Test LevelEExperiments initialization."""
        experiment_parameters = {
            "experiment_type": "soliton_formation",
            "duration": 10.0,
            "time_step": 0.01,
            "grid_size": 64,
        }

        experiments = LevelEExperiments(experiment_parameters)

        assert experiments.experiment_parameters == experiment_parameters
        assert experiments.experiment_results is None
        assert experiments.experiment_statistics is None

    def test_soliton_formation_experiment(self):
        """Test soliton formation experiment."""
        experiment_parameters = {
            "experiment_type": "soliton_formation",
            "duration": 10.0,
            "time_step": 0.01,
            "grid_size": 64,
        }

        experiments = LevelEExperiments(experiment_parameters)

        # Mock test data
        test_data = {
            "initial_field": np.random.rand(64, 64, 64),
            "formation_parameters": {
                "mass": 1.0,
                "charge": 1.0,
                "radius": 1.0,
            },
        }

        # Test soliton formation experiment
        results = experiments.run_soliton_formation_experiment(test_data)

        assert results is not None
        assert "experiment_results" in results
        assert "formation_analysis" in results
        assert "experiment_statistics" in results

        # Check experiment results
        experiment_results = results["experiment_results"]
        assert isinstance(experiment_results, dict)
        assert "final_field" in experiment_results
        assert "formation_time" in experiment_results

        # Check formation analysis
        formation_analysis = results["formation_analysis"]
        assert isinstance(formation_analysis, dict)
        assert "formation_success" in formation_analysis
        assert "formation_quality" in formation_analysis

        # Check experiment statistics
        experiment_statistics = results["experiment_statistics"]
        assert isinstance(experiment_statistics, dict)
        assert "execution_time" in experiment_statistics
        assert "memory_usage" in experiment_statistics

    def test_defect_formation_experiment(self):
        """Test defect formation experiment."""
        experiment_parameters = {
            "experiment_type": "defect_formation",
            "duration": 10.0,
            "time_step": 0.01,
            "grid_size": 64,
        }

        experiments = LevelEExperiments(experiment_parameters)

        # Mock test data
        test_data = {
            "initial_field": np.random.rand(64, 64, 64),
            "formation_parameters": {
                "defect_type": "vortex",
                "charge": 1.0,
                "radius": 1.0,
            },
        }

        # Test defect formation experiment
        results = experiments.run_defect_formation_experiment(test_data)

        assert results is not None
        assert "experiment_results" in results
        assert "formation_analysis" in results
        assert "experiment_statistics" in results

        # Check experiment results
        experiment_results = results["experiment_results"]
        assert isinstance(experiment_results, dict)
        assert "final_field" in experiment_results
        assert "formation_time" in experiment_results

        # Check formation analysis
        formation_analysis = results["formation_analysis"]
        assert isinstance(formation_analysis, dict)
        assert "formation_success" in formation_analysis
        assert "formation_quality" in formation_analysis

        # Check experiment statistics
        experiment_statistics = results["experiment_statistics"]
        assert isinstance(experiment_statistics, dict)
        assert "execution_time" in experiment_statistics
        assert "memory_usage" in experiment_statistics

    def test_soliton_interaction_experiment(self):
        """Test soliton interaction experiment."""
        experiment_parameters = {
            "experiment_type": "soliton_interaction",
            "duration": 10.0,
            "time_step": 0.01,
            "grid_size": 64,
        }

        experiments = LevelEExperiments(experiment_parameters)

        # Mock test data
        test_data = {
            "soliton1": {
                "position": np.array([0.0, 0.0, 0.0]),
                "velocity": np.array([0.1, 0.0, 0.0]),
            },
            "soliton2": {
                "position": np.array([2.0, 0.0, 0.0]),
                "velocity": np.array([-0.1, 0.0, 0.0]),
            },
            "interaction_parameters": {
                "interaction_strength": 1.0,
                "interaction_range": 1.0,
            },
        }

        # Test soliton interaction experiment
        results = experiments.run_soliton_interaction_experiment(test_data)

        assert results is not None
        assert "experiment_results" in results
        assert "interaction_analysis" in results
        assert "experiment_statistics" in results

        # Check experiment results
        experiment_results = results["experiment_results"]
        assert isinstance(experiment_results, dict)
        assert "final_states" in experiment_results
        assert "interaction_time" in experiment_results

        # Check interaction analysis
        interaction_analysis = results["interaction_analysis"]
        assert isinstance(interaction_analysis, dict)
        assert "scattering_angle" in interaction_analysis
        assert "energy_transfer" in interaction_analysis

        # Check experiment statistics
        experiment_statistics = results["experiment_statistics"]
        assert isinstance(experiment_statistics, dict)
        assert "execution_time" in experiment_statistics
        assert "memory_usage" in experiment_statistics

    def test_defect_interaction_experiment(self):
        """Test defect interaction experiment."""
        experiment_parameters = {
            "experiment_type": "defect_interaction",
            "duration": 10.0,
            "time_step": 0.01,
            "grid_size": 64,
        }

        experiments = LevelEExperiments(experiment_parameters)

        # Mock test data
        test_data = {
            "defect1": {
                "position": np.array([0.0, 0.0, 0.0]),
                "velocity": np.array([0.1, 0.0, 0.0]),
            },
            "defect2": {
                "position": np.array([2.0, 0.0, 0.0]),
                "velocity": np.array([-0.1, 0.0, 0.0]),
            },
            "interaction_parameters": {
                "interaction_strength": 1.0,
                "interaction_range": 1.0,
            },
        }

        # Test defect interaction experiment
        results = experiments.run_defect_interaction_experiment(test_data)

        assert results is not None
        assert "experiment_results" in results
        assert "interaction_analysis" in results
        assert "experiment_statistics" in results

        # Check experiment results
        experiment_results = results["experiment_results"]
        assert isinstance(experiment_results, dict)
        assert "final_states" in experiment_results
        assert "interaction_time" in experiment_results

        # Check interaction analysis
        interaction_analysis = results["interaction_analysis"]
        assert isinstance(interaction_analysis, dict)
        assert "scattering_angle" in interaction_analysis
        assert "energy_transfer" in interaction_analysis

        # Check experiment statistics
        experiment_statistics = results["experiment_statistics"]
        assert isinstance(experiment_statistics, dict)
        assert "execution_time" in experiment_statistics
        assert "memory_usage" in experiment_statistics

    def test_phase_transition_experiment(self):
        """Test phase transition experiment."""
        experiment_parameters = {
            "experiment_type": "phase_transition",
            "duration": 10.0,
            "time_step": 0.01,
            "grid_size": 64,
        }

        experiments = LevelEExperiments(experiment_parameters)

        # Mock test data
        test_data = {
            "initial_phase": np.random.rand(64, 64, 64),
            "transition_parameters": {
                "transition_temperature": 1.0,
                "transition_rate": 0.1,
            },
        }

        # Test phase transition experiment
        results = experiments.run_phase_transition_experiment(test_data)

        assert results is not None
        assert "experiment_results" in results
        assert "transition_analysis" in results
        assert "experiment_statistics" in results

        # Check experiment results
        experiment_results = results["experiment_results"]
        assert isinstance(experiment_results, dict)
        assert "final_phase" in experiment_results
        assert "transition_time" in experiment_results

        # Check transition analysis
        transition_analysis = results["transition_analysis"]
        assert isinstance(transition_analysis, dict)
        assert "transition_success" in transition_analysis
        assert "transition_quality" in transition_analysis

        # Check experiment statistics
        experiment_statistics = results["experiment_statistics"]
        assert isinstance(experiment_statistics, dict)
        assert "execution_time" in experiment_statistics
        assert "memory_usage" in experiment_statistics

    def test_experiment_validation(self):
        """Test experiment validation."""
        experiment_parameters = {
            "experiment_type": "soliton_formation",
            "duration": 10.0,
            "time_step": 0.01,
            "grid_size": 64,
        }

        experiments = LevelEExperiments(experiment_parameters)

        # Mock test data
        test_data = {
            "experiment_results": {
                "final_field": np.random.rand(64, 64, 64),
                "formation_time": 5.0,
            },
            "validation_criteria": {
                "energy_conservation": 1e-6,
                "momentum_conservation": 1e-6,
                "topological_charge": 1e-8,
            },
        }

        # Test experiment validation
        results = experiments.validate_experiment(test_data)

        assert results is not None
        assert "validation_results" in results
        assert "validation_metrics" in results
        assert "validation_status" in results

        # Check validation results
        validation_results = results["validation_results"]
        assert isinstance(validation_results, dict)
        assert "energy_conservation" in validation_results
        assert "momentum_conservation" in validation_results
        assert "topological_charge" in validation_results

        # Check validation metrics
        validation_metrics = results["validation_metrics"]
        assert isinstance(validation_metrics, dict)
        assert "overall_score" in validation_metrics
        assert "validation_errors" in validation_metrics

        # Check validation status
        validation_status = results["validation_status"]
        assert isinstance(validation_status, str)
        assert validation_status in ["valid", "invalid", "warning"]

    def test_experiment_report(self):
        """Test experiment report generation."""
        experiment_parameters = {
            "experiment_type": "soliton_formation",
            "duration": 10.0,
            "time_step": 0.01,
            "grid_size": 64,
        }

        experiments = LevelEExperiments(experiment_parameters)

        # Mock test data
        test_data = {
            "experiment_results": {
                "final_field": np.random.rand(64, 64, 64),
                "formation_time": 5.0,
            },
            "experiment_statistics": {
                "execution_time": 10.0,
                "memory_usage": 1000,
            },
        }

        # Test experiment report
        report = experiments.generate_experiment_report(test_data)

        assert report is not None
        assert "summary" in report
        assert "detailed_analysis" in report
        assert "recommendations" in report

        # Check report content
        summary = report["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0

        detailed_analysis = report["detailed_analysis"]
        assert isinstance(detailed_analysis, dict)
        assert len(detailed_analysis) > 0

        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
