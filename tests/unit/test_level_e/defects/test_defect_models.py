"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for defect models functionality.

This module tests the defect models functionality
for Level E experiments in 7D phase field theory.

Physical Meaning:
    Tests ensure that defect models correctly
    implement defect physics for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/defects/test_defect_models.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import DefectModel, VortexDefect, MultiDefectSystem


class TestDefectModels:
    """Test defect models functionality."""

    def test_defect_model_initialization(self):
        """Test DefectModel initialization."""
        model_parameters = {
            "defect_type": "vortex",
            "charge": 1.0,
            "radius": 1.0,
            "strength": 1.0,
        }

        model = DefectModel(model_parameters)

        assert model.model_parameters == model_parameters
        assert model.defect_properties is None
        assert model.defect_dynamics is None

    def test_defect_creation(self):
        """Test defect creation."""
        model_parameters = {
            "defect_type": "vortex",
            "charge": 1.0,
            "radius": 1.0,
            "strength": 1.0,
        }

        model = DefectModel(model_parameters)

        # Mock test data
        test_data = {
            "position": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0]),
            "field_amplitude": 1.0,
        }

        # Test defect creation
        results = model.create_defect(test_data)

        assert results is not None
        assert "defect_field" in results
        assert "defect_properties" in results
        assert "creation_energy" in results

        # Check defect field
        defect_field = results["defect_field"]
        assert isinstance(defect_field, np.ndarray)
        assert defect_field.shape == (64, 64, 64)

        # Check defect properties
        defect_properties = results["defect_properties"]
        assert isinstance(defect_properties, dict)
        assert "charge" in defect_properties
        assert "radius" in defect_properties
        assert "strength" in defect_properties

        # Check creation energy
        creation_energy = results["creation_energy"]
        assert isinstance(creation_energy, float)
        assert creation_energy > 0

    def test_defect_dynamics(self):
        """Test defect dynamics."""
        model_parameters = {
            "defect_type": "vortex",
            "charge": 1.0,
            "radius": 1.0,
            "strength": 1.0,
        }

        model = DefectModel(model_parameters)

        # Mock test data
        test_data = {
            "initial_position": np.array([0.0, 0.0, 0.0]),
            "initial_velocity": np.array([0.1, 0.0, 0.0]),
            "time_steps": 100,
            "time_step": 0.01,
        }

        # Test defect dynamics
        results = model.simulate_defect_dynamics(test_data)

        assert results is not None
        assert "trajectory" in results
        assert "velocity_history" in results
        assert "energy_history" in results

        # Check trajectory
        trajectory = results["trajectory"]
        assert isinstance(trajectory, np.ndarray)
        assert trajectory.shape == (100, 3)

        # Check velocity history
        velocity_history = results["velocity_history"]
        assert isinstance(velocity_history, np.ndarray)
        assert velocity_history.shape == (100, 3)

        # Check energy history
        energy_history = results["energy_history"]
        assert isinstance(energy_history, np.ndarray)
        assert energy_history.shape == (100,)

    def test_defect_interactions(self):
        """Test defect interactions."""
        model_parameters = {
            "defect_type": "vortex",
            "charge": 1.0,
            "radius": 1.0,
            "strength": 1.0,
        }

        model = DefectModel(model_parameters)

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
            "interaction_time": 1.0,
        }

        # Test defect interactions
        results = model.simulate_defect_interactions(test_data)

        assert results is not None
        assert "interaction_energy" in results
        assert "scattering_angle" in results
        assert "final_states" in results

        # Check interaction energy
        interaction_energy = results["interaction_energy"]
        assert isinstance(interaction_energy, float)
        assert interaction_energy >= 0

        # Check scattering angle
        scattering_angle = results["scattering_angle"]
        assert isinstance(scattering_angle, float)
        assert 0 <= scattering_angle <= np.pi

        # Check final states
        final_states = results["final_states"]
        assert isinstance(final_states, dict)
        assert "defect1_final" in final_states
        assert "defect2_final" in final_states

    def test_vortex_defect(self):
        """Test vortex defect model."""
        model_parameters = {
            "defect_type": "vortex",
            "charge": 1.0,
            "radius": 1.0,
            "strength": 1.0,
        }

        model = VortexDefect(model_parameters)

        # Mock test data
        test_data = {
            "position": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0]),
            "field_amplitude": 1.0,
        }

        # Test vortex defect creation
        results = model.create_vortex_defect(test_data)

        assert results is not None
        assert "vortex_field" in results
        assert "vortex_properties" in results
        assert "vortex_energy" in results

        # Check vortex field
        vortex_field = results["vortex_field"]
        assert isinstance(vortex_field, np.ndarray)
        assert vortex_field.shape == (64, 64, 64)

        # Check vortex properties
        vortex_properties = results["vortex_properties"]
        assert isinstance(vortex_properties, dict)
        assert "circulation" in vortex_properties
        assert "vorticity" in vortex_properties

        # Check vortex energy
        vortex_energy = results["vortex_energy"]
        assert isinstance(vortex_energy, float)
        assert vortex_energy > 0

    def test_multi_defect_system(self):
        """Test multi-defect system."""
        model_parameters = {
            "defect_type": "vortex",
            "charge": 1.0,
            "radius": 1.0,
            "strength": 1.0,
        }

        model = MultiDefectSystem(model_parameters)

        # Mock test data
        test_data = {
            "defect_positions": [
                np.array([0.0, 0.0, 0.0]),
                np.array([2.0, 0.0, 0.0]),
                np.array([0.0, 2.0, 0.0]),
            ],
            "defect_charges": [1.0, -1.0, 1.0],
            "interaction_strength": 1.0,
        }

        # Test multi-defect system
        results = model.create_multi_defect_system(test_data)

        assert results is not None
        assert "defect_system_field" in results
        assert "defect_system_properties" in results
        assert "system_energy" in results

        # Check defect system field
        defect_system_field = results["defect_system_field"]
        assert isinstance(defect_system_field, np.ndarray)
        assert defect_system_field.shape == (64, 64, 64)

        # Check defect system properties
        defect_system_properties = results["defect_system_properties"]
        assert isinstance(defect_system_properties, dict)
        assert "total_charge" in defect_system_properties
        assert "defect_count" in defect_system_properties

        # Check system energy
        system_energy = results["system_energy"]
        assert isinstance(system_energy, float)
        assert system_energy > 0

    def test_defect_stability(self):
        """Test defect stability analysis."""
        model_parameters = {
            "defect_type": "vortex",
            "charge": 1.0,
            "radius": 1.0,
            "strength": 1.0,
        }

        model = DefectModel(model_parameters)

        # Mock test data
        test_data = {
            "defect_field": np.random.rand(64, 64, 64),
            "perturbation_amplitude": 0.1,
            "stability_time": 10.0,
        }

        # Test defect stability
        results = model.analyze_defect_stability(test_data)

        assert results is not None
        assert "stability_metrics" in results
        assert "stability_analysis" in results
        assert "stability_recommendations" in results

        # Check stability metrics
        stability_metrics = results["stability_metrics"]
        assert isinstance(stability_metrics, dict)
        assert "stability_index" in stability_metrics
        assert "decay_rate" in stability_metrics

        # Check stability analysis
        stability_analysis = results["stability_analysis"]
        assert isinstance(stability_analysis, dict)
        assert "stable_regions" in stability_analysis
        assert "unstable_regions" in stability_analysis

        # Check stability recommendations
        stability_recommendations = results["stability_recommendations"]
        assert isinstance(stability_recommendations, list)
        assert len(stability_recommendations) >= 0

    def test_defect_report(self):
        """Test defect report generation."""
        model_parameters = {
            "defect_type": "vortex",
            "charge": 1.0,
            "radius": 1.0,
            "strength": 1.0,
        }

        model = DefectModel(model_parameters)

        # Mock test data
        test_data = {
            "defect_field": np.random.rand(64, 64, 64),
            "defect_properties": {
                "charge": 1.0,
                "radius": 1.0,
                "strength": 1.0,
            },
        }

        # Test defect report
        report = model.generate_defect_report(test_data)

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
