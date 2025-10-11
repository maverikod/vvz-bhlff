"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for soliton models functionality.

This module tests the soliton models functionality
for Level E experiments in 7D phase field theory.

Physical Meaning:
    Tests ensure that soliton models correctly
    implement soliton physics for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/solitons/test_soliton_models.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import SolitonModel, BaryonSoliton, SkyrmionSoliton


class TestSolitonModels:
    """Test soliton models functionality."""

    def test_soliton_model_initialization(self):
        """Test SolitonModel initialization."""
        model_parameters = {
            "mass": 1.0,
            "charge": 1.0,
            "radius": 1.0,
            "velocity": 0.1,
        }

        model = SolitonModel(model_parameters)

        assert model.model_parameters == model_parameters
        assert model.soliton_properties is None
        assert model.soliton_dynamics is None

    def test_soliton_creation(self):
        """Test soliton creation."""
        model_parameters = {
            "mass": 1.0,
            "charge": 1.0,
            "radius": 1.0,
            "velocity": 0.1,
        }

        model = SolitonModel(model_parameters)

        # Mock test data
        test_data = {
            "position": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0]),
            "field_amplitude": 1.0,
        }

        # Test soliton creation
        results = model.create_soliton(test_data)

        assert results is not None
        assert "soliton_field" in results
        assert "soliton_properties" in results
        assert "creation_energy" in results

        # Check soliton field
        soliton_field = results["soliton_field"]
        assert isinstance(soliton_field, np.ndarray)
        assert soliton_field.shape == (64, 64, 64)

        # Check soliton properties
        soliton_properties = results["soliton_properties"]
        assert isinstance(soliton_properties, dict)
        assert "mass" in soliton_properties
        assert "charge" in soliton_properties
        assert "radius" in soliton_properties

        # Check creation energy
        creation_energy = results["creation_energy"]
        assert isinstance(creation_energy, float)
        assert creation_energy > 0

    def test_soliton_dynamics(self):
        """Test soliton dynamics."""
        model_parameters = {
            "mass": 1.0,
            "charge": 1.0,
            "radius": 1.0,
            "velocity": 0.1,
        }

        model = SolitonModel(model_parameters)

        # Mock test data
        test_data = {
            "initial_position": np.array([0.0, 0.0, 0.0]),
            "initial_velocity": np.array([0.1, 0.0, 0.0]),
            "time_steps": 100,
            "time_step": 0.01,
        }

        # Test soliton dynamics
        results = model.simulate_soliton_dynamics(test_data)

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

    def test_soliton_interactions(self):
        """Test soliton interactions."""
        model_parameters = {
            "mass": 1.0,
            "charge": 1.0,
            "radius": 1.0,
            "velocity": 0.1,
        }

        model = SolitonModel(model_parameters)

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
            "interaction_time": 1.0,
        }

        # Test soliton interactions
        results = model.simulate_soliton_interactions(test_data)

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
        assert "soliton1_final" in final_states
        assert "soliton2_final" in final_states

    def test_baryon_soliton(self):
        """Test baryon soliton model."""
        model_parameters = {
            "mass": 1.0,
            "charge": 1.0,
            "radius": 1.0,
            "velocity": 0.1,
        }

        model = BaryonSoliton(model_parameters)

        # Mock test data
        test_data = {
            "position": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0]),
            "field_amplitude": 1.0,
        }

        # Test baryon soliton creation
        results = model.create_baryon_soliton(test_data)

        assert results is not None
        assert "baryon_field" in results
        assert "baryon_properties" in results
        assert "baryon_energy" in results

        # Check baryon field
        baryon_field = results["baryon_field"]
        assert isinstance(baryon_field, np.ndarray)
        assert baryon_field.shape == (64, 64, 64)

        # Check baryon properties
        baryon_properties = results["baryon_properties"]
        assert isinstance(baryon_properties, dict)
        assert "baryon_number" in baryon_properties
        assert "isospin" in baryon_properties

        # Check baryon energy
        baryon_energy = results["baryon_energy"]
        assert isinstance(baryon_energy, float)
        assert baryon_energy > 0

    def test_skyrmion_soliton(self):
        """Test skyrmion soliton model."""
        model_parameters = {
            "mass": 1.0,
            "charge": 1.0,
            "radius": 1.0,
            "velocity": 0.1,
        }

        model = SkyrmionSoliton(model_parameters)

        # Mock test data
        test_data = {
            "position": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0]),
            "field_amplitude": 1.0,
        }

        # Test skyrmion soliton creation
        results = model.create_skyrmion_soliton(test_data)

        assert results is not None
        assert "skyrmion_field" in results
        assert "skyrmion_properties" in results
        assert "skyrmion_energy" in results

        # Check skyrmion field
        skyrmion_field = results["skyrmion_field"]
        assert isinstance(skyrmion_field, np.ndarray)
        assert skyrmion_field.shape == (64, 64, 64)

        # Check skyrmion properties
        skyrmion_properties = results["skyrmion_properties"]
        assert isinstance(skyrmion_properties, dict)
        assert "topological_charge" in skyrmion_properties
        assert "winding_number" in skyrmion_properties

        # Check skyrmion energy
        skyrmion_energy = results["skyrmion_energy"]
        assert isinstance(skyrmion_energy, float)
        assert skyrmion_energy > 0

    def test_soliton_stability(self):
        """Test soliton stability analysis."""
        model_parameters = {
            "mass": 1.0,
            "charge": 1.0,
            "radius": 1.0,
            "velocity": 0.1,
        }

        model = SolitonModel(model_parameters)

        # Mock test data
        test_data = {
            "soliton_field": np.random.rand(64, 64, 64),
            "perturbation_amplitude": 0.1,
            "stability_time": 10.0,
        }

        # Test soliton stability
        results = model.analyze_soliton_stability(test_data)

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

    def test_soliton_report(self):
        """Test soliton report generation."""
        model_parameters = {
            "mass": 1.0,
            "charge": 1.0,
            "radius": 1.0,
            "velocity": 0.1,
        }

        model = SolitonModel(model_parameters)

        # Mock test data
        test_data = {
            "soliton_field": np.random.rand(64, 64, 64),
            "soliton_properties": {
                "mass": 1.0,
                "charge": 1.0,
                "radius": 1.0,
            },
        }

        # Test soliton report
        report = model.generate_soliton_report(test_data)

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
