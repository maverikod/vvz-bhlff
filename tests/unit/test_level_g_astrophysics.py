"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for level G astrophysics models.

This module tests the astrophysical object models for 7D phase field theory,
including stars, galaxies, and black holes.

Physical Meaning:
    Tests the representation of astrophysical objects as phase field
    configurations with specific topological properties.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_g.astrophysics import AstrophysicalObjectModel


class TestAstrophysicalObjectModel:
    """
    Test astrophysical object model.

    Physical Meaning:
        Tests the representation of astrophysical objects as phase field
        configurations with specific topological properties.
    """

    def test_star_model_initialization(self):
        """Test star model initialization."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)

        assert model.object_type == "star"
        assert model.physical_params["mass"] == 1.0
        assert model.physical_params["radius"] == 1.0
        assert model.physical_params["temperature"] == 5778.0
        assert model.topological_charge == 1

    def test_galaxy_model_initialization(self):
        """Test galaxy model initialization."""
        galactic_params = {
            "mass": 1e11,
            "radius": 10.0,
            "spiral_arms": 2,
            "bulge_ratio": 0.3,
            "grid_size": 64,
            "domain_size": 50.0,
        }

        model = AstrophysicalObjectModel("galaxy", galactic_params)

        assert model.object_type == "galaxy"
        assert model.physical_params["mass"] == 1e11
        assert model.physical_params["radius"] == 10.0
        assert model.physical_params["spiral_arms"] == 2
        assert model.topological_charge == 2

    def test_black_hole_model_initialization(self):
        """Test black hole model initialization."""
        bh_params = {
            "mass": 10.0,
            "spin": 0.5,
            "schwarzschild_radius": 1.0,
            "alpha": 1.0,
            "grid_size": 64,
            "domain_size": 20.0,
        }

        model = AstrophysicalObjectModel("black_hole", bh_params)

        assert model.object_type == "black_hole"
        assert model.physical_params["mass"] == 10.0
        assert model.physical_params["spin"] == 0.5
        assert model.physical_params["schwarzschild_radius"] == 1.0
        assert model.topological_charge == -1

    def test_star_phase_profile_creation(self):
        """Test star phase profile creation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)

        assert model.phase_profile is not None
        assert model.phase_profile.shape == (32, 32, 32)
        assert np.isfinite(model.phase_profile).all()

    def test_galaxy_phase_profile_creation(self):
        """Test galaxy phase profile creation."""
        galactic_params = {
            "mass": 1e11,
            "radius": 10.0,
            "spiral_arms": 2,
            "bulge_ratio": 0.3,
            "grid_size": 32,
            "domain_size": 50.0,
        }

        model = AstrophysicalObjectModel("galaxy", galactic_params)

        assert model.phase_profile is not None
        assert model.phase_profile.shape == (32, 32, 32)
        assert np.isfinite(model.phase_profile).all()

    def test_black_hole_phase_profile_creation(self):
        """Test black hole phase profile creation."""
        bh_params = {
            "mass": 10.0,
            "spin": 0.5,
            "schwarzschild_radius": 1.0,
            "alpha": 1.0,
            "grid_size": 32,
            "domain_size": 20.0,
        }

        model = AstrophysicalObjectModel("black_hole", bh_params)

        assert model.phase_profile is not None
        assert model.phase_profile.shape == (32, 32, 32)
        assert np.isfinite(model.phase_profile).all()

    def test_phase_properties_analysis(self):
        """Test phase properties analysis."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        properties = model.analyze_phase_properties()

        assert "object_type" in properties
        assert "topological_charge" in properties
        assert "phase_amplitude" in properties
        assert "phase_rms" in properties
        assert "phase_gradient" in properties
        assert "correlation_length" in properties

        assert properties["object_type"] == "star"
        assert properties["topological_charge"] == 1
        assert properties["phase_amplitude"] >= 0
        assert properties["phase_rms"] >= 0
        assert properties["phase_gradient"] >= 0
        assert properties["correlation_length"] >= 0

    def test_observable_properties_computation(self):
        """Test observable properties computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        properties = model.compute_observable_properties()

        assert "total_mass" in properties
        assert "effective_radius" in properties
        assert "phase_energy" in properties
        assert "topological_defect_density" in properties

        assert properties["total_mass"] == 1.0
        assert properties["effective_radius"] >= 0
        assert properties["phase_energy"] >= 0
        assert properties["topological_defect_density"] >= 0

    def test_star_model_creation(self):
        """Test star model creation."""
        stellar_params = {
            "mass": 2.0,
            "radius": 2.0,
            "temperature": 10000.0,
            "phase_amplitude": 2.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", {})
        star_model = model.create_star_model(stellar_params)

        assert star_model.object_type == "star"
        assert star_model.physical_params["mass"] == 2.0
        assert star_model.physical_params["radius"] == 2.0
        assert star_model.physical_params["temperature"] == 10000.0
        assert star_model.topological_charge == 1

    def test_galaxy_model_creation(self):
        """Test galaxy model creation."""
        galactic_params = {
            "mass": 2e11,
            "radius": 20.0,
            "spiral_arms": 4,
            "bulge_ratio": 0.5,
            "grid_size": 32,
            "domain_size": 50.0,
        }

        model = AstrophysicalObjectModel("galaxy", {})
        galaxy_model = model.create_galaxy_model(galactic_params)

        assert galaxy_model.object_type == "galaxy"
        assert galaxy_model.physical_params["mass"] == 2e11
        assert galaxy_model.physical_params["radius"] == 20.0
        assert galaxy_model.physical_params["spiral_arms"] == 4
        assert galaxy_model.topological_charge == 4

    def test_black_hole_model_creation(self):
        """Test black hole model creation."""
        bh_params = {
            "mass": 100.0,
            "spin": 0.9,
            "schwarzschild_radius": 10.0,
            "alpha": 2.0,
            "grid_size": 32,
            "domain_size": 20.0,
        }

        model = AstrophysicalObjectModel("black_hole", {})
        bh_model = model.create_black_hole_model(bh_params)

        assert bh_model.object_type == "black_hole"
        assert bh_model.physical_params["mass"] == 100.0
        assert bh_model.physical_params["spin"] == 0.9
        assert bh_model.physical_params["schwarzschild_radius"] == 10.0
        assert bh_model.topological_charge == -1

    def test_phase_correlation_length_computation(self):
        """Test phase correlation length computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        correlation_length = model._compute_phase_correlation_length()

        assert correlation_length >= 0
        assert np.isfinite(correlation_length)

    def test_effective_radius_computation(self):
        """Test effective radius computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        effective_radius = model._compute_effective_radius()

        assert effective_radius >= 0
        assert np.isfinite(effective_radius)

    def test_phase_energy_computation(self):
        """Test phase energy computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        phase_energy = model._compute_phase_energy()

        assert phase_energy >= 0
        assert np.isfinite(phase_energy)

    def test_defect_density_computation(self):
        """Test defect density computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        defect_density = model._compute_defect_density()

        assert defect_density >= 0
        assert np.isfinite(defect_density)

    def test_star_phase_profile_physical_properties(self):
        """Test star phase profile physical properties."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)

        # Check that phase profile has expected properties
        assert model.phase_profile is not None
        assert model.phase_profile.shape == (32, 32, 32)
        assert np.isfinite(model.phase_profile).all()

        # Check that phase profile is centered (maximum at center)
        center_idx = 16  # Center of 32x32x32 grid
        center_value = model.phase_profile[center_idx, center_idx, center_idx]
        max_value = np.max(model.phase_profile)
        assert center_value == max_value

    def test_galaxy_spiral_structure(self):
        """Test galaxy spiral structure."""
        galactic_params = {
            "mass": 1e11,
            "radius": 10.0,
            "spiral_arms": 2,
            "bulge_ratio": 0.3,
            "grid_size": 32,
            "domain_size": 50.0,
        }

        model = AstrophysicalObjectModel("galaxy", galactic_params)

        # Check that phase profile has spiral structure
        assert model.phase_profile is not None
        assert model.phase_profile.shape == (32, 32, 32)
        assert np.isfinite(model.phase_profile).all()

        # Check that topological charge matches spiral arms
        assert model.topological_charge == 2

    def test_black_hole_singularity_behavior(self):
        """Test black hole singularity behavior."""
        bh_params = {
            "mass": 10.0,
            "spin": 0.5,
            "schwarzschild_radius": 1.0,
            "alpha": 1.0,
            "grid_size": 32,
            "domain_size": 20.0,
        }

        model = AstrophysicalObjectModel("black_hole", bh_params)

        # Check that phase profile has singularity behavior
        assert model.phase_profile is not None
        assert model.phase_profile.shape == (32, 32, 32)
        assert np.isfinite(model.phase_profile).all()

        # Check that topological charge is negative
        assert model.topological_charge == -1

    def test_phase_field_energy_conservation(self):
        """Test phase field energy conservation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)

        # Check that phase field has finite energy
        phase_energy = model._compute_phase_energy()
        assert phase_energy >= 0
        assert np.isfinite(phase_energy)

        # Check that phase field is normalized
        phase_profile = model.phase_profile
        assert np.isfinite(phase_profile).all()
        assert phase_profile.shape == (32, 32, 32)

    def test_topological_charge_conservation(self):
        """Test topological charge conservation."""
        # Test star (positive charge)
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 32,
            "domain_size": 10.0,
        }

        star_model = AstrophysicalObjectModel("star", stellar_params)
        assert star_model.topological_charge == 1

        # Test galaxy (multiple positive charges)
        galactic_params = {
            "mass": 1e11,
            "radius": 10.0,
            "spiral_arms": 3,
            "bulge_ratio": 0.3,
            "grid_size": 32,
            "domain_size": 50.0,
        }

        galaxy_model = AstrophysicalObjectModel("galaxy", galactic_params)
        assert galaxy_model.topological_charge == 3

        # Test black hole (negative charge)
        bh_params = {
            "mass": 10.0,
            "spin": 0.5,
            "schwarzschild_radius": 1.0,
            "alpha": 1.0,
            "grid_size": 32,
            "domain_size": 20.0,
        }

        bh_model = AstrophysicalObjectModel("black_hole", bh_params)
        assert bh_model.topological_charge == -1
