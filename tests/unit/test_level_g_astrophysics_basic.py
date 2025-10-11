"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic unit tests for level G astrophysics models.

This module tests the basic astrophysical object models for 7D phase field theory,
including initialization, phase profile creation, and basic properties.

Physical Meaning:
    Tests the basic representation of astrophysical objects as phase field
    configurations with specific topological properties.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_g.astrophysics import AstrophysicalObjectModel


class TestAstrophysicalObjectModelBasic:
    """
    Basic test astrophysical object model.

    Physical Meaning:
        Tests the basic representation of astrophysical objects as phase field
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
            "mass": 1e12,
            "radius": 50.0,
            "spiral_arms": 2,
            "phase_amplitude": 0.5,
            "grid_size": 128,
            "domain_size": 100.0,
        }

        model = AstrophysicalObjectModel("galaxy", galactic_params)

        assert model.object_type == "galaxy"
        assert model.physical_params["mass"] == 1e12
        assert model.physical_params["radius"] == 50.0
        assert model.physical_params["spiral_arms"] == 2
        assert model.topological_charge == 2

    def test_black_hole_model_initialization(self):
        """Test black hole model initialization."""
        black_hole_params = {
            "mass": 1e6,
            "schwarzschild_radius": 1.0,
            "spin": 0.5,
            "phase_amplitude": 2.0,
            "grid_size": 32,
            "domain_size": 5.0,
        }

        model = AstrophysicalObjectModel("black_hole", black_hole_params)

        assert model.object_type == "black_hole"
        assert model.physical_params["mass"] == 1e6
        assert model.physical_params["schwarzschild_radius"] == 1.0
        assert model.physical_params["spin"] == 0.5
        assert model.topological_charge == 0

    def test_star_phase_profile_creation(self):
        """Test star phase profile creation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        phase_profile = model.create_phase_profile()

        assert phase_profile.shape == (64, 64, 64, 8, 8, 8, 64)
        assert np.all(np.isfinite(phase_profile))
        assert np.all(np.abs(phase_profile) <= 2 * np.pi)

    def test_galaxy_phase_profile_creation(self):
        """Test galaxy phase profile creation."""
        galactic_params = {
            "mass": 1e12,
            "radius": 50.0,
            "spiral_arms": 2,
            "phase_amplitude": 0.5,
            "grid_size": 128,
            "domain_size": 100.0,
        }

        model = AstrophysicalObjectModel("galaxy", galactic_params)
        phase_profile = model.create_phase_profile()

        assert phase_profile.shape == (128, 128, 128, 8, 8, 8, 128)
        assert np.all(np.isfinite(phase_profile))
        assert np.all(np.abs(phase_profile) <= 2 * np.pi)

    def test_black_hole_phase_profile_creation(self):
        """Test black hole phase profile creation."""
        black_hole_params = {
            "mass": 1e6,
            "schwarzschild_radius": 1.0,
            "spin": 0.5,
            "phase_amplitude": 2.0,
            "grid_size": 32,
            "domain_size": 5.0,
        }

        model = AstrophysicalObjectModel("black_hole", black_hole_params)
        phase_profile = model.create_phase_profile()

        assert phase_profile.shape == (32, 32, 32, 8, 8, 8, 32)
        assert np.all(np.isfinite(phase_profile))
        assert np.all(np.abs(phase_profile) <= 2 * np.pi)

    def test_phase_properties_analysis(self):
        """Test phase properties analysis."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        phase_profile = model.create_phase_profile()
        properties = model.analyze_phase_properties(phase_profile)

        assert "topological_charge" in properties
        assert "phase_coherence" in properties
        assert "defect_density" in properties
        assert "correlation_length" in properties

        assert properties["topological_charge"] == 1
        assert 0 <= properties["phase_coherence"] <= 1
        assert properties["defect_density"] >= 0
        assert properties["correlation_length"] > 0

    def test_observable_properties_computation(self):
        """Test observable properties computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        phase_profile = model.create_phase_profile()
        observable_props = model.compute_observable_properties(phase_profile)

        assert "luminosity" in observable_props
        assert "effective_temperature" in observable_props
        assert "surface_gravity" in observable_props
        assert "radius" in observable_props

        assert observable_props["luminosity"] > 0
        assert observable_props["effective_temperature"] > 0
        assert observable_props["surface_gravity"] > 0
        assert observable_props["radius"] > 0

    def test_star_model_creation(self):
        """Test star model creation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        star_model = model.create_star_model()

        assert star_model is not None
        assert hasattr(star_model, "mass")
        assert hasattr(star_model, "radius")
        assert hasattr(star_model, "temperature")
        assert star_model.mass == 1.0
        assert star_model.radius == 1.0
        assert star_model.temperature == 5778.0

    def test_galaxy_model_creation(self):
        """Test galaxy model creation."""
        galactic_params = {
            "mass": 1e12,
            "radius": 50.0,
            "spiral_arms": 2,
            "phase_amplitude": 0.5,
            "grid_size": 128,
            "domain_size": 100.0,
        }

        model = AstrophysicalObjectModel("galaxy", galactic_params)
        galaxy_model = model.create_galaxy_model()

        assert galaxy_model is not None
        assert hasattr(galaxy_model, "mass")
        assert hasattr(galaxy_model, "radius")
        assert hasattr(galaxy_model, "spiral_arms")
        assert galaxy_model.mass == 1e12
        assert galaxy_model.radius == 50.0
        assert galaxy_model.spiral_arms == 2

    def test_black_hole_model_creation(self):
        """Test black hole model creation."""
        black_hole_params = {
            "mass": 1e6,
            "schwarzschild_radius": 1.0,
            "spin": 0.5,
            "phase_amplitude": 2.0,
            "grid_size": 32,
            "domain_size": 5.0,
        }

        model = AstrophysicalObjectModel("black_hole", black_hole_params)
        black_hole_model = model.create_black_hole_model()

        assert black_hole_model is not None
        assert hasattr(black_hole_model, "mass")
        assert hasattr(black_hole_model, "schwarzschild_radius")
        assert hasattr(black_hole_model, "spin")
        assert black_hole_model.mass == 1e6
        assert black_hole_model.schwarzschild_radius == 1.0
        assert black_hole_model.spin == 0.5

    def test_phase_correlation_length_computation(self):
        """Test phase correlation length computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        phase_profile = model.create_phase_profile()
        correlation_length = model.compute_phase_correlation_length(phase_profile)

        assert correlation_length > 0
        assert correlation_length <= model.physical_params["domain_size"]

    def test_effective_radius_computation(self):
        """Test effective radius computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        phase_profile = model.create_phase_profile()
        effective_radius = model.compute_effective_radius(phase_profile)

        assert effective_radius > 0
        assert effective_radius <= model.physical_params["domain_size"]

    def test_phase_energy_computation(self):
        """Test phase energy computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        phase_profile = model.create_phase_profile()
        energy = model.compute_phase_energy(phase_profile)

        assert energy >= 0
        assert np.isfinite(energy)

    def test_defect_density_computation(self):
        """Test defect density computation."""
        stellar_params = {
            "mass": 1.0,
            "radius": 1.0,
            "temperature": 5778.0,
            "phase_amplitude": 1.0,
            "grid_size": 64,
            "domain_size": 10.0,
        }

        model = AstrophysicalObjectModel("star", stellar_params)
        phase_profile = model.create_phase_profile()
        defect_density = model.compute_defect_density(phase_profile)

        assert defect_density >= 0
        assert np.isfinite(defect_density)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
