"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for different nonlinear types.

This module contains tests for different types of nonlinearity
in the NonlinearEffects class in Level F models.

Physical Meaning:
    Tests verify that different types of nonlinearity are correctly
    implemented and tested.

Example:
    >>> pytest tests/unit/test_level_f/nonlinear/test_nonlinear_types.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.nonlinear import NonlinearEffects
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestNonlinearTypes:
    """
    Test cases for different nonlinear types.

    Physical Meaning:
        Tests verify the correct implementation of different
        types of nonlinearity including cubic, quartic, and
        sine-Gordon nonlinearities.
    """

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=20.0, N=16, N_phi=8, N_t=16, T=10.0, dimensions=7)

    @pytest.fixture
    def particles(self):
        """Create test particles."""
        return [
            Particle(position=np.array([5.0, 10.0, 10.0]), charge=1, phase=0.0),
            Particle(position=np.array([15.0, 10.0, 10.0]), charge=-1, phase=np.pi),
        ]

    @pytest.fixture
    def system(self, domain, particles):
        """Create test system."""
        return MultiParticleSystem(domain, particles)

    def test_different_nonlinear_types(self, system):
        """
        Test different nonlinear types.

        Physical Meaning:
            Tests that different types of nonlinearity are correctly
            implemented and can be distinguished.
        """
        # Test cubic nonlinearity
        cubic_params = {
            "cubic_coefficient": 0.1,
            "quartic_coefficient": 0.0,
            "sine_gordon_amplitude": 0.0,
            "nonlinear_threshold": 0.5,
        }
        cubic_nonlinear = NonlinearEffects(system, cubic_params)

        # Test quartic nonlinearity
        quartic_params = {
            "cubic_coefficient": 0.0,
            "quartic_coefficient": 0.01,
            "sine_gordon_amplitude": 0.0,
            "nonlinear_threshold": 0.5,
        }
        quartic_nonlinear = NonlinearEffects(system, quartic_params)

        # Test sine-Gordon nonlinearity
        sine_gordon_params = {
            "cubic_coefficient": 0.0,
            "quartic_coefficient": 0.0,
            "sine_gordon_amplitude": 1.0,
            "nonlinear_threshold": 0.5,
        }
        sine_gordon_nonlinear = NonlinearEffects(system, sine_gordon_params)

        # Verify different types
        assert cubic_nonlinear.cubic_coefficient == 0.1
        assert cubic_nonlinear.quartic_coefficient == 0.0
        assert cubic_nonlinear.sine_gordon_amplitude == 0.0

        assert quartic_nonlinear.cubic_coefficient == 0.0
        assert quartic_nonlinear.quartic_coefficient == 0.01
        assert quartic_nonlinear.sine_gordon_amplitude == 0.0

        assert sine_gordon_nonlinear.cubic_coefficient == 0.0
        assert sine_gordon_nonlinear.quartic_coefficient == 0.0
        assert sine_gordon_nonlinear.sine_gordon_amplitude == 1.0

    def test_parameter_dependence(self, system):
        """
        Test parameter dependence.

        Physical Meaning:
            Tests that nonlinear effects correctly depend on
            the parameters of the system.
        """
        # Test with different parameters
        params1 = {
            "cubic_coefficient": 0.1,
            "quartic_coefficient": 0.01,
            "sine_gordon_amplitude": 1.0,
            "nonlinear_threshold": 0.5,
        }
        nonlinear1 = NonlinearEffects(system, params1)

        params2 = {
            "cubic_coefficient": 0.2,
            "quartic_coefficient": 0.02,
            "sine_gordon_amplitude": 2.0,
            "nonlinear_threshold": 1.0,
        }
        nonlinear2 = NonlinearEffects(system, params2)

        # Verify parameter dependence
        assert nonlinear1.cubic_coefficient != nonlinear2.cubic_coefficient
        assert nonlinear1.quartic_coefficient != nonlinear2.quartic_coefficient
        assert nonlinear1.sine_gordon_amplitude != nonlinear2.sine_gordon_amplitude
        assert nonlinear1.nonlinear_threshold != nonlinear2.nonlinear_threshold

    def test_error_handling(self, system):
        """
        Test error handling.

        Physical Meaning:
            Tests that error handling is correctly implemented
            for invalid parameters and edge cases.
        """
        # Test with invalid parameters
        invalid_params = {
            "cubic_coefficient": -0.1,  # Negative coefficient
            "quartic_coefficient": 0.01,
            "sine_gordon_amplitude": 1.0,
            "nonlinear_threshold": 0.5,
        }

        with pytest.raises(ValueError):
            NonlinearEffects(system, invalid_params)

        # Test with missing parameters
        incomplete_params = {
            "cubic_coefficient": 0.1,
            "quartic_coefficient": 0.01,
            # Missing sine_gordon_amplitude
            "nonlinear_threshold": 0.5,
        }

        with pytest.raises(KeyError):
            NonlinearEffects(system, incomplete_params)

    def test_linear_stability_analysis(self, nonlinear):
        """
        Test linear stability analysis.

        Physical Meaning:
            Tests that linear stability analysis is correctly
            performed for the nonlinear system.
        """
        # Test linear stability analysis
        stability = nonlinear.analyze_linear_stability()

        assert stability is not None
        assert "eigenvalues" in stability
        assert "eigenvectors" in stability
        assert "stability_matrix" in stability

        # Check eigenvalues
        eigenvalues = stability["eigenvalues"]
        assert isinstance(eigenvalues, np.ndarray)
        assert len(eigenvalues) > 0

        # Check eigenvectors
        eigenvectors = stability["eigenvectors"]
        assert isinstance(eigenvectors, np.ndarray)
        assert eigenvectors.shape[0] == eigenvectors.shape[1]

        # Check stability matrix
        stability_matrix = stability["stability_matrix"]
        assert isinstance(stability_matrix, np.ndarray)
        assert stability_matrix.shape[0] == stability_matrix.shape[1]

    def test_boundary_energy_exchange(self, nonlinear):
        """
        Test boundary energy exchange.

        Physical Meaning:
            Tests that boundary energy exchange is correctly
            computed for the nonlinear system.
        """
        # Test boundary energy exchange
        energy_exchange = nonlinear.compute_boundary_energy_exchange()

        assert energy_exchange is not None
        assert "incoming_energy" in energy_exchange
        assert "outgoing_energy" in energy_exchange
        assert "net_energy_flux" in energy_exchange

        # Check energy values
        incoming_energy = energy_exchange["incoming_energy"]
        assert isinstance(incoming_energy, float)
        assert incoming_energy >= 0

        outgoing_energy = energy_exchange["outgoing_energy"]
        assert isinstance(outgoing_energy, float)
        assert outgoing_energy >= 0

        net_energy_flux = energy_exchange["net_energy_flux"]
        assert isinstance(net_energy_flux, float)

    def test_equations_of_motion_without_damping(self, nonlinear):
        """
        Test equations of motion without damping.

        Physical Meaning:
            Tests that equations of motion are correctly
            computed without damping terms.
        """
        # Test equations of motion without damping
        equations = nonlinear.compute_equations_of_motion_without_damping()

        assert equations is not None
        assert "acceleration_terms" in equations
        assert "nonlinear_terms" in equations
        assert "linear_terms" in equations

        # Check acceleration terms
        acceleration_terms = equations["acceleration_terms"]
        assert isinstance(acceleration_terms, np.ndarray)
        assert acceleration_terms.shape == (16, 16, 16)

        # Check nonlinear terms
        nonlinear_terms = equations["nonlinear_terms"]
        assert isinstance(nonlinear_terms, np.ndarray)
        assert nonlinear_terms.shape == (16, 16, 16)

        # Check linear terms
        linear_terms = equations["linear_terms"]
        assert isinstance(linear_terms, np.ndarray)
        assert linear_terms.shape == (16, 16, 16)
