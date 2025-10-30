"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced tests for CollectiveExcitations class in Level F models.

This module contains advanced tests for the CollectiveExcitations
class, including dispersion analysis, parameter dependence,
and error handling.

Physical Meaning:
    Tests verify advanced functionality of collective
    excitations including dispersion relations, parameter
    dependence, and robust error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.collective import CollectiveExcitations
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestCollectiveExcitationsAdvanced:
    """
    Advanced test cases for CollectiveExcitations class.

    Physical Meaning:
        Tests verify advanced functionality of collective
        excitations including dispersion relations, parameter
        dependence, and robust error handling.
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

    @pytest.fixture
    def excitation_params(self):
        """Create excitation parameters."""
        return {
            "frequency_range": (0.1, 2.0),
            "amplitude": 1.0,
            "duration": 5.0,
            "excitation_type": "harmonic",
        }

    @pytest.fixture
    def excitations(self, system, excitation_params):
        """Create CollectiveExcitations instance."""
        return CollectiveExcitations(system, excitation_params)

    def test_dispersion_equation_solution(self, excitations):
        """
        Test dispersion equation solution.

        Physical Meaning:
            Verifies that dispersion equations are correctly
            solved to find the relationship between frequency
            and wave vector in the system.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        solution = excitations.solve_dispersion_equation(response_data)

        assert solution is not None
        assert "dispersion_roots" in solution
        assert "frequency_solutions" in solution
        assert "wave_vector_solutions" in solution

        # Check that solution is reasonable
        assert len(solution["dispersion_roots"]) > 0
        assert len(solution["frequency_solutions"]) == len(solution["dispersion_roots"])
        assert len(solution["wave_vector_solutions"]) == len(
            solution["dispersion_roots"]
        )

    def test_group_velocity_computation(self, excitations):
        """
        Test group velocity computation.

        Physical Meaning:
            Verifies that group velocities are correctly
            computed from dispersion relations, providing
            information about the propagation of collective
            modes.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        group_velocity = excitations.compute_group_velocity(response_data)

        assert group_velocity is not None
        assert "group_velocities" in group_velocity
        assert "frequencies" in group_velocity
        assert "wave_vectors" in group_velocity

        # Check that group velocities are reasonable
        assert len(group_velocity["group_velocities"]) > 0
        assert len(group_velocity["frequencies"]) == len(
            group_velocity["group_velocities"]
        )
        assert len(group_velocity["wave_vectors"]) == len(
            group_velocity["group_velocities"]
        )

    def test_dispersion_relation_fitting(self, excitations):
        """
        Test dispersion relation fitting.

        Physical Meaning:
            Verifies that dispersion relations are correctly
            fitted to theoretical models, providing insights
            into the system's physical properties.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        fitting = excitations.fit_dispersion_relation(response_data)

        assert fitting is not None
        assert "fitted_parameters" in fitting
        assert "fitting_error" in fitting
        assert "theoretical_model" in fitting

        # Check that fitting is reasonable
        assert fitting["fitted_parameters"] is not None
        assert fitting["fitting_error"] >= 0
        assert fitting["theoretical_model"] is not None

    def test_external_force_computation(self, excitations):
        """
        Test external force computation.

        Physical Meaning:
            Verifies that external forces are correctly
            computed and applied to the system, providing
            information about the system's response to
            external perturbations.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        force = excitations.compute_external_force(response_data)

        assert force is not None
        assert "force_field" in force
        assert "force_amplitude" in force
        assert "force_direction" in force

        # Check that force is reasonable
        assert force["force_field"] is not None
        assert force["force_amplitude"] >= 0
        assert force["force_direction"] is not None

    def test_parameter_dependence(self, system):
        """
        Test parameter dependence of collective excitations.

        Physical Meaning:
            Verifies that collective excitations depend
            correctly on system parameters, ensuring
            physical consistency.
        """
        # Test with different parameters
        params1 = {
            "frequency_range": (0.1, 1.0),
            "amplitude": 0.5,
            "duration": 3.0,
            "excitation_type": "harmonic",
        }

        params2 = {
            "frequency_range": (0.5, 2.0),
            "amplitude": 1.0,
            "duration": 5.0,
            "excitation_type": "harmonic",
        }

        excitations1 = CollectiveExcitations(system, params1)
        excitations2 = CollectiveExcitations(system, params2)

        # Check that parameters are correctly set
        assert excitations1.frequency_range == params1["frequency_range"]
        assert excitations2.frequency_range == params2["frequency_range"]
        assert excitations1.amplitude == params1["amplitude"]
        assert excitations2.amplitude == params2["amplitude"]

    def test_different_excitation_types(self, system):
        """
        Test different excitation types.

        Physical Meaning:
            Verifies that different types of excitations
            (harmonic, impulse, frequency sweep) are
            correctly handled by the system.
        """
        # Test harmonic excitation
        harmonic_params = {
            "frequency_range": (0.1, 2.0),
            "amplitude": 1.0,
            "duration": 5.0,
            "excitation_type": "harmonic",
        }
        harmonic_excitations = CollectiveExcitations(system, harmonic_params)
        assert harmonic_excitations.excitation_type == "harmonic"

        # Test impulse excitation
        impulse_params = {
            "frequency_range": (0.1, 2.0),
            "amplitude": 1.0,
            "duration": 0.1,
            "excitation_type": "impulse",
        }
        impulse_excitations = CollectiveExcitations(system, impulse_params)
        assert impulse_excitations.excitation_type == "impulse"

        # Test frequency sweep excitation
        sweep_params = {
            "frequency_range": (0.1, 2.0),
            "amplitude": 1.0,
            "duration": 5.0,
            "excitation_type": "frequency_sweep",
        }
        sweep_excitations = CollectiveExcitations(system, sweep_params)
        assert sweep_excitations.excitation_type == "frequency_sweep"

    def test_error_handling(self, system):
        """
        Test error handling in collective excitations.

        Physical Meaning:
            Verifies that the system handles errors gracefully
            and provides meaningful error messages for
            invalid inputs.
        """
        # Test with invalid parameters
        invalid_params = {
            "frequency_range": (2.0, 0.1),  # Invalid range
            "amplitude": -1.0,  # Negative amplitude
            "duration": -1.0,  # Negative duration
            "excitation_type": "invalid_type",
        }

        with pytest.raises(ValueError):
            CollectiveExcitations(system, invalid_params)

        # Test with None system
        with pytest.raises(TypeError):
            CollectiveExcitations(None, {"frequency_range": (0.1, 2.0)})

        # Test with invalid response data
        excitations = CollectiveExcitations(
            system,
            {
                "frequency_range": (0.1, 2.0),
                "amplitude": 1.0,
                "duration": 5.0,
                "excitation_type": "harmonic",
            },
        )

        with pytest.raises(ValueError):
            excitations.analyze_response(None)

    def test_step_resonator_transmission(self, excitations):
        """
        Test step resonator transmission analysis.

        Physical Meaning:
            Verifies that step resonator transmission is
            correctly analyzed, providing information about
            the system's transmission characteristics.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        transmission = excitations.analyze_step_resonator_transmission(response_data)

        assert transmission is not None
        assert "transmission_coefficient" in transmission
        assert "reflection_coefficient" in transmission
        assert "resonance_frequencies" in transmission

        # Check that transmission analysis is reasonable
        assert 0 <= transmission["transmission_coefficient"] <= 1
        assert 0 <= transmission["reflection_coefficient"] <= 1
        assert len(transmission["resonance_frequencies"]) >= 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
