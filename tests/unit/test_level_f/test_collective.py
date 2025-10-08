"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for CollectiveExcitations class in Level F models.

This module contains comprehensive tests for the CollectiveExcitations
class, including tests for system excitation, response analysis,
and dispersion relations.

Physical Meaning:
    Tests verify that collective excitations are correctly
    applied to multi-particle systems, responses are properly
    analyzed, and dispersion relations are accurately computed.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.collective import CollectiveExcitations
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestCollectiveExcitations:
    """
    Test cases for CollectiveExcitations class.

    Physical Meaning:
        Tests verify the correct implementation of collective
        excitations including harmonic, impulse, and frequency
        sweep excitations.
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
        return MultiParticleSystem(domain, particles, interaction_range=5.0)

    @pytest.fixture
    def excitation_params(self):
        """Create test excitation parameters."""
        return {
            "frequency_range": [0.1, 10.0],
            "amplitude": 0.1,
            "type": "harmonic",
            "duration": 100.0,
        }

    @pytest.fixture
    def excitations(self, system, excitation_params):
        """Create test excitations."""
        return CollectiveExcitations(system, excitation_params)

    def test_initialization(self, system, excitation_params):
        """
        Test excitations initialization.

        Physical Meaning:
            Verifies that the excitations model is correctly
            initialized with the specified parameters.
        """
        excitations = CollectiveExcitations(system, excitation_params)

        assert excitations.system == system
        assert excitations.frequency_range == [0.1, 10.0]
        assert excitations.amplitude == 0.1
        assert excitations.excitation_type == "harmonic"
        assert excitations.duration == 100.0

    def test_harmonic_excitation(self, excitations):
        """
        Test harmonic excitation.

        Physical Meaning:
            Verifies that harmonic excitation is correctly
            applied to the system.
        """
        external_field = np.random.randn(64, 64, 64)

        response = excitations._harmonic_excitation(external_field)

        # Check that response is returned
        assert response is not None

        # Check that response has expected properties
        if hasattr(response, "shape"):
            assert len(response.shape) >= 1

    def test_impulse_excitation(self, excitations):
        """
        Test impulse excitation.

        Physical Meaning:
            Verifies that impulse excitation is correctly
            applied to the system.
        """
        external_field = np.random.randn(64, 64, 64)

        response = excitations._impulse_excitation(external_field)

        # Check that response is returned
        assert response is not None

        # Check that response has expected properties
        if hasattr(response, "shape"):
            assert len(response.shape) >= 1

    def test_frequency_sweep_excitation(self, excitations):
        """
        Test frequency sweep excitation.

        Physical Meaning:
            Verifies that frequency sweep excitation is
            correctly applied to the system.
        """
        external_field = np.random.randn(64, 64, 64)

        response = excitations._frequency_sweep_excitation(external_field)

        # Check that response is returned
        assert response is not None

        # Check that response has expected properties
        if hasattr(response, "shape"):
            assert len(response.shape) >= 1

    def test_system_excitation(self, excitations):
        """
        Test system excitation with different types.

        Physical Meaning:
            Verifies that the system is correctly excited
            with different types of external fields.
        """
        external_field = np.random.randn(64, 64, 64)

        # Test harmonic excitation
        excitations.excitation_type = "harmonic"
        response_harmonic = excitations.excite_system(external_field)
        assert response_harmonic is not None

        # Test impulse excitation
        excitations.excitation_type = "impulse"
        response_impulse = excitations.excite_system(external_field)
        assert response_impulse is not None

        # Test frequency sweep
        excitations.excitation_type = "sweep"
        response_sweep = excitations.excite_system(external_field)
        assert response_sweep is not None

    def test_response_analysis(self, excitations):
        """
        Test response analysis.

        Physical Meaning:
            Verifies that system response is correctly
            analyzed to extract collective mode properties.
        """
        # Create mock response data
        n_particles = len(excitations.system.particles)
        n_time = 1000
        response = np.random.randn(n_particles, n_time)

        analysis = excitations.analyze_response(response)

        # Check that analysis is returned
        assert "frequencies" in analysis
        assert "peaks" in analysis
        assert "transmission_analysis" in analysis
        assert "participation" in analysis
        assert "quality_factors" in analysis
        assert "spectrum" in analysis

        # Check that frequencies are returned
        assert isinstance(analysis["frequencies"], np.ndarray)

        # Check that peaks are identified
        assert "frequencies" in analysis["peaks"]
        assert "amplitudes" in analysis["peaks"]

        # Check that transmission analysis is performed
        assert "transmission_coefficients" in analysis["transmission_analysis"]
        assert "reflection_coefficients" in analysis["transmission_analysis"]

    def test_dispersion_relations(self, excitations):
        """
        Test dispersion relations computation.

        Physical Meaning:
            Verifies that dispersion relations are correctly
            computed for collective modes.
        """
        dispersion = excitations.compute_dispersion_relations()

        # Check that dispersion relations are returned
        assert "k_values" in dispersion
        assert "frequencies" in dispersion
        assert "group_velocities" in dispersion
        assert "phase_velocities" in dispersion
        assert "dispersion_fit" in dispersion

        # Check shapes
        k_values = dispersion["k_values"]
        frequencies = dispersion["frequencies"]
        group_velocities = dispersion["group_velocities"]
        phase_velocities = dispersion["phase_velocities"]

        assert len(frequencies) == len(k_values)
        assert len(group_velocities) == len(k_values)
        assert len(phase_velocities) == len(k_values)

        # Check that frequencies are non-negative
        assert np.all(frequencies >= 0)

        # Check that velocities are finite
        assert np.all(np.isfinite(group_velocities))
        assert np.all(np.isfinite(phase_velocities))

    def test_susceptibility_computation(self, excitations):
        """
        Test susceptibility computation.

        Physical Meaning:
            Verifies that the susceptibility function is
            correctly computed for collective excitations.
        """
        frequencies = np.linspace(0.1, 10.0, 100)

        susceptibility = excitations.compute_susceptibility(frequencies)

        # Check that susceptibility is returned
        assert isinstance(susceptibility, np.ndarray)
        assert len(susceptibility) == len(frequencies)

        # Check that susceptibility is complex
        assert np.iscomplexobj(susceptibility)

        # Check that susceptibility is finite
        assert np.all(np.isfinite(susceptibility))

    def test_spectral_peak_detection(self, excitations):
        """
        Test spectral peak detection.

        Physical Meaning:
            Verifies that spectral peaks are correctly
            identified in the response spectrum.
        """
        # Create mock spectrum with peaks
        n_freq = 1000
        frequencies = np.linspace(0, 10, n_freq)
        spectrum = np.random.randn(n_freq)

        # Add some peaks
        peak_indices = [100, 300, 700]
        for idx in peak_indices:
            spectrum[idx] += 10.0

        peaks = excitations._find_spectral_peaks(spectrum, frequencies)

        # Check that peaks are identified
        assert "indices" in peaks
        assert "frequencies" in peaks
        assert "amplitudes" in peaks

        # Check that peaks are found
        assert len(peaks["indices"]) > 0
        assert len(peaks["frequencies"]) > 0
        assert len(peaks["amplitudes"]) > 0

    def test_step_resonator_transmission_analysis(self, excitations):
        """
        Test step resonator transmission analysis.

        Physical Meaning:
            Verifies that transmission/reflection coefficients
            are correctly computed through step resonator boundaries.
        """
        # Create mock response with step-resonator transmission
        n_particles = len(excitations.system.particles)
        n_time = 1000
        t = np.linspace(0, 10, n_time)

        response = np.zeros((n_particles, n_time))
        for i in range(n_particles):
            signal = np.sin(2 * np.pi * t)
            # emulate step-resonator transmission
            signal[0] = 0.9 * signal[0] + 0.1 * signal[1]  # 90% transmission, 10% reflection
            signal[-1] = 0.9 * signal[-1] + 0.1 * signal[-2]
            response[i, :] = signal

        transmission_analysis = excitations._analyze_step_resonator_transmission(response)

        # Check that transmission analysis is returned
        assert "transmission_coefficients" in transmission_analysis
        assert "reflection_coefficients" in transmission_analysis
        assert "mean_transmission" in transmission_analysis
        assert "mean_reflection" in transmission_analysis

        # Check that transmission coefficients are computed
        assert len(transmission_analysis["transmission_coefficients"]) == n_particles
        assert len(transmission_analysis["reflection_coefficients"]) == n_particles

    def test_participation_ratios(self, excitations):
        """
        Test participation ratios computation.

        Physical Meaning:
            Verifies that participation ratios are correctly
            computed for collective modes.
        """
        # Create mock response
        n_particles = len(excitations.system.particles)
        n_time = 1000
        response = np.random.randn(n_particles, n_time)

        participation = excitations._compute_participation_ratios(response)

        # Check that participation ratios are returned
        assert isinstance(participation, np.ndarray)
        assert len(participation) == n_particles

        # Check that participation ratios are non-negative
        assert np.all(participation >= 0)

        # Check that participation ratios sum to 1
        assert np.isclose(np.sum(participation), 1.0, rtol=1e-10)

    def test_quality_factors(self, excitations):
        """
        Test quality factors computation.

        Physical Meaning:
            Verifies that quality factors are correctly
            computed for collective modes.
        """
        # Create mock peaks and damping analysis
        peaks = {"frequencies": [1.0, 2.0, 3.0], "amplitudes": [0.5, 0.8, 0.3]}

        transmission_analysis = {
            "transmission_coefficients": [0.9, 0.8, 0.85],
            "reflection_coefficients": [0.1, 0.2, 0.15],
        }

        quality_factors = excitations._compute_quality_factors(peaks, transmission_analysis)

        # Check that quality factors are returned
        assert isinstance(quality_factors, np.ndarray)
        assert len(quality_factors) == len(peaks["frequencies"])

        # Check that quality factors are finite
        assert np.all(np.isfinite(quality_factors))

    def test_dispersion_equation_solution(self, excitations):
        """
        Test dispersion equation solution.

        Physical Meaning:
            Verifies that the dispersion equation is
            correctly solved for given wave vectors.
        """
        k_values = [0.1, 0.5, 1.0, 2.0]

        for k in k_values:
            omega = excitations._solve_dispersion_equation(k)

            # Check that frequency is returned
            assert isinstance(omega, (int, float))
            assert omega >= 0  # Frequency should be non-negative
            assert np.isfinite(omega)

    def test_group_velocity_computation(self, excitations):
        """
        Test group velocity computation.

        Physical Meaning:
            Verifies that group velocities are correctly
            computed for collective modes.
        """
        k_values = [0.1, 0.5, 1.0, 2.0]

        for k in k_values:
            omega = excitations._solve_dispersion_equation(k)
            v_g = excitations._compute_group_velocity(k, omega)

            # Check that group velocity is returned
            assert isinstance(v_g, (int, float, np.integer, np.floating))
            assert np.isfinite(v_g)

    def test_dispersion_relation_fitting(self, excitations):
        """
        Test dispersion relation fitting.

        Physical Meaning:
            Verifies that dispersion relations are correctly
            fitted to computed data.
        """
        k_values = np.linspace(0.1, 2.0, 20)
        frequencies = np.sqrt(1.0 + 0.5 * k_values**2)  # Mock dispersion relation

        fit = excitations._fit_dispersion_relation(k_values, frequencies)

        # Check that fit is returned
        assert "omega_0" in fit
        assert "c" in fit
        assert "r_squared" in fit
        assert "coefficients" in fit

        # Check that parameters are finite
        assert np.isfinite(fit["omega_0"])
        assert np.isfinite(fit["c"])
        assert np.isfinite(fit["r_squared"])

        # Check that R² is reasonable
        assert 0 <= fit["r_squared"] <= 1

    def test_external_force_computation(self, excitations):
        """
        Test external force computation.

        Physical Meaning:
            Verifies that external forces are correctly
            computed for particles.
        """
        external_field = np.random.randn(64, 64, 64)
        excitation_amplitude = 0.5

        forces = excitations._compute_external_force(
            external_field, excitation_amplitude
        )

        # Check that forces are returned
        assert isinstance(forces, np.ndarray)
        assert len(forces) == len(excitations.system.particles)

        # Check that forces are finite
        assert np.all(np.isfinite(forces))

    def test_parameter_dependence(self, system):
        """
        Test dependence on excitation parameters.

        Physical Meaning:
            Verifies that the system response changes
            correctly with excitation parameters.
        """
        # Test different amplitudes
        amplitudes = [0.1, 0.5, 1.0]

        for amplitude in amplitudes:
            params = {
                "frequency_range": [0.1, 10.0],
                "amplitude": amplitude,
                "type": "harmonic",
                "duration": 100.0,
            }

            excitations = CollectiveExcitations(system, params)
            assert excitations.amplitude == amplitude

        # Test different frequency ranges
        frequency_ranges = [[0.1, 5.0], [0.5, 10.0], [1.0, 20.0]]

        for freq_range in frequency_ranges:
            params = {
                "frequency_range": freq_range,
                "amplitude": 0.1,
                "type": "harmonic",
                "duration": 100.0,
            }

            excitations = CollectiveExcitations(system, params)
            assert excitations.frequency_range == freq_range

    def test_different_excitation_types(self, system):
        """
        Test different excitation types.

        Physical Meaning:
            Verifies that different excitation types
            work correctly.
        """
        excitation_types = ["harmonic", "impulse", "sweep"]

        for exc_type in excitation_types:
            params = {
                "frequency_range": [0.1, 10.0],
                "amplitude": 0.1,
                "type": exc_type,
                "duration": 100.0,
            }

            excitations = CollectiveExcitations(system, params)
            assert excitations.excitation_type == exc_type

    def test_error_handling(self, system):
        """
        Test error handling for invalid parameters.

        Physical Meaning:
            Verifies that the system handles invalid
            parameters gracefully.
        """
        # Test invalid excitation type
        with pytest.raises(ValueError):
            params = {
                "frequency_range": [0.1, 10.0],
                "amplitude": 0.1,
                "type": "invalid_type",
                "duration": 100.0,
            }
            excitations = CollectiveExcitations(system, params)
            external_field = np.random.randn(64, 64, 64)
            excitations.excite_system(external_field)
