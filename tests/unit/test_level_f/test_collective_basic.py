"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic tests for CollectiveExcitations class in Level F models.

This module contains basic tests for the CollectiveExcitations
class, including initialization, basic excitation types,
and fundamental response analysis.

Physical Meaning:
    Tests verify that collective excitations are correctly
    applied to multi-particle systems and basic responses
    are properly analyzed.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.collective import CollectiveExcitations
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestCollectiveExcitationsBasic:
    """
    Basic test cases for CollectiveExcitations class.

    Physical Meaning:
        Tests verify the correct implementation of basic
        collective excitations including harmonic, impulse,
        and frequency sweep excitations.
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

    def test_initialization(self, system, excitation_params):
        """
        Test CollectiveExcitations initialization.

        Physical Meaning:
            Verifies that the CollectiveExcitations class is
            correctly initialized with proper parameters.
        """
        excitations = CollectiveExcitations(system, excitation_params)

        assert excitations.system == system
        assert excitations.excitation_params == excitation_params
        assert excitations.frequency_range == excitation_params["frequency_range"]
        assert excitations.amplitude == excitation_params["amplitude"]

    def test_harmonic_excitation(self, excitations):
        """
        Test harmonic excitation application.

        Physical Meaning:
            Verifies that harmonic excitations are correctly
            applied to the system with proper frequency and
            amplitude characteristics.
        """
        frequency = 1.0
        amplitude = 0.5
        duration = 3.0

        excitation = excitations.apply_harmonic_excitation(
            frequency, amplitude, duration
        )

        assert excitation is not None
        assert "frequency" in excitation
        assert "amplitude" in excitation
        assert "duration" in excitation
        assert excitation["frequency"] == frequency
        assert excitation["amplitude"] == amplitude
        assert excitation["duration"] == duration

    def test_impulse_excitation(self, excitations):
        """
        Test impulse excitation application.

        Physical Meaning:
            Verifies that impulse excitations are correctly
            applied to the system with proper timing and
            amplitude characteristics.
        """
        amplitude = 1.0
        duration = 0.1

        excitation = excitations.apply_impulse_excitation(amplitude, duration)

        assert excitation is not None
        assert "amplitude" in excitation
        assert "duration" in excitation
        assert excitation["amplitude"] == amplitude
        assert excitation["duration"] == duration

    def test_frequency_sweep_excitation(self, excitations):
        """
        Test frequency sweep excitation application.

        Physical Meaning:
            Verifies that frequency sweep excitations are
            correctly applied to the system with proper
            frequency range and sweep characteristics.
        """
        start_freq = 0.1
        end_freq = 2.0
        sweep_time = 5.0

        excitation = excitations.apply_frequency_sweep(start_freq, end_freq, sweep_time)

        assert excitation is not None
        assert "start_frequency" in excitation
        assert "end_frequency" in excitation
        assert "sweep_time" in excitation
        assert excitation["start_frequency"] == start_freq
        assert excitation["end_frequency"] == end_freq
        assert excitation["sweep_time"] == sweep_time

    def test_system_excitation(self, excitations):
        """
        Test system excitation with external field.

        Physical Meaning:
            Verifies that the system is correctly excited
            with external fields and the response is
            properly computed.
        """
        external_field = np.random.random((16, 16, 16, 8, 8, 8, 16))
        time_points = np.linspace(0, 10, 16)

        response = excitations.excite_system(external_field, time_points)

        assert response is not None
        assert "excitation_field" in response
        assert "response_field" in response
        assert "time_points" in response
        assert response["excitation_field"].shape == external_field.shape
        assert response["response_field"].shape == external_field.shape
        assert len(response["time_points"]) == len(time_points)

    def test_response_analysis(self, excitations):
        """
        Test response analysis functionality.

        Physical Meaning:
            Verifies that system responses are correctly
            analyzed to extract meaningful physical
            information about collective modes.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        analysis = excitations.analyze_response(response_data)

        assert analysis is not None
        assert "response_spectrum" in analysis
        assert "dominant_frequencies" in analysis
        assert "response_amplitude" in analysis
        assert "phase_shift" in analysis

        # Check that analysis results are reasonable
        assert len(analysis["dominant_frequencies"]) > 0
        assert analysis["response_amplitude"] > 0
        assert np.isfinite(analysis["phase_shift"])

    def test_dispersion_relations(self, excitations):
        """
        Test dispersion relation computation.

        Physical Meaning:
            Verifies that dispersion relations are correctly
            computed from the system response, revealing
            the relationship between frequency and wave vector.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        dispersion = excitations.compute_dispersion_relations(response_data)

        assert dispersion is not None
        assert "frequencies" in dispersion
        assert "wave_vectors" in dispersion
        assert "dispersion_relation" in dispersion

        # Check that dispersion relation is reasonable
        assert len(dispersion["frequencies"]) > 0
        assert len(dispersion["wave_vectors"]) > 0
        assert dispersion["dispersion_relation"] is not None

    def test_susceptibility_computation(self, excitations):
        """
        Test susceptibility computation.

        Physical Meaning:
            Verifies that the susceptibility function is
            correctly computed from the system response,
            providing information about the system's
            linear response characteristics.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        susceptibility = excitations.compute_susceptibility(response_data)

        assert susceptibility is not None
        assert "frequencies" in susceptibility
        assert "susceptibility" in susceptibility
        assert "phase" in susceptibility

        # Check that susceptibility is reasonable
        assert len(susceptibility["frequencies"]) > 0
        assert susceptibility["susceptibility"] is not None
        assert susceptibility["phase"] is not None

    def test_spectral_peak_detection(self, excitations):
        """
        Test spectral peak detection.

        Physical Meaning:
            Verifies that spectral peaks are correctly
            detected in the system response, identifying
            resonant frequencies and collective modes.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        peaks = excitations.detect_spectral_peaks(response_data)

        assert peaks is not None
        assert "peak_frequencies" in peaks
        assert "peak_amplitudes" in peaks
        assert "peak_quality_factors" in peaks

        # Check that peaks are reasonable
        assert len(peaks["peak_frequencies"]) >= 0
        assert len(peaks["peak_amplitudes"]) == len(peaks["peak_frequencies"])
        assert len(peaks["peak_quality_factors"]) == len(peaks["peak_frequencies"])

    def test_step_resonator_transmission_analysis(self, excitations):
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

    def test_participation_ratios(self, excitations):
        """
        Test participation ratio computation.

        Physical Meaning:
            Verifies that participation ratios are correctly
            computed, providing information about the
            localization of collective modes.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        participation = excitations.compute_participation_ratios(response_data)

        assert participation is not None
        assert "participation_ratios" in participation
        assert "mode_indices" in participation

        # Check that participation ratios are reasonable
        assert len(participation["participation_ratios"]) > 0
        assert len(participation["mode_indices"]) == len(
            participation["participation_ratios"]
        )
        assert all(0 <= ratio <= 1 for ratio in participation["participation_ratios"])

    def test_quality_factors(self, excitations):
        """
        Test quality factor computation.

        Physical Meaning:
            Verifies that quality factors are correctly
            computed for collective modes, providing
            information about mode damping and lifetime.
        """
        # Create mock response data
        response_data = {
            "excitation_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "response_field": np.random.random((16, 16, 16, 8, 8, 8, 16)),
            "time_points": np.linspace(0, 10, 16),
        }

        quality_factors = excitations.compute_quality_factors(response_data)

        assert quality_factors is not None
        assert "quality_factors" in quality_factors
        assert "mode_frequencies" in quality_factors

        # Check that quality factors are reasonable
        assert len(quality_factors["quality_factors"]) > 0
        assert len(quality_factors["mode_frequencies"]) == len(
            quality_factors["quality_factors"]
        )
        assert all(qf > 0 for qf in quality_factors["quality_factors"])


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
