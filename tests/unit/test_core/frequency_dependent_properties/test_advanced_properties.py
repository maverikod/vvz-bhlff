"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced frequency-dependent properties tests.

This module contains advanced tests for frequency-dependent properties
including complex scenarios and edge cases.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.constants.frequency_dependent_properties import (
    FrequencyDependentProperties,
)


class TestAdvancedProperties:
    """Advanced tests for frequency-dependent properties."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for constants testing."""
        return Domain(L=1.0, N=32, dimensions=7, N_phi=16, N_t=64, T=1.0)

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for testing."""
        config = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
                "carrier_frequency": 1.85e43,
            },
            "basic_material": {"mu": 1.0, "beta": 1.5, "lambda_param": 0.1, "nu": 1.0},
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def freq_props(self, domain_7d, bvp_constants):
        """Create frequency-dependent properties for testing."""
        return FrequencyDependentProperties(domain_7d, bvp_constants)

    def test_frequency_array_operations(self, freq_props):
        """Test operations on frequency arrays."""
        # Test frequency array properties
        frequencies = freq_props.frequencies
        omega = freq_props.omega

        # Check that omega = 2*pi*f
        expected_omega = 2 * np.pi * frequencies
        assert np.allclose(
            omega, expected_omega, rtol=1e-10
        ), "Angular frequency should be 2*pi times frequency"

        # Check that frequencies are properly spaced
        freq_spacing = np.diff(frequencies)
        assert np.allclose(
            freq_spacing, freq_spacing[0], rtol=1e-10
        ), "Frequencies should be evenly spaced"

    def test_susceptibility_frequency_dependence(self, freq_props):
        """Test frequency dependence of susceptibility."""
        frequencies = np.linspace(0.1, 10.0, 50)
        susceptibilities = [freq_props.compute_susceptibility(f) for f in frequencies]

        # Check that susceptibility varies with frequency
        chi_real = [np.real(chi) for chi in susceptibilities]
        chi_imag = [np.imag(chi) for chi in susceptibilities]

        # Real and imaginary parts should vary
        assert (
            np.std(chi_real) > 1e-10
        ), "Real part of susceptibility should vary with frequency"
        assert (
            np.std(chi_imag) > 1e-10
        ), "Imaginary part of susceptibility should vary with frequency"

    def test_dispersion_relation_consistency(self, freq_props):
        """Test consistency of dispersion relation."""
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            k = freq_props.compute_dispersion_relation(freq)
            v_phase = freq_props.compute_phase_velocity(freq)

            # Check consistency: v_phase = omega / k
            omega = 2 * np.pi * freq
            expected_v_phase = omega / k

            assert np.isclose(
                v_phase, expected_v_phase, rtol=1e-10
            ), f"Phase velocity should be consistent with dispersion relation at frequency {freq}"

    def test_velocity_relationships(self, freq_props):
        """Test relationships between different velocities."""
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            v_phase = freq_props.compute_phase_velocity(freq)
            v_group = freq_props.compute_group_velocity(freq)

            # Both velocities should be positive
            assert v_phase > 0, f"Phase velocity should be positive at frequency {freq}"
            assert v_group > 0, f"Group velocity should be positive at frequency {freq}"

            # Velocities should be finite
            assert np.isfinite(
                v_phase
            ), f"Phase velocity should be finite at frequency {freq}"
            assert np.isfinite(
                v_group
            ), f"Group velocity should be finite at frequency {freq}"

    def test_refractive_index_properties(self, freq_props):
        """Test properties of refractive index."""
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            n = freq_props.compute_refractive_index(freq)

            # Real part should be positive
            assert (
                np.real(n) > 0
            ), f"Real part of refractive index should be positive at frequency {freq}"

            # Imaginary part should be non-negative (for passive media)
            assert (
                np.imag(n) >= 0
            ), f"Imaginary part of refractive index should be non-negative at frequency {freq}"

            # Refractive index should be finite
            assert np.isfinite(
                n
            ), f"Refractive index should be finite at frequency {freq}"

    def test_absorption_properties(self, freq_props):
        """Test properties of absorption coefficient."""
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            alpha = freq_props.compute_absorption_coefficient(freq)

            # Absorption should be non-negative
            assert (
                alpha >= 0
            ), f"Absorption coefficient should be non-negative at frequency {freq}"

            # Absorption should be finite
            assert np.isfinite(
                alpha
            ), f"Absorption coefficient should be finite at frequency {freq}"

    def test_high_frequency_behavior(self, freq_props):
        """Test behavior at high frequencies."""
        high_frequencies = [100.0, 1000.0, 10000.0]

        for freq in high_frequencies:
            # All properties should remain finite at high frequencies
            chi = freq_props.compute_susceptibility(freq)
            k = freq_props.compute_dispersion_relation(freq)
            v_phase = freq_props.compute_phase_velocity(freq)
            v_group = freq_props.compute_group_velocity(freq)
            alpha = freq_props.compute_absorption_coefficient(freq)
            n = freq_props.compute_refractive_index(freq)

            assert np.isfinite(
                chi
            ), f"Susceptibility should be finite at high frequency {freq}"
            assert np.isfinite(
                k
            ), f"Wave number should be finite at high frequency {freq}"
            assert np.isfinite(
                v_phase
            ), f"Phase velocity should be finite at high frequency {freq}"
            assert np.isfinite(
                v_group
            ), f"Group velocity should be finite at high frequency {freq}"
            assert np.isfinite(
                alpha
            ), f"Absorption coefficient should be finite at high frequency {freq}"
            assert np.isfinite(
                n
            ), f"Refractive index should be finite at high frequency {freq}"

    def test_low_frequency_behavior(self, freq_props):
        """Test behavior at low frequencies."""
        low_frequencies = [1e-6, 1e-5, 1e-4]

        for freq in low_frequencies:
            # All properties should remain finite at low frequencies
            chi = freq_props.compute_susceptibility(freq)
            k = freq_props.compute_dispersion_relation(freq)
            v_phase = freq_props.compute_phase_velocity(freq)
            v_group = freq_props.compute_group_velocity(freq)
            alpha = freq_props.compute_absorption_coefficient(freq)
            n = freq_props.compute_refractive_index(freq)

            assert np.isfinite(
                chi
            ), f"Susceptibility should be finite at low frequency {freq}"
            assert np.isfinite(
                k
            ), f"Wave number should be finite at low frequency {freq}"
            assert np.isfinite(
                v_phase
            ), f"Phase velocity should be finite at low frequency {freq}"
            assert np.isfinite(
                v_group
            ), f"Group velocity should be finite at low frequency {freq}"
            assert np.isfinite(
                alpha
            ), f"Absorption coefficient should be finite at low frequency {freq}"
            assert np.isfinite(
                n
            ), f"Refractive index should be finite at low frequency {freq}"

    def test_parameter_sensitivity(self, domain_7d):
        """Test sensitivity to parameter changes."""
        # Test with different parameter sets
        configs = [
            {
                "envelope_equation": {
                    "kappa_0": 1.0,
                    "kappa_2": 0.1,
                    "chi_prime": 1.0,
                    "chi_double_prime_0": 0.01,
                    "k0_squared": 4.0,
                    "carrier_frequency": 1.85e43,
                },
                "basic_material": {
                    "mu": 1.0,
                    "beta": 1.5,
                    "lambda_param": 0.1,
                    "nu": 1.0,
                },
            },
            {
                "envelope_equation": {
                    "kappa_0": 2.0,  # Different kappa_0
                    "kappa_2": 0.1,
                    "chi_prime": 1.0,
                    "chi_double_prime_0": 0.01,
                    "k0_squared": 4.0,
                    "carrier_frequency": 1.85e43,
                },
                "basic_material": {
                    "mu": 1.0,
                    "beta": 1.5,
                    "lambda_param": 0.1,
                    "nu": 1.0,
                },
            },
        ]

        freq_props_list = []
        for config in configs:
            bvp_constants = BVPConstantsAdvanced(config)
            freq_props = FrequencyDependentProperties(domain_7d, bvp_constants)
            freq_props_list.append(freq_props)

        # Test that different parameters produce different results
        test_freq = 1.0
        chi1 = freq_props_list[0].compute_susceptibility(test_freq)
        chi2 = freq_props_list[1].compute_susceptibility(test_freq)

        # Results should be different for different parameters
        assert not np.isclose(
            chi1, chi2, rtol=1e-10
        ), "Susceptibility should depend on parameters"
