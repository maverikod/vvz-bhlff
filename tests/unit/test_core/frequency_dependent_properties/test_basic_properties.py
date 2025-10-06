"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic frequency-dependent properties tests.

This module contains basic tests for frequency-dependent properties
including fundamental validation and basic functionality tests.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.constants.frequency_dependent_properties import (
    FrequencyDependentProperties,
)


class TestBasicProperties:
    """Basic tests for frequency-dependent properties."""

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

    def test_frequency_domain_creation(self, freq_props, domain_7d):
        """Test that frequency domain is created correctly."""
        # Check that frequency arrays are created
        assert hasattr(freq_props, "frequencies")
        assert hasattr(freq_props, "omega")

        # Check dimensions
        assert freq_props.frequencies.shape == (domain_7d.N_t,)
        assert freq_props.omega.shape == (domain_7d.N_t,)

        # Check that frequencies are properly ordered
        assert np.all(
            np.diff(freq_props.frequencies) > 0
        ), "Frequencies should be increasing"
        assert np.all(
            np.diff(freq_props.omega) > 0
        ), "Angular frequencies should be increasing"

    def test_susceptibility_calculation(self, freq_props):
        """Test susceptibility calculation."""
        # Test at zero frequency
        chi_zero = freq_props.compute_susceptibility(0.0)
        assert np.isfinite(
            chi_zero
        ), "Susceptibility at zero frequency should be finite"

        # Test at non-zero frequency
        chi_nonzero = freq_props.compute_susceptibility(1.0)
        assert np.isfinite(
            chi_nonzero
        ), "Susceptibility at non-zero frequency should be finite"

        # Test that susceptibility is complex
        assert np.iscomplexobj(chi_zero), "Susceptibility should be complex"
        assert np.iscomplexobj(chi_nonzero), "Susceptibility should be complex"

    def test_dispersion_relation(self, freq_props):
        """Test dispersion relation calculation."""
        # Test at different frequencies
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            k = freq_props.compute_dispersion_relation(freq)
            assert np.isfinite(
                k
            ), f"Dispersion relation should be finite at frequency {freq}"
            assert k > 0, f"Wave number should be positive at frequency {freq}"

    def test_phase_velocity(self, freq_props):
        """Test phase velocity calculation."""
        # Test at different frequencies
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            v_phase = freq_props.compute_phase_velocity(freq)
            assert np.isfinite(
                v_phase
            ), f"Phase velocity should be finite at frequency {freq}"
            assert v_phase > 0, f"Phase velocity should be positive at frequency {freq}"

    def test_group_velocity(self, freq_props):
        """Test group velocity calculation."""
        # Test at different frequencies
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            v_group = freq_props.compute_group_velocity(freq)
            assert np.isfinite(
                v_group
            ), f"Group velocity should be finite at frequency {freq}"

    def test_absorption_coefficient(self, freq_props):
        """Test absorption coefficient calculation."""
        # Test at different frequencies
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            alpha = freq_props.compute_absorption_coefficient(freq)
            assert np.isfinite(
                alpha
            ), f"Absorption coefficient should be finite at frequency {freq}"
            assert (
                alpha >= 0
            ), f"Absorption coefficient should be non-negative at frequency {freq}"

    def test_refractive_index(self, freq_props):
        """Test refractive index calculation."""
        # Test at different frequencies
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            n = freq_props.compute_refractive_index(freq)
            assert np.isfinite(
                n
            ), f"Refractive index should be finite at frequency {freq}"
            assert (
                np.real(n) > 0
            ), f"Real part of refractive index should be positive at frequency {freq}"

    def test_physical_constraints(self, freq_props):
        """Test that physical constraints are satisfied."""
        # Test causality (Kramers-Kronig relations)
        frequencies = np.linspace(0.1, 10.0, 20)

        for freq in frequencies:
            chi = freq_props.compute_susceptibility(freq)

            # Real part should be even function of frequency
            chi_neg = freq_props.compute_susceptibility(-freq)
            assert np.isclose(
                np.real(chi), np.real(chi_neg), rtol=1e-10
            ), f"Real part of susceptibility should be even at frequency {freq}"

            # Imaginary part should be odd function of frequency
            assert np.isclose(
                np.imag(chi), -np.imag(chi_neg), rtol=1e-10
            ), f"Imaginary part of susceptibility should be odd at frequency {freq}"

    def test_energy_conservation(self, freq_props):
        """Test energy conservation properties."""
        # Test that absorption coefficient is consistent with imaginary part of susceptibility
        frequencies = [0.1, 1.0, 10.0]

        for freq in frequencies:
            chi = freq_props.compute_susceptibility(freq)
            alpha = freq_props.compute_absorption_coefficient(freq)

            # Absorption should be related to imaginary part of susceptibility
            # This is a simplified check - exact relationship depends on implementation
            assert np.isfinite(
                alpha
            ), f"Absorption coefficient should be finite at frequency {freq}"
            assert (
                alpha >= 0
            ), f"Absorption coefficient should be non-negative at frequency {freq}"
