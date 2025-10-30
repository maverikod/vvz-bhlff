"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic nonlinear coefficients tests.

This module contains basic tests for nonlinear coefficients
including fundamental validation and basic functionality tests.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.constants.nonlinear_coefficients import NonlinearCoefficients


class TestBasicCoefficients:
    """Basic tests for nonlinear coefficients."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for constants testing."""
        return Domain7DBVP(L_spatial=1.0, N_spatial=8, N_phase=4, T=1.0, N_t=8)

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
            "material_properties": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
                "nu": 1.0,
            },
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def nonlinear_coeffs(self, bvp_constants):
        """Create nonlinear coefficients for testing."""
        return NonlinearCoefficients(bvp_constants)

    def test_nonlinear_coefficients_creation(self, nonlinear_coeffs, domain_7d):
        """Test that nonlinear coefficients are created correctly."""
        # Check that coefficients are created
        assert hasattr(nonlinear_coeffs, "kappa_0")
        assert hasattr(nonlinear_coeffs, "kappa_2")
        assert hasattr(nonlinear_coeffs, "chi_prime")
        assert hasattr(nonlinear_coeffs, "chi_double_prime_0")

        # Check that coefficients are finite
        assert np.isfinite(nonlinear_coeffs.kappa_0), "kappa_0 should be finite"
        assert np.isfinite(nonlinear_coeffs.kappa_2), "kappa_2 should be finite"
        assert np.isfinite(nonlinear_coeffs.chi_prime), "chi_prime should be finite"
        assert np.isfinite(
            nonlinear_coeffs.chi_double_prime_0
        ), "chi_double_prime_0 should be finite"

    def test_kappa_coefficients(self, nonlinear_coeffs):
        """Test kappa coefficients properties."""
        # kappa_0 should be positive (stiffness coefficient)
        assert nonlinear_coeffs.kappa_0 > 0, "kappa_0 should be positive"

        # kappa_2 should be non-negative (nonlinear stiffness)
        assert nonlinear_coeffs.kappa_2 >= 0, "kappa_2 should be non-negative"

        # Both should be finite
        assert np.isfinite(nonlinear_coeffs.kappa_0), "kappa_0 should be finite"
        assert np.isfinite(nonlinear_coeffs.kappa_2), "kappa_2 should be finite"

    def test_chi_coefficients(self, nonlinear_coeffs):
        """Test chi coefficients properties."""
        # chi_prime should be positive (real part of susceptibility)
        assert nonlinear_coeffs.chi_prime > 0, "chi_prime should be positive"

        # chi_double_prime_0 should be non-negative (imaginary part of susceptibility)
        assert (
            nonlinear_coeffs.chi_double_prime_0 >= 0
        ), "chi_double_prime_0 should be non-negative"

        # Both should be finite
        assert np.isfinite(nonlinear_coeffs.chi_prime), "chi_prime should be finite"
        assert np.isfinite(
            nonlinear_coeffs.chi_double_prime_0
        ), "chi_double_prime_0 should be finite"

    def test_carrier_frequency(self, nonlinear_coeffs):
        """Test carrier frequency properties."""
        # Carrier frequency should be positive
        assert (
            nonlinear_coeffs.carrier_frequency > 0
        ), "Carrier frequency should be positive"

        # Should be finite
        assert np.isfinite(
            nonlinear_coeffs.carrier_frequency
        ), "Carrier frequency should be finite"

        # Should be a reasonable value (not too large or too small)
        assert (
            1e40 < nonlinear_coeffs.carrier_frequency < 1e50
        ), "Carrier frequency should be in reasonable range"

    def test_k0_squared(self, nonlinear_coeffs):
        """Test k0_squared properties."""
        # k0_squared should be positive
        assert nonlinear_coeffs.k0_squared > 0, "k0_squared should be positive"

        # Should be finite
        assert np.isfinite(nonlinear_coeffs.k0_squared), "k0_squared should be finite"

    def test_physical_constraints(self, nonlinear_coeffs):
        """Test that physical constraints are satisfied."""
        # All coefficients should be finite
        coeffs = [
            nonlinear_coeffs.kappa_0,
            nonlinear_coeffs.kappa_2,
            nonlinear_coeffs.chi_prime,
            nonlinear_coeffs.chi_double_prime_0,
            nonlinear_coeffs.carrier_frequency,
            nonlinear_coeffs.k0_squared,
        ]

        for coeff in coeffs:
            assert np.isfinite(coeff), f"Coefficient {coeff} should be finite"
            assert not np.isnan(coeff), f"Coefficient {coeff} should not be NaN"
            assert not np.isinf(coeff), f"Coefficient {coeff} should not be Inf"

    def test_parameter_relationships(self, nonlinear_coeffs):
        """Test relationships between parameters."""
        # kappa_0 should be larger than kappa_2 (linear stiffness > nonlinear stiffness)
        assert (
            nonlinear_coeffs.kappa_0 >= nonlinear_coeffs.kappa_2
        ), "Linear stiffness should be >= nonlinear stiffness"

        # chi_prime should be larger than chi_double_prime_0 (real part > imaginary part)
        assert (
            nonlinear_coeffs.chi_prime >= nonlinear_coeffs.chi_double_prime_0
        ), "Real part of susceptibility should be >= imaginary part"

    def test_dimensional_consistency(self, nonlinear_coeffs):
        """Test dimensional consistency of coefficients."""
        # All coefficients should have consistent dimensions
        # This is a simplified check - exact dimensional analysis would be more complex

        # kappa_0 and kappa_2 should have same dimensions
        assert np.isfinite(
            nonlinear_coeffs.kappa_0 / nonlinear_coeffs.kappa_2
        ), "kappa_0 and kappa_2 should have consistent dimensions"

        # chi_prime and chi_double_prime_0 should have same dimensions
        assert np.isfinite(
            nonlinear_coeffs.chi_prime / nonlinear_coeffs.chi_double_prime_0
        ), "chi_prime and chi_double_prime_0 should have consistent dimensions"

    def test_coefficient_scaling(self, domain_7d):
        """Test coefficient scaling with different parameters."""
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

        coeffs_list = []
        for config in configs:
            bvp_constants = BVPConstantsAdvanced(config)
            nonlinear_coeffs = NonlinearCoefficients(bvp_constants)
            coeffs_list.append(nonlinear_coeffs)

        # Test that different parameters produce different results
        assert (
            coeffs_list[0].kappa_0 != coeffs_list[1].kappa_0
        ), "Different parameters should produce different kappa_0 values"

        # Test that other coefficients remain the same
        assert (
            coeffs_list[0].kappa_2 == coeffs_list[1].kappa_2
        ), "kappa_2 should remain the same for different kappa_0"
        assert (
            coeffs_list[0].chi_prime == coeffs_list[1].chi_prime
        ), "chi_prime should remain the same for different kappa_0"
