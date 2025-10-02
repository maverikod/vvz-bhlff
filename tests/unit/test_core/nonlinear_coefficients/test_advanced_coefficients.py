"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced nonlinear coefficients tests.

This module contains advanced tests for nonlinear coefficients
including complex scenarios and edge cases.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.constants.nonlinear_coefficients import NonlinearCoefficients


class TestAdvancedCoefficients:
    """Advanced tests for nonlinear coefficients."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for constants testing."""
        return Domain(
            L=1.0,
            N=32,
            dimensions=7,
            N_phi=16,
            N_t=64,
            T=1.0
        )

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
                "carrier_frequency": 1.85e43
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
                "nu": 1.0
            }
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def nonlinear_coeffs(self, domain_7d, bvp_constants):
        """Create nonlinear coefficients for testing."""
        return NonlinearCoefficients(domain_7d, bvp_constants)

    def test_nonlinear_effects(self, nonlinear_coeffs):
        """Test nonlinear effects of coefficients."""
        # Test that nonlinear coefficients affect the system behavior
        # This is a simplified test - in practice, this would involve
        # solving the equations with different coefficient values
        
        # kappa_2 should represent nonlinear stiffness
        assert nonlinear_coeffs.kappa_2 >= 0, "Nonlinear stiffness should be non-negative"
        
        # The ratio kappa_2/kappa_0 should be reasonable
        ratio = nonlinear_coeffs.kappa_2 / nonlinear_coeffs.kappa_0
        assert 0 <= ratio <= 1, "Nonlinear to linear stiffness ratio should be reasonable"

    def test_susceptibility_properties(self, nonlinear_coeffs):
        """Test susceptibility coefficient properties."""
        # Test complex susceptibility properties
        chi_complex = nonlinear_coeffs.chi_prime + 1j * nonlinear_coeffs.chi_double_prime_0
        
        # Real part should be positive
        assert np.real(chi_complex) > 0, "Real part of susceptibility should be positive"
        
        # Imaginary part should be non-negative
        assert np.imag(chi_complex) >= 0, "Imaginary part of susceptibility should be non-negative"
        
        # Magnitude should be finite
        assert np.isfinite(np.abs(chi_complex)), "Susceptibility magnitude should be finite"

    def test_frequency_dependence(self, nonlinear_coeffs):
        """Test frequency dependence of coefficients."""
        # Test that coefficients are consistent with frequency
        omega = 2 * np.pi * nonlinear_coeffs.carrier_frequency
        
        # Frequency should be finite and positive
        assert np.isfinite(omega), "Angular frequency should be finite"
        assert omega > 0, "Angular frequency should be positive"
        
        # k0_squared should be related to frequency
        # This is a simplified relationship check
        assert np.isfinite(nonlinear_coeffs.k0_squared), "k0_squared should be finite"

    def test_material_properties(self, nonlinear_coeffs, bvp_constants):
        """Test material property relationships."""
        # Test relationships with basic material properties
        mu = bvp_constants.basic_material.mu
        beta = bvp_constants.basic_material.beta
        lambda_param = bvp_constants.basic_material.lambda_param
        nu = bvp_constants.basic_material.nu
        
        # All material properties should be positive
        assert mu > 0, "mu should be positive"
        assert beta > 0, "beta should be positive"
        assert lambda_param >= 0, "lambda_param should be non-negative"
        assert nu > 0, "nu should be positive"
        
        # Coefficients should be consistent with material properties
        assert np.isfinite(nonlinear_coeffs.kappa_0), "kappa_0 should be finite"
        assert np.isfinite(nonlinear_coeffs.chi_prime), "chi_prime should be finite"

    def test_extreme_parameter_values(self, domain_7d):
        """Test behavior with extreme parameter values."""
        # Test with very small kappa_2
        config_small = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 1e-10,  # Very small nonlinear coefficient
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
                "carrier_frequency": 1.85e43
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
                "nu": 1.0
            }
        }
        
        bvp_constants_small = BVPConstantsAdvanced(config_small)
        nonlinear_coeffs_small = NonlinearCoefficients(domain_7d, bvp_constants_small)
        
        # Should still be finite
        assert np.isfinite(nonlinear_coeffs_small.kappa_2), "Very small kappa_2 should be finite"
        assert nonlinear_coeffs_small.kappa_2 >= 0, "Very small kappa_2 should be non-negative"
        
        # Test with very large kappa_2
        config_large = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 1e10,  # Very large nonlinear coefficient
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
                "carrier_frequency": 1.85e43
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
                "nu": 1.0
            }
        }
        
        bvp_constants_large = BVPConstantsAdvanced(config_large)
        nonlinear_coeffs_large = NonlinearCoefficients(domain_7d, bvp_constants_large)
        
        # Should still be finite
        assert np.isfinite(nonlinear_coeffs_large.kappa_2), "Very large kappa_2 should be finite"
        assert nonlinear_coeffs_large.kappa_2 >= 0, "Very large kappa_2 should be non-negative"

    def test_zero_nonlinear_coefficient(self, domain_7d):
        """Test behavior with zero nonlinear coefficient."""
        config_zero = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.0,  # Zero nonlinear coefficient
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
                "carrier_frequency": 1.85e43
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
                "nu": 1.0
            }
        }
        
        bvp_constants_zero = BVPConstantsAdvanced(config_zero)
        nonlinear_coeffs_zero = NonlinearCoefficients(domain_7d, bvp_constants_zero)
        
        # Should be exactly zero
        assert nonlinear_coeffs_zero.kappa_2 == 0.0, "Zero kappa_2 should be exactly zero"
        
        # Other coefficients should remain finite
        assert np.isfinite(nonlinear_coeffs_zero.kappa_0), "kappa_0 should be finite with zero kappa_2"
        assert np.isfinite(nonlinear_coeffs_zero.chi_prime), "chi_prime should be finite with zero kappa_2"

    def test_susceptibility_limits(self, domain_7d):
        """Test susceptibility coefficient limits."""
        # Test with very small chi_double_prime_0
        config_small = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 1e-15,  # Very small imaginary part
                "k0_squared": 4.0,
                "carrier_frequency": 1.85e43
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
                "nu": 1.0
            }
        }
        
        bvp_constants_small = BVPConstantsAdvanced(config_small)
        nonlinear_coeffs_small = NonlinearCoefficients(domain_7d, bvp_constants_small)
        
        # Should still be finite
        assert np.isfinite(nonlinear_coeffs_small.chi_double_prime_0), "Very small chi_double_prime_0 should be finite"
        assert nonlinear_coeffs_small.chi_double_prime_0 >= 0, "Very small chi_double_prime_0 should be non-negative"

    def test_carrier_frequency_limits(self, domain_7d):
        """Test carrier frequency limits."""
        # Test with different carrier frequencies
        frequencies = [1e40, 1e43, 1e46]
        
        for freq in frequencies:
            config = {
                "envelope_equation": {
                    "kappa_0": 1.0,
                    "kappa_2": 0.1,
                    "chi_prime": 1.0,
                    "chi_double_prime_0": 0.01,
                    "k0_squared": 4.0,
                    "carrier_frequency": freq
                },
                "basic_material": {
                    "mu": 1.0,
                    "beta": 1.5,
                    "lambda_param": 0.1,
                    "nu": 1.0
                }
            }
            
            bvp_constants = BVPConstantsAdvanced(config)
            nonlinear_coeffs = NonlinearCoefficients(domain_7d, bvp_constants)
            
            # Should be finite for all frequencies
            assert np.isfinite(nonlinear_coeffs.carrier_frequency), f"Carrier frequency should be finite for {freq}"
            assert nonlinear_coeffs.carrier_frequency > 0, f"Carrier frequency should be positive for {freq}"

    def test_coefficient_consistency(self, nonlinear_coeffs):
        """Test consistency between different coefficients."""
        # Test that all coefficients are consistent with each other
        coeffs = [
            nonlinear_coeffs.kappa_0,
            nonlinear_coeffs.kappa_2,
            nonlinear_coeffs.chi_prime,
            nonlinear_coeffs.chi_double_prime_0,
            nonlinear_coeffs.carrier_frequency,
            nonlinear_coeffs.k0_squared
        ]
        
        # All coefficients should be finite
        for coeff in coeffs:
            assert np.isfinite(coeff), f"Coefficient {coeff} should be finite"
        
        # Test specific relationships
        # kappa_0 should be larger than kappa_2
        assert nonlinear_coeffs.kappa_0 >= nonlinear_coeffs.kappa_2, \
            "Linear stiffness should be >= nonlinear stiffness"
        
        # chi_prime should be larger than chi_double_prime_0
        assert nonlinear_coeffs.chi_prime >= nonlinear_coeffs.chi_double_prime_0, \
            "Real part of susceptibility should be >= imaginary part"
        
        # carrier_frequency should be positive
        assert nonlinear_coeffs.carrier_frequency > 0, "Carrier frequency should be positive"
        
        # k0_squared should be positive
        assert nonlinear_coeffs.k0_squared > 0, "k0_squared should be positive"
