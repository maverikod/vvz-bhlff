"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for Carrier Primacy Postulate.

This module provides comprehensive physical validation tests for the
Carrier Primacy Postulate, ensuring it correctly implements the theoretical
foundations of high-frequency carrier dominance in the BVP theory.

Physical Meaning:
    Tests validate that the high-frequency carrier dominates the field
    structure, ensuring the BVP is truly a high-frequency field
    with envelope modulation.

Mathematical Foundation:
    Tests that |a_carrier| >> |a_envelope| where a_carrier is the
    high-frequency component and a_envelope is the slow modulation.

Example:
    >>> pytest tests/unit/test_core/test_carrier_primacy_postulate_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.postulates.carrier_primacy_postulate import BVPPostulate1_CarrierPrimacy


class TestCarrierPrimacyPostulatePhysics:
    """Physical validation tests for Carrier Primacy Postulate."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for postulate testing."""
        return Domain(
            L=1.0,
            N=8,
            dimensions=3,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for postulate testing."""
        config = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
            }
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def test_envelope(self, domain_7d):
        """Create test envelope for postulate validation."""
        envelope = np.zeros(domain_7d.shape)
        
        # Create envelope with known properties
        center = domain_7d.N // 2
        envelope[center-4:center+5, center-4:center+5, center-4:center+5,
                :, :, :, :] = 1.0
        
        # Add phase structure
        phi1 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
        phi2 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
        phi3 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
        
        PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing='ij')
        phase_factor = np.exp(1j * (PHI1 + PHI2 + PHI3))
        
        envelope = envelope * phase_factor
        
        return envelope

    def test_carrier_primacy_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Carrier Primacy Postulate physics.
        
        Physical Meaning:
            Validates that the high-frequency carrier dominates the field
            structure, ensuring the BVP is truly a high-frequency field
            with envelope modulation.
            
        Mathematical Foundation:
            Tests that |a_carrier| >> |a_envelope| where a_carrier is the
            high-frequency component and a_envelope is the slow modulation.
        """
        postulate = BVPPostulate1_CarrierPrimacy(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "Carrier Primacy postulate not satisfied"
        
        # Physical validation 2: Carrier dominance ratio should be > 1
        carrier_dominance = result['carrier_dominance_ratio']
        assert carrier_dominance > 1.0, f"Carrier not dominant: ratio = {carrier_dominance}"
        
        # Physical validation 3: Carrier frequency should be high
        carrier_frequency = result['carrier_frequency']
        assert carrier_frequency > 1.0, f"Carrier frequency too low: {carrier_frequency}"
        
        # Physical validation 4: Envelope should be smooth
        envelope_smoothness = result['envelope_smoothness']
        assert envelope_smoothness > 0.5, f"Envelope not smooth: {envelope_smoothness}"
