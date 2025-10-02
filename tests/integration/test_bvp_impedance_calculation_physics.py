"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP impedance calculation.

This module provides comprehensive integration tests for BVP impedance
calculation, ensuring physical consistency and theoretical correctness
of impedance computation.

Physical Meaning:
    Tests validate BVP impedance calculation:
    - Impedance calculation correctly computes field impedance
    - Physical consistency of impedance properties
    - Proper impedance bounds and characteristics

Mathematical Foundation:
    Tests impedance calculation: Z = V/I
    and validates impedance properties.

Example:
    >>> pytest tests/integration/test_bvp_impedance_calculation_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_core import BVPCore
from bhlff.core.bvp.bvp_impedance_calculator import BVPImpedanceCalculator
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


class TestBVPImpedanceCalculationPhysics:
    """BVP impedance calculation physical validation tests."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for complete pipeline testing."""
        return Domain(
            L=2.0,  # Larger domain for better physics
            N=64,   # Higher resolution
            dimensions=3,
            N_phi=32,  # More phase points
            N_t=128,   # More time points
            T=2.0      # Longer evolution
        )

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for complete pipeline testing."""
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
    def bvp_core(self, domain_7d, bvp_constants):
        """Create BVP core for complete pipeline testing."""
        return BVPCore(domain_7d, bvp_constants)

    def test_bvp_impedance_calculation_physics(self, domain_7d, bvp_core):
        """
        Test BVP impedance calculation physics.
        
        Physical Meaning:
            Validates that impedance calculation correctly computes
            the field impedance and maintains physical consistency.
            
        Mathematical Foundation:
            Tests impedance calculation: Z = V/I
            and validates impedance properties.
        """
        # Create test source
        source = self._generate_physical_source(domain_7d)
        
        # Solve envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Calculate impedance
        impedance_calculator = BVPImpedanceCalculator(domain_7d, bvp_core.constants)
        impedance = impedance_calculator.compute_impedance(envelope)
        
        # Physical validation 1: Impedance should be finite
        assert np.all(np.isfinite(impedance)), "Impedance contains non-finite values"
        
        # Physical validation 2: Real part should be positive (resistance)
        real_impedance = np.real(impedance)
        assert np.all(real_impedance >= 0), "Negative real impedance"
        
        # Physical validation 3: Imaginary part should be reasonable
        imag_impedance = np.imag(impedance)
        assert np.all(np.isfinite(imag_impedance)), "Non-finite imaginary impedance"
        
        # Physical validation 4: Impedance should be bounded
        max_impedance = np.max(np.abs(impedance))
        assert max_impedance < 1e6, f"Impedance too large: {max_impedance}"

    def _generate_physical_source(self, domain: Domain) -> np.ndarray:
        """Generate a physical source for testing."""
        source = np.zeros(domain.shape)
        
        # Create localized source in center
        center = domain.N // 2
        source[center-2:center+3, center-2:center+3, center-2:center+3, 
               :, :, :, :] = 1.0
        
        return source
