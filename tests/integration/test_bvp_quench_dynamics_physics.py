"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP quench dynamics.

This module provides comprehensive integration tests for BVP quench
dynamics, ensuring physical consistency and theoretical correctness
of quench detection and evolution.

Physical Meaning:
    Tests validate BVP quench dynamics:
    - Quench detection correctly identifies phase transition regions
    - Physical consistency of quench evolution
    - Correlation with field gradients

Mathematical Foundation:
    Tests quench dynamics: |∇a|² > threshold
    and validates quench evolution.

Example:
    >>> pytest tests/integration/test_bvp_quench_dynamics_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_core.bvp_core import BVPCore
from bhlff.core.bvp.quench_detector import QuenchDetector
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


class TestBVPQuenchDynamicsPhysics:
    """BVP quench dynamics physical validation tests."""

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

    def test_bvp_quench_dynamics_physics(self, domain_7d, bvp_core):
        """
        Test BVP quench dynamics physics.
        
        Physical Meaning:
            Validates that quench detection correctly identifies
            phase transition regions and maintains physical consistency.
            
        Mathematical Foundation:
            Tests quench dynamics: |∇a|² > threshold
            and validates quench evolution.
        """
        # Create source with known quench regions
        source = self._generate_source_with_quenches(domain_7d)
        
        # Solve envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Detect quenches
        quench_detector = QuenchDetector(domain_7d, bvp_core.constants)
        quench_map = quench_detector.detect_quenches(envelope)
        
        # Physical validation 1: Quench map should be binary
        assert np.all((quench_map == 0) | (quench_map == 1)), "Quench map not binary"
        
        # Physical validation 2: Quenches should be localized
        quench_fraction = np.mean(quench_map)
        assert 0 < quench_fraction < 0.5, f"Quench fraction out of range: {quench_fraction}"
        
        # Physical validation 3: Quenches should correlate with high gradients
        gradient_magnitude = self._compute_gradient_magnitude(envelope, domain_7d)
        quench_gradient_correlation = np.corrcoef(quench_map.flatten(), 
                                                 gradient_magnitude.flatten())[0, 1]
        assert quench_gradient_correlation > 0.3, "Quenches don't correlate with gradients"

    def _generate_source_with_quenches(self, domain: Domain) -> np.ndarray:
        """Generate source with known quench regions."""
        source = np.zeros(domain.shape)
        
        # Create sharp gradients (quenches)
        source[domain.N//4:3*domain.N//4, domain.N//4:3*domain.N//4, 
               domain.N//4:3*domain.N//4, :, :, :, :] = 10.0
        
        return source

    def _compute_gradient_magnitude(self, envelope: np.ndarray, domain: Domain) -> np.ndarray:
        """Compute gradient magnitude."""
        grad_x = np.gradient(envelope, axis=0)
        grad_y = np.gradient(envelope, axis=1)
        grad_z = np.gradient(envelope, axis=2)
        
        return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
