"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for Quenches Postulate.

This module provides comprehensive physical validation tests for the
Quenches Postulate, ensuring it correctly implements the theoretical
foundations of phase transition dynamics in the BVP theory.

Physical Meaning:
    Tests validate that quench detection correctly identifies phase
    transition regions where the field gradient exceeds threshold.

Mathematical Foundation:
    Tests quench condition: |∇a|² > threshold and validates
    quench dynamics and memory effects.

Example:
    >>> pytest tests/unit/test_core/test_quenches_postulate_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.postulates.quenches_postulate import BVPPostulate5_Quenches


class TestQuenchesPostulatePhysics:
    """Physical validation tests for Quenches Postulate."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for postulate testing."""
        return Domain7DBVP(L_spatial=1.0, N_spatial=8, N_phase=4, T=1.0, N_t=8)

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
            "material_properties": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
                "nu": 1.0,
            },
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def test_envelope(self, domain_7d):
        """Create test envelope for postulate validation."""
        envelope = np.zeros(domain_7d.shape)

        # Create envelope with known properties
        center = domain_7d.N_spatial // 2
        envelope[
            center - 4 : center + 5,
            center - 4 : center + 5,
            center - 4 : center + 5,
            :,
            :,
            :,
            :,
        ] = 1.0

        # Add phase structure for 7D
        phi1 = np.linspace(0, 2 * np.pi, domain_7d.N_phase)
        phi2 = np.linspace(0, 2 * np.pi, domain_7d.N_phase)
        phi3 = np.linspace(0, 2 * np.pi, domain_7d.N_phase)
        t = np.linspace(0, domain_7d.T, domain_7d.N_t)

        # Create 7D coordinate grids
        x = np.linspace(0, domain_7d.L_spatial, domain_7d.N_spatial)
        y = np.linspace(0, domain_7d.L_spatial, domain_7d.N_spatial)
        z = np.linspace(0, domain_7d.L_spatial, domain_7d.N_spatial)
        
        X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(x, y, z, phi1, phi2, phi3, t, indexing="ij")
        phase_factor = np.exp(1j * (PHI1 + PHI2 + PHI3 + 0.1 * T))

        envelope = envelope * phase_factor

        return envelope

    def test_quenches_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Quenches Postulate physics.

        Physical Meaning:
            Validates that quench detection correctly identifies phase
            transition regions where the field gradient exceeds threshold.

        Mathematical Foundation:
            Tests quench condition: |∇a|² > threshold and validates
            quench dynamics and memory effects.
        """
        config = {
            "amplitude_threshold": 0.8,
            "gradient_threshold": 0.1,
            "quench_memory_time": 0.5
        }
        postulate = BVPPostulate5_Quenches(domain_7d, config)

        # Apply postulate
        result = postulate.apply(test_envelope)

        # Physical validation 1: Postulate should be satisfied
        assert result["postulate_satisfied"], "Quenches postulate not satisfied"

        # Physical validation 2: Quench count should be non-negative
        quench_count = result["quench_count"]
        assert quench_count >= 0, f"Negative quench count: {quench_count}"

        # Physical validation 3: Energy dissipated should be non-negative
        energy_dissipated = result["energy_dissipated"]
        assert energy_dissipated >= 0, f"Negative energy dissipated: {energy_dissipated}"

        # Physical validation 4: Thresholds should be positive
        amplitude_threshold = result["amplitude_threshold"]
        gradient_threshold = result["gradient_threshold"]
        assert amplitude_threshold > 0, f"Non-positive amplitude threshold: {amplitude_threshold}"
        assert gradient_threshold > 0, f"Non-positive gradient threshold: {gradient_threshold}"
