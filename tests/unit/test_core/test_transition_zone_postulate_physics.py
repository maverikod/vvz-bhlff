"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for Transition Zone Postulate.

This module provides comprehensive physical validation tests for the
Transition Zone Postulate, ensuring it correctly implements the theoretical
foundations of nonlinear interface behavior in the BVP theory.

Physical Meaning:
    Tests validate that the transition zone between different field
    regions exhibits proper nonlinear interface behavior.

Mathematical Foundation:
    Tests nonlinear interface equations and validates transition
    zone dynamics.

Example:
    >>> pytest tests/unit/test_core/test_transition_zone_postulate_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.postulates.transition_zone_postulate import (
    BVPPostulate7_TransitionZone,
)


class TestTransitionZonePostulatePhysics:
    """Physical validation tests for Transition Zone Postulate."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for postulate testing."""
        return Domain(L=1.0, N=8, dimensions=3, N_phi=4, N_t=8, T=1.0)

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
            },
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def test_envelope(self, domain_7d):
        """Create test envelope for postulate validation."""
        envelope = np.zeros(domain_7d.shape)

        # Create envelope with known properties
        center = domain_7d.N // 2
        envelope[
            center - 4 : center + 5,
            center - 4 : center + 5,
            center - 4 : center + 5,
            :,
            :,
            :,
            :,
        ] = 1.0

        # Add phase structure
        phi1 = np.linspace(0, 2 * np.pi, domain_7d.N_phi)
        phi2 = np.linspace(0, 2 * np.pi, domain_7d.N_phi)
        phi3 = np.linspace(0, 2 * np.pi, domain_7d.N_phi)

        PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing="ij")
        phase_factor = np.exp(1j * (PHI1 + PHI2 + PHI3))

        envelope = envelope * phase_factor

        return envelope

    def test_transition_zone_postulate_physics(
        self, domain_7d, bvp_constants, test_envelope
    ):
        """
        Test Transition Zone Postulate physics.

        Physical Meaning:
            Validates that the transition zone between different field
            regions exhibits proper nonlinear interface behavior.

        Mathematical Foundation:
            Tests nonlinear interface equations and validates transition
            zone dynamics.
        """
        postulate = BVPPostulate7_TransitionZone(domain_7d, bvp_constants)

        # Apply postulate
        result = postulate.apply(test_envelope)

        # Physical validation 1: Postulate should be satisfied
        assert result["satisfied"], "Transition Zone postulate not satisfied"

        # Physical validation 2: Interface thickness should be positive
        interface_thickness = result["interface_thickness"]
        assert (
            interface_thickness > 0
        ), f"Zero interface thickness: {interface_thickness}"

        # Physical validation 3: Transition energy should be positive
        transition_energy = result["transition_energy"]
        assert (
            transition_energy >= 0
        ), f"Negative transition energy: {transition_energy}"

        # Physical validation 4: Nonlinear effects should be present
        nonlinear_strength = result["nonlinear_strength"]
        assert (
            nonlinear_strength >= 0
        ), f"Negative nonlinear strength: {nonlinear_strength}"
