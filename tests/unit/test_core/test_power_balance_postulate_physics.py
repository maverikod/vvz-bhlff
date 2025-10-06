"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for Power Balance Postulate.

This module provides comprehensive physical validation tests for the
Power Balance Postulate, ensuring it correctly implements the theoretical
foundations of energy conservation in the BVP theory.

Physical Meaning:
    Tests validate that energy is conserved in the BVP system,
    ensuring the fundamental conservation law is satisfied.

Mathematical Foundation:
    Tests energy conservation: ∂E/∂t + ∇·S = 0 and validates
    power balance at boundaries.

Example:
    >>> pytest tests/unit/test_core/test_power_balance_postulate_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.postulates.power_balance.power_balance_postulate import (
    PowerBalancePostulate,
)


class TestPowerBalancePostulatePhysics:
    """Physical validation tests for Power Balance Postulate."""

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

    def test_power_balance_postulate_physics(
        self, domain_7d, bvp_constants, test_envelope
    ):
        """
        Test Power Balance Postulate physics.

        Physical Meaning:
            Validates that energy is conserved in the BVP system,
            ensuring the fundamental conservation law is satisfied.

        Mathematical Foundation:
            Tests energy conservation: ∂E/∂t + ∇·S = 0 and validates
            power balance at boundaries.
        """
        postulate = PowerBalancePostulate(domain_7d, bvp_constants)

        # Apply postulate
        result = postulate.apply(test_envelope)

        # Physical validation 1: Postulate should be satisfied
        assert result["satisfied"], "Power Balance postulate not satisfied"

        # Physical validation 2: Energy should be conserved
        energy_conservation_error = result["energy_conservation_error"]
        assert (
            energy_conservation_error < 1e-3
        ), f"Energy not conserved: error = {energy_conservation_error}"

        # Physical validation 3: Power flux should be balanced
        power_flux_balance = result["power_flux_balance"]
        assert (
            power_flux_balance > 0.9
        ), f"Power flux not balanced: {power_flux_balance}"

        # Physical validation 4: Total energy should be positive
        total_energy = result["total_energy"]
        assert total_energy > 0, f"Negative total energy: {total_energy}"
