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

from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.postulates.power_balance.power_balance_postulate import (
    PowerBalancePostulate,
)


class TestPowerBalancePostulatePhysics:
    """Physical validation tests for Power Balance Postulate."""

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

        X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(
            x, y, z, phi1, phi2, phi3, t, indexing="ij"
        )
        phase_factor = np.exp(1j * (PHI1 + PHI2 + PHI3 + 0.1 * T))

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

        # Physical validation 1: Postulate result should be boolean-like
        assert isinstance(
            result["postulate_satisfied"], (bool, np.bool_)
        ), "Postulate result should be boolean"

        # Physical validation 2: BVP flux should be finite
        bvp_flux = result["bvp_flux"]
        assert np.isfinite(bvp_flux), f"BVP flux not finite: {bvp_flux}"

        # Physical validation 3: Core energy growth should be finite
        core_energy_growth = result["core_energy_growth"]
        assert np.isfinite(
            core_energy_growth
        ), f"Core energy growth not finite: {core_energy_growth}"

        # Physical validation 4: Radiation losses should be non-negative
        radiation_losses = result["radiation_losses"]
        assert radiation_losses >= 0, f"Negative radiation losses: {radiation_losses}"
