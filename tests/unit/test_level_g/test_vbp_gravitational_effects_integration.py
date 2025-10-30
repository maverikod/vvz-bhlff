"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Integration tests for VBP gravitational effects in 7D phase field theory.

This module tests the integration of all VBP gravitational effects,
including envelope curvature, gravitational waves, and effective metric.

Theoretical Background:
    Integration of all VBP gravitational effects to ensure physical
    consistency and proper interaction between components.

Physical Tests:
    - VBP gravitational effects integration
    - Physical consistency validation
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.models.level_g.gravity import VBPGravitationalEffectsModel


class TestVBPGravitationalEffectsIntegration:
    """Test integration of all VBP gravitational effects."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=64, dimensions=7)

    @pytest.fixture
    def gravity_params(self):
        """Create gravitational parameters."""
        return {
            "c_phi": 1.0,
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": 1.0,
            "resolution": 64,
            "tolerance": 1e-12,
            "max_iterations": 1000,
        }

    @pytest.fixture
    def mock_system(self, domain_7d):
        """Create mock system with phase field."""

        class MockSystem:
            def __init__(self, domain):
                self.domain = domain
                # Create a simple 7D phase field (smaller size)
                self.phase_field = np.zeros((8, 8, 8, 8, 8, 8, 8), dtype=complex)

                # Add spatial variation (test data generation for comparison with 7D BVP theory)
                for i in range(8):
                    for j in range(8):
                        for k in range(8):
                            self.phase_field[i, j, k, :, :, :, :] = np.exp(
                                1j * 2 * np.pi * (i + j + k) / 8
                            )

        return MockSystem(domain_7d)

    def test_vbp_gravitational_effects_integration(self, mock_system, gravity_params):
        """Test integration of all VBP gravitational effects."""
        gravity_model = VBPGravitationalEffectsModel(mock_system, gravity_params)

        # Compute all envelope effects
        envelope_effects = gravity_model.compute_envelope_effects()

        # Should contain all required components
        required_components = [
            "envelope_curvature",
            "gravitational_waves",
            "envelope_solution",
            "effective_metric",
            "curvature_descriptors",
        ]

        for component in required_components:
            assert (
                component in envelope_effects
            ), f"Envelope effects should contain {component}"

        # Test envelope curvature
        curvature = envelope_effects["envelope_curvature"]
        assert (
            "envelope_curvature_scalar" in curvature
        ), "Envelope curvature should contain scalar"
        assert (
            "anisotropy_index" in curvature
        ), "Envelope curvature should contain anisotropy index"
        assert (
            "focusing_rate" in curvature
        ), "Envelope curvature should contain focusing rate"

        # Test gravitational waves
        waves = envelope_effects["gravitational_waves"]
        assert "c_T" in waves, "Gravitational waves should contain c_T"
        assert "c_phi" in waves, "Gravitational waves should contain c_phi"
        assert waves["c_T"] == waves["c_phi"], "Should have c_T = c_phi"

        # Test effective metric
        g_eff = envelope_effects["effective_metric"]
        assert g_eff.shape == (
            7,
            7,
        ), f"Effective metric should be 7x7, got {g_eff.shape}"
        assert g_eff[0, 0] < 0, "Time component should be negative"

    def test_physical_consistency(self, mock_system, gravity_params):
        """Test physical consistency of VBP gravitational effects."""
        gravity_model = VBPGravitationalEffectsModel(mock_system, gravity_params)

        # Compute all envelope effects
        envelope_effects = gravity_model.compute_envelope_effects()

        # Test envelope curvature consistency
        curvature = envelope_effects["envelope_curvature"]
        assert (
            curvature["envelope_curvature_scalar"] >= 0
        ), "Envelope curvature scalar should be non-negative"
        assert (
            0 <= curvature["anisotropy_index"] <= 1
        ), "Anisotropy index should be bounded [0,1]"

        # Test gravitational waves consistency
        waves = envelope_effects["gravitational_waves"]
        assert waves["amplitude"] >= 0, "Wave amplitude should be non-negative"
        assert waves["c_T"] == waves["c_phi"], "Should have c_T = c_phi"

        # Test effective metric consistency
        g_eff = envelope_effects["effective_metric"]
        assert g_eff[0, 0] < 0, "Time component should be negative"
        for i in range(1, 7):
            assert g_eff[i, i] > 0, f"Component g{i}{i} should be positive"
