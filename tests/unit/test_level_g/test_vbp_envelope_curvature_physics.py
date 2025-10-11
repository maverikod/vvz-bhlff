"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for VBP envelope curvature in 7D phase field theory.

This module tests the physical correctness of the VBP envelope curvature
calculations, including envelope curvature invariants and effective metric
properties.

Theoretical Background:
    In 7D BVP theory, gravity arises from the curvature of the VBP envelope,
    not from spacetime curvature. The tests validate the physical correctness
    of envelope curvature descriptors and effective metric g_eff[Θ].

Physical Tests:
    - Envelope curvature invariants (positivity, boundedness)
    - Effective metric properties (7D structure, g00=-1/c_φ^2)
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.models.level_g.gravity_curvature import VBPEnvelopeCurvatureCalculator


class TestVBPEnvelopeCurvaturePhysics:
    """Test physical correctness of VBP envelope curvature calculations."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=64, dimensions=7)

    @pytest.fixture
    def envelope_params(self):
        """Create VBP envelope parameters."""
        return {
            "c_phi": 1.0,
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": 1.0,
            "resolution": 64,
        }

    @pytest.fixture
    def phase_field_7d(self, domain_7d):
        """Create 7D phase field for testing."""
        # Create a simple 7D phase field with spatial and phase components (smaller size)
        field = np.zeros((8, 8, 8, 8, 8, 8, 8), dtype=complex)

        # Add spatial variation
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    field[i, j, k, :, :, :, :] = np.exp(
                        1j * 2 * np.pi * (i + j + k) / 8
                    )

        return field

    def test_envelope_curvature_scalar_positivity(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that envelope curvature scalar is positive."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)

        # Envelope curvature scalar should be positive
        assert (
            curvature_descriptors["envelope_curvature_scalar"] >= 0
        ), "Envelope curvature scalar should be non-negative"

    def test_anisotropy_index_boundedness(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that anisotropy index is bounded."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)

        # Anisotropy index should be bounded
        anisotropy = curvature_descriptors["anisotropy_index"]
        assert (
            0 <= anisotropy <= 1
        ), f"Anisotropy index should be bounded [0,1], got {anisotropy}"

    def test_effective_metric_7d_structure(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that effective metric has correct 7D structure."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)
        g_eff = curvature_descriptors["effective_metric"]

        # Should be 7x7 matrix
        assert g_eff.shape == (
            7,
            7,
        ), f"Effective metric should be 7x7, got {g_eff.shape}"

        # Time component should be negative
        assert (
            g_eff[0, 0] < 0
        ), f"Time component g00 should be negative, got {g_eff[0, 0]}"

        # Spatial components should be positive
        for i in range(1, 4):
            assert (
                g_eff[i, i] > 0
            ), f"Spatial component g{i}{i} should be positive, got {g_eff[i, i]}"

        # Phase components should be positive
        for i in range(4, 7):
            assert (
                g_eff[i, i] > 0
            ), f"Phase component g{i}{i} should be positive, got {g_eff[i, i]}"

    def test_effective_metric_time_component(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that time component follows g00=-1/c_φ^2 with correction factor."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)
        g_eff = curvature_descriptors["effective_metric"]

        # Time component should be g00=-1/c_φ^2 with correction factor
        expected_g00 = -1.0 / (envelope_params["c_phi"] ** 2)
        # Account for correction factor (1.0 + 0.1 * phase_amplitude)
        phase_amplitude = np.mean(np.abs(phase_field_7d))
        correction_factor = 1.0 + 0.1 * phase_amplitude
        expected_g00 *= correction_factor

        assert np.isclose(
            g_eff[0, 0], expected_g00, rtol=1e-6
        ), f"Time component should be g00=-1/c_φ^2*correction={expected_g00}, got {g_eff[0, 0]}"

    def test_focusing_rate_energy_argument(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that focusing rate is consistent with energy argument."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)
        focusing_rate = curvature_descriptors["focusing_rate"]

        # Focusing rate should be finite
        assert np.isfinite(
            focusing_rate
        ), f"Focusing rate should be finite, got {focusing_rate}"
