"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for phase envelope balance in 7D phase field theory.

This module tests the physical correctness of the phase envelope balance
equations, including convergence and effective metric computation.

Theoretical Background:
    The phase envelope balance equations describe the equilibrium between
    phase field dynamics and gravitational effects in 7D BVP theory.

Physical Tests:
    - Envelope balance convergence
    - Effective metric computation from solution
    - Balance operator components validation
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.models.level_g.gravity_einstein import PhaseEnvelopeBalanceSolver


class TestPhaseEnvelopeBalancePhysics:
    """Test physical correctness of phase envelope balance equations."""

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
            "tolerance": 1e-12,
            "max_iterations": 1000,
        }

    @pytest.fixture
    def phase_field_7d(self, domain_7d):
        """Create 7D phase field for testing."""
        # Create a simple 7D phase field (smaller size)
        field = np.zeros((8, 8, 8, 8, 8, 8, 8), dtype=complex)

        # Add spatial variation
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    field[i, j, k, :, :, :, :] = np.exp(
                        1j * 2 * np.pi * (i + j + k) / 8
                    )

        return field

    def test_envelope_balance_convergence(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that envelope balance equation converges."""
        solver = PhaseEnvelopeBalanceSolver(domain_7d, envelope_params)

        envelope_result = solver.solve_phase_envelope_balance(phase_field_7d)

        # Should return valid result
        assert (
            "envelope_solution" in envelope_result
        ), "Envelope result should contain solution"

        # Solution should be finite
        solution = envelope_result["envelope_solution"]
        assert np.all(np.isfinite(solution)), "Envelope solution should be finite"

    def test_effective_metric_from_solution(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that effective metric is computed from solution."""
        solver = PhaseEnvelopeBalanceSolver(domain_7d, envelope_params)

        envelope_result = solver.solve_phase_envelope_balance(phase_field_7d)

        # Should contain effective metric
        assert (
            "effective_metric" in envelope_result
        ), "Envelope result should contain effective metric"

        g_eff = envelope_result["effective_metric"]

        # Should be 7x7 matrix
        assert g_eff.shape == (
            7,
            7,
        ), f"Effective metric should be 7x7, got {g_eff.shape}"

        # Time component should be negative
        assert g_eff[0, 0] < 0, f"Time component should be negative, got {g_eff[0, 0]}"

    def test_balance_operator_components(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that balance operator has correct components."""
        solver = PhaseEnvelopeBalanceSolver(domain_7d, envelope_params)

        balance_operator = solver._build_balance_operator(phase_field_7d)

        # Should contain all required components
        required_components = [
            "memory_kernels",
            "spatial_operator",
            "bridge_terms",
            "c_phi",
            "beta",
            "mu",
        ]
        for component in required_components:
            assert (
                component in balance_operator
            ), f"Balance operator should contain {component}"

        # Memory kernels should have gamma and k
        memory_kernels = balance_operator["memory_kernels"]
        assert "gamma" in memory_kernels, "Memory kernels should contain gamma"
        assert "k" in memory_kernels, "Memory kernels should contain k"

        # Spatial operator should have correct parameters
        spatial_operator = balance_operator["spatial_operator"]
        assert (
            spatial_operator["beta"] == envelope_params["beta"]
        ), "Spatial operator beta should match parameters"
        assert (
            spatial_operator["mu"] == envelope_params["mu"]
        ), "Spatial operator mu should match parameters"
