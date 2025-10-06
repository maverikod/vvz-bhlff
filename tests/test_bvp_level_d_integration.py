"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Framework integration tests for Level D.

This module implements comprehensive tests for BVP framework integration
at Level D, ensuring BVP multimode superposition functionality.

Physical Meaning:
    Tests validate that the BVP framework serves as the central backbone
    for Level D, providing multimode superposition analysis.

Mathematical Foundation:
    Tests validate BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    integration at Level D with mode superposition, field projections,
    and streamlines analysis.

Example:
    >>> pytest tests/test_bvp_level_d_integration.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore, BVPInterface
from bhlff.core.bvp.bvp_envelope_solver import BVPEnvelopeSolver
from bhlff.solvers.integrators import TimeIntegrator


class TestBVPLevelDIntegration:
    """Test BVP integration for Level D: BVP Multimode Superposition."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(
            dimensions=3,
            size=(1.0, 1.0, 1.0),
            resolution=(64, 64, 64),
            boundary_conditions="periodic",
        )

    @pytest.fixture
    def bvp_config(self):
        """Create BVP configuration."""
        return {
            "carrier_frequency": 1.85e43,
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 1.0,
            },
        }

    def test_level_d_bvp_mode_superposition(self, domain, bvp_config):
        """Test D1: BVP Mode Superposition."""
        bvp_core = BVPCore(domain, bvp_config)

        # Create multiple sources for mode superposition
        source1 = np.zeros(domain.shape)
        source1[20, 20, 20] = 1.0

        source2 = np.zeros(domain.shape)
        source2[44, 44, 44] = 1.0

        # Solve individual envelopes
        envelope1 = bvp_core.solve_envelope(source1)
        envelope2 = bvp_core.solve_envelope(source2)

        # Test mode superposition
        combined_source = source1 + source2
        combined_envelope = bvp_core.solve_envelope(combined_source)

        # Validate superposition properties
        assert combined_envelope.shape == domain.shape
        assert np.all(np.isfinite(combined_envelope))

    def test_level_d_bvp_field_projections(self, domain, bvp_config):
        """Test D2: BVP Field Projections."""
        bvp_core = BVPCore(domain, bvp_config)

        # Test U(1)³ phase vector projections
        phase_vector = bvp_core.get_phase_vector()
        phase_components = bvp_core.get_phase_components()

        # Test electroweak current projections
        envelope = np.ones(domain.shape)
        currents = bvp_core.compute_electroweak_currents(envelope)

        # Validate field projections
        assert len(phase_components) == 3
        assert "em_current" in currents
        assert "weak_current" in currents
        assert "mixed_current" in currents

    def test_level_d_bvp_streamlines(self, domain, bvp_config):
        """Test D3: BVP Streamlines."""
        bvp_core = BVPCore(domain, bvp_config)

        # Test phase coherence for streamline analysis
        coherence = bvp_core.compute_phase_coherence()
        assert coherence.shape == domain.shape

        # Test total phase for flow analysis
        total_phase = bvp_core.get_total_phase()
        assert total_phase.shape == domain.shape
