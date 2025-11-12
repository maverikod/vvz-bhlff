"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Framework integration tests for Level B.

This module implements comprehensive tests for BVP framework integration
at Level B, ensuring BVP fundamental properties functionality.

Physical Meaning:
    Tests validate that the BVP framework serves as the central backbone
    for Level B, providing fundamental properties analysis.

Mathematical Foundation:
    Tests validate BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    integration at Level B with power law tails, topological charge,
    and zone separation analysis.

Example:
    >>> pytest tests/test_bvp_level_b_integration.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore, BVPInterface
from bhlff.core.bvp.bvp_envelope_solver import BVPEnvelopeSolver
from bhlff.solvers.integrators import TimeIntegrator


class TestBVPLevelBIntegration:
    """Test BVP integration for Level B: BVP Fundamental Properties."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        # Create 7D domain for BVP tests with block processing
        # 7D domain: 3 spatial + 3 phase + 1 temporal
        # Block processing should handle memory efficiently
        return Domain(
            L=1.0,  # Spatial domain size
            N=16,  # Reduced for testing
            N_phi=4,  # Phase dimension resolution
            N_t=4,  # Temporal dimension resolution
            T=1.0,  # Temporal domain size
            dimensions=7,  # 7D domain required for BVP
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

    def test_level_b_bvp_power_law_tails(self, domain, bvp_config):
        """Test B1: BVP Power Law Tails."""
        bvp_core = BVPCore(domain, bvp_config)

        # Create point source for power law analysis (7D domain)
        # Use center of spatial dimensions, average over phase and time
        source = np.zeros(domain.shape)
        center_idx = domain.N // 2
        # Place source at spatial center, average over phase/temporal dimensions
        source[center_idx, center_idx, center_idx, :, :, :, :] = 1.0

        # Solve BVP envelope
        envelope = bvp_core.solve_envelope(source)

        # Analyze power law behavior in spatial dimensions only
        center = np.array([center_idx, center_idx, center_idx])
        radial_profile = []

        # Average over phase and temporal dimensions for radial analysis
        envelope_spatial = np.mean(np.abs(envelope), axis=(3, 4, 5, 6))

        for r in range(1, min(10, center_idx)):
            count = 0
            total_amp = 0
            for i in range(domain.N):
                for j in range(domain.N):
                    for k in range(domain.N):
                        dist = np.linalg.norm(np.array([i, j, k]) - center)
                        if abs(dist - r) < 0.5:
                            total_amp += envelope_spatial[i, j, k]
                            count += 1
            if count > 0:
                radial_profile.append(total_amp / count)

        # Validate power law behavior
        assert len(radial_profile) > 0
        assert radial_profile[0] > radial_profile[-1]  # Decreasing with distance

    def test_level_b_bvp_topological_charge(self, domain, bvp_config):
        """Test B2: BVP Topological Charge."""
        bvp_core = BVPCore(domain, bvp_config)

        # Test U(1)³ phase vector structure
        phase_vector_obj = bvp_core.get_phase_vector()
        assert phase_vector_obj is not None
        
        # Get phase components from phase vector
        phase_components = phase_vector_obj.get_phase_components()
        assert len(phase_components) == 3

        # Test topological charge calculation
        total_phase = phase_vector_obj.get_total_phase()
        assert total_phase.shape == domain.shape

        # Test electroweak current generation
        envelope = np.ones(domain.shape)
        currents = phase_vector_obj.compute_electroweak_currents(envelope)
        assert "em_current" in currents
        assert "weak_current" in currents

    def test_level_b_bvp_zone_separation(self, domain, bvp_config):
        """Test B3: BVP Zone Separation."""
        bvp_core = BVPCore(domain, bvp_config)

        # Create test envelope (7D domain)
        source = np.zeros(domain.shape)
        center_idx = domain.N // 2
        # Place source at spatial center, average over phase/temporal dimensions
        source[center_idx, center_idx, center_idx, :, :, :, :] = 1.0
        envelope = bvp_core.solve_envelope(source)

        # Test impedance calculation for zone analysis
        impedance = bvp_core.compute_impedance(envelope)

        # Validate zone separation capabilities
        assert "peaks" in impedance
        assert "admittance" in impedance
