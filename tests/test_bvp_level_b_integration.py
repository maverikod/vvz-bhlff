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
        return Domain(
            dimensions=3,
            size=(2.0, 2.0, 2.0),
            resolution=(128, 128, 128),
            boundary_conditions="periodic"
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
                "k0_squared": 1.0
            }
        }

    def test_level_b_bvp_power_law_tails(self, domain, bvp_config):
        """Test B1: BVP Power Law Tails."""
        bvp_core = BVPCore(domain, bvp_config)
        
        # Create point source for power law analysis
        source = np.zeros(domain.shape)
        source[64, 64, 64] = 1.0
        
        # Solve BVP envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Analyze power law behavior
        center = np.array([64, 64, 64])
        radial_profile = []
        
        for r in range(1, 30):
            count = 0
            total_amp = 0
            for i in range(domain.shape[0]):
                for j in range(domain.shape[1]):
                    for k in range(domain.shape[2]):
                        dist = np.linalg.norm(np.array([i, j, k]) - center)
                        if abs(dist - r) < 0.5:
                            total_amp += np.abs(envelope[i, j, k])
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
        phase_vector = bvp_core.get_phase_vector()
        phase_components = bvp_core.get_phase_components()
        assert len(phase_components) == 3
        
        # Test topological charge calculation
        total_phase = bvp_core.get_total_phase()
        assert total_phase.shape == domain.shape
        
        # Test electroweak current generation
        envelope = np.ones(domain.shape)
        currents = bvp_core.compute_electroweak_currents(envelope)
        assert "em_current" in currents
        assert "weak_current" in currents

    def test_level_b_bvp_zone_separation(self, domain, bvp_config):
        """Test B3: BVP Zone Separation."""
        bvp_core = BVPCore(domain, bvp_config)
        
        # Create test envelope
        source = np.zeros(domain.shape)
        source[64, 64, 64] = 1.0
        envelope = bvp_core.solve_envelope(source)
        
        # Test impedance calculation for zone analysis
        impedance = bvp_core.compute_impedance(envelope)
        
        # Validate zone separation capabilities
        assert "peaks" in impedance
        assert "admittance" in impedance
