"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Framework integration tests for Level F.

This module implements comprehensive tests for BVP framework integration
at Level F, ensuring BVP collective effects functionality.

Physical Meaning:
    Tests validate that the BVP framework serves as the central backbone
    for Level F, providing collective effects analysis.

Mathematical Foundation:
    Tests validate BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    integration at Level F with multi-particle systems, collective modes,
    and nonlinear effects analysis.

Example:
    >>> pytest tests/test_bvp_level_f_integration.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore, BVPInterface
from bhlff.solvers.spectral import FFTSolver3D
from bhlff.solvers.integrators import TimeIntegrator


class TestBVPLevelFIntegration:
    """Test BVP integration for Level F: BVP Collective Effects."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(
            dimensions=3,
            size=(1.0, 1.0, 1.0),
            resolution=(64, 64, 64),
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

    def test_level_f_bvp_multi_particle_systems(self, domain, bvp_config):
        """Test F1: BVP Multi-Particle Systems."""
        bvp_core = BVPCore(domain, bvp_config)
        
        # Create multiple particle sources
        source = np.zeros(domain.shape)
        source[20, 20, 20] = 1.0
        source[44, 44, 44] = 1.0
        
        # Solve BVP envelope for multi-particle system
        envelope = bvp_core.solve_envelope(source)
        
        # Test collective mode analysis
        assert envelope.shape == domain.shape
        assert np.all(np.isfinite(envelope))

    def test_level_f_bvp_collective_modes(self, domain, bvp_config):
        """Test F2: BVP Collective Modes."""
        bvp_core = BVPCore(domain, bvp_config)
        
        # Test U(1)³ phase vector for collective modes
        phase_vector = bvp_core.get_phase_vector()
        coherence = bvp_core.compute_phase_coherence()
        
        # Validate collective mode capabilities
        assert coherence.shape == domain.shape

    def test_level_f_bvp_nonlinear_effects(self, domain, bvp_config):
        """Test F3: BVP Nonlinear Effects."""
        bvp_core = BVPCore(domain, bvp_config)
        
        # Test nonlinear envelope equation
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        
        envelope = bvp_core.solve_envelope(source)
        
        # Test nonlinear effects through envelope parameters
        params = bvp_core.get_envelope_parameters()
        assert "kappa_2" in params  # Nonlinear stiffness coefficient
        
        # Test quench detection for nonlinear effects
        quenches = bvp_core.detect_quenches(envelope)
        assert isinstance(quenches, dict)
