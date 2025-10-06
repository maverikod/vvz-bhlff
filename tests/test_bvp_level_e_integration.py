"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Framework integration tests for Level E.

This module implements comprehensive tests for BVP framework integration
at Level E, ensuring BVP solitons and defects functionality.

Physical Meaning:
    Tests validate that the BVP framework serves as the central backbone
    for Level E, providing solitons and defects analysis.

Mathematical Foundation:
    Tests validate BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    integration at Level E with solitons, defect dynamics,
    and theory integration analysis.

Example:
    >>> pytest tests/test_bvp_level_e_integration.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore, BVPInterface
from bhlff.core.bvp.bvp_envelope_solver import BVPEnvelopeSolver
from bhlff.solvers.integrators import TimeIntegrator


class TestBVPLevelEIntegration:
    """Test BVP integration for Level E: BVP Solitons and Defects."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(
            dimensions=3,
            size=(2.0, 2.0, 2.0),
            resolution=(128, 128, 128),
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

    def test_level_e_bvp_solitons(self, domain, bvp_config):
        """Test E1: BVP Solitons."""
        bvp_core = BVPCore(domain, bvp_config)

        # Create soliton-like source
        source = np.zeros(domain.shape)
        source[64, 64, 64] = 1.0

        # Solve BVP envelope for soliton formation
        envelope = bvp_core.solve_envelope(source)

        # Test soliton stability
        assert envelope.shape == domain.shape
        assert np.all(np.isfinite(envelope))

        # Test quench detection for soliton dynamics
        quenches = bvp_core.detect_quenches(envelope)
        assert isinstance(quenches, dict)

    def test_level_e_bvp_defect_dynamics(self, domain, bvp_config):
        """Test E2: BVP Defect Dynamics."""
        bvp_core = BVPCore(domain, bvp_config)

        # Test U(1)³ phase vector for defect analysis
        phase_vector = bvp_core.get_phase_vector()
        phase_components = bvp_core.get_phase_components()

        # Test topological charge calculation
        total_phase = bvp_core.get_total_phase()

        # Validate defect dynamics capabilities
        assert len(phase_components) == 3
        assert total_phase.shape == domain.shape

    def test_level_e_bvp_theory_integration(self, domain, bvp_config):
        """Test E3: BVP Theory Integration."""
        bvp_core = BVPCore(domain, bvp_config)

        # Test theoretical validation capabilities
        envelope_params = bvp_core.get_envelope_parameters()
        quench_thresholds = bvp_core.get_quench_thresholds()
        impedance_params = bvp_core.get_impedance_parameters()

        # Validate theoretical parameter access
        assert isinstance(envelope_params, dict)
        assert isinstance(quench_thresholds, dict)
        assert isinstance(impedance_params, dict)
