"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Framework integration tests for Level C.

This module implements comprehensive tests for BVP framework integration
at Level C, ensuring BVP boundaries and resonators functionality.

Physical Meaning:
    Tests validate that the BVP framework serves as the central backbone
    for Level C, providing boundaries and resonators analysis.

Mathematical Foundation:
    Tests validate BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    integration at Level C with boundary effects, resonator chains,
    and quench memory analysis.

Example:
    >>> pytest tests/test_bvp_level_c_integration.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore, BVPInterface
from bhlff.core.bvp.bvp_envelope_solver import BVPEnvelopeSolver
from bhlff.solvers.integrators import TimeIntegrator


class TestBVPLevelCIntegration:
    """Test BVP integration for Level C: BVP Boundaries and Resonators."""

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
            "impedance_calculation": {
                "frequency_range": [1e15, 1e20],
                "frequency_points": 1000,
            },
        }

    def test_level_c_bvp_boundary_effects(self, domain, bvp_config):
        """Test C1: BVP Boundary Effects."""
        bvp_core = BVPCore(domain, bvp_config)

        # Create envelope with boundary effects
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        envelope = bvp_core.solve_envelope(source)

        # Test BVP impedance calculation
        impedance = bvp_core.compute_impedance(envelope)

        # Validate boundary function calculation
        assert "admittance" in impedance
        assert "reflection" in impedance
        assert "transmission" in impedance

    def test_level_c_bvp_resonator_chains(self, domain, bvp_config):
        """Test C2: BVP Resonator Chains."""
        bvp_core = BVPCore(domain, bvp_config)

        # Test BVP interface for resonator chains
        bvp_interface = BVPInterface(bvp_core)

        # Create test envelope
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        envelope = bvp_core.solve_envelope(source)

        # Test interface with tail resonators
        tail_data = bvp_interface.interface_with_tail(envelope)
        assert isinstance(tail_data, dict)

        # Test interface with transition zone
        transition_data = bvp_interface.interface_with_transition_zone(envelope)
        assert isinstance(transition_data, dict)

    def test_level_c_bvp_quench_memory(self, domain, bvp_config):
        """Test C3: BVP Quench Memory."""
        bvp_core = BVPCore(domain, bvp_config)

        # Create envelope with quench events
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        envelope = bvp_core.solve_envelope(source)

        # Test quench detection
        quenches = bvp_core.detect_quenches(envelope)

        # Validate quench memory effects
        assert "quench_locations" in quenches
        assert "energy_dumped" in quenches

        # Test quench threshold modification
        new_thresholds = {
            "amplitude_threshold": 0.9,
            "detuning_threshold": 0.2,
            "gradient_threshold": 0.6,
        }
        bvp_core.set_quench_thresholds(new_thresholds)
