"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP phase vector.

This module provides comprehensive integration tests for BVP phase vector,
ensuring physical consistency and theoretical correctness
of U(1)³ phase structure implementation.

Physical Meaning:
    Tests validate BVP phase vector:
    - U(1)³ phase structure implementation
    - Phase coherence maintenance
    - Topological charge quantization

Mathematical Foundation:
    Tests U(1)³ phase structure: a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃)
    and validates phase coherence.

Example:
    >>> pytest tests/integration/test_bvp_phase_vector_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_core import BVPCore
from bhlff.core.bvp.phase_vector.phase_vector import PhaseVector
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


class TestBVPPhaseVectorPhysics:
    """BVP phase vector physical validation tests."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for complete pipeline testing."""
        return Domain(
            L=1.0,  # Smaller domain for testing
            N=2,  # Very low resolution for memory efficiency
            dimensions=7,  # 7D structure
        )

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for complete pipeline testing."""
        config = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
            },
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def bvp_core(self, domain_7d, bvp_constants):
        """Create BVP core for complete pipeline testing."""
        return BVPCore(domain_7d, bvp_constants)

    def test_bvp_phase_vector_physics(self, domain_7d, bvp_core):
        """
        Test BVP phase vector physics.

        Physical Meaning:
            Validates that phase vector correctly implements
            U(1)³ phase structure and maintains physical consistency.

        Mathematical Foundation:
            Tests U(1)³ phase structure: a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃)
            and validates phase coherence.
        """
        # Create test source
        source = self._generate_physical_source(domain_7d)

        # Solve envelope
        envelope = bvp_core.solve_envelope(source)

        # Create phase vector
        phase_vector = PhaseVector(domain_7d, bvp_core.constants)

        # Test phase decomposition
        amplitude, phases = phase_vector.decompose_phase_structure(envelope)

        # Physical validation 1: Amplitude should be non-negative
        assert np.all(amplitude >= 0), "Phase vector amplitude contains negative values"

        # Physical validation 2: Phases should be in [0, 2π)
        for phase in phases:
            assert np.all(phase >= 0) and np.all(
                phase < 2 * np.pi
            ), "Phases out of range"

        # Physical validation 3: Phase coherence should be maintained
        coherence = phase_vector.compute_phase_coherence(envelope)
        assert 0 <= coherence <= 1, f"Phase coherence out of range: {coherence}"

        # Physical validation 4: Topological charge should be quantized
        topological_charge = phase_vector.compute_topological_charge(envelope)
        assert np.isclose(
            topological_charge, np.round(topological_charge), atol=1e-6
        ), f"Topological charge not quantized: {topological_charge}"

    def _generate_physical_source(self, domain: Domain) -> np.ndarray:
        """Generate a physical source for testing."""
        source = np.zeros(domain.shape)

        # Create localized source in center
        center = domain.N // 2
        source[
            center - 2 : center + 3,
            center - 2 : center + 3,
            center - 2 : center + 3,
            :,
            :,
            :,
            :,
        ] = 1.0

        return source
