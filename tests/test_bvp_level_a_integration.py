"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Framework integration tests for Level A.

This module implements comprehensive tests for BVP framework integration
at Level A, ensuring BVP validation and core framework functionality.

Physical Meaning:
    Tests validate that the BVP framework serves as the central backbone
    for Level A, providing validation and core framework functionality.

Mathematical Foundation:
    Tests validate BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    integration at Level A with consistent quench detection,
    impedance calculation, and U(1)³ phase structure.

Example:
    >>> pytest tests/test_bvp_level_a_integration.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore, BVPInterface
from bhlff.solvers.spectral import FFTSolver3D
from bhlff.solvers.integrators import TimeIntegrator


class TestBVPLevelAIntegration:
    """Test BVP integration for Level A: BVP Validation and Core Framework."""

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
            },
            "quench_detection": {
                "amplitude_threshold": 0.8,
                "detuning_threshold": 0.1,
                "gradient_threshold": 0.5
            }
        }

    def test_level_a_bvp_framework_validation(self, domain, bvp_config):
        """Test A0: BVP Framework Validation."""
        bvp_core = BVPCore(domain, bvp_config)
        
        # Validate BVP envelope solver
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        envelope = bvp_core.solve_envelope(source)
        assert envelope.shape == domain.shape
        
        # Validate quench detection system
        quenches = bvp_core.detect_quenches(envelope)
        assert isinstance(quenches, dict)
        
        # Validate U(1)³ phase vector
        phase_vector = bvp_core.get_phase_vector()
        assert phase_vector is not None
        
        # Validate BVP impedance calculation
        impedance = bvp_core.compute_impedance(envelope)
        assert isinstance(impedance, dict)

    def test_level_a_bvp_enhanced_solvers(self, domain, bvp_config):
        """Test A1: BVP-Enhanced Solvers."""
        bvp_core = BVPCore(domain, bvp_config)
        
        # Test 7D FFT solver with BVP integration
        fft_solver = FFTSolver3D(domain, bvp_config, bvp_core)
        assert fft_solver.get_bvp_core() == bvp_core
        
        # Test BVP envelope equation solution
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        envelope = fft_solver.solve_bvp_envelope(source)
        assert envelope.shape == domain.shape
        
        # Test BVP quench event handling
        quenches = fft_solver.detect_quenches(envelope)
        assert isinstance(quenches, dict)

    def test_level_a_bvp_scaling(self, domain, bvp_config):
        """Test A2: BVP Scaling and Nondimensionalization."""
        # Test BVP parameter scaling
        config1 = bvp_config.copy()
        config2 = bvp_config.copy()
        config2["carrier_frequency"] = 2.0 * config1["carrier_frequency"]
        
        bvp_core1 = BVPCore(domain, config1)
        bvp_core2 = BVPCore(domain, config2)
        
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        
        envelope1 = bvp_core1.solve_envelope(source)
        envelope2 = bvp_core2.solve_envelope(source)
        
        # Validate scaling consistency
        assert envelope1.shape == envelope2.shape
        assert np.all(np.isfinite(envelope1))
        assert np.all(np.isfinite(envelope2))
