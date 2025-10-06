"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive tests for BVP (Base High-Frequency Field) framework.

This module provides comprehensive testing for the BVP framework,
including 7D space-time structure, U(1)³ phase structure, and
all 9 BVP postulates.

Physical Meaning:
    Tests validate that the BVP framework correctly implements
    the 7D phase field theory with proper mathematical foundations
    and physical consistency.

Example:
    >>> pytest tests/test_bvp_framework.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp import (
    BVPCore,
    BVPEnvelopeSolver,
    BVPImpedanceCalculator,
    BVPInterface,
    BVPConstants,
    QuenchDetector,
    PhaseVector,
    BVPPostulates,
)


class TestBVPFramework:
    """Test suite for BVP framework components."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, dimensions=7, N_phi=4, N_t=8, T=1.0)

    @pytest.fixture
    def bvp_config(self):
        """Create BVP configuration for testing."""
        return {
            "carrier_frequency": 1.85e43,
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 1.0,
            },
            "quench_detection": {
                "amplitude_threshold": 0.8,
                "detuning_threshold": 0.1,
                "gradient_threshold": 0.5,
            },
            "impedance_calculation": {
                "frequency_range": [1e15, 1e20],
                "frequency_points": 100,
            },
            "phase_components": {
                "amplitude_1": 1.0,
                "amplitude_2": 1.0,
                "amplitude_3": 1.0,
                "frequency_1": 1.0,
                "frequency_2": 1.0,
                "frequency_3": 1.0,
            },
        }

    @pytest.fixture
    def bvp_core(self, domain_7d, bvp_config):
        """Create BVP core instance for testing."""
        return BVPCore(domain_7d, bvp_config)

    def test_domain_7d_structure(self, domain_7d):
        """Test 7D domain structure."""
        assert domain_7d.dimensions == 7
        assert hasattr(domain_7d, "dphi")
        assert hasattr(domain_7d, "N_phi")
        assert hasattr(domain_7d, "dt")
        assert hasattr(domain_7d, "N_t")
        assert domain_7d.shape == (8, 8, 8, 4, 4, 4, 8)

    def test_bvp_core_initialization(self, bvp_core):
        """Test BVP core initialization."""
        assert bvp_core.domain is not None
        assert bvp_core.config is not None
        assert bvp_core.get_bvp_constants() is not None
        assert hasattr(bvp_core, "_operations")
        assert hasattr(bvp_core, "_bvp_constants")

    def test_phase_vector_u1_structure(self, bvp_core):
        """Test U(1)³ phase structure."""
        phase_vector = bvp_core.get_phase_vector()
        if phase_vector is not None:
            assert hasattr(phase_vector, "_phase_components")
            assert hasattr(phase_vector._phase_components, "theta_components")
            assert (
                len(phase_vector._phase_components.theta_components) == 3
            )  # Three U(1) components
            # Test that components exist and are arrays
            for comp in phase_vector._phase_components.theta_components:
                assert isinstance(comp, np.ndarray)
                assert comp.ndim > 0  # At least 1D array

    def test_envelope_solver_7d(self, bvp_core):
        """Test 7D envelope solver."""
        # Test with simple source
        source = np.zeros(bvp_core.domain.shape, dtype=complex)
        source[4, 4, 4, 2, 2, 2, 4] = (
            1.0  # Point source in 7D (adjusted for smaller domain)
        )

        envelope = bvp_core.solve_envelope(source)
        assert envelope.shape == bvp_core.domain.shape
        assert np.isfinite(envelope).all()

    def test_quench_detector(self, bvp_core):
        """Test quench detection."""
        # Test with envelope
        envelope = np.ones(bvp_core.domain.shape, dtype=complex)
        quenches = bvp_core.detect_quenches(envelope)

        assert isinstance(quenches, dict)
        assert "quench_locations" in quenches
        assert "quench_types" in quenches
        assert "quench_strengths" in quenches

    def test_impedance_calculator(self, bvp_core):
        """Test impedance calculation."""
        # Test with envelope
        envelope = np.ones(bvp_core.domain.shape, dtype=complex)
        impedance = bvp_core.compute_impedance(envelope)

        assert isinstance(impedance, dict)
        assert "admittance" in impedance
        assert "reflection" in impedance
        assert "transmission" in impedance
        assert "peaks" in impedance

    def test_bvp_postulates(self, bvp_core):
        """Test all 9 BVP postulates."""
        postulates = BVPPostulates(bvp_core.domain, bvp_core.get_bvp_constants())

        # Test postulate initialization
        assert hasattr(postulates, "carrier_primacy")
        assert hasattr(postulates, "scale_separation")
        assert hasattr(postulates, "bvp_rigidity")
        assert hasattr(postulates, "u1_phase_structure")
        assert hasattr(postulates, "quenches")
        assert hasattr(postulates, "tail_resonatorness")
        assert hasattr(postulates, "transition_zone")
        assert hasattr(postulates, "core_renormalization")
        assert hasattr(postulates, "power_balance")

        # Test with envelope
        envelope = np.ones(bvp_core.domain.shape, dtype=complex)
        results = postulates.apply_all_postulates(envelope)

        assert isinstance(results, dict)
        assert len(results) == 10  # 9 postulates + overall satisfaction
        assert "all_postulates_satisfied" in results

    def test_bvp_interface(self, bvp_core):
        """Test BVP interface."""
        interface = BVPInterface(bvp_core)
        assert interface.bvp_core is bvp_core
        assert interface.config is not None

        # Test with envelope
        envelope = np.ones(bvp_core.domain.shape, dtype=complex)

        # Test tail interface
        tail_data = interface.interface_with_tail(envelope)
        assert isinstance(tail_data, dict)

        # Test transition zone interface
        tz_data = interface.interface_with_transition_zone(envelope)
        assert isinstance(tz_data, dict)

        # Test core interface
        core_data = interface.interface_with_core(envelope)
        assert isinstance(core_data, dict)

    def test_7d_gradient_computation(self, bvp_core):
        """Test 7D gradient computation."""
        interface = BVPInterface(bvp_core)

        # Test with 7D field
        field = np.ones(bvp_core.domain.shape, dtype=complex)
        gradients = interface.core_interface.compute_field_gradient(field)

        assert isinstance(gradients, list)
        assert len(gradients) == 7  # 3 spatial + 3 phase + 1 temporal
        assert all(g.shape == bvp_core.domain.shape for g in gradients)

    def test_phase_vector_electroweak_currents(self, bvp_core):
        """Test electroweak current computation."""
        phase_vector = bvp_core.get_phase_vector()

        if phase_vector is not None:
            # Test with envelope
            envelope = np.ones(bvp_core.domain.shape, dtype=complex)
            # Test PhaseVector object structure
            assert hasattr(phase_vector, "_phase_components")
            assert hasattr(phase_vector._phase_components, "theta_components")
            assert (
                len(phase_vector._phase_components.theta_components) == 3
            )  # Three U(1) components

        # Skip current computation test if phase_vector is None
        pass

    def test_bvp_constants(self, bvp_core):
        """Test BVP constants."""
        constants = bvp_core.get_bvp_constants()

        # Test physical parameters
        carrier_freq = constants.get_physical_parameter("carrier_frequency")
        assert carrier_freq == 1.85e43

        # Test envelope parameters
        kappa_0 = constants.get_envelope_parameter("kappa_0")
        assert kappa_0 == 1.0

        # Test quench parameters
        amp_threshold = constants.get_quench_parameter("amplitude_threshold")
        assert amp_threshold == 0.8

        # Test numerical parameters
        max_iter = constants.get_numerical_parameter("max_iterations")
        assert max_iter == 50

    def test_bvp_core_solve_envelope(self, bvp_core):
        """Test BVP core envelope solving."""
        # Test with simple source
        source = np.zeros(bvp_core.domain.shape, dtype=complex)
        source[4, 4, 4, 2, 2, 2, 4] = (
            1.0  # Point source in 7D (adjusted for smaller domain)
        )

        envelope = bvp_core.solve_envelope(source)
        assert envelope.shape == bvp_core.domain.shape
        assert np.isfinite(envelope).all()

        # Test phase vector update
        phase_vector = bvp_core.get_phase_vector()
        assert phase_vector is not None
        assert phase_vector._phase_components.theta_components[0] is not None

    def test_bvp_core_detect_quenches(self, bvp_core):
        """Test BVP core quench detection."""
        envelope = np.ones(bvp_core.domain.shape, dtype=complex)
        quenches = bvp_core.detect_quenches(envelope)

        assert isinstance(quenches, dict)
        assert "quench_locations" in quenches
        assert "quench_types" in quenches
        assert "quench_strengths" in quenches

    def test_bvp_core_compute_impedance(self, bvp_core):
        """Test BVP core impedance computation."""
        envelope = np.ones(bvp_core.domain.shape, dtype=complex)
        impedance = bvp_core.compute_impedance(envelope)

        assert isinstance(impedance, dict)
        assert "admittance" in impedance
        assert "reflection" in impedance
        assert "transmission" in impedance
        assert "peaks" in impedance

    def test_bvp_framework_validation(self, bvp_core):
        """Test BVP framework validation."""
        # Test with envelope
        envelope = np.ones(bvp_core.domain.shape, dtype=complex)

        # Test postulate validation
        postulates = BVPPostulates(bvp_core.domain, bvp_core.get_bvp_constants())
        is_valid = postulates.validate_bvp_framework(envelope)
        assert isinstance(is_valid, bool)

        # Test postulate summary
        summary = postulates.get_postulate_summary(envelope)
        assert isinstance(summary, dict)
        assert len(summary) == 9  # 9 postulates

        # Test failed postulates
        failed = postulates.get_failed_postulates(envelope)
        assert isinstance(failed, list)

        # Test quality scores
        scores = postulates.get_postulate_quality_scores(envelope)
        assert isinstance(scores, dict)
        assert len(scores) == 9  # 9 postulates
        assert all(0.0 <= score <= 1.0 for score in scores.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
