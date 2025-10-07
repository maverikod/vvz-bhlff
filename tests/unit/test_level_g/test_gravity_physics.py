"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for gravitational effects in Level G experiments.

This module contains comprehensive physical tests for gravitational
effects models, verifying their behavior against theoretical predictions
and physical constraints.

Theoretical Background:
    Tests verify the physical correctness of gravitational calculations
    including Einstein equations, spacetime curvature, and gravitational
    waves.

Example:
    >>> pytest tests/unit/test_level_g/test_gravity_physics.py -v
"""

import pytest
import numpy as np
from bhlff.core.domain.domain import Domain
from bhlff.models.level_g.cosmology import EnvelopeEffectiveMetric


class TestEnvelopeEffectiveMetricGravity:
    """
    Physics tests aligned with envelope-derived effective metric (no GR curvature).
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=4.0, N=64, dimensions=7)  # Domain size  # Grid points

    @pytest.fixture
    def gravity_params(self):
        """Create realistic gravity parameters."""
        return {
            "G": 6.67430e-11,  # Gravitational constant
            "c": 299792458.0,  # Speed of light
            "phase_gravity_coupling": 1.0,
            "resolution": 64,
            "domain_size": 4.0,
            "precision": 1e-12,
            "tolerance": 1e-12,
            "max_iterations": 1000,
            "field_mass_squared": 1.0,
            "update_factor": 0.01,
            "frequency_range": (1e-4, 1e3),
            "detection_sensitivity": 1e-21,
            "wave_speed": 299792458.0,
            "source_distance": 1e6,
        }

    @pytest.fixture
    def mock_system(self, domain_7d):
        """Create mock system for testing."""

        class MockSystem:
            def __init__(self, domain):
                self.domain = domain
                self.phase_field = np.ones((64, 64, 64), dtype=complex)

        return MockSystem(domain_7d)

    def test_effective_metric_basic_properties(self, gravity_params):
        metric = EnvelopeEffectiveMetric({"c_phi": 2.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 3.0})
        assert g.shape == (7, 7)
        assert g[0, 0] == pytest.approx(-1.0 / (2.0**2))
        assert g[1, 1] == pytest.approx(3.0)
        assert g[2, 2] == pytest.approx(3.0)
        assert g[3, 3] == pytest.approx(3.0)
        assert np.all(np.diag(g[4:, 4:]) == 1.0)

    def test_metric_sensitivity_to_invariants(self):
        metric = EnvelopeEffectiveMetric({"c_phi": 1.5})
        g1 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        g2 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 2.0})
        assert g2[1, 1] > g1[1, 1]

    def test_metric_is_real_and_finite(self):
        metric = EnvelopeEffectiveMetric({})
        g = metric.compute_effective_metric_from_vbp_envelope({})
        assert np.isfinite(g).all()

    def test_time_component_negative(self):
        metric = EnvelopeEffectiveMetric({"c_phi": 3.0})
        g = metric.compute_effective_metric_from_vbp_envelope({})
        assert g[0, 0] < 0

    def test_phase_block_is_identity(self):
        metric = EnvelopeEffectiveMetric({})
        g = metric.compute_effective_metric_from_vbp_envelope({})
        assert np.allclose(np.diag(g[4:, 4:]), np.ones(3))

    def test_metric_stability_under_small_param_change(self):
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g1 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        metric.params["c_phi"] = 1.01
        g2 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        assert np.all(np.isfinite(g1)) and np.all(np.isfinite(g2))

    def test_diagonal_dominance(self):
        metric = EnvelopeEffectiveMetric({})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 2.0})
        assert np.allclose(g, np.diag(np.diag(g)))

    def test_metric_output_consistency(self):
        metric = EnvelopeEffectiveMetric({"c_phi": 2.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.5})
        assert g[0, 0] == pytest.approx(-1.0 / (2.0**2))

        # Check curvature analysis
        curvature = effects["curvature"]
        assert "riemann_tensor" in curvature, "Should contain Riemann tensor"
        assert "ricci_tensor" in curvature, "Should contain Ricci tensor"
        assert "scalar_curvature" in curvature, "Should contain scalar curvature"
        assert "weyl_tensor" in curvature, "Should contain Weyl tensor"
        assert (
            "curvature_invariants" in curvature
        ), "Should contain curvature invariants"

        # Check gravitational waves
        waves = effects["gravitational_waves"]
        assert "strain_tensor" in waves, "Should contain strain tensor"
        assert "amplitude" in waves, "Should contain amplitude"
        assert "frequency_spectrum" in waves, "Should contain frequency spectrum"
        assert "polarization" in waves, "Should contain polarization"

    def test_metric_scaling_changes_with_params(self):
        m1 = EnvelopeEffectiveMetric({"c_phi": 1.0})
        m2 = EnvelopeEffectiveMetric({"c_phi": 0.5})
        g1 = m1.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        g2 = m2.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        assert g2[0, 0] != g1[0, 0]

    def test_metric_diagonal_structure(self):
        metric = EnvelopeEffectiveMetric({})
        g = metric.compute_effective_metric_from_vbp_envelope({})
        assert np.allclose(g, np.diag(np.diag(g)))
