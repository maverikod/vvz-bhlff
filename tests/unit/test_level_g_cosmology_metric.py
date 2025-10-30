"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for level G cosmology envelope effective metric.

This module tests the envelope-derived effective metric for 7D phase field theory,
including metric initialization, scale factors computation, and metric tensor computation.

Physical Meaning:
    Tests the envelope-derived effective metric that describes the
    effective spacetime geometry without classical spacetime curvature.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_g.cosmology import EnvelopeEffectiveMetric


class TestEnvelopeEffectiveMetric:
    """
    Tests the envelope-derived effective metric (no spacetime curvature).
    """

    def test_metric_initialization(self):
        """Test metric initialization."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }

        metric = EnvelopeEffectiveMetric({"c_phi": 2.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 3.0})
        assert g.shape == (7, 7)
        assert g[0, 0] == pytest.approx(-1.0 / (2.0**2))

    def test_scale_factors_computation(self):
        """Test scale factors computation."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g1 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        g2 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 2.0})
        assert g2[1, 1] > g1[1, 1]

    def test_metric_tensor_computation(self):
        """Test metric tensor computation."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        # Check metric properties
        assert g.shape == (7, 7)
        assert np.allclose(g, g.T)  # Symmetric
        assert g[0, 0] < 0  # Time component negative
        assert g[1, 1] > 0  # Spatial components positive

    def test_metric_determinant(self):
        """Test metric determinant computation."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        det_g = np.linalg.det(g)
        assert det_g != 0  # Non-degenerate metric

    def test_metric_inverse(self):
        """Test metric inverse computation."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        g_inv = np.linalg.inv(g)
        identity = np.dot(g, g_inv)
        assert np.allclose(identity, np.eye(7))

    def test_metric_contraction(self):
        """Test metric contraction properties."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        # Test metric contraction with itself
        trace = np.trace(g)
        assert isinstance(trace, (int, float))

    def test_metric_eigenvalues(self):
        """Test metric eigenvalues."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        eigenvalues = np.linalg.eigvals(g)
        # Should have one negative eigenvalue (time) and six positive (space)
        negative_eigenvals = eigenvalues[eigenvalues < 0]
        positive_eigenvals = eigenvalues[eigenvalues > 0]

        assert len(negative_eigenvals) == 1
        assert len(positive_eigenvals) == 6

    def test_metric_parameter_dependence(self):
        """Test metric dependence on parameters."""
        metric1 = EnvelopeEffectiveMetric({"c_phi": 1.0})
        metric2 = EnvelopeEffectiveMetric({"c_phi": 2.0})

        g1 = metric1.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        g2 = metric2.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        # Different parameters should give different metrics
        assert not np.allclose(g1, g2)

    def test_metric_chi_dependence(self):
        """Test metric dependence on chi_over_kappa."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})

        g1 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        g2 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 2.0})

        # Different chi_over_kappa should give different metrics
        assert not np.allclose(g1, g2)

    def test_metric_consistency(self):
        """Test metric consistency across different computations."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})

        # Same parameters should give same metric
        g1 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        g2 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        assert np.allclose(g1, g2)

    def test_metric_physical_meaning(self):
        """Test metric physical meaning."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        # Time component should be negative (timelike)
        assert g[0, 0] < 0

        # Spatial components should be positive (spacelike)
        for i in range(1, 7):
            assert g[i, i] > 0

    def test_metric_off_diagonal_terms(self):
        """Test metric off-diagonal terms."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        # Off-diagonal terms should be zero for diagonal metric
        for i in range(7):
            for j in range(7):
                if i != j:
                    assert abs(g[i, j]) < 1e-10

    def test_metric_units(self):
        """Test metric units consistency."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        # All components should have consistent units
        assert all(
            isinstance(g[i, j], (int, float)) for i in range(7) for j in range(7)
        )

    def test_metric_boundary_conditions(self):
        """Test metric boundary conditions."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})

        # Test with extreme values
        g_min = metric.compute_effective_metric_from_vbp_envelope(
            {"chi_over_kappa": 0.001}
        )
        g_max = metric.compute_effective_metric_from_vbp_envelope(
            {"chi_over_kappa": 1000.0}
        )

        # Metrics should be finite
        assert np.all(np.isfinite(g_min))
        assert np.all(np.isfinite(g_max))

    def test_metric_symmetry(self):
        """Test metric symmetry properties."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        # Metric should be symmetric
        assert np.allclose(g, g.T, atol=1e-10)

    def test_metric_regularity(self):
        """Test metric regularity."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})

        # Metric should be regular (no NaN or inf)
        assert np.all(np.isfinite(g))
        assert not np.any(np.isnan(g))
        assert not np.any(np.isinf(g))


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
