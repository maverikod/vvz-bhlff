"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for EnvelopeEffectiveMetric (no spacetime curvature, no exponentials).

These tests validate that the effective metric g_eff[Θ] is constructed solely
from VBP envelope invariants and parameters, without using cosmological scale
factors or spacetime curvature.
"""

import numpy as np
from bhlff.models.level_g.cosmology import EnvelopeEffectiveMetric


def test_effective_metric_basic_isotropic():
    params = {"c_phi": 2.0, "chi_kappa": 1.5}
    metric = EnvelopeEffectiveMetric(params)
    invariants = {"chi_over_kappa": 1.2}

    g = metric.compute_effective_metric_from_vbp_envelope(invariants)

    assert g.shape == (7, 7)
    # Time component
    assert np.isclose(g[0, 0], -1.0 / (params["c_phi"] ** 2))
    # Spatial diagonal equals chi_over_kappa (overrides chi_kappa)
    assert np.isclose(g[1, 1], invariants["chi_over_kappa"])  # x
    assert np.isclose(g[2, 2], invariants["chi_over_kappa"])  # y
    assert np.isclose(g[3, 3], invariants["chi_over_kappa"])  # z
    # Phase space diagonal unity
    assert np.isclose(g[4, 4], 1.0)
    assert np.isclose(g[5, 5], 1.0)
    assert np.isclose(g[6, 6], 1.0)
    # Off-diagonal zeros
    off_diag = g - np.diag(np.diag(g))
    assert np.allclose(off_diag, 0.0)


def test_effective_metric_fallback_params():
    params = {"c_phi": 3.0, "chi_kappa": 2.0}
    metric = EnvelopeEffectiveMetric(params)
    invariants = {}  # No chi_over_kappa provided → fallback to params

    g = metric.compute_effective_metric_from_vbp_envelope(invariants)

    assert np.isclose(g[0, 0], -1.0 / (params["c_phi"] ** 2))
    # Spatial uses params["chi_kappa"] when invariants are missing
    assert np.isclose(g[1, 1], params["chi_kappa"])  # x
    assert np.isclose(g[2, 2], params["chi_kappa"])  # y
    assert np.isclose(g[3, 3], params["chi_kappa"])  # z
    # Phase space diagonal unity
    assert np.isclose(g[4, 4], 1.0)
    assert np.isclose(g[5, 5], 1.0)
    assert np.isclose(g[6, 6], 1.0)
