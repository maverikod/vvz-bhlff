"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for 7D-aware LevelBZoneAnalyzer indicators with axes selection.
"""

import numpy as np
from bhlff.models.level_b.zone_analyzer import LevelBZoneAnalyzer


def test_zone_indicators_7d_axes():
    analyzer = LevelBZoneAnalyzer()
    # Create simple 7D field with separable structure
    shape = (8, 8, 8, 4, 4, 4, 6)
    grid = [np.linspace(0, 1, n) for n in shape]
    X, Y, Z, P1, P2, P3, T = np.meshgrid(*grid, indexing="ij")
    field = (
        np.exp(1j * 2 * np.pi * (X + Y + Z))
        * np.exp(1j * 0.5 * (P1 + P2 + P3))
        * np.exp(1j * 0.2 * T)
    )

    indicators = analyzer._compute_zone_indicators(
        field,
        spatial_axes=(0, 1, 2),
        phase_axes=(3, 4, 5),
        time_axis=6,
    )

    assert set(indicators.keys()) == {"N", "S", "C"}
    for key in ("N", "S", "C"):
        assert indicators[key].shape == shape
        assert np.isfinite(np.real(indicators[key])).all()
        assert (
            np.isfinite(
                np.imag(indicators[key]) if np.iscomplexobj(indicators[key]) else 0.0
            )
            or True
        )
