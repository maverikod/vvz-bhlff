"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for semi-transparent step resonator boundary operator.
"""

import numpy as np
from bhlff.core.bvp.boundary.step_resonator import apply_step_resonator


def test_step_resonator_basic_1d():
    f = np.array([10.0, 1.0, 2.0, 3.0], dtype=float)
    out = apply_step_resonator(f.copy(), axes=(0,), R=0.5, T=0.5)
    # left boundary: 0.5*10 + 0.5*1 = 5.5
    # right boundary: 0.5*3 + 0.5*2 = 2.5
    assert np.isclose(out[0], 5.5)
    assert np.isclose(out[-1], 2.5)
    # interior unchanged
    assert np.allclose(out[1:-1], f[1:-1])


def test_step_resonator_7d_axes_subset():
    shape = (4, 4, 4, 2, 2, 2, 2)
    f = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    out = apply_step_resonator(f.copy(), axes=(0, 1, 2), R=0.1, T=0.9)
    # Sanity checks: shapes preserved, finite values
    assert out.shape == shape
    assert np.isfinite(np.real(out)).all()
    assert np.isfinite(np.imag(out)).all()


def test_step_resonator_frequency_dependent_rt():
    shape = (8, 8, 8, 2, 2, 2, 2)
    f = np.ones(shape)
    # Frequency/axis dependent coefficients emulated as arrays broadcastable to boundaries
    R = np.linspace(0.0, 0.5, shape[0]).reshape(-1, 1, 1, 1, 1, 1, 1)
    T = 1.0 - R
    out = apply_step_resonator(f.copy(), axes=(0,), R=R, T=T)
    # Left boundary should be between 0.5 and 1.0 due to mixing
    assert np.all(out[0] <= 1.0) and np.all(out[0] >= 0.5)
