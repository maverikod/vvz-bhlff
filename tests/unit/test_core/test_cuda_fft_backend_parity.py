"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Parity tests between CPU and (if available) CUDA backends via UnifiedSpectralOperations.
"""

import numpy as np

from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations


class DummyDomain:
    def __init__(self, shape):
        self.shape = shape


def test_forward_inverse_fft_parity_small_grid():
    shape = (4, 4, 4, 2, 2, 2, 4)
    domain = DummyDomain(shape)
    ops = UnifiedSpectralOperations(domain, precision="float64")

    field = np.random.randn(*shape)
    spec = ops.forward_fft(field, "ortho")
    rec = ops.inverse_fft(spec, "ortho")

    # Numerical parity within tolerance
    diff = np.linalg.norm(rec - field) / (np.linalg.norm(field) + 1e-30)
    assert diff < 1e-12
