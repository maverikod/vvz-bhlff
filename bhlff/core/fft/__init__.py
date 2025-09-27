"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

FFT package for BHLFF framework.

This package provides FFT operations and spectral methods for the 7D phase
field theory.

Physical Meaning:
    FFT components implement spectral methods for efficient computation
    of phase field equations in frequency space.

Mathematical Foundation:
    Implements FFT-based spectral methods for solving phase field equations
    including spectral operations and FFT planning for optimization.
"""

from .fft_backend import FFTBackend
from .spectral_operations import SpectralOperations

__all__ = [
    "FFTBackend",
    "SpectralOperations",
]
