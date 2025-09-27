"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Spectral solvers package.

This package provides spectral methods for solving phase field equations
using FFT and other spectral techniques.

Physical Meaning:
    Spectral solvers implement high-accuracy numerical methods for solving
    phase field equations in frequency space, providing efficient computation
    of fractional operators and related equations.

Mathematical Foundation:
    Implements spectral methods including FFT-based solvers for the fractional
    Riesz operator and related equations in frequency space.
"""

from .fft_solver_3d import FFTSolver3D

__all__ = [
    "FFTSolver3D",
]
