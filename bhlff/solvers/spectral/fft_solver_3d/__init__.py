"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

3D FFT solver package for BVP envelope equations.

This package provides modular components for 3D FFT solving of BVP
envelope equations, including spectral operations and boundary handling.

Physical Meaning:
    3D FFT solver implements spectral methods for solving BVP envelope
    equations in 3D space, providing efficient computation of fractional
    operators and BVP envelope modulations.

Mathematical Foundation:
    Implements 3D spectral methods including FFT-based solvers for the
    BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x) in 3D frequency space.

Example:
    >>> from .fft_solver_3d_core import FFTSolver3D
    >>> from .spectral_operations import SpectralOperations
    >>> solver = FFTSolver3D(domain, config)
    >>> solution = solver.solve(source)
"""

from .fft_solver_3d_core import FFTSolver3D
from .spectral_operations import SpectralOperations
from .boundary_handler import BoundaryHandler
from .bvp_integration import BVPIntegration

__all__ = [
    "FFTSolver3D",
    "SpectralOperations",
    "BoundaryHandler",
    "BVPIntegration"
]
