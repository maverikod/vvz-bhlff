"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

3D FFT solver implementation with BVP framework integration.

This module implements the 3D FFT solver for the 7D phase field theory,
providing efficient spectral methods for 3D problems with full BVP framework
integration.

Physical Meaning:
    3D FFT solver implements spectral methods for solving BVP envelope
    equations in 3D space, providing efficient computation of fractional
    operators and BVP envelope modulations.

Mathematical Foundation:
    Implements 3D spectral methods including FFT-based solvers for the
    BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x) in 3D frequency space.

Example:
    >>> solver = FFTSolver3D(domain, config)
    >>> bvp_envelope = solver.solve_bvp_envelope(source)
    >>> quenches = solver.detect_quenches(bvp_envelope)
"""

from .fft_solver_3d_core import FFTSolver3D
