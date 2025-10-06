"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Facade for FFT solver 7D modules.

This module provides a unified interface for all FFT solver 7D
functionality, delegating to specialized modules for different
aspects of FFT solver operations.
"""

from .fft_solver_7d_basic import FFTSolver7DBasic
from .fft_solver_7d_advanced import FFTSolver7DAdvanced

# Alias for backward compatibility
FFTSolver7D = FFTSolver7DAdvanced

__all__ = ["FFTSolver7DBasic", "FFTSolver7DAdvanced", "FFTSolver7D"]
