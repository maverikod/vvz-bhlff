"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced FFT solver facade for 7D space-time.

This module provides a unified interface for advanced FFT solving functionality,
delegating to specialized modules for different aspects of advanced solving.
"""

from .advanced import FFTAdvancedCore

# Alias for backward compatibility
FFTSolver7DAdvanced = FFTAdvancedCore

__all__ = ["FFTAdvancedCore", "FFTSolver7DAdvanced"]
