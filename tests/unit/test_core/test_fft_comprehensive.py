"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive unit tests for FFT module - Facade.

This module provides a facade for comprehensive unit tests for the FFT module,
importing and organizing all FFT-related test classes.
"""

# Import all FFT test modules
from test_fft_backend import TestFFTBackend
from test_spectral_operations import TestSpectralOperations
from test_spectral_derivatives import TestSpectralDerivatives
from test_spectral_filtering import TestSpectralFiltering
from test_fft_plan_manager import TestFFTPlanManager
from test_fft_butterfly_computer import TestFFTButterflyComputer
from test_fft_twiddle_computer import TestFFTTwiddleComputer

# Export all test classes
__all__ = [
    "TestFFTBackend",
    "TestSpectralOperations",
    "TestSpectralDerivatives",
    "TestSpectralFiltering",
    "TestFFTPlanManager",
    "TestFFTButterflyComputer",
    "TestFFTTwiddleComputer",
]
