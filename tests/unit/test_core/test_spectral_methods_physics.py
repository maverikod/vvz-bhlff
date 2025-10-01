"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for spectral methods in 7D BVP theory - Facade.

This module provides a facade for comprehensive physical validation tests for spectral
methods used in the 7D BVP theory, ensuring mathematical correctness and physical
consistency of FFT-based computations.
"""

# Import all spectral methods physics test modules
from test_fft_physics import TestFFTPhysics
from test_spectral_derivatives_physics import TestSpectralDerivativesPhysics
from test_spectral_laplacian_physics import TestSpectralLaplacianPhysics
from test_spectral_boundary_conditions_physics import TestSpectralBoundaryConditionsPhysics
from test_spectral_convergence_physics import TestSpectralConvergencePhysics
from test_spectral_energy_spectrum_physics import TestSpectralEnergySpectrumPhysics

# Export all test classes
__all__ = [
    'TestFFTPhysics',
    'TestSpectralDerivativesPhysics',
    'TestSpectralLaplacianPhysics',
    'TestSpectralBoundaryConditionsPhysics',
    'TestSpectralConvergencePhysics',
    'TestSpectralEnergySpectrumPhysics',
]
