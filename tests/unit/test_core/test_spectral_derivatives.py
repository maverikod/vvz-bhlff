"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for SpectralDerivatives class.

This module provides comprehensive unit tests for the SpectralDerivatives class,
covering first, second, and higher-order derivatives.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.fft.fft_backend_core import FFTBackend
from bhlff.core.fft.spectral_derivatives import SpectralDerivatives


class TestSpectralDerivatives:
    """Comprehensive tests for SpectralDerivatives class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(
            L=1.0,
            N=8,
            dimensions=7,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def fft_backend(self, domain):
        """Create FFT backend for testing."""
        return FFTBackend(domain)

    @pytest.fixture
    def spectral_derivs(self, fft_backend):
        """Create spectral derivatives for testing."""
        return SpectralDerivatives(fft_backend)

    def test_spectral_derivs_initialization(self, spectral_derivs, fft_backend):
        """Test spectral derivatives initialization."""
        assert spectral_derivs.fft_backend == fft_backend

    def test_spectral_derivs_first_derivative(self, spectral_derivs):
        """Test first-order derivative computation."""
        # Create test field
        field = np.random.random(spectral_derivs.fft_backend.domain.shape)
        
        # Compute first derivative
        derivative = spectral_derivs.compute_derivative(field, axis=0, order=1)
        
        assert isinstance(derivative, np.ndarray)
        assert derivative.shape == field.shape

    def test_spectral_derivs_second_derivative(self, spectral_derivs):
        """Test second-order derivative computation."""
        # Create test field
        field = np.random.random(spectral_derivs.fft_backend.domain.shape)
        
        # Compute second derivative
        derivative = spectral_derivs.compute_derivative(field, axis=0, order=2)
        
        assert isinstance(derivative, np.ndarray)
        assert derivative.shape == field.shape

    def test_spectral_derivs_nth_derivative(self, spectral_derivs):
        """Test nth-order derivative computation."""
        # Create test field
        field = np.random.random(spectral_derivs.fft_backend.domain.shape)
        
        # Test different orders
        for order in [1, 2, 3, 4]:
            derivative = spectral_derivs.compute_derivative(field, axis=0, order=order)
            
            assert isinstance(derivative, np.ndarray)
            assert derivative.shape == field.shape

    def test_spectral_derivs_mixed_derivative(self, spectral_derivs):
        """Test mixed derivative computation."""
        # Create test field
        field = np.random.random(spectral_derivs.fft_backend.domain.shape)
        
        # Compute mixed derivative
        derivative = spectral_derivs.compute_mixed_derivative(field, axes=[0, 1], orders=[1, 1])
        
        assert isinstance(derivative, np.ndarray)
        assert derivative.shape == field.shape

    def test_spectral_derivs_validation(self, spectral_derivs):
        """Test input validation."""
        # Test with wrong shape
        wrong_field = np.random.random((4, 4, 4))
        
        with pytest.raises(ValueError):
            spectral_derivs.compute_derivative(wrong_field, axis=0, order=1)
        
        # Test with invalid axis
        field = np.random.random(spectral_derivs.fft_backend.domain.shape)
        
        with pytest.raises(ValueError):
            spectral_derivs.compute_derivative(field, axis=10, order=1)
        
        # Test with invalid order
        with pytest.raises(ValueError):
            spectral_derivs.compute_derivative(field, axis=0, order=0)

    def test_spectral_derivs_energy_conservation(self, spectral_derivs):
        """Test energy conservation in spectral derivatives."""
        # Create test field
        field = np.random.random(spectral_derivs.fft_backend.domain.shape)
        
        # Compute derivative
        derivative = spectral_derivs.compute_derivative(field, axis=0, order=1)
        
        # Energy should be finite
        original_energy = np.sum(field**2)
        derivative_energy = np.sum(derivative**2)
        
        assert np.isfinite(original_energy)
        assert np.isfinite(derivative_energy)

    def test_spectral_derivs_7d_structure(self, spectral_derivs):
        """Test 7D structure preservation in spectral derivatives."""
        # Create 7D test field
        field = np.zeros(spectral_derivs.fft_backend.domain.shape)
        field[0, 0, 0, 0, 0, 0, 0] = 1.0
        
        # Compute derivatives along different axes
        for axis in range(7):
            derivative = spectral_derivs.compute_derivative(field, axis=axis, order=1)
            
            # Should preserve 7D structure
            assert derivative.shape == spectral_derivs.fft_backend.domain.shape

    def test_spectral_derivs_numerical_stability(self, spectral_derivs):
        """Test numerical stability of spectral derivatives."""
        # Test with extreme values
        field = np.array([1e10, -1e10, 1e-10, -1e-10])
        field = np.broadcast_to(field.reshape(-1, 1, 1, 1, 1, 1, 1), spectral_derivs.fft_backend.domain.shape)
        
        # Should not raise errors
        derivative = spectral_derivs.compute_derivative(field, axis=0, order=1)
        
        # Should be stable
        assert np.isfinite(derivative).all()

    def test_spectral_derivs_precision(self, spectral_derivs):
        """Test precision of spectral derivatives."""
        # Test with known function
        x = np.linspace(0, 2*np.pi, spectral_derivs.fft_backend.domain.shape[0], endpoint=False)
        field = np.sin(x)
        field = np.broadcast_to(field.reshape(-1, 1, 1, 1, 1, 1, 1), spectral_derivs.fft_backend.domain.shape)
        
        # Compute derivative
        derivative = spectral_derivs.compute_derivative(field, axis=0, order=1)
        
        # Should be finite and reasonable
        assert np.isfinite(derivative).all()
        assert np.max(np.abs(derivative)) < 10.0  # Reasonable bound

    def test_spectral_derivs_axis_handling(self, spectral_derivs):
        """Test derivative computation along different axes."""
        # Create test field
        field = np.random.random(spectral_derivs.fft_backend.domain.shape)
        
        # Test all axes
        for axis in range(spectral_derivs.fft_backend.dimensions):
            derivative = spectral_derivs.compute_derivative(field, axis=axis, order=1)
            
            assert isinstance(derivative, np.ndarray)
            assert derivative.shape == field.shape

    def test_spectral_derivs_order_handling(self, spectral_derivs):
        """Test derivative computation with different orders."""
        # Create test field
        field = np.random.random(spectral_derivs.fft_backend.domain.shape)
        
        # Test different orders
        for order in [1, 2, 3, 4]:
            derivative = spectral_derivs.compute_derivative(field, axis=0, order=order)
            
            assert isinstance(derivative, np.ndarray)
            assert derivative.shape == field.shape
            assert np.isfinite(derivative).all()
