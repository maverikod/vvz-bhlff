"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for FFTBackend class.

This module provides comprehensive unit tests for the FFTBackend class,
covering initialization, transforms, and energy conservation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.fft.fft_backend_core import FFTBackend


class TestFFTBackend:
    """Comprehensive tests for FFTBackend class."""

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

    def test_fft_backend_initialization(self, fft_backend, domain):
        """Test FFT backend initialization."""
        assert fft_backend.domain == domain
        assert fft_backend.N == domain.N
        assert fft_backend.dimensions == domain.dimensions

    def test_fft_backend_forward_transform(self, fft_backend):
        """Test forward FFT transform."""
        # Create test field
        field = np.random.random(fft_backend.domain.shape)
        
        # Forward transform
        spectral_field = fft_backend.forward_transform(field)
        
        assert isinstance(spectral_field, np.ndarray)
        assert spectral_field.shape == field.shape
        assert np.iscomplexobj(spectral_field)

    def test_fft_backend_inverse_transform(self, fft_backend):
        """Test inverse FFT transform."""
        # Create test spectral field
        spectral_field = np.random.random(fft_backend.domain.shape) + 1j * np.random.random(fft_backend.domain.shape)
        
        # Inverse transform
        field = fft_backend.inverse_transform(spectral_field)
        
        assert isinstance(field, np.ndarray)
        assert field.shape == spectral_field.shape
        # In 7D BVP theory, result can be complex if it contains BVP phase information
        # assert np.isrealobj(field)  # Removed - BVP can have complex structure

    def test_fft_backend_round_trip(self, fft_backend):
        """Test round-trip FFT transform."""
        # Create test field
        original_field = np.random.random(fft_backend.domain.shape)
        
        # Forward then inverse transform
        spectral_field = fft_backend.forward_transform(original_field)
        reconstructed_field = fft_backend.inverse_transform(spectral_field)
        
        # Should be close to original (within numerical precision)
        assert np.allclose(original_field, reconstructed_field, atol=1e-10)

    def test_fft_backend_energy_conservation(self, fft_backend):
        """Test FFT energy conservation (Parseval's theorem)."""
        # Create test field
        field = np.random.random(fft_backend.domain.shape)
        
        # Compute energy in real space
        real_energy = np.sum(field**2)
        
        # Transform to spectral space
        spectral_field = fft_backend.forward_transform(field)
        
        # Compute energy in spectral space (with proper normalization)
        spectral_energy = np.sum(np.abs(spectral_field)**2) / np.prod(field.shape)
        
        # Energy should be conserved (Parseval's theorem)
        assert np.allclose(real_energy, spectral_energy, atol=1e-10)

    def test_fft_backend_get_wave_vectors(self, fft_backend):
        """Test wave vector computation."""
        # Get wave vectors
        wave_vectors = fft_backend.get_wave_vectors()
        
        # Should return tuple of wave vectors
        assert isinstance(wave_vectors, tuple)
        assert len(wave_vectors) == fft_backend.dimensions
        
        # Each wave vector should have correct shape
        for i, k in enumerate(wave_vectors):
            assert isinstance(k, np.ndarray)
            # Wave vector shape should match domain shape for that dimension
            expected_shape = fft_backend.domain.shape[i]
            assert k.shape == (expected_shape,)

    def test_fft_backend_spectral_operations(self, fft_backend):
        """Test spectral operations."""
        # Create test field
        field = np.random.random(fft_backend.domain.shape)
        
        # Transform to spectral space
        spectral_field = fft_backend.forward_transform(field)
        
        # Test spectral operations
        assert isinstance(spectral_field, np.ndarray)
        assert np.iscomplexobj(spectral_field)
        assert spectral_field.shape == field.shape

    def test_fft_backend_error_handling(self, fft_backend):
        """Test error handling for invalid inputs."""
        # Test with wrong shape
        wrong_field = np.random.random((4, 4, 4))  # Wrong shape
        
        with pytest.raises(ValueError):
            fft_backend.forward_transform(wrong_field)
        
        with pytest.raises(ValueError):
            fft_backend.inverse_transform(wrong_field)

    def test_fft_backend_memory_efficiency(self, fft_backend):
        """Test memory efficiency of FFT operations."""
        # Create large field
        field = np.random.random(fft_backend.domain.shape)
        
        # Transform should not modify original
        original_field = field.copy()
        spectral_field = fft_backend.forward_transform(field)
        
        # Original should be unchanged
        assert np.array_equal(field, original_field)
        
        # Inverse transform should not modify spectral field
        original_spectral = spectral_field.copy()
        reconstructed = fft_backend.inverse_transform(spectral_field)
        
        # Spectral field should be unchanged
        assert np.array_equal(spectral_field, original_spectral)

    def test_fft_backend_numerical_stability(self, fft_backend):
        """Test numerical stability of FFT operations."""
        # Test with extreme values
        field = np.array([1e10, -1e10, 1e-10, -1e-10])
        field = np.broadcast_to(field.reshape(-1, 1, 1, 1, 1, 1, 1), fft_backend.domain.shape)
        
        # Should not raise errors
        spectral_field = fft_backend.forward_transform(field)
        reconstructed = fft_backend.inverse_transform(spectral_field)
        
        # Should be stable
        assert np.isfinite(spectral_field).all()
        assert np.isfinite(reconstructed).all()

    def test_fft_backend_precision(self, fft_backend):
        """Test FFT precision with known functions."""
        # Test with sinusoidal function
        x = np.linspace(0, 2*np.pi, fft_backend.domain.shape[0], endpoint=False)
        field = np.sin(x)
        field = np.broadcast_to(field.reshape(-1, 1, 1, 1, 1, 1, 1), fft_backend.domain.shape)
        
        # Transform and back
        spectral_field = fft_backend.forward_transform(field)
        reconstructed = fft_backend.inverse_transform(spectral_field)
        
        # Should be very close to original
        assert np.allclose(field, reconstructed, atol=1e-12)

    def test_fft_backend_7d_structure(self, fft_backend):
        """Test 7D structure preservation in FFT operations."""
        # Create 7D test field with specific structure
        field = np.zeros(fft_backend.domain.shape)
        
        # Set specific values in 7D space
        field[0, 0, 0, 0, 0, 0, 0] = 1.0  # Origin
        field[-1, -1, -1, -1, -1, -1, -1] = 1.0  # Corner
        
        # Transform
        spectral_field = fft_backend.forward_transform(field)
        
        # Should preserve 7D structure
        assert spectral_field.shape == fft_backend.domain.shape
        assert np.iscomplexobj(spectral_field)
        
        # Transform back
        reconstructed = fft_backend.inverse_transform(spectral_field)
        
        # Should reconstruct original structure
        assert np.allclose(field, reconstructed, atol=1e-10)
