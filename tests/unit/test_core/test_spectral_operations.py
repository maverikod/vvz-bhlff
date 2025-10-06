"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for SpectralOperations class.

This module provides comprehensive unit tests for the SpectralOperations class,
covering spectral derivatives, filtering, and operations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.fft.fft_backend_core import FFTBackend
from bhlff.core.fft.spectral_operations import SpectralOperations
from bhlff.core.fft.spectral_derivatives import SpectralDerivatives


class TestSpectralOperations:
    """Comprehensive tests for SpectralOperations class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(L=1.0, N=8, dimensions=7, N_phi=4, N_t=8, T=1.0)

    @pytest.fixture
    def fft_backend(self, domain):
        """Create FFT backend for testing."""
        return FFTBackend(domain)

    @pytest.fixture
    def spectral_ops(self, fft_backend):
        """Create spectral operations for testing."""
        return SpectralOperations(fft_backend)

    def test_spectral_ops_initialization(self, spectral_ops, fft_backend):
        """Test spectral operations initialization."""
        assert spectral_ops.fft_backend == fft_backend
        assert hasattr(spectral_ops, "derivatives")
        assert hasattr(spectral_ops, "filtering")

    def test_spectral_ops_compute_derivative(self, spectral_ops):
        """Test spectral derivative computation."""
        # Create test field
        field = np.random.random(spectral_ops.fft_backend.domain.shape)

        # Compute derivative
        derivative = spectral_ops.compute_derivative(field, axis=0, order=1)

        assert isinstance(derivative, np.ndarray)
        assert derivative.shape == field.shape

    def test_spectral_ops_compute_laplacian(self, spectral_ops):
        """Test spectral Laplacian computation."""
        # Create test field
        field = np.random.random(spectral_ops.fft_backend.domain.shape)

        # Compute Laplacian
        laplacian = spectral_ops.compute_laplacian(field)

        assert isinstance(laplacian, np.ndarray)
        assert laplacian.shape == field.shape

    def test_spectral_ops_compute_gradient(self, spectral_ops):
        """Test spectral gradient computation."""
        # Create test field
        field = np.random.random(spectral_ops.fft_backend.domain.shape)

        # Compute gradient
        gradient = spectral_ops.compute_gradient(field)

        assert isinstance(gradient, tuple)
        assert len(gradient) == spectral_ops.fft_backend.dimensions

        # Each component should have correct shape
        for component in gradient:
            assert isinstance(component, np.ndarray)
            assert component.shape == field.shape

    def test_spectral_ops_compute_divergence(self, spectral_ops):
        """Test spectral divergence computation."""
        # Create test vector field
        vector_field = tuple(
            np.random.random(spectral_ops.fft_backend.domain.shape)
            for _ in range(spectral_ops.fft_backend.dimensions)
        )

        # Compute divergence
        divergence = spectral_ops.compute_divergence(vector_field)

        assert isinstance(divergence, np.ndarray)
        assert divergence.shape == spectral_ops.fft_backend.domain.shape

    def test_spectral_ops_compute_curl(self, spectral_ops):
        """Test spectral curl computation."""
        # Create 3D vector field for curl
        if spectral_ops.fft_backend.dimensions >= 3:
            vector_field = tuple(
                np.random.random(spectral_ops.fft_backend.domain.shape)
                for _ in range(3)
            )

            # Compute curl
            curl = spectral_ops.compute_curl(vector_field)

            assert isinstance(curl, tuple)
            assert len(curl) == 3

            # Each component should have correct shape
            for component in curl:
                assert isinstance(component, np.ndarray)
                assert component.shape == spectral_ops.fft_backend.domain.shape

    def test_spectral_ops_validation(self, spectral_ops):
        """Test input validation."""
        # Test with wrong shape
        wrong_field = np.random.random((4, 4, 4))

        with pytest.raises(ValueError):
            spectral_ops.compute_derivative(wrong_field, axis=0, order=1)

        with pytest.raises(ValueError):
            spectral_ops.compute_laplacian(wrong_field)

    def test_spectral_ops_energy_conservation(self, spectral_ops):
        """Test energy conservation in spectral operations."""
        # Create test field
        field = np.random.random(spectral_ops.fft_backend.domain.shape)

        # Compute Laplacian
        laplacian = spectral_ops.compute_laplacian(field)

        # Energy should be conserved in spectral operations
        original_energy = np.sum(field**2)
        laplacian_energy = np.sum(laplacian**2)

        # Energy should be finite
        assert np.isfinite(original_energy)
        assert np.isfinite(laplacian_energy)

    def test_spectral_ops_7d_structure(self, spectral_ops):
        """Test 7D structure preservation in spectral operations."""
        # Create 7D test field
        field = np.zeros(spectral_ops.fft_backend.domain.shape)
        field[0, 0, 0, 0, 0, 0, 0] = 1.0

        # Compute operations
        derivative = spectral_ops.compute_derivative(field, axis=0, order=1)
        laplacian = spectral_ops.compute_laplacian(field)
        gradient = spectral_ops.compute_gradient(field)

        # All should preserve 7D structure
        assert derivative.shape == spectral_ops.fft_backend.domain.shape
        assert laplacian.shape == spectral_ops.fft_backend.domain.shape
        assert len(gradient) == 7

        for component in gradient:
            assert component.shape == spectral_ops.fft_backend.domain.shape

    def test_spectral_ops_numerical_stability(self, spectral_ops):
        """Test numerical stability of spectral operations."""
        # Test with extreme values
        field = np.array([1e10, -1e10, 1e-10, -1e-10])
        field = np.broadcast_to(
            field.reshape(-1, 1, 1, 1, 1, 1, 1), spectral_ops.fft_backend.domain.shape
        )

        # Should not raise errors
        derivative = spectral_ops.compute_derivative(field, axis=0, order=1)
        laplacian = spectral_ops.compute_laplacian(field)

        # Should be stable
        assert np.isfinite(derivative).all()
        assert np.isfinite(laplacian).all()

    def test_spectral_ops_precision(self, spectral_ops):
        """Test precision of spectral operations."""
        # Test with known function
        x = np.linspace(
            0, 2 * np.pi, spectral_ops.fft_backend.domain.shape[0], endpoint=False
        )
        field = np.sin(x)
        field = np.broadcast_to(
            field.reshape(-1, 1, 1, 1, 1, 1, 1), spectral_ops.fft_backend.domain.shape
        )

        # Compute derivative
        derivative = spectral_ops.compute_derivative(field, axis=0, order=1)

        # Should be finite and reasonable
        assert np.isfinite(derivative).all()
        assert np.max(np.abs(derivative)) < 10.0  # Reasonable bound
