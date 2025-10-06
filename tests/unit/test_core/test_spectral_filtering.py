"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for SpectralFiltering class.

This module provides comprehensive unit tests for the SpectralFiltering class,
covering low-pass, high-pass, band-pass, and Gaussian filtering.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.fft.fft_backend_core import FFTBackend
from bhlff.core.fft.spectral_filtering import SpectralFiltering


class TestSpectralFiltering:
    """Comprehensive tests for SpectralFiltering class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(L=1.0, N=8, dimensions=7, N_phi=4, N_t=8, T=1.0)

    @pytest.fixture
    def fft_backend(self, domain):
        """Create FFT backend for testing."""
        return FFTBackend(domain)

    @pytest.fixture
    def spectral_filtering(self, fft_backend):
        """Create spectral filtering for testing."""
        return SpectralFiltering(fft_backend)

    def test_spectral_filtering_initialization(self, spectral_filtering, fft_backend):
        """Test spectral filtering initialization."""
        assert spectral_filtering.fft_backend == fft_backend

    def test_spectral_filtering_low_pass(self, spectral_filtering):
        """Test low-pass filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        # Apply low-pass filter
        filtered_field = spectral_filtering.low_pass_filter(field, cutoff=0.5)

        assert isinstance(filtered_field, np.ndarray)
        assert filtered_field.shape == field.shape

    def test_spectral_filtering_high_pass(self, spectral_filtering):
        """Test high-pass filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        # Apply high-pass filter
        filtered_field = spectral_filtering.high_pass_filter(field, cutoff=0.5)

        assert isinstance(filtered_field, np.ndarray)
        assert filtered_field.shape == field.shape

    def test_spectral_filtering_band_pass(self, spectral_filtering):
        """Test band-pass filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        # Apply band-pass filter
        filtered_field = spectral_filtering.band_pass_filter(
            field, low_cutoff=0.2, high_cutoff=0.8
        )

        assert isinstance(filtered_field, np.ndarray)
        assert filtered_field.shape == field.shape

    def test_spectral_filtering_gaussian(self, spectral_filtering):
        """Test Gaussian filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        # Apply Gaussian filter
        filtered_field = spectral_filtering.gaussian_filter(field, sigma=1.0)

        assert isinstance(filtered_field, np.ndarray)
        assert filtered_field.shape == field.shape

    def test_spectral_filtering_validation(self, spectral_filtering):
        """Test input validation."""
        # Test with wrong shape
        wrong_field = np.random.random((4, 4, 4))

        with pytest.raises(ValueError):
            spectral_filtering.low_pass_filter(wrong_field, cutoff=0.5)

        # Test with invalid cutoff
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        with pytest.raises(ValueError):
            spectral_filtering.low_pass_filter(field, cutoff=-0.1)

        with pytest.raises(ValueError):
            spectral_filtering.low_pass_filter(field, cutoff=1.1)

    def test_spectral_filtering_energy_conservation(self, spectral_filtering):
        """Test energy conservation in spectral filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        # Apply filter
        filtered_field = spectral_filtering.low_pass_filter(field, cutoff=0.5)

        # Energy should be finite
        original_energy = np.sum(field**2)
        filtered_energy = np.sum(filtered_field**2)

        assert np.isfinite(original_energy)
        assert np.isfinite(filtered_energy)

        # Filtered energy should be less than or equal to original
        assert filtered_energy <= original_energy

    def test_spectral_filtering_7d_structure(self, spectral_filtering):
        """Test 7D structure preservation in spectral filtering."""
        # Create 7D test field
        field = np.zeros(spectral_filtering.fft_backend.domain.shape)
        field[0, 0, 0, 0, 0, 0, 0] = 1.0

        # Apply different filters
        low_pass = spectral_filtering.low_pass_filter(field, cutoff=0.5)
        high_pass = spectral_filtering.high_pass_filter(field, cutoff=0.5)
        band_pass = spectral_filtering.band_pass_filter(
            field, low_cutoff=0.2, high_cutoff=0.8
        )
        gaussian = spectral_filtering.gaussian_filter(field, sigma=1.0)

        # All should preserve 7D structure
        assert low_pass.shape == spectral_filtering.fft_backend.domain.shape
        assert high_pass.shape == spectral_filtering.fft_backend.domain.shape
        assert band_pass.shape == spectral_filtering.fft_backend.domain.shape
        assert gaussian.shape == spectral_filtering.fft_backend.domain.shape

    def test_spectral_filtering_numerical_stability(self, spectral_filtering):
        """Test numerical stability of spectral filtering."""
        # Test with extreme values
        field = np.array([1e10, -1e10, 1e-10, -1e-10])
        field = np.broadcast_to(
            field.reshape(-1, 1, 1, 1, 1, 1, 1),
            spectral_filtering.fft_backend.domain.shape,
        )

        # Should not raise errors
        filtered_field = spectral_filtering.low_pass_filter(field, cutoff=0.5)

        # Should be stable
        assert np.isfinite(filtered_field).all()

    def test_spectral_filtering_precision(self, spectral_filtering):
        """Test precision of spectral filtering."""
        # Test with known function
        x = np.linspace(
            0, 2 * np.pi, spectral_filtering.fft_backend.domain.shape[0], endpoint=False
        )
        field = np.sin(x)
        field = np.broadcast_to(
            field.reshape(-1, 1, 1, 1, 1, 1, 1),
            spectral_filtering.fft_backend.domain.shape,
        )

        # Apply filter
        filtered_field = spectral_filtering.low_pass_filter(field, cutoff=0.5)

        # Should be finite and reasonable
        assert np.isfinite(filtered_field).all()
        assert np.max(np.abs(filtered_field)) < 10.0  # Reasonable bound

    def test_spectral_filtering_cutoff_effects(self, spectral_filtering):
        """Test effects of different cutoff frequencies."""
        # Create test field
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        # Test different cutoffs
        cutoffs = [0.1, 0.3, 0.5, 0.7, 0.9]

        for cutoff in cutoffs:
            filtered_field = spectral_filtering.low_pass_filter(field, cutoff=cutoff)

            assert isinstance(filtered_field, np.ndarray)
            assert filtered_field.shape == field.shape
            assert np.isfinite(filtered_field).all()

    def test_spectral_filtering_band_pass_validation(self, spectral_filtering):
        """Test band-pass filter validation."""
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        # Test invalid band parameters
        with pytest.raises(ValueError):
            spectral_filtering.band_pass_filter(field, low_cutoff=0.8, high_cutoff=0.2)

        with pytest.raises(ValueError):
            spectral_filtering.band_pass_filter(field, low_cutoff=-0.1, high_cutoff=0.5)

        with pytest.raises(ValueError):
            spectral_filtering.band_pass_filter(field, low_cutoff=0.5, high_cutoff=1.1)

    def test_spectral_filtering_gaussian_validation(self, spectral_filtering):
        """Test Gaussian filter validation."""
        field = np.random.random(spectral_filtering.fft_backend.domain.shape)

        # Test invalid sigma
        with pytest.raises(ValueError):
            spectral_filtering.gaussian_filter(field, sigma=-1.0)

        with pytest.raises(ValueError):
            spectral_filtering.gaussian_filter(field, sigma=0.0)
