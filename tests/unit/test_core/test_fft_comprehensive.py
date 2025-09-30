"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive unit tests for FFT module.

This module provides comprehensive unit tests for the FFT module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.fft.fft_backend_core import FFTBackend
from bhlff.core.fft.spectral_operations import SpectralOperations
from bhlff.core.fft.spectral_derivatives import SpectralDerivatives
from bhlff.core.fft.spectral_filtering import SpectralFiltering
from bhlff.core.fft.fft_plan_manager import FFTPlanManager
from bhlff.core.fft.fft_butterfly_computer import FFTButterflyComputer
from bhlff.core.fft.fft_twiddle_computer import FFTTwiddleComputer


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
        assert np.isrealobj(field)

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
        
        # Should be equal (Parseval's theorem)
        assert np.allclose(real_energy, spectral_energy, atol=1e-10)

    def test_fft_backend_get_wave_vectors(self, fft_backend):
        """Test wave vector generation."""
        kx = fft_backend.get_wave_vectors(0)
        ky = fft_backend.get_wave_vectors(1)
        kz = fft_backend.get_wave_vectors(2)
        
        assert len(kx) == fft_backend.N
        assert len(ky) == fft_backend.N
        assert len(kz) == fft_backend.N
        
        # Check symmetry
        assert np.allclose(kx, -kx[::-1])
        assert np.allclose(ky, -ky[::-1])
        assert np.allclose(kz, -kz[::-1])

    def test_fft_backend_get_wave_vector_magnitude(self, fft_backend):
        """Test wave vector magnitude computation."""
        k_magnitude = fft_backend.get_wave_vector_magnitude()
        
        assert k_magnitude.shape == (fft_backend.N, fft_backend.N, fft_backend.N, fft_backend.N_phi, fft_backend.N_phi, fft_backend.N_phi, fft_backend.N_t)
        assert np.all(k_magnitude >= 0)
        assert k_magnitude[0, 0, 0, 0, 0, 0, 0] == 0.0  # DC component

    def test_fft_backend_validation(self, fft_backend):
        """Test FFT backend validation."""
        with pytest.raises(ValueError):
            fft_backend.forward_transform(np.zeros((4, 4, 4)))  # Wrong shape
        
        with pytest.raises(ValueError):
            fft_backend.inverse_transform(np.zeros((4, 4, 4)))  # Wrong shape


class TestSpectralOperations:
    """Comprehensive tests for SpectralOperations class."""

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
    def spectral_ops(self, domain, fft_backend):
        """Create spectral operations for testing."""
        return SpectralOperations(domain, fft_backend)

    def test_spectral_ops_initialization(self, spectral_ops, domain, fft_backend):
        """Test spectral operations initialization."""
        assert spectral_ops.domain == domain
        assert spectral_ops.fft_backend == fft_backend

    def test_spectral_ops_compute_derivative(self, spectral_ops):
        """Test spectral derivative computation."""
        # Create test field with known derivative
        x = spectral_ops.domain.get_coordinates(0)
        y = spectral_ops.domain.get_coordinates(1)
        z = spectral_ops.domain.get_coordinates(2)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        field = X + Y + Z
        
        # Broadcast to full 7D shape
        field_7d = field[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        
        # Compute derivative
        derivative = spectral_ops.compute_derivative(field_7d, order=1, axis=0)
        
        assert derivative.shape == field_7d.shape
        assert np.allclose(derivative, 1.0, atol=1e-10)

    def test_spectral_ops_compute_laplacian(self, spectral_ops):
        """Test spectral Laplacian computation."""
        # Create test field with known Laplacian
        x = spectral_ops.domain.get_coordinates(0)
        y = spectral_ops.domain.get_coordinates(1)
        z = spectral_ops.domain.get_coordinates(2)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        field = X**2 + Y**2 + Z**2
        
        # Broadcast to full 7D shape
        field_7d = field[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        
        # Compute Laplacian
        laplacian = spectral_ops.compute_laplacian(field_7d)
        
        assert laplacian.shape == field_7d.shape
        assert np.allclose(laplacian, 6.0, atol=1e-10)

    def test_spectral_ops_compute_gradient(self, spectral_ops):
        """Test spectral gradient computation."""
        # Create test field
        field = np.random.random(spectral_ops.domain.shape)
        
        # Compute gradient
        gradient = spectral_ops.compute_gradient(field)
        
        assert gradient.shape == (3,) + field.shape
        assert isinstance(gradient, np.ndarray)

    def test_spectral_ops_compute_divergence(self, spectral_ops):
        """Test spectral divergence computation."""
        # Create test vector field
        field_x = np.random.random(spectral_ops.domain.shape)
        field_y = np.random.random(spectral_ops.domain.shape)
        field_z = np.random.random(spectral_ops.domain.shape)
        vector_field = np.stack([field_x, field_y, field_z], axis=0)
        
        # Compute divergence
        divergence = spectral_ops.compute_divergence(vector_field)
        
        assert divergence.shape == spectral_ops.domain.shape
        assert isinstance(divergence, np.ndarray)

    def test_spectral_ops_compute_curl(self, spectral_ops):
        """Test spectral curl computation."""
        # Create test vector field
        field_x = np.random.random(spectral_ops.domain.shape)
        field_y = np.random.random(spectral_ops.domain.shape)
        field_z = np.random.random(spectral_ops.domain.shape)
        vector_field = np.stack([field_x, field_y, field_z], axis=0)
        
        # Compute curl
        curl = spectral_ops.compute_curl(vector_field)
        
        assert curl.shape == (3,) + spectral_ops.domain.shape
        assert isinstance(curl, np.ndarray)

    def test_spectral_ops_validation(self, spectral_ops):
        """Test spectral operations validation."""
        with pytest.raises(ValueError):
            spectral_ops.compute_derivative(np.zeros((4, 4, 4)), order=1, axis=0)  # Wrong shape
        
        with pytest.raises(ValueError):
            spectral_ops.compute_derivative(np.zeros(spectral_ops.domain.shape), order=1, axis=3)  # Invalid axis


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
    def spectral_derivs(self, domain, fft_backend):
        """Create spectral derivatives for testing."""
        return SpectralDerivatives(domain, fft_backend)

    def test_spectral_derivs_initialization(self, spectral_derivs, domain, fft_backend):
        """Test spectral derivatives initialization."""
        assert spectral_derivs.domain == domain
        assert spectral_derivs.fft_backend == fft_backend

    def test_spectral_derivs_first_derivative(self, spectral_derivs):
        """Test first derivative computation."""
        # Create test field
        field = np.random.random(spectral_derivs.domain.shape)
        
        # Compute first derivative
        derivative = spectral_derivs.compute_first_derivative(field, axis=0)
        
        assert derivative.shape == field.shape
        assert isinstance(derivative, np.ndarray)

    def test_spectral_derivs_second_derivative(self, spectral_derivs):
        """Test second derivative computation."""
        # Create test field
        field = np.random.random(spectral_derivs.domain.shape)
        
        # Compute second derivative
        derivative = spectral_derivs.compute_second_derivative(field, axis=0)
        
        assert derivative.shape == field.shape
        assert isinstance(derivative, np.ndarray)

    def test_spectral_derivs_nth_derivative(self, spectral_derivs):
        """Test nth derivative computation."""
        # Create test field
        field = np.random.random(spectral_derivs.domain.shape)
        
        # Compute nth derivative
        derivative = spectral_derivs.compute_nth_derivative(field, order=3, axis=0)
        
        assert derivative.shape == field.shape
        assert isinstance(derivative, np.ndarray)

    def test_spectral_derivs_mixed_derivative(self, spectral_derivs):
        """Test mixed derivative computation."""
        # Create test field
        field = np.random.random(spectral_derivs.domain.shape)
        
        # Compute mixed derivative
        derivative = spectral_derivs.compute_mixed_derivative(field, orders=[1, 1, 0])
        
        assert derivative.shape == field.shape
        assert isinstance(derivative, np.ndarray)

    def test_spectral_derivs_validation(self, spectral_derivs):
        """Test spectral derivatives validation."""
        with pytest.raises(ValueError):
            spectral_derivs.compute_first_derivative(np.zeros((4, 4, 4)), axis=0)  # Wrong shape
        
        with pytest.raises(ValueError):
            spectral_derivs.compute_first_derivative(np.zeros(spectral_derivs.domain.shape), axis=3)  # Invalid axis


class TestSpectralFiltering:
    """Comprehensive tests for SpectralFiltering class."""

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
    def spectral_filtering(self, domain, fft_backend):
        """Create spectral filtering for testing."""
        return SpectralFiltering(domain, fft_backend)

    def test_spectral_filtering_initialization(self, spectral_filtering, domain, fft_backend):
        """Test spectral filtering initialization."""
        assert spectral_filtering.domain == domain
        assert spectral_filtering.fft_backend == fft_backend

    def test_spectral_filtering_low_pass(self, spectral_filtering):
        """Test low-pass filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.domain.shape)
        
        # Apply low-pass filter
        filtered_field = spectral_filtering.apply_low_pass_filter(field, cutoff=0.5)
        
        assert filtered_field.shape == field.shape
        assert isinstance(filtered_field, np.ndarray)

    def test_spectral_filtering_high_pass(self, spectral_filtering):
        """Test high-pass filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.domain.shape)
        
        # Apply high-pass filter
        filtered_field = spectral_filtering.apply_high_pass_filter(field, cutoff=0.5)
        
        assert filtered_field.shape == field.shape
        assert isinstance(filtered_field, np.ndarray)

    def test_spectral_filtering_band_pass(self, spectral_filtering):
        """Test band-pass filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.domain.shape)
        
        # Apply band-pass filter
        filtered_field = spectral_filtering.apply_band_pass_filter(field, low_cutoff=0.2, high_cutoff=0.8)
        
        assert filtered_field.shape == field.shape
        assert isinstance(filtered_field, np.ndarray)

    def test_spectral_filtering_gaussian(self, spectral_filtering):
        """Test Gaussian filtering."""
        # Create test field
        field = np.random.random(spectral_filtering.domain.shape)
        
        # Apply Gaussian filter
        filtered_field = spectral_filtering.apply_gaussian_filter(field, sigma=0.5)
        
        assert filtered_field.shape == field.shape
        assert isinstance(filtered_field, np.ndarray)

    def test_spectral_filtering_validation(self, spectral_filtering):
        """Test spectral filtering validation."""
        with pytest.raises(ValueError):
            spectral_filtering.apply_low_pass_filter(np.zeros((4, 4, 4)), cutoff=0.5)  # Wrong shape


class TestFFTPlanManager:
    """Comprehensive tests for FFTPlanManager class."""

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
    def plan_manager(self, domain):
        """Create FFT plan manager for testing."""
        return FFTPlanManager(domain)

    def test_plan_manager_initialization(self, plan_manager, domain):
        """Test FFT plan manager initialization."""
        assert plan_manager.domain == domain

    def test_plan_manager_create_plan(self, plan_manager):
        """Test FFT plan creation."""
        # Create test field
        field = np.random.random(plan_manager.domain.shape)
        
        # Create plan
        plan = plan_manager.create_plan(field)
        
        assert plan is not None

    def test_plan_manager_get_plan(self, plan_manager):
        """Test FFT plan retrieval."""
        # Create test field
        field = np.random.random(plan_manager.domain.shape)
        
        # Get plan
        plan = plan_manager.get_plan(field)
        
        assert plan is not None

    def test_plan_manager_clear_plans(self, plan_manager):
        """Test FFT plan clearing."""
        # Create test field
        field = np.random.random(plan_manager.domain.shape)
        
        # Create plan
        plan_manager.create_plan(field)
        
        # Clear plans
        plan_manager.clear_plans()
        
        # Should work without error
        assert True


class TestFFTButterflyComputer:
    """Comprehensive tests for FFTButterflyComputer class."""

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
    def butterfly_computer(self, domain):
        """Create FFT butterfly computer for testing."""
        return FFTButterflyComputer(domain)

    def test_butterfly_computer_initialization(self, butterfly_computer, domain):
        """Test FFT butterfly computer initialization."""
        assert butterfly_computer.domain == domain

    def test_butterfly_computer_compute_butterfly(self, butterfly_computer):
        """Test butterfly computation."""
        # Create test data
        data = np.random.random(butterfly_computer.domain.shape) + 1j * np.random.random(butterfly_computer.domain.shape)
        
        # Compute butterfly
        result = butterfly_computer.compute_butterfly(data)
        
        assert result.shape == data.shape
        assert isinstance(result, np.ndarray)

    def test_butterfly_computer_compute_inverse_butterfly(self, butterfly_computer):
        """Test inverse butterfly computation."""
        # Create test data
        data = np.random.random(butterfly_computer.domain.shape) + 1j * np.random.random(butterfly_computer.domain.shape)
        
        # Compute inverse butterfly
        result = butterfly_computer.compute_inverse_butterfly(data)
        
        assert result.shape == data.shape
        assert isinstance(result, np.ndarray)


class TestFFTTwiddleComputer:
    """Comprehensive tests for FFTTwiddleComputer class."""

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
    def twiddle_computer(self, domain):
        """Create FFT twiddle computer for testing."""
        return FFTTwiddleComputer(domain)

    def test_twiddle_computer_initialization(self, twiddle_computer, domain):
        """Test FFT twiddle computer initialization."""
        assert twiddle_computer.domain == domain

    def test_twiddle_computer_compute_twiddle_factors(self, twiddle_computer):
        """Test twiddle factors computation."""
        # Compute twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()
        
        assert isinstance(twiddle_factors, np.ndarray)
        assert np.iscomplexobj(twiddle_factors)

    def test_twiddle_computer_get_twiddle_factor(self, twiddle_computer):
        """Test individual twiddle factor retrieval."""
        # Get twiddle factor
        twiddle_factor = twiddle_computer.get_twiddle_factor(0, 1)
        
        assert isinstance(twiddle_factor, complex)

    def test_twiddle_computer_compute_inverse_twiddle_factors(self, twiddle_computer):
        """Test inverse twiddle factors computation."""
        # Compute inverse twiddle factors
        inverse_twiddle_factors = twiddle_computer.compute_inverse_twiddle_factors()
        
        assert isinstance(inverse_twiddle_factors, np.ndarray)
        assert np.iscomplexobj(inverse_twiddle_factors)
