"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for FFTButterflyComputer class.

This module provides comprehensive unit tests for the FFTButterflyComputer class,
covering butterfly operations and computations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.fft.fft_backend_core import FFTBackend
from bhlff.core.fft.fft_butterfly_computer import FFTButterflyComputer


class TestFFTButterflyComputer:
    """Comprehensive tests for FFTButterflyComputer class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(L=1.0, N=8, dimensions=7, N_phi=4, N_t=8, T=1.0)

    @pytest.fixture
    def fft_backend(self, domain):
        """Create FFT backend for testing."""
        return FFTBackend(domain)

    @pytest.fixture
    def butterfly_computer(self, fft_backend):
        """Create FFT butterfly computer for testing."""
        return FFTButterflyComputer(fft_backend)

    def test_butterfly_computer_initialization(self, butterfly_computer, fft_backend):
        """Test FFT butterfly computer initialization."""
        assert butterfly_computer.fft_backend == fft_backend

    def test_butterfly_computer_compute_butterfly(self, butterfly_computer):
        """Test butterfly computation."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Compute butterfly
        result = butterfly_computer.compute_butterfly(data)

        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape

    def test_butterfly_computer_compute_inverse_butterfly(self, butterfly_computer):
        """Test inverse butterfly computation."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Compute inverse butterfly
        result = butterfly_computer.compute_inverse_butterfly(data)

        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape

    def test_butterfly_computer_butterfly_round_trip(self, butterfly_computer):
        """Test butterfly round-trip computation."""
        # Create test data
        original_data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Forward then inverse butterfly
        butterfly_data = butterfly_computer.compute_butterfly(original_data)
        reconstructed_data = butterfly_computer.compute_inverse_butterfly(
            butterfly_data
        )

        # Should be close to original (within numerical precision)
        assert np.allclose(original_data, reconstructed_data, atol=1e-10)

    def test_butterfly_computer_butterfly_energy_conservation(self, butterfly_computer):
        """Test energy conservation in butterfly operations."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Compute butterfly
        butterfly_data = butterfly_computer.compute_butterfly(data)

        # Energy should be conserved
        original_energy = np.sum(data**2)
        butterfly_energy = np.sum(butterfly_data**2)

        assert np.allclose(original_energy, butterfly_energy, atol=1e-10)

    def test_butterfly_computer_butterfly_validation(self, butterfly_computer):
        """Test input validation for butterfly operations."""
        # Test with wrong shape
        wrong_data = np.random.random((4, 4, 4))

        with pytest.raises(ValueError):
            butterfly_computer.compute_butterfly(wrong_data)

        with pytest.raises(ValueError):
            butterfly_computer.compute_inverse_butterfly(wrong_data)

    def test_butterfly_computer_butterfly_7d_structure(self, butterfly_computer):
        """Test 7D structure preservation in butterfly operations."""
        # Create 7D test data
        data = np.zeros(butterfly_computer.fft_backend.domain.shape)
        data[0, 0, 0, 0, 0, 0, 0] = 1.0

        # Compute butterfly
        butterfly_data = butterfly_computer.compute_butterfly(data)

        # Should preserve 7D structure
        assert butterfly_data.shape == butterfly_computer.fft_backend.domain.shape

        # Compute inverse
        reconstructed_data = butterfly_computer.compute_inverse_butterfly(
            butterfly_data
        )

        # Should reconstruct original structure
        assert np.allclose(data, reconstructed_data, atol=1e-10)

    def test_butterfly_computer_butterfly_numerical_stability(self, butterfly_computer):
        """Test numerical stability of butterfly operations."""
        # Test with extreme values
        data = np.array([1e10, -1e10, 1e-10, -1e-10])
        data = np.broadcast_to(
            data.reshape(-1, 1, 1, 1, 1, 1, 1),
            butterfly_computer.fft_backend.domain.shape,
        )

        # Should not raise errors
        butterfly_data = butterfly_computer.compute_butterfly(data)

        # Should be stable
        assert np.isfinite(butterfly_data).all()

    def test_butterfly_computer_butterfly_precision(self, butterfly_computer):
        """Test precision of butterfly operations."""
        # Test with known function
        x = np.linspace(
            0, 2 * np.pi, butterfly_computer.fft_backend.domain.shape[0], endpoint=False
        )
        data = np.sin(x)
        data = np.broadcast_to(
            data.reshape(-1, 1, 1, 1, 1, 1, 1),
            butterfly_computer.fft_backend.domain.shape,
        )

        # Compute butterfly
        butterfly_data = butterfly_computer.compute_butterfly(data)

        # Should be finite and reasonable
        assert np.isfinite(butterfly_data).all()
        assert np.max(np.abs(butterfly_data)) < 10.0  # Reasonable bound

    def test_butterfly_computer_butterfly_performance(self, butterfly_computer):
        """Test performance of butterfly operations."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Measure performance
        start_time = time.time()
        butterfly_data = butterfly_computer.compute_butterfly(data)
        end_time = time.time()

        # Should be reasonable performance
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should be fast for small domain

    def test_butterfly_computer_butterfly_memory(self, butterfly_computer):
        """Test memory usage of butterfly operations."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Compute butterfly
        butterfly_data = butterfly_computer.compute_butterfly(data)

        # Should not use excessive memory
        assert butterfly_data.nbytes <= data.nbytes * 2  # Reasonable memory usage

    def test_butterfly_computer_butterfly_statistics(self, butterfly_computer):
        """Test butterfly operation statistics."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Get statistics
        stats = butterfly_computer.get_butterfly_statistics(data)

        assert isinstance(stats, dict)
        assert "input_energy" in stats
        assert "output_energy" in stats
        assert "energy_conservation" in stats

    def test_butterfly_computer_butterfly_optimization(self, butterfly_computer):
        """Test butterfly operation optimization."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Compute optimized butterfly
        optimized_result = butterfly_computer.compute_optimized_butterfly(data)

        assert isinstance(optimized_result, np.ndarray)
        assert optimized_result.shape == data.shape

    def test_butterfly_computer_butterfly_parallel(self, butterfly_computer):
        """Test parallel butterfly computation."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Compute parallel butterfly
        parallel_result = butterfly_computer.compute_parallel_butterfly(data)

        assert isinstance(parallel_result, np.ndarray)
        assert parallel_result.shape == data.shape

    def test_butterfly_computer_butterfly_vectorized(self, butterfly_computer):
        """Test vectorized butterfly computation."""
        # Create test data
        data = np.random.random(butterfly_computer.fft_backend.domain.shape)

        # Compute vectorized butterfly
        vectorized_result = butterfly_computer.compute_vectorized_butterfly(data)

        assert isinstance(vectorized_result, np.ndarray)
        assert vectorized_result.shape == data.shape

    def test_butterfly_computer_butterfly_error_handling(self, butterfly_computer):
        """Test error handling in butterfly operations."""
        # Test with None input
        with pytest.raises(ValueError):
            butterfly_computer.compute_butterfly(None)

        # Test with empty array
        empty_data = np.array([])

        with pytest.raises(ValueError):
            butterfly_computer.compute_butterfly(empty_data)

    def test_butterfly_computer_butterfly_edge_cases(self, butterfly_computer):
        """Test edge cases in butterfly operations."""
        # Test with single element
        single_data = np.array([1.0])
        single_data = np.broadcast_to(
            single_data.reshape(-1, 1, 1, 1, 1, 1, 1),
            butterfly_computer.fft_backend.domain.shape,
        )

        butterfly_data = butterfly_computer.compute_butterfly(single_data)

        assert isinstance(butterfly_data, np.ndarray)
        assert butterfly_data.shape == single_data.shape

        # Test with all zeros
        zero_data = np.zeros(butterfly_computer.fft_backend.domain.shape)

        butterfly_data = butterfly_computer.compute_butterfly(zero_data)

        assert isinstance(butterfly_data, np.ndarray)
        assert butterfly_data.shape == zero_data.shape
        assert np.allclose(butterfly_data, zero_data, atol=1e-10)

    def test_butterfly_computer_butterfly_complex_data(self, butterfly_computer):
        """Test butterfly operations with complex data."""
        # Create complex test data
        real_data = np.random.random(butterfly_computer.fft_backend.domain.shape)
        imag_data = np.random.random(butterfly_computer.fft_backend.domain.shape)
        complex_data = real_data + 1j * imag_data

        # Compute butterfly
        butterfly_data = butterfly_computer.compute_butterfly(complex_data)

        assert isinstance(butterfly_data, np.ndarray)
        assert butterfly_data.shape == complex_data.shape
        assert np.iscomplexobj(butterfly_data)
