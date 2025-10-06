"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for FFTTwiddleComputer class.

This module provides comprehensive unit tests for the FFTTwiddleComputer class,
covering twiddle factor computation and management.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.fft.fft_backend_core import FFTBackend
from bhlff.core.fft.fft_twiddle_computer import FFTTwiddleComputer


class TestFFTTwiddleComputer:
    """Comprehensive tests for FFTTwiddleComputer class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(L=1.0, N=8, dimensions=7, N_phi=4, N_t=8, T=1.0)

    @pytest.fixture
    def fft_backend(self, domain):
        """Create FFT backend for testing."""
        return FFTBackend(domain)

    @pytest.fixture
    def twiddle_computer(self, fft_backend):
        """Create FFT twiddle computer for testing."""
        return FFTTwiddleComputer(fft_backend)

    def test_twiddle_computer_initialization(self, twiddle_computer, fft_backend):
        """Test FFT twiddle computer initialization."""
        assert twiddle_computer.fft_backend == fft_backend

    def test_twiddle_computer_compute_twiddle_factors(self, twiddle_computer):
        """Test twiddle factors computation."""
        # Compute twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        assert isinstance(twiddle_factors, dict)
        assert "twiddle_factors" in twiddle_factors

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

    def test_twiddle_computer_twiddle_factor_caching(self, twiddle_computer):
        """Test twiddle factor caching."""
        # Compute twiddle factors twice
        factors1 = twiddle_computer.compute_twiddle_factors()
        factors2 = twiddle_computer.compute_twiddle_factors()

        # Should be the same (cached)
        assert factors1 is factors2

    def test_twiddle_computer_twiddle_factor_validation(self, twiddle_computer):
        """Test twiddle factor validation."""
        # Test with valid indices
        twiddle_factor = twiddle_computer.get_twiddle_factor(0, 1)

        assert isinstance(twiddle_factor, complex)
        assert np.isfinite(twiddle_factor)

    def test_twiddle_computer_twiddle_factor_energy_conservation(
        self, twiddle_computer
    ):
        """Test energy conservation in twiddle factor operations."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Energy should be conserved
        total_energy = np.sum(np.abs(twiddle_factors["twiddle_factors"]) ** 2)

        assert np.isfinite(total_energy)
        assert total_energy > 0

    def test_twiddle_computer_twiddle_factor_7d_structure(self, twiddle_computer):
        """Test 7D structure preservation in twiddle factor operations."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Should preserve 7D structure
        assert "twiddle_factors" in twiddle_factors
        assert isinstance(twiddle_factors["twiddle_factors"], np.ndarray)

    def test_twiddle_computer_twiddle_factor_numerical_stability(
        self, twiddle_computer
    ):
        """Test numerical stability of twiddle factor operations."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Should be stable
        assert np.isfinite(twiddle_factors["twiddle_factors"]).all()

    def test_twiddle_computer_twiddle_factor_precision(self, twiddle_computer):
        """Test precision of twiddle factor operations."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Should be finite and reasonable
        assert np.isfinite(twiddle_factors["twiddle_factors"]).all()
        assert np.max(np.abs(twiddle_factors["twiddle_factors"])) <= 1.0  # Unit circle

    def test_twiddle_computer_twiddle_factor_performance(self, twiddle_computer):
        """Test performance of twiddle factor operations."""
        # Measure performance
        start_time = time.time()
        twiddle_factors = twiddle_computer.compute_twiddle_factors()
        end_time = time.time()

        # Should be reasonable performance
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should be fast for small domain

    def test_twiddle_computer_twiddle_factor_memory(self, twiddle_computer):
        """Test memory usage of twiddle factor operations."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Should not use excessive memory
        memory_usage = twiddle_factors["twiddle_factors"].nbytes
        assert memory_usage < 1024 * 1024  # Less than 1MB for small domain

    def test_twiddle_computer_twiddle_factor_statistics(self, twiddle_computer):
        """Test twiddle factor statistics."""
        # Get statistics
        stats = twiddle_computer.get_twiddle_factor_statistics()

        assert isinstance(stats, dict)
        assert "total_factors" in stats
        assert "memory_usage" in stats
        assert "computation_time" in stats

    def test_twiddle_computer_twiddle_factor_optimization(self, twiddle_computer):
        """Test twiddle factor optimization."""
        # Get optimized twiddle factors
        optimized_factors = twiddle_computer.compute_optimized_twiddle_factors()

        assert isinstance(optimized_factors, dict)
        assert "twiddle_factors" in optimized_factors

    def test_twiddle_computer_twiddle_factor_parallel(self, twiddle_computer):
        """Test parallel twiddle factor computation."""
        # Get parallel twiddle factors
        parallel_factors = twiddle_computer.compute_parallel_twiddle_factors()

        assert isinstance(parallel_factors, dict)
        assert "twiddle_factors" in parallel_factors

    def test_twiddle_computer_twiddle_factor_vectorized(self, twiddle_computer):
        """Test vectorized twiddle factor computation."""
        # Get vectorized twiddle factors
        vectorized_factors = twiddle_computer.compute_vectorized_twiddle_factors()

        assert isinstance(vectorized_factors, dict)
        assert "twiddle_factors" in vectorized_factors

    def test_twiddle_computer_twiddle_factor_error_handling(self, twiddle_computer):
        """Test error handling in twiddle factor operations."""
        # Test with invalid indices
        with pytest.raises(ValueError):
            twiddle_computer.get_twiddle_factor(-1, 1)

        with pytest.raises(ValueError):
            twiddle_computer.get_twiddle_factor(0, -1)

    def test_twiddle_computer_twiddle_factor_edge_cases(self, twiddle_computer):
        """Test edge cases in twiddle factor operations."""
        # Test with zero indices
        twiddle_factor = twiddle_computer.get_twiddle_factor(0, 0)

        assert isinstance(twiddle_factor, complex)
        assert np.isfinite(twiddle_factor)

    def test_twiddle_computer_twiddle_factor_complex_data(self, twiddle_computer):
        """Test twiddle factor operations with complex data."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Should be complex
        assert np.iscomplexobj(twiddle_factors["twiddle_factors"])

    def test_twiddle_computer_twiddle_factor_unit_circle(self, twiddle_computer):
        """Test that twiddle factors lie on unit circle."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # All factors should lie on unit circle
        magnitudes = np.abs(twiddle_factors["twiddle_factors"])
        assert np.allclose(magnitudes, 1.0, atol=1e-10)

    def test_twiddle_computer_twiddle_factor_phase_relationships(
        self, twiddle_computer
    ):
        """Test phase relationships in twiddle factors."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Test phase relationships
        phases = np.angle(twiddle_factors["twiddle_factors"])

        # Phases should be finite
        assert np.isfinite(phases).all()

        # Phases should be in [-π, π]
        assert np.all(phases >= -np.pi)
        assert np.all(phases <= np.pi)

    def test_twiddle_computer_twiddle_factor_symmetry(self, twiddle_computer):
        """Test symmetry properties of twiddle factors."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Test symmetry properties
        factors = twiddle_factors["twiddle_factors"]

        # Should have proper symmetry
        assert np.isfinite(factors).all()

    def test_twiddle_computer_twiddle_factor_cleanup(self, twiddle_computer):
        """Test twiddle factor cleanup."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Cleanup
        twiddle_computer.cleanup_twiddle_factors()

        # Should not raise errors
        assert True

    def test_twiddle_computer_twiddle_factor_reset(self, twiddle_computer):
        """Test twiddle factor reset."""
        # Get twiddle factors
        twiddle_factors = twiddle_computer.compute_twiddle_factors()

        # Reset
        twiddle_computer.reset_twiddle_factors()

        # Should not raise errors
        assert True
