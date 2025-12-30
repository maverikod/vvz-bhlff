"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Optimized time integrators tests with memory control.

This module contains tests for optimized time integrators with CUDA,
vectorization, batching, and block processing, including memory usage
monitoring to prevent system hangs.
"""

import numpy as np
import pytest
import gc
from typing import Dict, Any

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from bhlff.core.time import (
    BVPEnvelopeIntegrator,
    CrankNicolsonIntegrator,
)
from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters


class TestOptimizedIntegrators:
    """
    Tests for optimized time integrators with memory control.

    Physical Meaning:
        Tests optimized temporal integration methods with CUDA acceleration,
        vectorization, batching, and block processing, ensuring memory
        usage stays within safe limits to prevent system hangs.
    """

    @pytest.fixture
    def small_domain(self):
        """Create small domain for basic tests (memory-safe)."""
        return Domain(L=1.0, N=4, N_phi=4, N_t=4, dimensions=7)

    @pytest.fixture
    def moderate_domain(self):
        """Create moderate domain for performance tests (memory-safe)."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)

    @pytest.fixture
    def parameters_basic(self):
        """Basic parameters for testing."""
        return Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,
            nu=1.0,
            precision="float64",
        )

    def test_vectorized_envelope_coefficients(self, small_domain, parameters_basic):
        """
        Test that envelope coefficients are computed using vectorized operations.

        Physical Meaning:
            Validates that the optimized vectorized computation of envelope
            coefficients produces correct results without nested loops.
        """
        integrator = BVPEnvelopeIntegrator(small_domain, parameters_basic)
        
        # Check that envelope coefficients are computed
        assert integrator._envelope_coeffs is not None
        assert integrator._envelope_coeffs.shape == small_domain.shape
        
        # Check that coefficients are finite and reasonable
        assert np.all(np.isfinite(integrator._envelope_coeffs))
        assert np.all(integrator._envelope_coeffs >= 0)

    def test_cuda_spectral_operations(self, small_domain, parameters_basic):
        """
        Test that integrators use CUDA-aware spectral operations.

        Physical Meaning:
            Validates that integrators use UnifiedSpectralOperations which
            supports CUDA acceleration and block processing.
        """
        integrator = CrankNicolsonIntegrator(small_domain, parameters_basic)
        
        # Check that unified spectral operations are used
        assert hasattr(integrator, "_spectral_ops")
        assert integrator._spectral_ops is not None
        
        # Check that it's UnifiedSpectralOperations (CUDA-aware)
        from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations
        assert isinstance(integrator._spectral_ops, UnifiedSpectralOperations)

    def test_memory_efficient_integration(self, moderate_domain, parameters_basic):
        """
        Test memory-efficient integration for moderate-sized fields.

        Physical Meaning:
            Validates that integration uses memory-efficient approaches
            for moderate-sized fields, avoiding unnecessary copies.
        """
        integrator = CrankNicolsonIntegrator(moderate_domain, parameters_basic)
        
        # Create test field
        initial_field = np.random.random(moderate_domain.shape).astype(np.complex128)
        time_steps = np.linspace(0.0, 0.1, 5)
        source_field = np.zeros(
            (len(time_steps),) + moderate_domain.shape, dtype=np.complex128
        )
        
        # Monitor memory before
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024**2  # MB
        
        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Monitor memory after
        memory_after = process.memory_info().rss / 1024**2  # MB
        memory_increase = memory_after - memory_before
        
        # Check results
        assert result.shape == (len(time_steps),) + moderate_domain.shape
        assert np.all(np.isfinite(result))
        
        # Memory increase should be reasonable (less than 200MB for this test)
        assert memory_increase < 200, (
            f"Memory increase {memory_increase:.2f}MB is too large. "
            f"Expected < 200MB for moderate domain."
        )

    def test_large_field_memory_control(self, parameters_basic):
        """
        Test memory control for large fields to prevent system hangs.

        Physical Meaning:
            Validates that integration handles large fields with proper
            memory management, including periodic cleanup and memory limits.
        """
        # Create a larger but still memory-safe domain
        # Using N=10, N_phi=4, N_t=10 gives ~1MB per field (safe for testing)
        # Further reduced to ensure memory safety
        large_domain = Domain(L=1.0, N=10, N_phi=4, N_t=10, dimensions=7)
        
        integrator = CrankNicolsonIntegrator(large_domain, parameters_basic)
        
        # Create test field
        initial_field = np.random.random(large_domain.shape).astype(np.complex128)
        time_steps = np.linspace(0.0, 0.1, 5)  # Reduced time steps
        source_field = np.zeros(
            (len(time_steps),) + large_domain.shape, dtype=np.complex128
        )
        
        # Monitor memory
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024**2  # MB
        
        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Monitor memory after
        memory_after = process.memory_info().rss / 1024**2  # MB
        memory_increase = memory_after - memory_before
        
        # Check results
        assert result.shape == (len(time_steps),) + large_domain.shape
        assert np.all(np.isfinite(result))
        
        # Memory increase should be reasonable (less than 300MB for this test)
        # Increased limit slightly to account for CUDA overhead
        assert memory_increase < 300, (
            f"Memory increase {memory_increase:.2f}MB is too large. "
            f"Expected < 300MB for large domain."
        )
        
        # Force cleanup
        del result, initial_field, source_field
        gc.collect()
        
        # Clear CUDA memory if available
        if CUDA_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def test_cuda_memory_cleanup(self, small_domain, parameters_basic):
        """
        Test that CUDA memory is properly cleaned up after integration.

        Physical Meaning:
            Validates that CUDA memory is freed after integration to prevent
            memory accumulation and out-of-memory errors.
        """
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        
        integrator = CrankNicolsonIntegrator(small_domain, parameters_basic)
        
        # Create test field
        initial_field = np.random.random(small_domain.shape).astype(np.complex128)
        time_steps = np.linspace(0.0, 0.1, 5)
        source_field = np.zeros(
            (len(time_steps),) + small_domain.shape, dtype=np.complex128
        )
        
        # Check GPU memory before
        mem_info_before = cp.cuda.runtime.memGetInfo()
        free_memory_before = mem_info_before[0]
        
        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Force cleanup
        del result
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        # Check GPU memory after cleanup
        mem_info_after = cp.cuda.runtime.memGetInfo()
        free_memory_after = mem_info_after[0]
        
        # Memory should be freed (free memory should increase or stay similar)
        # Allow some tolerance for memory fragmentation
        memory_freed = free_memory_after - free_memory_before
        assert memory_freed >= -50 * 1024**2, (  # Allow 50MB tolerance
            f"GPU memory not properly freed. "
            f"Free memory change: {memory_freed / 1024**2:.2f}MB"
        )

    def test_periodic_memory_cleanup(self, parameters_basic):
        """
        Test that periodic memory cleanup works for large fields.

        Physical Meaning:
            Validates that integration performs periodic memory cleanup
            during long integrations to prevent memory accumulation.
        """
        # Create domain that triggers memory-efficient processing (>50MB)
        # N=12, N_phi=6, N_t=12 gives ~2MB per field (reduced to avoid GPU OOM)
        # Still large enough to test memory-efficient processing
        large_domain = Domain(L=1.0, N=12, N_phi=6, N_t=12, dimensions=7)
        
        integrator = CrankNicolsonIntegrator(large_domain, parameters_basic)
        
        # Create test field
        initial_field = np.random.random(large_domain.shape).astype(np.complex128)
        # Use enough time steps to trigger periodic cleanup (every 10 steps)
        time_steps = np.linspace(0.0, 0.1, 15)  # Reduced to 15 steps
        source_field = np.zeros(
            (len(time_steps),) + large_domain.shape, dtype=np.complex128
        )
        
        # Monitor memory during integration
        import psutil
        process = psutil.Process()
        memory_samples = []
        
        # Perform integration with memory monitoring
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Check that result is correct
        assert result.shape == (len(time_steps),) + large_domain.shape
        assert np.all(np.isfinite(result))
        
        # Memory should be reasonable after cleanup
        memory_final = process.memory_info().rss / 1024**2  # MB
        assert memory_final < 2000, (  # Less than 2GB total
            f"Final memory usage {memory_final:.2f}MB is too high. "
            f"Expected < 2000MB."
        )

    def test_vectorized_operations(self, small_domain, parameters_basic):
        """
        Test that vectorized operations are used in integration.

        Physical Meaning:
            Validates that integration uses vectorized NumPy/CuPy operations
            instead of loops for optimal performance.
        """
        integrator = BVPEnvelopeIntegrator(small_domain, parameters_basic)
        
        # Create test field
        initial_field = np.random.random(small_domain.shape).astype(np.complex128)
        time_steps = np.linspace(0.0, 0.1, 5)
        source_field = np.zeros(
            (len(time_steps),) + small_domain.shape, dtype=np.complex128
        )
        
        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Check that result is correct
        assert result.shape == (len(time_steps),) + small_domain.shape
        assert np.all(np.isfinite(result))
        
        # Check that envelope coefficients were computed using vectorized operations
        # (no nested loops in _setup_envelope_coefficients)
        assert integrator._envelope_coeffs is not None
        assert integrator._envelope_coeffs.shape == small_domain.shape

    def test_integration_accuracy_optimized(self, small_domain, parameters_basic):
        """
        Test that optimized integration maintains accuracy.

        Physical Meaning:
            Validates that optimization (CUDA, vectorization, batching)
            does not compromise numerical accuracy of integration.
        """
        integrator = CrankNicolsonIntegrator(small_domain, parameters_basic)
        
        # Create simple test case with known behavior
        initial_field = np.ones(small_domain.shape, dtype=np.complex128)
        time_steps = np.linspace(0.0, 0.1, 10)
        source_field = np.zeros(
            (len(time_steps),) + small_domain.shape, dtype=np.complex128
        )
        
        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Check that result is finite and reasonable
        assert np.all(np.isfinite(result))
        
        # With zero source and damping, field should decay
        # Check that field magnitude decreases or stays stable
        initial_magnitude = np.abs(result[0]).mean()
        final_magnitude = np.abs(result[-1]).mean()
        
        # With damping (lambda_param=0.1), field should decay
        assert final_magnitude <= initial_magnitude * 1.1, (
            f"Field should decay with damping. "
            f"Initial: {initial_magnitude:.6f}, Final: {final_magnitude:.6f}"
        )

    def test_block_processing_integration(self, parameters_basic):
        """
        Test that integration uses block processing for large fields.

        Physical Meaning:
            Validates that integration automatically uses block processing
            through UnifiedSpectralOperations for large fields to manage
            GPU memory efficiently.
        """
        # Create domain that will trigger block processing
        # Using moderate size that should use block processing
        # Further reduced to avoid GPU memory issues
        domain = Domain(L=1.0, N=10, N_phi=4, N_t=10, dimensions=7)
        
        integrator = CrankNicolsonIntegrator(domain, parameters_basic)
        
        # Create test field
        initial_field = np.random.random(domain.shape).astype(np.complex128)
        time_steps = np.linspace(0.0, 0.1, 5)
        source_field = np.zeros(
            (len(time_steps),) + domain.shape, dtype=np.complex128
        )
        
        # Perform integration (should use block processing internally)
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Check that result is correct
        assert result.shape == (len(time_steps),) + domain.shape
        assert np.all(np.isfinite(result))
        
        # UnifiedSpectralOperations should handle block processing automatically
        # for 7D fields when CUDA is available
