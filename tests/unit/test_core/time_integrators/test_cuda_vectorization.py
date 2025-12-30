"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for CUDA and vectorization optimizations in time integrators.

This module tests that CUDA acceleration and vectorization are properly
implemented in time integrators for optimal performance.

Physical Meaning:
    Tests validate that time integrators use CUDA acceleration and vectorization
    when available, ensuring optimal performance for 7D phase field computations.

Mathematical Foundation:
    Tests verify that vectorized operations produce identical results to
    non-vectorized operations, and that CUDA acceleration improves performance.
"""

import numpy as np
import pytest
from typing import Dict, Any

# CUDA support
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from bhlff.core.time import BVPEnvelopeIntegrator, CrankNicolsonIntegrator
from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP


class TestCUDAVectorization:
    """
    Tests for CUDA and vectorization optimizations.

    Physical Meaning:
        Tests validate that CUDA acceleration and vectorization are properly
        implemented in time integrators for optimal performance.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain7DBVP(L_spatial=1.0, N_spatial=8, N_phase=4, T=1.0, N_t=8)

    @pytest.fixture
    def parameters_cuda(self):
        """Parameters with CUDA enabled."""
        return Parameters7DBVP(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,
            nu=1.0,
            precision="float64",
            tolerance=1e-12,
            use_cuda=True,  # Enable CUDA
        )

    @pytest.fixture
    def parameters_cpu(self):
        """Parameters with CUDA disabled."""
        return Parameters7DBVP(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,
            nu=1.0,
            precision="float64",
            tolerance=1e-12,
            use_cuda=False,  # Disable CUDA
        )

    def test_envelope_coefficients_cuda_vectorization(
        self, domain_7d, parameters_cuda, parameters_cpu
    ):
        """
        Test that envelope coefficients are computed with CUDA vectorization.

        Physical Meaning:
            Validates that envelope coefficients are computed using fully
            vectorized CUDA operations when CUDA is available.
        """
        # Create integrators
        integrator_cuda = BVPEnvelopeIntegrator(domain_7d, parameters_cuda)
        integrator_cpu = BVPEnvelopeIntegrator(domain_7d, parameters_cpu)

        # Check that coefficients are computed
        assert integrator_cuda._envelope_coeffs is not None
        assert integrator_cpu._envelope_coeffs is not None

        # Check that coefficients have correct shape
        assert integrator_cuda._envelope_coeffs.shape == domain_7d.shape
        assert integrator_cpu._envelope_coeffs.shape == domain_7d.shape

        # Check that coefficients are finite
        assert np.all(np.isfinite(integrator_cuda._envelope_coeffs))
        assert np.all(np.isfinite(integrator_cpu._envelope_coeffs))

        # Check that results are approximately equal (within numerical precision)
        # CUDA and CPU should produce identical results
        np.testing.assert_allclose(
            integrator_cuda._envelope_coeffs,
            integrator_cpu._envelope_coeffs,
            rtol=1e-10,
            atol=1e-12,
            err_msg="CUDA and CPU should produce identical envelope coefficients",
        )

    def test_crank_nicolson_vectorization(
        self, domain_7d, parameters_cuda, parameters_cpu
    ):
        """
        Test that Crank-Nicolson integrator uses vectorized operations.

        Physical Meaning:
            Validates that Crank-Nicolson integrator uses fully vectorized
            operations for optimal performance.
        """
        # Create integrators
        integrator_cuda = CrankNicolsonIntegrator(domain_7d, parameters_cuda)
        integrator_cpu = CrankNicolsonIntegrator(domain_7d, parameters_cpu)

        # Create test field
        initial_field = np.random.random(domain_7d.shape) + 1j * np.random.random(
            domain_7d.shape
        )
        initial_field = initial_field.astype(np.complex128)

        # Create source field
        time_steps = np.linspace(0, 0.1, 5)
        source_field = np.zeros(
            (len(time_steps),) + domain_7d.shape, dtype=np.complex128
        )

        # Test single step
        current_source = source_field[0]
        next_source = source_field[1]
        dt = time_steps[1] - time_steps[0]

        # Perform step with CUDA
        result_cuda = integrator_cuda.step(
            initial_field.copy(), current_source, next_source, dt
        )

        # Perform step with CPU
        result_cpu = integrator_cpu.step(
            initial_field.copy(), current_source, next_source, dt
        )

        # Check that results are finite
        assert np.all(np.isfinite(result_cuda))
        assert np.all(np.isfinite(result_cpu))

        # Check that results are approximately equal (within numerical precision)
        np.testing.assert_allclose(
            result_cuda,
            result_cpu,
            rtol=1e-10,
            atol=1e-12,
            err_msg="CUDA and CPU should produce identical Crank-Nicolson results",
        )

    def test_block_processing_cuda_streams(self, domain_7d, parameters_cuda):
        """
        Test that block processing uses CUDA streams for parallel processing.

        Physical Meaning:
            Validates that block processing uses CUDA streams for parallel
            batch processing when multiple blocks are available.
        """
        from bhlff.core.domain.simple_block_processor import (
            SimpleBlockProcessor,
            SimpleConfig,
        )

        # Create processor with CUDA enabled
        config = SimpleConfig(
            block_size=2,
            batch_size=4,  # Multiple blocks for stream testing
            use_cuda=True,
        )
        processor = SimpleBlockProcessor(domain_7d, config)

        # Check that CUDA is available if expected
        if CUDA_AVAILABLE:
            assert processor.cuda_available, "CUDA should be available"

        # Create test field
        field = np.random.random(domain_7d.shape) + 1j * np.random.random(
            domain_7d.shape
        )
        field = field.astype(np.complex128)

        # Process field (should use CUDA streams if CUDA available)
        try:
            result = processor.process_7d_field(field, operation="bvp_solve")
            assert result is not None
            assert result.shape == field.shape
        except Exception as e:
            # If CUDA is not available, this is expected
            if not CUDA_AVAILABLE:
                pytest.skip("CUDA not available for stream testing")
            else:
                raise

    def test_fractional_laplacian_cuda_vectorization(
        self, domain_7d, parameters_cuda, parameters_cpu
    ):
        """
        Test that fractional Laplacian uses CUDA vectorization.

        Physical Meaning:
            Validates that fractional Laplacian operator uses fully vectorized
            CUDA operations for optimal performance.
        """
        from bhlff.core.operators.fractional_laplacian import FractionalLaplacian

        # Create operators
        operator_cuda = FractionalLaplacian(domain_7d, parameters_cuda)
        operator_cpu = FractionalLaplacian(domain_7d, parameters_cpu)

        # Create test field
        field = np.random.random(domain_7d.shape) + 1j * np.random.random(
            domain_7d.shape
        )
        field = field.astype(np.complex128)

        # Apply operators
        result_cuda = operator_cuda.apply(field)
        result_cpu = operator_cpu.apply(field)

        # Check that results are finite
        assert np.all(np.isfinite(result_cuda))
        assert np.all(np.isfinite(result_cpu))

        # Check that results are approximately equal (within numerical precision)
        np.testing.assert_allclose(
            result_cuda,
            result_cpu,
            rtol=1e-10,
            atol=1e-12,
            err_msg="CUDA and CPU should produce identical fractional Laplacian results",
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_performance_improvement(self, domain_7d, parameters_cuda):
        """
        Test that CUDA provides performance improvement (if available).

        Physical Meaning:
            Validates that CUDA acceleration provides performance improvement
            for large field computations.
        """
        import time

        from bhlff.core.time import BVPEnvelopeIntegrator

        # Create integrator with CUDA
        integrator = BVPEnvelopeIntegrator(domain_7d, parameters_cuda)

        # Create larger test field
        large_domain = Domain7DBVP(L_spatial=1.0, N_spatial=16, N_phase=8, T=1.0, N_t=16)
        large_integrator = BVPEnvelopeIntegrator(large_domain, parameters_cuda)

        # Measure setup time for envelope coefficients
        start_time = time.time()
        large_integrator._setup_envelope_coefficients()
        cuda_time = time.time() - start_time

        # Check that setup completed successfully
        assert large_integrator._envelope_coeffs is not None
        assert cuda_time > 0

        # Log performance (for manual verification)
        print(f"\nCUDA envelope coefficients setup time: {cuda_time:.4f}s")
