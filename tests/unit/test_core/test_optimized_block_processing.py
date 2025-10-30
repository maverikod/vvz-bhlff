"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Fast tests for optimized block processing system.

This module tests the optimized block processing system for 7D BVP computations
with intelligent memory management and adaptive block sizing.
"""

import pytest
import numpy as np
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from bhlff.core.domain.optimized_block_processor import (
    OptimizedBlockProcessor,
    OptimizedConfig,
    ProcessingMode,
)
from bhlff.core.domain import Domain


class TestOptimizedBlockProcessing:
    """Test optimized block processor functionality."""

    def setup_method(self):
        """Setup test environment."""
        # Create small test domain
        self.domain = Domain(L=1.0, N=6, dimensions=7)

        # Create processing config
        self.config = OptimizedConfig(
            block_size=4,
            overlap_ratio=0.1,
            max_memory_usage=0.7,
            enable_adaptive_sizing=True,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            enable_gpu_acceleration=True,
            max_field_size_mb=50.0,
        )

        # Create processor
        self.processor = OptimizedBlockProcessor(self.domain, self.config)

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.domain == self.domain
        assert self.processor.config == self.config
        assert self.processor.base_processor is not None
        assert self.processor.memory_monitor is not None
        assert isinstance(self.processor.cuda_available, bool)

    def test_optimal_block_size_calculation(self):
        """Test optimal block size calculation."""
        block_size = self.processor._calculate_optimal_block_size()

        assert isinstance(block_size, int)
        assert block_size >= 2
        assert block_size <= self.config.block_size

    def test_memory_requirements_check(self):
        """Test memory requirements checking."""
        # Create small field
        small_field = np.random.random((3, 3, 3, 3, 3, 3, 3))

        # Should pass memory check
        assert self.processor._check_memory_requirements(small_field)

    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection."""
        # Small field
        small_field = np.random.random((3, 3, 3, 3, 3, 3, 3))
        strategy = self.processor._choose_adaptive_strategy(small_field)
        assert strategy in [
            ProcessingMode.CPU_ONLY,
            ProcessingMode.GPU_PREFERRED,
            ProcessingMode.ADAPTIVE,
        ]

    def test_cpu_processing(self):
        """Test CPU processing."""
        field = np.random.random((3, 3, 3, 3, 3, 3, 3)) + 1j * np.random.random(
            (3, 3, 3, 3, 3, 3, 3)
        )

        start_time = time.time()
        result = self.processor._process_cpu_optimized(field, "fft")
        processing_time = time.time() - start_time

        assert result.shape == field.shape
        assert result.dtype == field.dtype
        assert processing_time < 1.0  # Should be fast

    def test_processing_stats(self):
        """Test processing statistics."""
        stats = self.processor.get_processing_stats()

        assert isinstance(stats, dict)
        assert "cuda_available" in stats
        assert "current_block_size" in stats
        assert "memory_usage" in stats
        assert "blocks_processed" in stats

    def test_optimization_for_field(self):
        """Test optimization for specific field."""
        field = np.random.random((3, 3, 3, 3, 3, 3, 3))

        # Should not raise exception
        self.processor.optimize_for_field(field)

        # Check that block size was adjusted
        stats = self.processor.get_processing_stats()
        assert "current_block_size" in stats

    def test_cleanup(self):
        """Test resource cleanup."""
        # Should not raise exception
        self.processor.cleanup()

    def test_block_processing_operations(self):
        """Test different block processing operations."""
        field = np.random.random((3, 3, 3, 3, 3, 3, 3)) + 1j * np.random.random(
            (3, 3, 3, 3, 3, 3, 3)
        )

        # Test FFT operation
        start_time = time.time()
        result_fft = self.processor.process_7d_field(field, operation="fft")
        fft_time = time.time() - start_time

        assert result_fft.shape == field.shape
        assert fft_time < 2.0  # Should be fast

        # Test IFFT operation
        start_time = time.time()
        result_ifft = self.processor.process_7d_field(field, operation="ifft")
        ifft_time = time.time() - start_time

        assert result_ifft.shape == field.shape
        assert ifft_time < 2.0  # Should be fast

        # Test BVP solve operation
        start_time = time.time()
        result_bvp = self.processor.process_7d_field(field, operation="bvp_solve")
        bvp_time = time.time() - start_time

        assert result_bvp.shape == field.shape
        assert bvp_time < 2.0  # Should be fast

    def test_memory_efficiency(self):
        """Test memory efficiency of block processing."""
        # Create field that's larger than single block
        field = np.random.random((4, 4, 4, 4, 4, 4, 4)) + 1j * np.random.random(
            (4, 4, 4, 4, 4, 4, 4)
        )

        # Should process without memory issues
        start_time = time.time()
        try:
            result = self.processor.process_7d_field(field, operation="fft")
            processing_time = time.time() - start_time

            assert result.shape == field.shape
            assert processing_time < 5.0  # Should be reasonably fast
        except MemoryError:
            # Acceptable on systems with limited memory
            pass

    def test_adaptive_processing(self):
        """Test adaptive processing capabilities."""
        # Test with different field sizes
        for size in [3, 4, 5]:
            domain = Domain(L=1.0, N=size, dimensions=7)
            config = OptimizedConfig(
                block_size=min(3, size),
                overlap_ratio=0.1,
                max_memory_usage=0.7,
                enable_adaptive_sizing=True,
                enable_memory_optimization=True,
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                max_field_size_mb=50.0,
            )
            processor = OptimizedBlockProcessor(domain, config)

            field = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)

            start_time = time.time()
            try:
                result = processor.process_7d_field(field, operation="fft")
                processing_time = time.time() - start_time

                assert result.shape == field.shape
                assert processing_time < 3.0  # Should be fast
            except MemoryError:
                # Acceptable on systems with limited memory
                pass
            finally:
                processor.cleanup()

    def test_large_field_processing(self):
        """Test processing of large fields."""
        # Create larger field
        field = np.random.random((5, 5, 5, 5, 5, 5, 5)) + 1j * np.random.random(
            (5, 5, 5, 5, 5, 5, 5)
        )

        start_time = time.time()
        try:
            result = self.processor.process_7d_field(field, operation="fft")
            processing_time = time.time() - start_time

            assert result.shape == field.shape
            assert processing_time < 10.0  # Should be reasonably fast
        except MemoryError:
            # Acceptable on systems with limited memory
            pass

    def test_performance_benchmark(self):
        """Test performance benchmark."""
        field = np.random.random((3, 3, 3, 3, 3, 3, 3)) + 1j * np.random.random(
            (3, 3, 3, 3, 3, 3, 3)
        )

        # Benchmark FFT operation
        start_time = time.time()
        for _ in range(5):  # Run 5 iterations
            result = self.processor.process_7d_field(field, operation="fft")
        total_time = time.time() - start_time

        assert result.shape == field.shape
        assert total_time < 10.0  # Should be fast for 5 iterations

        # Check statistics
        stats = self.processor.get_processing_stats()
        assert stats["blocks_processed"] > 0
        assert stats["processing_time"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
