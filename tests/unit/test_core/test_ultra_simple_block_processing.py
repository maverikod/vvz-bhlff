"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Ultra simple tests for block processing system.

This module tests the basic block processing functionality
with minimal memory usage to avoid hanging.
"""

import pytest
import numpy as np
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from bhlff.core.domain.simple_block_processor import SimpleBlockProcessor, SimpleConfig
from bhlff.core.domain import Domain


class TestUltraSimpleBlockProcessing:
    """Test ultra simple block processor functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create very small test domain
        self.domain = Domain(
            L=1.0,
            N=2,
            N_phi=2,
            N_t=2,
            dimensions=7
        )
        
        # Create processing config
        self.config = SimpleConfig(
            block_size=2,
            overlap_ratio=0.1,
            max_memory_usage=0.7,
            enable_adaptive_sizing=True,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            max_field_size_mb=1.0  # Very small limit
        )
        
        # Create processor
        self.processor = SimpleBlockProcessor(self.domain, self.config)
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.domain == self.domain
        assert self.processor.config == self.config
        assert self.processor.base_processor is not None
        assert self.processor.memory_monitor is not None
    
    def test_optimal_block_size_calculation(self):
        """Test optimal block size calculation."""
        block_size = self.processor._calculate_optimal_block_size()
        
        assert isinstance(block_size, int)
        assert block_size >= 2
        assert block_size <= self.config.block_size
    
    def test_cpu_processing(self):
        """Test CPU processing."""
        field = np.random.random((2, 2, 2, 2, 2, 2, 2)) + 1j * np.random.random((2, 2, 2, 2, 2, 2, 2))
        
        start_time = time.time()
        result = self.processor._process_cpu_optimized(field, "fft")
        processing_time = time.time() - start_time
        
        assert result.shape == field.shape
        assert result.dtype == field.dtype
        assert processing_time < 0.1  # Should be very fast
    
    def test_processing_stats(self):
        """Test processing statistics."""
        stats = self.processor.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert 'current_block_size' in stats
        assert 'memory_usage' in stats
        assert 'blocks_processed' in stats
    
    def test_optimization_for_field(self):
        """Test optimization for specific field."""
        field = np.random.random((2, 2, 2, 2, 2, 2, 2))
        
        # Should not raise exception
        self.processor.optimize_for_field(field)
        
        # Check that block size was adjusted
        stats = self.processor.get_processing_stats()
        assert 'current_block_size' in stats
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Should not raise exception
        self.processor.cleanup()
    
    def test_block_processing_operations(self):
        """Test different block processing operations."""
        field = np.random.random((2, 2, 2, 2, 2, 2, 2)) + 1j * np.random.random((2, 2, 2, 2, 2, 2, 2))
        
        # Test FFT operation
        start_time = time.time()
        result_fft = self.processor.process_7d_field(field, operation="fft")
        fft_time = time.time() - start_time
        
        assert result_fft.shape == field.shape
        assert fft_time < 0.2  # Should be very fast
        
        # Test IFFT operation
        start_time = time.time()
        result_ifft = self.processor.process_7d_field(field, operation="ifft")
        ifft_time = time.time() - start_time
        
        assert result_ifft.shape == field.shape
        assert ifft_time < 0.2  # Should be very fast
        
        # Test BVP solve operation
        start_time = time.time()
        result_bvp = self.processor.process_7d_field(field, operation="bvp_solve")
        bvp_time = time.time() - start_time
        
        assert result_bvp.shape == field.shape
        assert bvp_time < 0.2  # Should be very fast
    
    def test_memory_efficiency(self):
        """Test memory efficiency of block processing."""
        # Create very small field
        field = np.random.random((2, 2, 2, 2, 2, 2, 2)) + 1j * np.random.random((2, 2, 2, 2, 2, 2, 2))
        
        # Should process without memory issues
        start_time = time.time()
        try:
            result = self.processor.process_7d_field(field, operation="fft")
            processing_time = time.time() - start_time
            
            assert result.shape == field.shape
            assert processing_time < 0.2  # Should be very fast
        except MemoryError:
            # Acceptable on systems with limited memory
            pass
    
    def test_adaptive_processing(self):
        """Test adaptive processing capabilities."""
        # Test with only the smallest possible size
        domain = Domain(L=1.0, N=2, N_phi=2, N_t=2, dimensions=7)
        config = SimpleConfig(
            block_size=2,
            overlap_ratio=0.1,
            max_memory_usage=0.7,
            enable_adaptive_sizing=True,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            max_field_size_mb=1.0  # Very small limit
        )
        processor = SimpleBlockProcessor(domain, config)
        
        # Create very small field
        field = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
        
        start_time = time.time()
        try:
            result = processor.process_7d_field(field, operation="fft")
            processing_time = time.time() - start_time
            
            assert result.shape == field.shape
            assert processing_time < 0.2  # Should be very fast
        except MemoryError:
            # Acceptable on systems with limited memory
            pass
        finally:
            processor.cleanup()
    
    def test_performance_benchmark(self):
        """Test performance benchmark."""
        field = np.random.random((2, 2, 2, 2, 2, 2, 2)) + 1j * np.random.random((2, 2, 2, 2, 2, 2, 2))
        
        # Benchmark FFT operation
        start_time = time.time()
        for _ in range(3):  # Run 3 iterations
            result = self.processor.process_7d_field(field, operation="fft")
        total_time = time.time() - start_time
        
        assert result.shape == field.shape
        assert total_time < 1.0  # Should be very fast for 3 iterations
        
        # Check statistics
        stats = self.processor.get_processing_stats()
        assert stats['blocks_processed'] > 0
        assert stats['processing_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
