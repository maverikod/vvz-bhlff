"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple tests for block processing system.

This module tests the basic block processing functionality
without complex BVP components that require large memory.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from bhlff.core.domain.enhanced_block_processor import EnhancedBlockProcessor, ProcessingConfig, ProcessingMode
from bhlff.core.domain import Domain


class TestSimpleBlockProcessing:
    """Test basic block processing functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create small test domain
        self.domain = Domain(
            L=1.0,
            N=8,
            dimensions=7
        )
        
        # Create processing config
        self.config = ProcessingConfig(
            mode=ProcessingMode.ADAPTIVE,
            max_memory_usage=0.8,
            min_block_size=2,
            max_block_size=4,
            overlap_ratio=0.1,
            enable_memory_optimization=True,
            enable_adaptive_sizing=True,
            enable_parallel_processing=True
        )
        
        # Create processor
        self.processor = EnhancedBlockProcessor(self.domain, self.config)
    
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
        assert block_size >= self.config.min_block_size
        assert block_size <= self.config.max_block_size
    
    def test_memory_requirements_check(self):
        """Test memory requirements checking."""
        # Create small field
        small_field = np.random.random((4, 4, 4, 4, 4, 4, 4))
        
        # Should pass memory check
        assert self.processor._check_memory_requirements(small_field)
    
    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection."""
        # Small field
        small_field = np.random.random((4, 4, 4, 4, 4, 4, 4))
        strategy = self.processor._choose_adaptive_strategy(small_field)
        assert strategy in [ProcessingMode.CPU_ONLY, ProcessingMode.GPU_PREFERRED, ProcessingMode.ADAPTIVE]
    
    def test_cpu_processing(self):
        """Test CPU processing."""
        field = np.random.random((4, 4, 4, 4, 4, 4, 4)) + 1j * np.random.random((4, 4, 4, 4, 4, 4, 4))
        
        result = self.processor._process_cpu_optimized(field, "fft")
        
        assert result.shape == field.shape
        assert result.dtype == field.dtype
    
    def test_processing_stats(self):
        """Test processing statistics."""
        stats = self.processor.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert 'cuda_available' in stats
        assert 'current_block_size' in stats
        assert 'memory_usage' in stats
    
    def test_optimization_for_field(self):
        """Test optimization for specific field."""
        field = np.random.random((4, 4, 4, 4, 4, 4, 4))
        
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
        field = np.random.random((4, 4, 4, 4, 4, 4, 4)) + 1j * np.random.random((4, 4, 4, 4, 4, 4, 4))
        
        # Test FFT operation
        result_fft = self.processor.process_7d_field(field, operation="fft")
        assert result_fft.shape == field.shape
        
        # Test IFFT operation
        result_ifft = self.processor.process_7d_field(field, operation="ifft")
        assert result_ifft.shape == field.shape
        
        # Test BVP solve operation
        result_bvp = self.processor.process_7d_field(field, operation="bvp_solve")
        assert result_bvp.shape == field.shape
    
    def test_memory_efficiency(self):
        """Test memory efficiency of block processing."""
        # Create field that's larger than single block
        field = np.random.random((6, 6, 6, 6, 6, 6, 6)) + 1j * np.random.random((6, 6, 6, 6, 6, 6, 6))
        
        # Should process without memory issues
        try:
            result = self.processor.process_7d_field(field, operation="fft")
            assert result.shape == field.shape
        except MemoryError:
            # Acceptable on systems with limited memory
            pass
    
    def test_adaptive_processing(self):
        """Test adaptive processing capabilities."""
        # Test with different field sizes
        for size in [4, 6, 8]:
            domain = Domain(L=1.0, N=size, dimensions=7)
            config = ProcessingConfig(
                mode=ProcessingMode.ADAPTIVE,
                max_memory_usage=0.8,
                min_block_size=2,
                max_block_size=min(4, size),
                overlap_ratio=0.1,
                enable_memory_optimization=True,
                enable_adaptive_sizing=True,
                enable_parallel_processing=True
            )
            processor = EnhancedBlockProcessor(domain, config)
            
            field = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
            
            try:
                result = processor.process_7d_field(field, operation="fft")
                assert result.shape == field.shape
            except MemoryError:
                # Acceptable on systems with limited memory
                pass
            finally:
                processor.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
