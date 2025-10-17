"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for enhanced block processing system.

This module tests the enhanced block processing system for 7D BVP computations
with intelligent memory management and adaptive block sizing.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from bhlff.core.domain.enhanced_block_processor import EnhancedBlockProcessor, ProcessingConfig, ProcessingMode
from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_block_processing_system import BVPBlockProcessingSystem, BVPBlockConfig


class TestEnhancedBlockProcessor:
    """Test enhanced block processor functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create test domain
        self.domain = Domain(
            L=1.0,
            N=16,
            dimensions=7
        )
        
        # Create processing config
        self.config = ProcessingConfig(
            mode=ProcessingMode.ADAPTIVE,
            max_memory_usage=0.8,
            min_block_size=4,
            max_block_size=16,
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
        small_field = np.random.random((8, 8, 8, 8, 8, 8, 8))
        
        # Should pass memory check
        assert self.processor._check_memory_requirements(small_field)
        
        # Create large field (if memory allows)
        try:
            large_field = np.random.random((16, 16, 16, 16, 16, 16, 16))
            # This might fail on systems with limited memory
            result = self.processor._check_memory_requirements(large_field)
            assert isinstance(result, bool)
        except MemoryError:
            # Expected on systems with limited memory
            pass
    
    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection."""
        # Small field
        small_field = np.random.random((8, 8, 8, 8, 8, 8, 8))
        strategy = self.processor._choose_adaptive_strategy(small_field)
        assert strategy in [ProcessingMode.CPU_ONLY, ProcessingMode.GPU_PREFERRED, ProcessingMode.ADAPTIVE]
        
        # Medium field
        medium_field = np.random.random((12, 12, 12, 12, 12, 12, 12))
        strategy = self.processor._choose_adaptive_strategy(medium_field)
        assert strategy in [ProcessingMode.CPU_ONLY, ProcessingMode.GPU_PREFERRED, ProcessingMode.ADAPTIVE]
    
    def test_cpu_processing(self):
        """Test CPU processing."""
        field = np.random.random((8, 8, 8, 8, 8, 8, 8)) + 1j * np.random.random((8, 8, 8, 8, 8, 8, 8))
        
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
        field = np.random.random((12, 12, 12, 12, 12, 12, 12))
        
        # Should not raise exception
        self.processor.optimize_for_field(field)
        
        # Check that block size was adjusted
        stats = self.processor.get_processing_stats()
        assert 'current_block_size' in stats
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Should not raise exception
        self.processor.cleanup()


class TestBVPBlockProcessingSystem:
    """Test BVP block processing system."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create test domain
        self.domain = Domain(
            L=1.0,
            N=16,
            dimensions=7
        )
        
        # Create BVP config
        self.config = BVPBlockConfig(
            block_size=16,
            overlap_ratio=0.1,
            max_memory_usage=0.8,
            envelope_tolerance=1e-6,
            max_envelope_iterations=10,
            quench_detection_enabled=False,  # Disable for testing
            impedance_calculation_enabled=False,  # Disable for testing
            enable_adaptive_sizing=True,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            enable_gpu_acceleration=True
        )
        
        # Create BVP processor
        self.bvp_processor = BVPBlockProcessingSystem(self.domain, self.config)
    
    def test_initialization(self):
        """Test BVP processor initialization."""
        assert self.bvp_processor.domain == self.domain
        assert self.bvp_processor.config == self.config
        assert self.bvp_processor.block_processor is not None
        assert self.bvp_processor.memory_monitor is not None
    
    def test_envelope_solution(self):
        """Test BVP envelope solution."""
        # Create test source
        source = np.random.random(self.domain.shape) + 1j * np.random.random(self.domain.shape)
        
        # Solve envelope equation
        envelope = self.bvp_processor.solve_envelope_blocked(
            source, 
            max_iterations=5,  # Reduced for testing
            tolerance=1e-4
        )
        
        assert envelope.shape == source.shape
        assert envelope.dtype == source.dtype
    
    def test_block_extraction(self):
        """Test block extraction methods."""
        field = np.random.random(self.domain.shape)
        
        # Test source block extraction
        for block_data, block_info in self.bvp_processor.block_processor.base_processor.iterate_blocks():
            source_block = self.bvp_processor._extract_source_block(field, block_info)
            assert source_block.shape == block_data.shape
            
            envelope_block = self.bvp_processor._extract_envelope_block(field, block_info)
            assert envelope_block.shape == block_data.shape
    
    def test_block_bvp_solution(self):
        """Test BVP solution for a single block."""
        # Create test data
        envelope_block = np.random.random((4, 4, 4, 4, 4, 4, 4)) + 1j * np.random.random((4, 4, 4, 4, 4, 4, 4))
        source_block = np.random.random((4, 4, 4, 4, 4, 4, 4)) + 1j * np.random.random((4, 4, 4, 4, 4, 4, 4))
        
        # Create mock block info
        class MockBlockInfo:
            def __init__(self):
                self.start_indices = (0, 0, 0, 0, 0, 0, 0)
                self.end_indices = (4, 4, 4, 4, 4, 4, 4)
        
        block_info = MockBlockInfo()
        
        # Solve BVP for block
        solution = self.bvp_processor._solve_block_bvp(envelope_block, source_block, block_info)
        
        assert solution.shape == envelope_block.shape
        assert solution.dtype == envelope_block.dtype
    
    def test_convergence_check(self):
        """Test convergence checking."""
        # Test convergence
        old_envelope = np.ones((4, 4, 4, 4, 4, 4, 4))
        new_envelope = np.ones((4, 4, 4, 4, 4, 4, 4)) * 1.001  # Small change
        tolerance = 1e-3
        
        converged = self.bvp_processor._check_envelope_convergence(
            old_envelope, new_envelope, tolerance
        )
        assert converged
        
        # Test non-convergence
        new_envelope = np.ones((4, 4, 4, 4, 4, 4, 4)) * 2.0  # Large change
        converged = self.bvp_processor._check_envelope_convergence(
            old_envelope, new_envelope, tolerance
        )
        assert not converged
    
    def test_processing_stats(self):
        """Test processing statistics."""
        stats = self.bvp_processor.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert 'envelope_solves' in stats
        assert 'quench_detections' in stats
        assert 'impedance_calculations' in stats
        assert 'blocks_processed' in stats
        assert 'memory_usage' in stats
    
    def test_optimization_for_field(self):
        """Test optimization for specific field."""
        field = np.random.random((12, 12, 12, 12, 12, 12, 12))
        
        # Should not raise exception
        self.bvp_processor.optimize_for_field(field)
        
        # Check that settings were adjusted
        assert self.bvp_processor.config.envelope_tolerance is not None
        assert self.bvp_processor.config.max_envelope_iterations is not None
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Should not raise exception
        self.bvp_processor.cleanup()


class TestBlockProcessingIntegration:
    """Test integration of block processing components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.domain = Domain(L=1.0, N=16, dimensions=7)
        self.config = BVPBlockConfig(block_size=8)
        self.bvp_processor = BVPBlockProcessingSystem(self.domain, self.config)
    
    def test_end_to_end_processing(self):
        """Test end-to-end block processing."""
        # Create test source
        source = np.random.random(self.domain.shape) + 1j * np.random.random(self.domain.shape)
        
        # Process with block processing
        result = self.bvp_processor.block_processor.process_7d_field(
            source, operation="bvp_solve"
        )
        
        assert result.shape == source.shape
        assert result.dtype == source.dtype
    
    def test_memory_efficiency(self):
        """Test memory efficiency of block processing."""
        # Create larger field
        large_domain = Domain(L=1.0, N=32, dimensions=7)
        large_config = BVPBlockConfig(block_size=8)
        large_processor = BVPBlockProcessingSystem(large_domain, large_config)
        
        source = np.random.random(large_domain.shape) + 1j * np.random.random(large_domain.shape)
        
        # Should process without memory issues
        try:
            result = large_processor.block_processor.process_7d_field(
                source, operation="bvp_solve"
            )
            assert result.shape == source.shape
        except MemoryError:
            # Acceptable on systems with limited memory
            pass
        finally:
            large_processor.cleanup()
    
    def test_adaptive_processing(self):
        """Test adaptive processing capabilities."""
        # Test with different field sizes
        for size in [16, 32, 64]:
            if size > 32:  # Skip large sizes on limited memory systems
                continue
                
            domain = Domain(L=1.0, N=size, dimensions=7)
            config = BVPBlockConfig(block_size=min(8, size))
            processor = BVPBlockProcessingSystem(domain, config)
            
            source = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
            
            try:
                result = processor.block_processor.process_7d_field(
                    source, operation="bvp_solve"
                )
                assert result.shape == source.shape
            except MemoryError:
                # Acceptable on systems with limited memory
                pass
            finally:
                processor.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
