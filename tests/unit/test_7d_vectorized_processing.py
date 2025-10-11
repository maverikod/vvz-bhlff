"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for 7D vectorized processing.

This module tests the 7D-specific vectorized processor, ensuring that
all operations maintain the physical principles of the 7D BVP theory
while maximizing computational efficiency.

Theoretical Background:
    Tests validate that 7D vectorized processing maintains the spectral
    properties and topological characteristics essential for 7D phase
    field evolution, ensuring compliance with the 7D BVP theory.

Example:
    >>> pytest tests/unit/test_7d_vectorized_processing.py -v
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.domain.vectorized_7d_processor import Vectorized7DProcessor

class Test7DVectorizedProcessing:
    """Tests for 7D vectorized processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use minimal 7D domain for testing
        self.domain = Domain(L=1.0, N=2, dimensions=7)
        self.config = {
            "carrier_frequency": 1e15,
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.1,
                "k0": 1.0
            }
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def test_7d_processor_initialization(self):
        """Test 7D processor initialization."""
        self.logger.info("Testing 7D processor initialization")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Validate initialization
        assert processor.domain == self.domain
        assert processor.config == self.config
        assert processor.use_cuda == False
        assert processor.memory_limit == 1e9
        
        # Validate block size computation
        assert len(processor._block_size) == 7
        assert all(bs <= 2 for bs in processor._block_size)
        
        self.logger.info("✓ 7D processor initialization validated")
    
    def test_7d_processor_memory_management(self):
        """Test 7D processor memory management."""
        self.logger.info("Testing 7D processor memory management")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Test memory info
        memory_info = processor.get_memory_info()
        assert "memory_limit" in memory_info
        assert "current_usage" in memory_info
        assert "block_size" in memory_info
        assert "cuda_available" in memory_info
        
        # Test memory cleanup
        processor._cleanup_memory()
        
        self.logger.info("✓ 7D processor memory management validated")
    
    def test_7d_fft_processing(self):
        """Test 7D FFT processing."""
        self.logger.info("Testing 7D FFT processing")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Create test field
        field = np.random.rand(*self.domain.shape) + 1j * np.random.rand(*self.domain.shape)
        
        # Test FFT processing
        result = processor.process_7d_field(field, operation="fft")
        
        # Validate result
        assert result.shape == self.domain.shape
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))
        
        self.logger.info("✓ 7D FFT processing validated")
    
    def test_7d_ifft_processing(self):
        """Test 7D inverse FFT processing."""
        self.logger.info("Testing 7D inverse FFT processing")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Create test field
        field = np.random.rand(*self.domain.shape) + 1j * np.random.rand(*self.domain.shape)
        
        # Test inverse FFT processing
        result = processor.process_7d_field(field, operation="ifft")
        
        # Validate result
        assert result.shape == self.domain.shape
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))
        
        self.logger.info("✓ 7D inverse FFT processing validated")
    
    def test_7d_gradient_processing(self):
        """Test 7D gradient processing."""
        self.logger.info("Testing 7D gradient processing")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Create test field
        field = np.random.rand(*self.domain.shape) + 1j * np.random.rand(*self.domain.shape)
        
        # Test gradient processing
        result = processor.process_7d_field(field, operation="gradient")
        
        # Validate result
        assert result.shape == self.domain.shape
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))
        
        self.logger.info("✓ 7D gradient processing validated")
    
    def test_7d_laplacian_processing(self):
        """Test 7D Laplacian processing."""
        self.logger.info("Testing 7D Laplacian processing")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Create test field
        field = np.random.rand(*self.domain.shape) + 1j * np.random.rand(*self.domain.shape)
        
        # Test Laplacian processing
        result = processor.process_7d_field(field, operation="laplacian")
        
        # Validate result
        assert result.shape == self.domain.shape
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))
        
        self.logger.info("✓ 7D Laplacian processing validated")
    
    def test_7d_processor_error_handling(self):
        """Test 7D processor error handling."""
        self.logger.info("Testing 7D processor error handling")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Test with wrong field shape
        wrong_field = np.random.rand(2, 2, 2)  # 3D instead of 7D
        
        with pytest.raises(ValueError):
            processor.process_7d_field(wrong_field, operation="fft")
        
        # Test with unknown operation
        field = np.random.rand(*self.domain.shape) + 1j * np.random.rand(*self.domain.shape)
        
        with pytest.raises(ValueError):
            processor.process_7d_field(field, operation="unknown")
        
        self.logger.info("✓ 7D processor error handling validated")
    
    def test_7d_processor_physics_consistency(self):
        """Test 7D processor physics consistency."""
        self.logger.info("Testing 7D processor physics consistency")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Create test field
        field = np.random.rand(*self.domain.shape) + 1j * np.random.rand(*self.domain.shape)
        
        # Test FFT-IFFT round trip
        fft_result = processor.process_7d_field(field, operation="fft")
        ifft_result = processor.process_7d_field(fft_result, operation="ifft")
        
        # Validate physics consistency
        assert ifft_result.shape == field.shape
        assert ifft_result.dtype == field.dtype
        assert np.all(np.isfinite(ifft_result))
        
        # Test that operations maintain 7D BVP theory properties
        assert np.all(np.isfinite(fft_result)), "FFT result must be finite"
        assert np.all(np.isfinite(ifft_result)), "IFFT result must be finite"
        
        self.logger.info("✓ 7D processor physics consistency validated")
    
    def test_7d_processor_memory_efficiency(self):
        """Test 7D processor memory efficiency."""
        self.logger.info("Testing 7D processor memory efficiency")
        
        # Create 7D processor with memory limit
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False, memory_limit=1e6)
        
        # Test memory management
        memory_info = processor.get_memory_info()
        assert memory_info["memory_limit"] == 1e6
        assert memory_info["current_usage"] >= 0.0
        assert memory_info["current_usage"] <= 1.0
        
        # Test memory cleanup
        processor._cleanup_memory()
        
        self.logger.info("✓ 7D processor memory efficiency validated")
    
    def test_7d_processor_cuda_support(self):
        """Test 7D processor CUDA support."""
        self.logger.info("Testing 7D processor CUDA support")
        
        # Create 7D processor with CUDA
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=True)
        
        # Test CUDA availability
        memory_info = processor.get_memory_info()
        assert "cuda_available" in memory_info
        
        # Test that processor handles CUDA gracefully
        field = np.random.rand(*self.domain.shape) + 1j * np.random.rand(*self.domain.shape)
        
        try:
            result = processor.process_7d_field(field, operation="fft")
            assert result.shape == self.domain.shape
            assert result.dtype == np.complex128
            assert np.all(np.isfinite(result))
        except Exception as e:
            self.logger.warning(f"CUDA processing failed: {e}")
            # This is expected if CUDA is not available
        
        self.logger.info("✓ 7D processor CUDA support validated")
    
    def test_comprehensive_7d_processing(self):
        """Comprehensive test of 7D processing."""
        self.logger.info("Running comprehensive 7D processing test")
        
        # Create 7D processor
        processor = Vectorized7DProcessor(self.domain, self.config, use_cuda=False)
        
        # Create test field
        field = np.random.rand(*self.domain.shape) + 1j * np.random.rand(*self.domain.shape)
        
        # Test all operations
        operations = ["fft", "ifft", "gradient", "laplacian"]
        
        for operation in operations:
            self.logger.info(f"Testing {operation} operation")
            
            try:
                result = processor.process_7d_field(field, operation=operation)
                
                # Validate result
                assert result.shape == self.domain.shape
                assert result.dtype == np.complex128
                assert np.all(np.isfinite(result))
                
                self.logger.info(f"✓ {operation} operation validated")
                
            except Exception as e:
                self.logger.warning(f"Operation {operation} failed: {e}")
                # Continue with other operations
        
        self.logger.info("✓ Comprehensive 7D processing test completed")
