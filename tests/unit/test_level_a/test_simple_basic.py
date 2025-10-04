"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple basic test to verify that the core components can be imported and initialized.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging

from bhlff.core.domain import Domain, Parameters


class TestSimpleBasic:
    """
    Simple basic test for Level A components.
    
    Physical Meaning:
        Tests that the basic components can be imported and initialized
        without errors, ensuring the framework is properly set up.
    """
    
    def test_domain_creation(self):
        """Test that Domain can be created."""
        domain = Domain(L=1.0, N=8, N_phi=4, N_t=8, T=1.0)
        
        assert domain.L == 1.0
        assert domain.N == 8
        assert domain.N_phi == 4
        assert domain.N_t == 8
        assert domain.T == 1.0
        assert domain.dimensions == 7
        
        print(f"Domain created successfully: {domain.shape}")
    
    def test_parameters_creation(self):
        """Test that Parameters can be created."""
        parameters = Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1
        )
        
        assert parameters.mu == 1.0
        assert parameters.beta == 1.0
        assert parameters.lambda_param == 0.1
        
        print(f"Parameters created successfully: mu={parameters.mu}, beta={parameters.beta}")
    
    def test_simple_fft_operation(self):
        """Test basic FFT operations."""
        # Create small test array
        N = 8
        test_array = np.random.randn(N, N, N, 4, 4, 4, 8)
        
        # Test FFT
        fft_result = np.fft.fftn(test_array)
        ifft_result = np.fft.ifftn(fft_result)
        
        # Check that FFT is reversible
        error = np.max(np.abs(test_array - ifft_result.real))
        
        assert error < 1e-12, f"FFT error too large: {error}"
        
        print(f"FFT operation successful, error: {error:.2e}")
    
    def test_fractional_laplacian_import(self):
        """Test that FractionalLaplacian can be imported."""
        try:
            from bhlff.core.fft.fractional_laplacian import FractionalLaplacian
            
            domain = Domain(L=1.0, N=8, N_phi=4, N_t=8, T=1.0)
            laplacian = FractionalLaplacian(domain, beta=1.0, lambda_param=0.1)
            
            assert laplacian.beta == 1.0
            assert laplacian.lambda_param == 0.1
            
            print("FractionalLaplacian imported and created successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to import FractionalLaplacian: {e}")
    
    def test_spectral_operations_import(self):
        """Test that SpectralOperations can be imported."""
        try:
            from bhlff.core.fft.spectral_operations import SpectralOperations
            
            domain = Domain(L=1.0, N=8, N_phi=4, N_t=8, T=1.0)
            parameters = Parameters(mu=1.0, beta=1.0, lambda_param=0.1)
            
            spectral_ops = SpectralOperations(domain, parameters)
            
            print("SpectralOperations imported and created successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to import SpectralOperations: {e}")
    
    def test_memory_manager_import(self):
        """Test that MemoryManager7D can be imported."""
        try:
            from bhlff.core.fft.memory_manager_7d import MemoryManager7D
            
            domain = Domain(L=1.0, N=8, N_phi=4, N_t=8, T=1.0)
            memory_manager = MemoryManager7D(domain.shape, 1.0)  # 1GB max memory
            
            print("MemoryManager7D imported and created successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to import MemoryManager7D: {e}")
    
    def test_fft_plan_import(self):
        """Test that FFTPlan7D can be imported."""
        try:
            from bhlff.core.fft.fft_plan_7d import FFTPlan7D
            
            domain = Domain(L=1.0, N=8, N_phi=4, N_t=8, T=1.0)
            parameters = Parameters(mu=1.0, beta=1.0, lambda_param=0.1)
            
            fft_plan = FFTPlan7D(domain, parameters)
            
            print("FFTPlan7D imported and created successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to import FFTPlan7D: {e}")
    
    def test_spectral_cache_import(self):
        """Test that SpectralCoefficientCache can be imported."""
        try:
            from bhlff.core.fft.spectral_coefficient_cache import SpectralCoefficientCache
            
            cache = SpectralCoefficientCache(10)  # 10 entries max
            
            print("SpectralCoefficientCache imported and created successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to import SpectralCoefficientCache: {e}")


if __name__ == "__main__":
    # Run tests
    test = TestSimpleBasic()
    
    try:
        test.test_domain_creation()
        test.test_parameters_creation()
        test.test_simple_fft_operation()
        test.test_fractional_laplacian_import()
        test.test_spectral_operations_import()
        test.test_memory_manager_import()
        test.test_fft_plan_import()
        test.test_spectral_cache_import()
        
        print("\n✅ All basic tests PASSED!")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        raise

