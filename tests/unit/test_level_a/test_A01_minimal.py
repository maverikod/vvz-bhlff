"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Minimal test for Level A - just check that components work without errors.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging

from bhlff.core.domain import Domain, Parameters
from bhlff.core.fft.fractional_laplacian import FractionalLaplacian


class TestA01Minimal:
    """
    Minimal test for Level A components.
    
    Physical Meaning:
        Tests that the basic components can be created and used
        without errors, ensuring the framework is functional.
    """
    
    def setup_method(self):
        """Setup test parameters."""
        # Very small domain for testing
        self.L = 1.0
        self.N = 4  # Very small
        self.domain = Domain(L=self.L, N=self.N, N_phi=2, N_t=4, T=1.0)
        
        # Physics parameters
        self.mu = 1.0
        self.beta = 1.0
        self.lambda_param = 0.1
        
        # Create parameters object
        self.parameters = Parameters(
            mu=self.mu,
            beta=self.beta,
            lambda_param=self.lambda_param
        )
        
        # Initialize fractional Laplacian
        self.laplacian = FractionalLaplacian(self.domain, self.beta, self.lambda_param)
        
    def test_domain_creation(self):
        """Test that domain is created correctly."""
        assert self.domain.L == 1.0
        assert self.domain.N == 4
        assert self.domain.N_phi == 2
        assert self.domain.N_t == 4
        assert self.domain.T == 1.0
        assert self.domain.dimensions == 7
        
        print(f"Domain created: {self.domain.shape}")
    
    def test_parameters_creation(self):
        """Test that parameters are created correctly."""
        assert self.parameters.mu == 1.0
        assert self.parameters.beta == 1.0
        assert self.parameters.lambda_param == 0.1
        
        print(f"Parameters created: mu={self.parameters.mu}, beta={self.parameters.beta}")
    
    def test_fractional_laplacian_creation(self):
        """Test that fractional Laplacian is created correctly."""
        assert self.laplacian.beta == 1.0
        assert self.laplacian.lambda_param == 0.1
        
        print("Fractional Laplacian created successfully")
    
    def test_spectral_coefficients(self):
        """Test that spectral coefficients can be computed."""
        spectral_coeffs = self.laplacian.get_spectral_coefficients()
        
        assert spectral_coeffs.shape == self.domain.shape
        assert np.all(spectral_coeffs >= 0)
        
        print(f"Spectral coefficients computed: shape {spectral_coeffs.shape}")
    
    def test_simple_field_application(self):
        """Test that fractional Laplacian can be applied to a simple field."""
        # Create simple test field
        test_field = np.ones(self.domain.shape, dtype=complex)
        
        # Apply fractional Laplacian
        result = self.laplacian.apply(test_field)
        
        # Check that result is computed
        assert result.shape == self.domain.shape
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
        
        print("Fractional Laplacian applied successfully")
    
    def test_plane_wave_creation(self):
        """Test that plane wave can be created."""
        # Create simple plane wave
        k_mode = [1, 0, 0]
        
        # Create coordinate grids
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create plane wave
        kx, ky, kz = k_mode
        k_dot_r = 2 * np.pi * (kx * X + ky * Y + kz * Z) / self.L
        plane_wave = np.exp(1j * k_dot_r)
        
        # Create 7D array
        source = np.zeros(self.domain.shape, dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    source[i, j, k, :, :, :, :] = plane_wave[i, j, k]
        
        # Apply fractional Laplacian
        result = self.laplacian.apply(source)
        
        # Check that result is computed
        assert result.shape == self.domain.shape
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
        
        print("Plane wave created and processed successfully")
    
    def test_fft_operations(self):
        """Test basic FFT operations."""
        # Create simple test array
        test_array = np.random.randn(*self.domain.shape)
        
        # Test FFT
        fft_result = np.fft.fftn(test_array)
        ifft_result = np.fft.ifftn(fft_result)
        
        # Check that FFT is reversible
        error = np.max(np.abs(test_array - ifft_result.real))
        
        assert error < 1e-12, f"FFT error too large: {error}"
        
        print(f"FFT operations successful, error: {error:.2e}")
    
    def test_spectral_solution(self):
        """Test basic spectral solution approach."""
        # Create simple source
        source = np.ones(self.domain.shape, dtype=complex)
        
        # Get spectral coefficients
        spectral_coeffs = self.laplacian.get_spectral_coefficients()
        
        # Simple spectral solution
        source_fft = np.fft.fftn(source)
        solution_fft = source_fft / spectral_coeffs
        solution = np.fft.ifftn(solution_fft)
        
        # Check that solution is computed
        assert solution.shape == self.domain.shape
        assert not np.isnan(solution).any()
        assert not np.isinf(solution).any()
        
        print("Spectral solution computed successfully")


if __name__ == "__main__":
    # Run tests
    test = TestA01Minimal()
    test.setup_method()
    
    try:
        test.test_domain_creation()
        test.test_parameters_creation()
        test.test_fractional_laplacian_creation()
        test.test_spectral_coefficients()
        test.test_simple_field_application()
        test.test_plane_wave_creation()
        test.test_fft_operations()
        test.test_spectral_solution()
        
        print("\n✅ All minimal tests PASSED!")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        raise
