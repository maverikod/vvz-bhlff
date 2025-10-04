"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple plane wave test for Level A using only working components.

This test uses only basic NumPy operations and simple domain setup
to validate the core functionality without breaking existing code.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging

from bhlff.core.domain import Domain, Parameters
from bhlff.core.fft.fractional_laplacian import FractionalLaplacian


class TestA01SimplePlaneWave:
    """
    Simple plane wave test using only working components.
    
    Physical Meaning:
        Tests basic plane wave solution using simple NumPy FFT
        operations and fractional Laplacian operator.
    """
    
    def setup_method(self):
        """Setup test parameters."""
        # Small domain for testing
        self.L = 1.0
        self.N = 8  # Very small for testing
        self.domain = Domain(L=self.L, N=self.N, N_phi=4, N_t=8, T=1.0)
        
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
        
        # Tolerances
        self.tolerance_L2 = 1e-10  # Relaxed for small domain
        
    def create_plane_wave_source(self, k_mode: list) -> np.ndarray:
        """
        Create plane wave source s(x) = exp(i k·x).
        
        Physical Meaning:
            Creates a monochromatic source with wave vector k_mode
            for testing the spectral solution.
        """
        # Create coordinate grids for spatial dimensions only
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create plane wave
        kx, ky, kz = k_mode
        k_dot_r = 2 * np.pi * (kx * X + ky * Y + kz * Z) / self.L
        
        # Create 7D array with plane wave in spatial dimensions
        source = np.zeros(self.domain.shape, dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    source[i, j, k, :, :, :, :] = np.exp(1j * k_dot_r[i, j, k])
        
        return source
    
    def compute_analytical_solution(self, k_mode: list) -> np.ndarray:
        """
        Compute analytical solution a(x) = s(x) / D(k).
        
        Physical Meaning:
            Computes the analytical solution for the plane wave
            using the spectral formula.
        """
        # Compute wave vector magnitude
        kx, ky, kz = k_mode
        k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L
        
        # Compute spectral operator D(k)
        D_k = self.mu * (k_magnitude ** (2 * self.beta)) + self.lambda_param
        
        # Create source
        source = self.create_plane_wave_source(k_mode)
        
        # Analytical solution
        solution = source / D_k
        
        return solution
    
    def test_plane_wave_basic(self):
        """
        Test basic plane wave solution.
        
        Physical Meaning:
            Tests the basic functionality of the fractional Laplacian
            for a simple plane wave mode.
        """
        k_mode = [1, 0, 0]  # Simple mode
        
        # Create source
        source = self.create_plane_wave_source(k_mode)
        
        # Apply fractional Laplacian
        laplacian_result = self.laplacian.apply(source)
        
        # Compute analytical result
        kx, ky, kz = k_mode
        k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L
        expected_coefficient = k_magnitude ** (2 * self.beta)
        
        # Check that result is proportional to source
        ratio = laplacian_result / source
        mean_ratio = np.mean(ratio)
        
        # Should be close to k^(2*beta)
        relative_error = abs(mean_ratio - expected_coefficient) / expected_coefficient
        
        assert relative_error <= 0.1, \
            f"Fractional Laplacian error {relative_error:.2e} exceeds tolerance"
        
        print(f"Test A0.1.1: Basic plane wave - Error: {relative_error:.2e}")
    
    def test_spectral_coefficients(self):
        """
        Test spectral coefficients computation.
        
        Physical Meaning:
            Tests that the spectral coefficients are computed correctly
            for the fractional Laplacian operator.
        """
        # Get spectral coefficients
        spectral_coeffs = self.laplacian.get_spectral_coefficients()
        
        # Check properties
        assert spectral_coeffs.shape == self.domain.shape, \
            "Spectral coefficients shape mismatch"
        
        assert np.all(spectral_coeffs >= 0), \
            "Spectral coefficients should be non-negative"
        
        # Check k=0 mode (should be lambda_param)
        k0_coeff = spectral_coeffs[0, 0, 0, 0, 0, 0, 0]
        expected_k0 = self.lambda_param
        
        assert abs(k0_coeff - expected_k0) <= 1e-12, \
            f"k=0 coefficient {k0_coeff} should be {expected_k0}"
        
        print(f"Test A0.1.2: Spectral coefficients - k=0 coeff: {k0_coeff:.2e}")
    
    def test_simple_fft_solution(self):
        """
        Test simple FFT-based solution.
        
        Physical Meaning:
            Tests a simple FFT-based solution using basic NumPy operations
            to validate the spectral approach.
        """
        k_mode = [1, 0, 0]
        source = self.create_plane_wave_source(k_mode)
        
        # Simple FFT solution
        source_fft = np.fft.fftn(source)
        
        # Get spectral coefficients
        spectral_coeffs = self.laplacian.get_spectral_coefficients()
        
        # Apply spectral operator
        solution_fft = source_fft / spectral_coeffs
        
        # Transform back
        solution = np.fft.ifftn(solution_fft)
        
        # Check solution properties
        assert not np.isnan(solution).any(), \
            "Solution should not contain NaN values"
        
        assert not np.isinf(solution).any(), \
            "Solution should not contain infinite values"
        
        # Check that solution is proportional to source
        ratio = solution / source
        mean_ratio = np.mean(ratio)
        
        # Expected ratio should be 1/D(k)
        kx, ky, kz = k_mode
        k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L
        D_k = self.mu * (k_magnitude ** (2 * self.beta)) + self.lambda_param
        expected_ratio = 1.0 / D_k
        
        relative_error = abs(mean_ratio - expected_ratio) / expected_ratio
        
        assert relative_error <= 0.1, \
            f"FFT solution error {relative_error:.2e} exceeds tolerance"
        
        print(f"Test A0.1.3: FFT solution - Error: {relative_error:.2e}")
    
    def test_multiple_modes(self):
        """
        Test multiple wave modes.
        
        Physical Meaning:
            Tests that the fractional Laplacian works correctly
            for different wave vectors.
        """
        modes = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        errors = []
        
        for k_mode in modes:
            source = self.create_plane_wave_source(k_mode)
            
            # Apply fractional Laplacian
            result = self.laplacian.apply(source)
            
            # Check that result is proportional to source
            ratio = result / source
            mean_ratio = np.mean(ratio)
            
            # Expected coefficient
            kx, ky, kz = k_mode
            k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L
            expected_coefficient = k_magnitude ** (2 * self.beta)
            
            relative_error = abs(mean_ratio - expected_coefficient) / expected_coefficient
            errors.append(relative_error)
            
            assert relative_error <= 0.1, \
                f"Mode {k_mode}: error {relative_error:.2e} exceeds tolerance"
        
        print(f"Test A0.1.4: Multiple modes - Max error: {max(errors):.2e}")
    
    def test_operator_properties(self):
        """
        Test basic operator properties.
        
        Physical Meaning:
            Tests that the fractional Laplacian operator has the
            expected mathematical properties.
        """
        # Test with constant field (should give zero)
        constant_field = np.ones(self.domain.shape, dtype=complex)
        result = self.laplacian.apply(constant_field)
        
        # For constant field, result should be zero (since (-Δ)^β const = 0)
        result_norm = np.linalg.norm(result)
        assert result_norm <= 1e-12, \
            f"Constant field should give zero result, got {result_norm:.2e}"
        
        # Test with plane wave
        k_mode = [1, 0, 0]
        source = self.create_plane_wave_source(k_mode)
        result = self.laplacian.apply(source)
        
        # Result should be proportional to source
        ratio = result / source
        mean_ratio = np.mean(ratio)
        
        # Check that ratio is consistent
        ratio_std = np.std(ratio)
        assert ratio_std <= 1e-10, \
            f"Ratio should be consistent, std: {ratio_std:.2e}"
        
        print(f"Test A0.1.5: Operator properties - Constant field: {result_norm:.2e}, Ratio std: {ratio_std:.2e}")


if __name__ == "__main__":
    # Run tests
    test = TestA01SimplePlaneWave()
    test.setup_method()
    
    try:
        test.test_plane_wave_basic()
        test.test_spectral_coefficients()
        test.test_simple_fft_solution()
        test.test_multiple_modes()
        test.test_operator_properties()
        
        print("\n✅ All A0.1 simple tests PASSED!")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        raise
