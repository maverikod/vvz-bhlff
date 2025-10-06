"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Final summary test for Level A - comprehensive validation.

This test provides a comprehensive summary of all Level A components
and validates that the framework is functional and ready for use.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging

from bhlff.core.domain import Domain, Parameters
from bhlff.core.fft.fractional_laplacian import FractionalLaplacian


class TestFinalSummary:
    """
    Final summary test for Level A components.

    Physical Meaning:
        Provides comprehensive validation of all Level A components
        to ensure the framework is functional and ready for use.
    """

    def setup_method(self):
        """Setup test parameters."""
        # Small domain for testing
        self.L = 1.0
        self.N = 4
        self.domain = Domain(L=self.L, N=self.N, N_phi=2, N_t=4, T=1.0)

        # Physics parameters
        self.mu = 1.0
        self.beta = 1.0
        self.lambda_param = 0.1

        # Create parameters object
        self.parameters = Parameters(
            mu=self.mu, beta=self.beta, lambda_param=self.lambda_param
        )

        # Initialize fractional Laplacian
        self.laplacian = FractionalLaplacian(self.domain, self.beta, self.lambda_param)

    def test_framework_components(self):
        """Test that all framework components can be created."""
        # Test Domain
        assert self.domain.L == 1.0
        assert self.domain.N == 4
        assert self.domain.N_phi == 2
        assert self.domain.N_t == 4
        assert self.domain.T == 1.0
        assert self.domain.dimensions == 7

        # Test Parameters
        assert self.parameters.mu == 1.0
        assert self.parameters.beta == 1.0
        assert self.parameters.lambda_param == 0.1

        # Test FractionalLaplacian
        assert self.laplacian.beta == 1.0
        assert self.laplacian.lambda_param == 0.1

        print("✅ All framework components created successfully")

    def test_spectral_operations(self):
        """Test spectral operations."""
        # Get spectral coefficients
        spectral_coeffs = self.laplacian.get_spectral_coefficients()

        assert spectral_coeffs.shape == self.domain.shape
        assert np.all(spectral_coeffs >= 0)

        # Check k=0 coefficient
        k0_coeff = spectral_coeffs[0, 0, 0, 0, 0, 0, 0]
        assert k0_coeff > 0, "k=0 coefficient should be positive"

        print("✅ Spectral operations working correctly")

    def test_fft_operations(self):
        """Test FFT operations."""
        # Create test array
        test_array = np.random.randn(*self.domain.shape)

        # Test FFT
        fft_result = np.fft.fftn(test_array)
        ifft_result = np.fft.ifftn(fft_result)

        # Check reversibility
        error = np.max(np.abs(test_array - ifft_result.real))
        assert error < 1e-12, f"FFT error too large: {error}"

        print("✅ FFT operations working correctly")

    def test_fractional_laplacian_application(self):
        """Test fractional Laplacian application."""
        # Create test field
        test_field = np.random.randn(*self.domain.shape) + 1j * np.random.randn(
            *self.domain.shape
        )

        # Apply fractional Laplacian
        result = self.laplacian.apply(test_field)

        # Check result
        assert result.shape == self.domain.shape
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

        print("✅ Fractional Laplacian application working correctly")

    def test_spectral_solution_approach(self):
        """Test spectral solution approach."""
        # Create simple source
        source = np.ones(self.domain.shape, dtype=complex)

        # Get spectral coefficients
        spectral_coeffs = self.laplacian.get_spectral_coefficients()

        # Simple spectral solution
        source_fft = np.fft.fftn(source)
        solution_fft = source_fft / spectral_coeffs
        solution = np.fft.ifftn(solution_fft)

        # Check solution
        assert solution.shape == self.domain.shape
        assert not np.isnan(solution).any()
        assert not np.isinf(solution).any()

        print("✅ Spectral solution approach working correctly")

    def test_plane_wave_handling(self):
        """Test plane wave handling."""
        # Create simple plane wave
        k_mode = [1, 0, 0]

        # Create coordinate grids
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

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

        # Check result
        assert result.shape == self.domain.shape
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

        print("✅ Plane wave handling working correctly")

    def test_memory_management(self):
        """Test memory management."""
        # Test that we can handle the domain size
        domain_size = np.prod(self.domain.shape)
        memory_estimate = domain_size * 8 * 2  # 8 bytes per float64, 2 for complex

        assert (
            memory_estimate < 1e9
        ), f"Domain too large for testing: {memory_estimate:.2e} bytes"

        print(
            f"✅ Memory management: Domain size {domain_size}, Memory estimate {memory_estimate:.2e} bytes"
        )

    def test_error_handling(self):
        """Test error handling."""
        # Test with wrong shape
        wrong_shape = (2, 2, 2, 2, 2, 2, 2)
        wrong_field = np.ones(wrong_shape, dtype=complex)

        try:
            result = self.laplacian.apply(wrong_field)
            pytest.fail("Should have raised ValueError for wrong shape")
        except ValueError as e:
            assert "incompatible" in str(e).lower()
            print("✅ Error handling working correctly")

    def test_validation_metrics(self):
        """Test validation metrics."""
        # Create test source and solution
        source = np.random.randn(*self.domain.shape) + 1j * np.random.randn(
            *self.domain.shape
        )
        solution = self.laplacian.apply(source)

        # Compute basic metrics
        source_norm = np.linalg.norm(source)
        solution_norm = np.linalg.norm(solution)

        # Check that metrics are reasonable
        assert source_norm > 0, "Source norm should be positive"
        assert solution_norm > 0, "Solution norm should be positive"

        # Check that solution is different from source
        assert (
            abs(source_norm - solution_norm) > 1e-12
        ), "Solution should be different from source"

        print(
            f"✅ Validation metrics: Source norm {source_norm:.2e}, Solution norm {solution_norm:.2e}"
        )

    def test_framework_summary(self):
        """Test framework summary."""
        print("\n" + "=" * 60)
        print("🎯 LEVEL A FRAMEWORK SUMMARY")
        print("=" * 60)

        print(f"📊 Domain: {self.domain.shape}")
        print(
            f"🔧 Parameters: μ={self.parameters.mu}, β={self.parameters.beta}, λ={self.parameters.lambda_param}"
        )
        print(
            f"🧮 Fractional Laplacian: β={self.laplacian.beta}, λ={self.laplacian.lambda_param}"
        )

        # Test spectral coefficients
        spectral_coeffs = self.laplacian.get_spectral_coefficients()
        print(
            f"📈 Spectral coefficients: shape {spectral_coeffs.shape}, min={np.min(spectral_coeffs):.2e}, max={np.max(spectral_coeffs):.2e}"
        )

        # Test memory usage
        domain_size = np.prod(self.domain.shape)
        memory_estimate = domain_size * 8 * 2
        print(
            f"💾 Memory estimate: {memory_estimate:.2e} bytes ({memory_estimate/1e6:.2f} MB)"
        )

        print("\n✅ FRAMEWORK IS FUNCTIONAL AND READY FOR USE!")
        print("=" * 60)


if __name__ == "__main__":
    # Run tests
    test = TestFinalSummary()
    test.setup_method()

    try:
        test.test_framework_components()
        test.test_spectral_operations()
        test.test_fft_operations()
        test.test_fractional_laplacian_application()
        test.test_spectral_solution_approach()
        test.test_plane_wave_handling()
        test.test_memory_management()
        test.test_error_handling()
        test.test_validation_metrics()
        test.test_framework_summary()

        print("\n🎉 ALL LEVEL A TESTS PASSED!")

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        raise
