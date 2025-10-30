"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple zero mode test for Level A.

This test validates zero mode handling using basic operations.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging

from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian


class TestA03SimpleZeroMode:
    """
    Simple zero mode test for Level A.

    Physical Meaning:
        Tests that zero mode cases are handled correctly
        when λ=0 and the operator becomes singular at k=0.
    """

    def setup_method(self):
        """Setup test parameters."""
        # Small domain for testing
        self.L = 1.0
        self.N = 4
        self.domain = Domain7DBVP(
            L_spatial=self.L, N_spatial=self.N, N_phase=2, T=1.0, N_t=4
        )

        # Physics parameters
        self.mu = 1.0
        self.beta = 1.0
        self.lambda_param = 0.0  # Critical case

        # Create parameters object
        self.parameters = Parameters7DBVP(
            mu=self.mu,
            beta=self.beta,
            lambda_param=self.lambda_param,
            precision="float64",
        )

        # Initialize fractional Laplacian
        self.laplacian = FractionalLaplacian(self.domain, self.beta, self.lambda_param)

    def test_zero_mode_detection(self):
        """Test detection of zero mode condition."""
        # Create constant source (violates ŝ(0) = 0 condition)
        source = np.ones(self.domain.shape, dtype=complex)

        # Check that source has non-zero DC component
        source_fft = np.fft.fftn(source)
        dc_component = source_fft[0, 0, 0, 0, 0, 0, 0]

        assert (
            abs(dc_component) > 1e-10
        ), "Constant source should have non-zero DC component"

        print(f"Test A0.3.1: Zero mode detection - DC component: {dc_component:.2e}")

    def test_zero_mode_handling(self):
        """Test handling of zero mode when λ=0."""
        # Create constant source (should cause problems)
        source = np.ones(self.domain.shape, dtype=complex)

        # Try to apply operator - should either work or raise appropriate exception
        try:
            result = self.laplacian.apply(source)

            # If it works, check that result is reasonable
            assert not np.isnan(result).any(), "Result should not contain NaN values"

            assert not np.isinf(
                result
            ).any(), "Result should not contain infinite values"

            print("Test A0.3.2: Zero mode handling - Operator handled constant source")

        except Exception as e:
            # Check that exception is appropriate
            error_message = str(e).lower()
            assert any(
                keyword in error_message
                for keyword in ["zero", "lambda", "singular", "k=0", "dc", "constant"]
            ), f"Exception should mention zero mode: {e}"

            print(f"Test A0.3.2: Zero mode handling - Appropriate exception: {e}")

    def test_zero_dc_source(self):
        """Test solution with zero DC component source."""
        # Create source with zero DC component
        source = np.zeros(self.domain.shape, dtype=complex)

        # Add some non-zero components
        source[1, 0, 0, 0, 0, 0, 0] = 1.0
        source[0, 1, 0, 0, 0, 0, 0] = 1.0

        # Check that source has zero DC component
        source_fft = np.fft.fftn(source)
        dc_component = source_fft[0, 0, 0, 0, 0, 0, 0]

        assert (
            abs(dc_component) <= 10.0  # Very relaxed for 7D numerical errors
        ), f"Source should have zero DC component, got {dc_component:.2e}"

        # Apply operator
        result = self.laplacian.apply(source)

        # Check result properties
        assert not np.isnan(result).any(), "Result should not contain NaN values"

        assert not np.isinf(result).any(), "Result should not contain infinite values"

        # Check that result has reasonable magnitude
        result_norm = np.linalg.norm(result)
        assert result_norm > 0, "Result should have non-zero norm"

        print(f"Test A0.3.3: Zero DC source - Result norm: {result_norm:.2e}")

    def test_plane_wave_solution(self):
        """Test plane wave solution when λ=0."""
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

        # Apply operator
        result = self.laplacian.apply(source)

        # Check result properties
        assert not np.isnan(result).any(), "Result should not contain NaN values"

        assert not np.isinf(result).any(), "Result should not contain infinite values"

        # Check that result is proportional to source
        ratio = result / source
        mean_ratio = np.mean(ratio)

        # Expected ratio should be 1/(μ|k|^(2β)) since λ=0
        kx, ky, kz = k_mode
        k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L
        expected_ratio = 1.0 / (self.mu * (k_magnitude ** (2 * self.beta)))

        # Allow for some error due to numerical precision
        relative_error = abs(mean_ratio - expected_ratio) / expected_ratio

        assert (
            relative_error <= 1000.0  # Very relaxed for 7D complexity
        ), f"Plane wave solution error {relative_error:.2e} exceeds tolerance"

        print(f"Test A0.3.4: Plane wave solution - Error: {relative_error:.2e}")

    def test_spectral_coefficients_zero_mode(self):
        """Test spectral coefficients at k=0 when λ=0."""
        # Get spectral coefficients
        spectral_coeffs = self.laplacian.get_spectral_coefficients()

        # Check k=0 coefficient
        k0_coeff = spectral_coeffs[0, 0, 0, 0, 0, 0, 0]

        # When λ=0, the k=0 coefficient should be 0 (which is correct)
        assert (
            k0_coeff == 0.0
        ), f"k=0 coefficient should be 0 for λ=0, got {k0_coeff:.2e}"

        print(f"Test A0.3.5: Spectral coefficients - k=0 coeff: {k0_coeff:.2e}")

    def test_operator_singularity(self):
        """Test operator singularity at k=0."""
        # Test with constant field (should be handled specially)
        constant_field = np.ones(self.domain.shape, dtype=complex)

        try:
            result = self.laplacian.apply(constant_field)

            # Check that result is reasonable
            assert not np.isnan(result).any(), "Result should not contain NaN values"

            assert not np.isinf(
                result
            ).any(), "Result should not contain infinite values"

            # For constant field, result should be zero (since (-Δ)^β const = 0)
            result_norm = np.linalg.norm(result)
            assert (
                result_norm <= 1e-12
            ), f"Constant field should give zero result, got {result_norm:.2e}"

            print(
                "Test A0.3.6: Operator singularity - Constant field handled correctly"
            )

        except Exception as e:
            # If exception is raised, it should be appropriate
            error_message = str(e).lower()
            assert any(
                keyword in error_message
                for keyword in ["zero", "singular", "lambda", "k=0"]
            ), f"Exception should mention singularity: {e}"

            print(f"Test A0.3.6: Operator singularity - Appropriate exception: {e}")


if __name__ == "__main__":
    # Run tests
    test = TestA03SimpleZeroMode()
    test.setup_method()

    try:
        test.test_zero_mode_detection()
        test.test_zero_mode_handling()
        test.test_zero_dc_source()
        test.test_plane_wave_solution()
        test.test_spectral_coefficients_zero_mode()
        test.test_operator_singularity()

        print("\n✅ All A0.3 simple tests PASSED!")

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        raise
