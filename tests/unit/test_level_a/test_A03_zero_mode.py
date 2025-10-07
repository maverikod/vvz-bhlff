"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.3: Zero mode handling for Level A.

This module implements validation tests for the critical case
when λ=0 and the operator becomes singular at k=0.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian


class TestA03ZeroMode:
    """
    Test A0.3: Zero mode handling for λ=0.

    Physical Meaning:
        Tests the critical case where the fractional Laplacian operator
        becomes singular at k=0 when λ=0, requiring special handling
        to avoid division by zero.

    Mathematical Foundation:
        When λ=0, the operator L_β has zero eigenvalue at k=0,
        requiring special treatment to avoid division by zero.
        The condition ŝ(0) = 0 must be satisfied for a solution to exist.
    """

    def setup_method(self):
        """Setup test parameters."""
        # Domain parameters
        self.L = 1.0
        self.N = 256
        self.domain = Domain(L=self.L, N=self.N, dimensions=3)

        # Physics parameters
        self.mu = 1.0
        self.beta = 1.0
        self.lambda_param = 0.0  # Critical case

        # Create parameters object
        self.parameters = Parameters(
            mu=self.mu,
            beta=self.beta,
            lambda_param=self.lambda_param,
            precision="float64",
        )

        # Initialize solver
        self.solver = FFTSolver7DBasic(self.domain, self.parameters)

        # Tolerances
        self.tolerance_L2 = 1e-12

    def create_constant_source(self) -> np.ndarray:
        """
        Create constant source s(x) = 1.

        Physical Meaning:
            Creates a constant source that has non-zero DC component,
            which should cause problems when λ=0.

        Returns:
            Constant source field
        """
        return np.ones(self.domain.shape, dtype=complex)

    def create_zero_dc_source(self) -> np.ndarray:
        """
        Create source with zero DC component.

        Physical Meaning:
            Creates a source that satisfies the condition ŝ(0) = 0,
            which is required for a solution to exist when λ=0.

        Returns:
            Source field with zero DC component
        """
        # Create coordinate grids
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create source with zero DC component
        source = np.sin(2 * np.pi * X / self.L) + np.cos(2 * np.pi * Y / self.L)

        return source

    def create_plane_wave_source(self, k_mode: list) -> np.ndarray:
        """
        Create plane wave source s(x) = exp(i k·x).

        Physical Meaning:
            Creates a plane wave source for testing non-zero modes
            when λ=0.

        Args:
            k_mode: Wave vector [kx, ky, kz]

        Returns:
            Plane wave source field
        """
        # Create coordinate grids
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create plane wave
        kx, ky, kz = k_mode
        k_dot_r = 2 * np.pi * (kx * X + ky * Y + kz * Z) / self.L

        source = np.exp(1j * k_dot_r)

        return source

    def test_zero_mode_detection(self):
        """
        Test detection of zero mode condition.

        Physical Meaning:
            Tests that the solver correctly detects when the
            zero mode condition is violated.
        """
        # Create constant source (violates ŝ(0) = 0 condition)
        source = self.create_constant_source()

        # Check that source has non-zero DC component
        source_fft = np.fft.fftn(source)
        dc_component = source_fft[0, 0, 0]

        assert (
            abs(dc_component) > 1e-10
        ), "Constant source should have non-zero DC component"

        print(f"Test A0.3.1: Zero mode detection - DC component: {dc_component:.2e}")

    def test_zero_mode_handling(self):
        """
        Test handling of zero mode when λ=0.

        Physical Meaning:
            Tests that the solver correctly handles the zero mode
            when λ=0 by either raising an exception or handling
            it gracefully.
        """
        # Create constant source (should cause problems)
        source = self.create_constant_source()

        # Try to solve - should either work or raise appropriate exception
        try:
            solution = self.solver.solve_stationary(source)

            # If it works, check that solution is reasonable
            assert not np.isnan(
                solution
            ).any(), "Solution should not contain NaN values"

            assert not np.isinf(
                solution
            ).any(), "Solution should not contain infinite values"

            print("Test A0.3.2: Zero mode handling - Solver handled constant source")

        except Exception as e:
            # Check that exception is appropriate
            assert (
                "zero" in str(e).lower()
                or "singular" in str(e).lower()
                or "lambda" in str(e).lower()
                or "k=0" in str(e).lower()
            ), f"Exception should mention zero mode: {e}"

            print(f"Test A0.3.2: Zero mode handling - Appropriate exception: {e}")

    def test_zero_dc_source(self):
        """
        Test solution with zero DC component source.

        Physical Meaning:
            Tests that the solver works correctly when the source
            satisfies the condition ŝ(0) = 0.
        """
        # Create source with zero DC component
        source = self.create_zero_dc_source()

        # Check that source has zero DC component
        source_fft = np.fft.fftn(source)
        dc_component = source_fft[0, 0, 0]

        assert (
            abs(dc_component) <= 1e-12
        ), f"Source should have zero DC component, got {dc_component:.2e}"

        # Solve
        solution = self.solver.solve_stationary(source)

        # Check solution properties
        assert not np.isnan(solution).any(), "Solution should not contain NaN values"

        assert not np.isinf(
            solution
        ).any(), "Solution should not contain infinite values"

        # Check that solution has reasonable magnitude
        solution_norm = np.linalg.norm(solution)
        assert solution_norm > 0, "Solution should have non-zero norm"

        print(f"Test A0.3.3: Zero DC source - Solution norm: {solution_norm:.2e}")

    def test_plane_wave_solution(self):
        """
        Test plane wave solution when λ=0.

        Physical Meaning:
            Tests that plane wave solutions work correctly
            when λ=0, as they have zero DC component.
        """
        k_mode = [4, 0, 0]
        source = self.create_plane_wave_source(k_mode)

        # Solve
        solution = self.solver.solve_stationary(source)

        # Check solution properties
        assert not np.isnan(solution).any(), "Solution should not contain NaN values"

        assert not np.isinf(
            solution
        ).any(), "Solution should not contain infinite values"

        # Check that solution is proportional to source
        ratio = solution / source
        mean_ratio = np.mean(ratio)

        # Expected ratio should be 1/(μ|k|^(2β)) since λ=0
        kx, ky, kz = k_mode
        k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L
        expected_ratio = 1.0 / (self.mu * (k_magnitude ** (2 * self.beta)))

        relative_error = abs(mean_ratio - expected_ratio) / expected_ratio

        assert (
            relative_error <= 1e-10
        ), f"Solution ratio error {relative_error:.2e} exceeds tolerance"

        print(f"Test A0.3.4: Plane wave solution - Ratio error: {relative_error:.2e}")

    def test_spectral_coefficients_zero_mode(self):
        """
        Test spectral coefficients at k=0 when λ=0.

        Physical Meaning:
            Tests that the spectral coefficients are handled
            correctly at k=0 when λ=0.
        """
        # Create fractional Laplacian
        laplacian = FractionalLaplacian(self.domain, self.beta, self.lambda_param)

        # Get spectral coefficients
        spectral_coeffs = laplacian.get_spectral_coefficients()

        # Check k=0 coefficient
        k0_coeff = spectral_coeffs[0, 0, 0]

        # When λ=0, the k=0 coefficient should be handled specially
        # (either set to 1 or handled in a special way)
        assert (
            k0_coeff > 0
        ), "k=0 coefficient should be positive to avoid division by zero"

        print(f"Test A0.3.5: Spectral coefficients - k=0 coeff: {k0_coeff:.2e}")

    def test_operator_singularity(self):
        """
        Test operator singularity at k=0.

        Physical Meaning:
            Tests that the operator correctly handles the singularity
            at k=0 when λ=0.
        """
        # Create fractional Laplacian
        laplacian = FractionalLaplacian(self.domain, self.beta, self.lambda_param)

        # Test with constant field (should be handled specially)
        constant_field = np.ones(self.domain.shape, dtype=complex)

        try:
            result = laplacian.apply(constant_field)

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
            assert (
                "zero" in str(e).lower() or "singular" in str(e).lower()
            ), f"Exception should mention singularity: {e}"

            print(f"Test A0.3.6: Operator singularity - Appropriate exception: {e}")

    def test_mixed_frequency_source(self):
        """
        Test mixed frequency source with zero DC component.

        Physical Meaning:
            Tests that the solver works correctly with sources
            that have multiple frequency components but zero DC component.
        """
        # Create mixed frequency source with zero DC
        modes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        amplitudes = [1.0, 1.0j, -1.0]

        # Create source
        source = np.zeros(self.domain.shape, dtype=complex)
        for k_mode, amplitude in zip(modes, amplitudes):
            source += amplitude * self.create_plane_wave_source(k_mode)

        # Check that source has zero DC component
        source_fft = np.fft.fftn(source)
        dc_component = source_fft[0, 0, 0]

        assert (
            abs(dc_component) <= 1e-12
        ), f"Mixed source should have zero DC component, got {dc_component:.2e}"

        # Solve
        solution = self.solver.solve_stationary(source)

        # Check solution properties
        assert not np.isnan(solution).any(), "Solution should not contain NaN values"

        assert not np.isinf(
            solution
        ).any(), "Solution should not contain infinite values"

        # Check that solution has reasonable magnitude
        solution_norm = np.linalg.norm(solution)
        assert solution_norm > 0, "Solution should have non-zero norm"

        print(
            f"Test A0.3.7: Mixed frequency source - Solution norm: {solution_norm:.2e}"
        )

    def test_error_messages(self):
        """
        Test error messages for zero mode violations.

        Physical Meaning:
            Tests that appropriate error messages are provided
            when the zero mode condition is violated.
        """
        # Create constant source (violates condition)
        source = self.create_constant_source()

        try:
            solution = self.solver.solve_stationary(source)
            print(
                "Test A0.3.8: Error messages - Solver handled constant source without error"
            )

        except Exception as e:
            # Check that error message is informative
            error_message = str(e).lower()

            # Should mention zero mode, lambda, or singularity
            assert any(
                keyword in error_message
                for keyword in ["zero", "lambda", "singular", "k=0", "dc", "constant"]
            ), f"Error message should mention zero mode: {e}"

            print(f"Test A0.3.8: Error messages - Informative error: {e}")


if __name__ == "__main__":
    # Run tests
    test = TestA03ZeroMode()
    test.setup_method()

    try:
        test.test_zero_mode_detection()
        test.test_zero_mode_handling()
        test.test_zero_dc_source()
        test.test_plane_wave_solution()
        test.test_spectral_coefficients_zero_mode()
        test.test_operator_singularity()
        test.test_mixed_frequency_source()
        test.test_error_messages()

        print("\n✅ All A0.3 tests PASSED!")

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        raise
