"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.1: Plane wave validation for Level A.

This module implements validation tests for the basic FFT solver
and fractional Laplacian operator using plane wave solutions.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging

from bhlff.core.domain import Domain, Parameters
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian


class TestA01PlaneWave:
    """
    Test A0.1: Plane wave validation.

    Physical Meaning:
        Validates the spectral solution for monochromatic excitation,
        ensuring correct implementation of the fractional Laplacian
        operator in k-space.

    Mathematical Foundation:
        Tests the formula a_hat(k) = s_hat(k) / D(k) where
        D(k) = mu|k|^(2*beta) + lambda is the spectral operator.
    """

    def setup_method(self):
        """Setup test parameters."""
        # Domain parameters
        self.L = 1.0
        self.N = 16  # Much smaller for testing
        self.domain = Domain(L=self.L, N=self.N, N_phi=8, N_t=16, T=1.0)

        # Physics parameters
        self.mu = 1.0
        self.beta = 1.0
        self.lambda_param = 0.1

        # Create parameters object
        self.parameters = Parameters(
            mu=self.mu,
            beta=self.beta,
            lambda_param=self.lambda_param,
            precision="float64",
        )

        # Initialize solver
        self.solver = FFTSolver7DBasic(self.domain, self.parameters)

        # Test wave vectors
        self.test_modes = [[4, 0, 0], [0, 4, 0], [3, 3, 2], [2, 2, 4]]

        # Tolerances
        self.tolerance_L2 = 1e-12
        self.tolerance_anisotropy = 1e-12

    def create_plane_wave_source(self, k_mode: list) -> np.ndarray:
        """
        Create plane wave source s(x) = exp(i k·x).

        Physical Meaning:
            Creates a monochromatic source with wave vector k_mode
            for testing the spectral solution.

        Args:
            k_mode: Wave vector [kx, ky, kz]

        Returns:
            Complex source field
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

    def compute_analytical_solution(self, k_mode: list) -> np.ndarray:
        """
        Compute analytical solution a(x) = s(x) / D(k).

        Physical Meaning:
            Computes the analytical solution for the plane wave
            using the spectral formula.

        Args:
            k_mode: Wave vector [kx, ky, kz]

        Returns:
            Analytical solution field
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

    def test_plane_wave_single_mode(self):
        """
        Test plane wave solution for single mode.

        Physical Meaning:
            Tests the basic functionality of the FFT solver
            for a single plane wave mode.
        """
        k_mode = [4, 0, 0]

        # Create source
        source = self.create_plane_wave_source(k_mode)

        # Solve numerically
        solution_numerical = self.solver.solve_stationary(source)

        # Compute analytical solution
        solution_analytical = self.compute_analytical_solution(k_mode)

        # Compute error
        error = np.abs(solution_numerical - solution_analytical)
        relative_error_L2 = np.linalg.norm(error) / np.linalg.norm(solution_analytical)

        # Check tolerance
        assert (
            relative_error_L2 <= self.tolerance_L2
        ), f"L2 error {relative_error_L2:.2e} exceeds tolerance {self.tolerance_L2:.2e}"

        print(f"Test A0.1.1: Single mode - L2 error: {relative_error_L2:.2e}")

    def test_plane_wave_multiple_modes(self):
        """
        Test plane wave solution for multiple modes.

        Physical Meaning:
            Tests the solver for different wave vectors to ensure
            correct spectral handling.
        """
        errors = []

        for i, k_mode in enumerate(self.test_modes):
            # Create source
            source = self.create_plane_wave_source(k_mode)

            # Solve numerically
            solution_numerical = self.solver.solve_stationary(source)

            # Compute analytical solution
            solution_analytical = self.compute_analytical_solution(k_mode)

            # Compute error
            error = np.abs(solution_numerical - solution_analytical)
            relative_error_L2 = np.linalg.norm(error) / np.linalg.norm(
                solution_analytical
            )

            errors.append(relative_error_L2)

            # Check tolerance
            assert (
                relative_error_L2 <= self.tolerance_L2
            ), f"Mode {k_mode}: L2 error {relative_error_L2:.2e} exceeds tolerance {self.tolerance_L2:.2e}"

        print(f"Test A0.1.2: Multiple modes - Max L2 error: {max(errors):.2e}")

    def test_anisotropy_check(self):
        """
        Test anisotropy for modes with same |k|.

        Physical Meaning:
            Tests that modes with the same wave vector magnitude
            produce solutions with the same amplitude, ensuring
            isotropy of the operator.
        """
        # Modes with same |k| = 4
        modes_same_k = [[4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 2]]

        amplitudes = []

        for k_mode in modes_same_k:
            # Create source
            source = self.create_plane_wave_source(k_mode)

            # Solve numerically
            solution = self.solver.solve_stationary(source)

            # Compute amplitude
            amplitude = np.abs(solution)
            mean_amplitude = np.mean(amplitude)
            amplitudes.append(mean_amplitude)

        # Check anisotropy
        max_amplitude = max(amplitudes)
        min_amplitude = min(amplitudes)
        anisotropy = (max_amplitude - min_amplitude) / max_amplitude

        assert (
            anisotropy <= self.tolerance_anisotropy
        ), f"Anisotropy {anisotropy:.2e} exceeds tolerance {self.tolerance_anisotropy:.2e}"

        print(f"Test A0.1.3: Anisotropy - {anisotropy:.2e}")

    def test_grid_convergence(self):
        """
        Test convergence with grid refinement.

        Physical Meaning:
            Tests that the solution converges as the grid is refined,
            ensuring numerical accuracy.
        """
        k_mode = [4, 0, 0]
        grid_sizes = [64, 128, 256]
        errors = []

        for N in grid_sizes:
            # Create domain
            domain = Domain(L=self.L, N=N, dimensions=3)

            # Create solver
            solver = FFTSolver7DBasic(domain, self.parameters)

            # Create source
            source = self.create_plane_wave_source(k_mode)

            # Solve numerically
            solution_numerical = solver.solve_stationary(source)

            # Compute analytical solution
            solution_analytical = self.compute_analytical_solution(k_mode)

            # Compute error
            error = np.abs(solution_numerical - solution_analytical)
            relative_error_L2 = np.linalg.norm(error) / np.linalg.norm(
                solution_analytical
            )

            errors.append(relative_error_L2)

        # Check convergence (error should decrease with increasing N)
        for i in range(1, len(errors)):
            assert (
                errors[i] <= errors[i - 1]
            ), f"Error increased from N={grid_sizes[i-1]} to N={grid_sizes[i]}"

        print(f"Test A0.1.4: Grid convergence - Errors: {errors}")

    def test_fractional_laplacian_operator(self):
        """
        Test fractional Laplacian operator directly.

        Physical Meaning:
            Tests the fractional Laplacian operator implementation
            to ensure correct spectral coefficients.
        """
        # Create fractional Laplacian
        laplacian = FractionalLaplacian(self.domain, self.beta, self.lambda_param)

        # Test with plane wave
        k_mode = [4, 0, 0]
        source = self.create_plane_wave_source(k_mode)

        # Apply operator
        result = laplacian.apply(source)

        # Compute expected result
        kx, ky, kz = k_mode
        k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L
        expected_coefficient = k_magnitude ** (2 * self.beta)

        # Check that result is proportional to source
        ratio = result / source
        mean_ratio = np.mean(ratio)

        # Should be close to k^(2*beta)
        expected_ratio = expected_coefficient
        relative_error = abs(mean_ratio - expected_ratio) / expected_ratio

        assert (
            relative_error <= 1e-10
        ), f"Fractional Laplacian error {relative_error:.2e} exceeds tolerance"

        print(f"Test A0.1.5: Fractional Laplacian - Error: {relative_error:.2e}")

    def test_spectral_coefficients(self):
        """
        Test spectral coefficients computation.

        Physical Meaning:
            Tests that the spectral coefficients are computed correctly
            for the fractional Laplacian operator.
        """
        # Create fractional Laplacian
        laplacian = FractionalLaplacian(self.domain, self.beta, self.lambda_param)

        # Get spectral coefficients
        spectral_coeffs = laplacian.get_spectral_coefficients()

        # Check properties
        assert (
            spectral_coeffs.shape == self.domain.shape
        ), "Spectral coefficients shape mismatch"

        assert np.all(
            spectral_coeffs >= 0
        ), "Spectral coefficients should be non-negative"

        # Check k=0 mode (should be lambda_param)
        k0_coeff = spectral_coeffs[0, 0, 0]
        expected_k0 = self.lambda_param

        assert (
            abs(k0_coeff - expected_k0) <= 1e-12
        ), f"k=0 coefficient {k0_coeff} should be {expected_k0}"

        print(f"Test A0.1.6: Spectral coefficients - k=0 coeff: {k0_coeff}")

    def test_solver_validation(self):
        """
        Test solver validation functionality.

        Physical Meaning:
            Tests the built-in validation methods to ensure
            solution quality.
        """
        k_mode = [4, 0, 0]
        source = self.create_plane_wave_source(k_mode)

        # Solve
        solution = self.solver.solve_stationary(source)

        # Validate solution
        validation_results = self.solver.validate_solution(solution, source)

        # Check validation results
        assert (
            "residual_norm" in validation_results
        ), "Validation should include residual norm"

        assert (
            "relative_error" in validation_results
        ), "Validation should include relative error"

        residual_norm = validation_results["residual_norm"]
        relative_error = validation_results["relative_error"]

        assert (
            residual_norm <= 1e-12
        ), f"Residual norm {residual_norm:.2e} exceeds tolerance"

        assert (
            relative_error <= 1e-12
        ), f"Relative error {relative_error:.2e} exceeds tolerance"

        print(
            f"Test A0.1.7: Solver validation - Residual: {residual_norm:.2e}, Error: {relative_error:.2e}"
        )

    def test_solver_info(self):
        """
        Test solver information retrieval.

        Physical Meaning:
            Tests that solver information is correctly provided
            for debugging and monitoring.
        """
        # Get solver info
        solver_info = self.solver.get_solver_info()

        # Check required fields
        required_fields = [
            "domain_shape",
            "parameters",
            "spectral_coefficients_computed",
            "fft_plan_setup",
            "solver_type",
        ]

        for field in required_fields:
            assert field in solver_info, f"Solver info should include {field}"

        # Check values
        assert solver_info["domain_shape"] == self.domain.shape, "Domain shape mismatch"

        assert solver_info["solver_type"] == "basic", "Solver type should be 'basic'"

        assert solver_info[
            "spectral_coefficients_computed"
        ], "Spectral coefficients should be computed"

        assert solver_info["fft_plan_setup"], "FFT plan should be setup"

        print(f"Test A0.1.8: Solver info - {solver_info}")


if __name__ == "__main__":
    # Run tests
    test = TestA01PlaneWave()
    test.setup_method()

    try:
        test.test_plane_wave_single_mode()
        test.test_plane_wave_multiple_modes()
        test.test_anisotropy_check()
        test.test_grid_convergence()
        test.test_fractional_laplacian_operator()
        test.test_spectral_coefficients()
        test.test_solver_validation()
        test.test_solver_info()

        print("\n✅ All A0.1 tests PASSED!")

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        raise
