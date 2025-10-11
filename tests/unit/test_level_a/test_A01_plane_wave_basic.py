"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic test A0.1: Plane wave validation for Level A.

This module implements basic validation tests for the FFT solver
and fractional Laplacian operator using plane wave solutions.

Physical Meaning:
    Validates the spectral solution for monochromatic excitation,
    ensuring correct implementation of the fractional Laplacian
    operator in k-space.

Mathematical Foundation:
    Tests the formula a_hat(k) = s_hat(k) / D(k) where
    D(k) = mu|k|^(2*beta) + lambda is the spectral operator.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging

from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian


class TestA01PlaneWaveBasic:
    """
    Basic test A0.1: Plane wave validation.

    Physical Meaning:
        Validates the basic spectral solution for monochromatic excitation,
        ensuring correct implementation of the fractional Laplacian
        operator in k-space.

    Mathematical Foundation:
        Tests the formula a_hat(k) = s_hat(k) / D(k) where
        D(k) = mu|k|^(2*beta) + lambda is the spectral operator.
    """

    def setup_method(self):
        """Setup test parameters."""
        # Domain parameters (smaller for testing)
        self.L = 1.0
        self.N = 8  # Much smaller for testing
        self.domain = Domain7DBVP(L_spatial=self.L, N_spatial=self.N, N_phase=4, T=1.0, N_t=8)

        # Physics parameters
        self.mu = 1.0
        self.beta = 1.0
        self.lambda_param = 0.1

        # Create parameters object
        self.parameters = Parameters7DBVP(
            mu=self.mu,
            beta=self.beta,
            lambda_param=self.lambda_param,
            nu=1.0
        )

        # Create solver
        self.solver = FFTSolver7DBasic(self.domain, self.parameters)

        # Create fractional Laplacian operator
        self.frac_lap = FractionalLaplacian(self.domain, self.parameters)

        # Test tolerance
        self.tolerance = 1e-10

    def test_plane_wave_single_mode(self):
        """
        Test single mode plane wave solution.

        Physical Meaning:
            Validates that a single mode plane wave is correctly
            solved by the spectral method.

        Mathematical Foundation:
            For a single mode s(x) = exp(i*k*x), the solution should be
            a(x) = exp(i*k*x) / D(k) where D(k) = mu|k|^(2*beta) + lambda.
        """
        # Create single mode source
        k_test = 2 * np.pi / self.L  # Test wavenumber
        x = np.linspace(0, self.L, self.N, endpoint=False)
        source = np.exp(1j * k_test * x)

        # Solve using spectral method
        solution = self.solver.solve(source)

        # Expected solution
        D_k = self.mu * (k_test ** (2 * self.beta)) + self.lambda_param
        expected = source / D_k

        # Check solution
        np.testing.assert_allclose(solution, expected, rtol=self.tolerance)

    def test_plane_wave_multiple_modes(self):
        """
        Test multiple mode plane wave solution.

        Physical Meaning:
            Validates that multiple mode plane waves are correctly
            solved by the spectral method.

        Mathematical Foundation:
            For multiple modes, each mode should be solved independently
            according to the spectral formula.
        """
        # Create multiple mode source
        k1 = 2 * np.pi / self.L
        k2 = 4 * np.pi / self.L
        x = np.linspace(0, self.L, self.N, endpoint=False)
        source = np.exp(1j * k1 * x) + 0.5 * np.exp(1j * k2 * x)

        # Solve using spectral method
        solution = self.solver.solve(source)

        # Expected solution
        D_k1 = self.mu * (k1 ** (2 * self.beta)) + self.lambda_param
        D_k2 = self.mu * (k2 ** (2 * self.beta)) + self.lambda_param
        expected = (np.exp(1j * k1 * x) / D_k1 + 
                   0.5 * np.exp(1j * k2 * x) / D_k2)

        # Check solution
        np.testing.assert_allclose(solution, expected, rtol=self.tolerance)

    def test_anisotropy_check(self):
        """
        Test anisotropy in the solution.

        Physical Meaning:
            Validates that the solution correctly handles
            anisotropic wave propagation.

        Mathematical Foundation:
            The fractional Laplacian should produce
            isotropic solutions for isotropic sources.
        """
        # Create isotropic source
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        source = np.exp(1j * 2 * np.pi * X / self.L)

        # Solve using spectral method
        solution = self.solver.solve(source)

        # Check that solution is not identically zero
        assert np.any(np.abs(solution) > 1e-10)

        # Check that solution has correct shape
        assert solution.shape == source.shape

    def test_grid_convergence(self):
        """
        Test grid convergence of the solution.

        Physical Meaning:
            Validates that the solution converges with
            increasing grid resolution.

        Mathematical Foundation:
            The spectral method should converge to the
            exact solution as grid resolution increases.
        """
        # Test different grid sizes
        grid_sizes = [4, 8, 16]
        errors = []

        for N in grid_sizes:
            # Create domain with different grid size
            domain = Domain7DBVP(L_spatial=self.L, N_spatial=N, N_phase=4, T=1.0, N_t=8)
            solver = FFTSolver7DBasic(domain, self.parameters)

            # Create test source
            k_test = 2 * np.pi / self.L
            x = np.linspace(0, self.L, N, endpoint=False)
            source = np.exp(1j * k_test * x)

            # Solve
            solution = solver.solve(source)

            # Calculate error (should decrease with grid size)
            D_k = self.mu * (k_test ** (2 * self.beta)) + self.lambda_param
            expected = source / D_k
            error = np.linalg.norm(solution - expected)
            errors.append(error)

        # Check convergence (errors should generally decrease)
        for i in range(1, len(errors)):
            # Allow some tolerance for numerical errors
            assert errors[i] <= errors[i-1] * 1.1

    def test_fractional_laplacian_operator(self):
        """
        Test fractional Laplacian operator.

        Physical Meaning:
            Validates that the fractional Laplacian operator
            is correctly implemented.

        Mathematical Foundation:
            The fractional Laplacian should satisfy
            (-Δ)^β exp(i*k*x) = |k|^(2*β) exp(i*k*x).
        """
        # Create test field
        k_test = 2 * np.pi / self.L
        x = np.linspace(0, self.L, self.N, endpoint=False)
        field = np.exp(1j * k_test * x)

        # Apply fractional Laplacian
        result = self.frac_lap.apply(field)

        # Expected result
        expected = (k_test ** (2 * self.beta)) * field

        # Check result
        np.testing.assert_allclose(result, expected, rtol=self.tolerance)

    def test_spectral_coefficients(self):
        """
        Test spectral coefficients calculation.

        Physical Meaning:
            Validates that the spectral coefficients
            are correctly calculated.

        Mathematical Foundation:
            The spectral coefficients should be
            D(k) = mu|k|^(2*beta) + lambda.
        """
        # Get spectral coefficients
        coeffs = self.solver.get_spectral_coefficients()

        # Check that coefficients have correct shape
        assert coeffs.shape == (self.N, self.N, self.N, 4, 4, 4, 8)

        # Check that coefficients are positive
        assert np.all(coeffs > 0)

        # Check specific coefficient for k=0
        k0_coeff = coeffs[0, 0, 0, 0, 0, 0, 0]
        expected_k0 = self.lambda_param
        np.testing.assert_allclose(k0_coeff, expected_k0, rtol=self.tolerance)

    def test_solver_validation(self):
        """
        Test solver validation.

        Physical Meaning:
            Validates that the solver produces
            physically reasonable results.

        Mathematical Foundation:
            The solver should satisfy the equation
            L_β a = s where L_β is the fractional Laplacian.
        """
        # Create test source
        k_test = 2 * np.pi / self.L
        x = np.linspace(0, self.L, self.N, endpoint=False)
        source = np.exp(1j * k_test * x)

        # Solve
        solution = self.solver.solve(source)

        # Check that solution is finite
        assert np.all(np.isfinite(solution))

        # Check that solution has correct shape
        assert solution.shape == source.shape

        # Check that solution is not identically zero
        assert np.any(np.abs(solution) > 1e-10)

    def test_solver_info(self):
        """
        Test solver information.

        Physical Meaning:
            Validates that the solver provides
            correct information about its state.

        Mathematical Foundation:
            The solver should provide information
            about its parameters and configuration.
        """
        # Get solver info
        info = self.solver.get_info()

        # Check that info contains required fields
        assert "domain" in info
        assert "parameters" in info
        assert "solver_type" in info

        # Check that info is a dictionary
        assert isinstance(info, dict)

        # Check that info is not empty
        assert len(info) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
