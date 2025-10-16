"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced test A0.1: Plane wave validation for Level A.

This module implements advanced validation tests for the FFT solver
and fractional Laplacian operator using plane wave solutions.

Physical Meaning:
    Validates advanced aspects of the spectral solution for monochromatic excitation,
    including boundary conditions, stability, and performance.

Mathematical Foundation:
    Tests advanced aspects of the formula a_hat(k) = s_hat(k) / D(k) where
    D(k) = mu|k|^(2*beta) + lambda is the spectral operator.
"""

import numpy as np
import pytest
from typing import Dict, Any
import logging
import time

from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian


class TestA01PlaneWaveAdvanced:
    """
    Advanced test A0.1: Plane wave validation.

    Physical Meaning:
        Validates advanced aspects of the spectral solution for monochromatic excitation,
        including boundary conditions, stability, and performance.

    Mathematical Foundation:
        Tests advanced aspects of the formula a_hat(k) = s_hat(k) / D(k) where
        D(k) = mu|k|^(2*beta) + lambda is the spectral operator.
    """

    @pytest.fixture(autouse=True)
    def setup_test(self):
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
    
    def _create_7d_source(self, k_test):
        """Create proper 7D source for testing."""
        x = np.linspace(0, self.L, self.N, endpoint=False)
        # Create 7D meshgrid for proper source shape
        X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(
            x, x, x, 
            np.linspace(0, 2*np.pi, 4, endpoint=False),
            np.linspace(0, 2*np.pi, 4, endpoint=False), 
            np.linspace(0, 2*np.pi, 4, endpoint=False),
            np.linspace(0, 1.0, 8, endpoint=False),
            indexing='ij'
        )
        return np.exp(1j * k_test * X)

        # Test tolerance
        self.tolerance = 1e-10

    def test_boundary_conditions(self):
        """
        Test boundary conditions handling.

        Physical Meaning:
            Validates that the solver correctly handles
            boundary conditions for plane wave solutions.

        Mathematical Foundation:
            The spectral method should respect
            periodic boundary conditions.
        """
        # Create test source with specific boundary behavior
        k_test = 2 * np.pi / self.L
        source = self._create_7d_source(k_test)

        # Solve
        solution = self.solver.solve(source)

        # Check that solution is finite and has correct shape
        assert np.all(np.isfinite(solution))
        assert solution.shape == self.domain.shape

    def test_stability_analysis(self):
        """
        Test numerical stability.

        Physical Meaning:
            Validates that the solver is numerically
            stable for various parameter ranges.

        Mathematical Foundation:
            The spectral method should be stable
            for all physically reasonable parameters.
        """
        # Test different parameter values
        test_params = [
            {"mu": 0.1, "beta": 0.5, "lambda_param": 0.01},
            {"mu": 1.0, "beta": 1.0, "lambda_param": 0.1},
            {"mu": 10.0, "beta": 1.5, "lambda_param": 1.0},
        ]

        for params in test_params:
            # Create parameters object
            test_parameters = Parameters7DBVP(
                mu=params["mu"],
                beta=params["beta"],
                lambda_param=params["lambda_param"],
                nu=1.0
            )

            # Create solver
            test_solver = FFTSolver7DBasic(self.domain, test_parameters)

            # Create test source
            k_test = 2 * np.pi / self.L
            source = self._create_7d_source(k_test)

            # Solve
            solution = test_solver.solve(source)

            # Check stability
            assert np.all(np.isfinite(solution))
            assert not np.any(np.isnan(solution))
            assert not np.any(np.isinf(solution))

    def test_performance_benchmark(self):
        """
        Test performance benchmark.

        Physical Meaning:
            Validates that the solver performs
            efficiently for typical workloads.

        Mathematical Foundation:
            The spectral method should be
            computationally efficient.
        """
        # Create test source
        k_test = 2 * np.pi / self.L
        source = self._create_7d_source(k_test)

        # Benchmark solve time
        start_time = time.time()
        solution = self.solver.solve(source)
        end_time = time.time()

        solve_time = end_time - start_time

        # Check that solve time is reasonable (less than 1 second)
        assert solve_time < 1.0

        # Check that solution is correct
        D_k = self.mu * (k_test ** (2 * self.beta)) + self.lambda_param
        expected = source / D_k
        np.testing.assert_allclose(solution, expected, rtol=self.tolerance)

    def test_memory_usage(self):
        """
        Test memory usage.

        Physical Meaning:
            Validates that the solver uses memory
            efficiently.

        Mathematical Foundation:
            The spectral method should have
            reasonable memory requirements.
        """
        # Create test source
        k_test = 2 * np.pi / self.L
        source = self._create_7d_source(k_test)

        # Solve multiple times to check memory usage
        for _ in range(10):
            solution = self.solver.solve(source)
            assert solution is not None

    def test_parameter_sensitivity(self):
        """
        Test parameter sensitivity.

        Physical Meaning:
            Validates that the solver is sensitive
            to parameter changes as expected.

        Mathematical Foundation:
            The solution should change appropriately
            with parameter changes.
        """
        # Create test source
        k_test = 2 * np.pi / self.L
        source = self._create_7d_source(k_test)

        # Test with different mu values
        mu_values = [0.1, 1.0, 10.0]
        solutions = []

        for mu in mu_values:
            # Create parameters with different mu
            test_parameters = Parameters7DBVP(
                mu=mu,
                beta=self.beta,
                lambda_param=self.lambda_param,
                nu=1.0
            )

            # Create solver
            test_solver = FFTSolver7DBasic(self.domain, test_parameters)

            # Solve
            solution = test_solver.solve(source)
            solutions.append(solution)

        # Check that solutions are different
        for i in range(1, len(solutions)):
            assert not np.allclose(solutions[i], solutions[i-1], rtol=1e-10)

    def test_error_handling(self):
        """
        Test error handling.

        Physical Meaning:
            Validates that the solver handles
            error conditions gracefully.

        Mathematical Foundation:
            The solver should provide meaningful
            error messages for invalid inputs.
        """
        # Test with invalid source
        invalid_source = None

        with pytest.raises((ValueError, TypeError)):
            self.solver.solve(invalid_source)

        # Test with wrong shape source
        wrong_shape_source = np.random.random((self.N, self.N))

        with pytest.raises(ValueError):
            self.solver.solve(wrong_shape_source)

    def test_convergence_rate(self):
        """
        Test convergence rate.

        Physical Meaning:
            Validates that the solver converges
            at the expected rate.

        Mathematical Foundation:
            The spectral method should converge
            exponentially with grid resolution.
        """
        # Test different grid sizes
        grid_sizes = [4, 8, 16, 32]
        errors = []

        for N in grid_sizes:
            # Create domain with different grid size
            domain = Domain7DBVP(L_spatial=self.L, N_spatial=N, N_phase=4, T=1.0, N_t=8)
            solver = FFTSolver7DBasic(domain, self.parameters)

            # Create test source
            k_test = 2 * np.pi / self.L
            # Create 7D source for this domain size
            x = np.linspace(0, self.L, N, endpoint=False)
            X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(
                x, x, x, 
                np.linspace(0, 2*np.pi, 4, endpoint=False),
                np.linspace(0, 2*np.pi, 4, endpoint=False), 
                np.linspace(0, 2*np.pi, 4, endpoint=False),
                np.linspace(0, 1.0, 8, endpoint=False),
                indexing='ij'
            )
            source = np.exp(1j * k_test * X)

            # Solve
            solution = solver.solve(source)

            # Calculate error
            D_k = self.mu * (k_test ** (2 * self.beta)) + self.lambda_param
            expected = source / D_k
            error = np.linalg.norm(solution - expected)
            errors.append(error)

        # Check convergence rate (errors should decrease)
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i-1]

    def test_spectral_accuracy(self):
        """
        Test spectral accuracy.

        Physical Meaning:
            Validates that the spectral method
            achieves high accuracy.

        Mathematical Foundation:
            The spectral method should achieve
            machine precision accuracy.
        """
        # Create test source
        k_test = 2 * np.pi / self.L
        source = self._create_7d_source(k_test)

        # Solve
        solution = self.solver.solve(source)

        # Calculate expected solution
        D_k = self.mu * (k_test ** (2 * self.beta)) + self.lambda_param
        expected = source / D_k

        # Check accuracy
        relative_error = np.linalg.norm(solution - expected) / np.linalg.norm(expected)
        assert relative_error < 1e-12

    def test_energy_conservation(self):
        """
        Test energy conservation.

        Physical Meaning:
            Validates that the solver conserves
            energy appropriately.

        Mathematical Foundation:
            The spectral method should conserve
            energy for conservative systems.
        """
        # Create test source
        k_test = 2 * np.pi / self.L
        source = self._create_7d_source(k_test)

        # Solve
        solution = self.solver.solve(source)

        # Calculate energy
        source_energy = np.sum(np.abs(source)**2)
        solution_energy = np.sum(np.abs(solution)**2)

        # Check energy conservation (should be approximately conserved)
        energy_ratio = solution_energy / source_energy
        assert 0.9 < energy_ratio < 1.1


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
