"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Numerical validation tests for 7D FFT Solver.

This module contains numerical validation tests including convergence tests,
boundary condition tests, and numerical stability tests.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian
from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP


class TestNumericalValidation:
    """
    Numerical validation tests for 7D FFT Solver.

    Physical Meaning:
        Tests numerical accuracy, convergence, and stability
        of the 7D FFT solver implementation.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain7DBVP(L_spatial=1.0, N_spatial=8, N_phase=4, T=1.0, N_t=8)

    @pytest.fixture
    def parameters_basic(self):
        """Basic parameters for testing."""
        return Parameters7DBVP(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,
            precision="float64",
            tolerance=1e-12,
        )

    @pytest.fixture
    def solver(self, domain_7d, parameters_basic):
        """Create FFT solver for testing."""
        return FFTSolver7DBasic(domain_7d, parameters_basic)

    def _create_gaussian_source(
        self,
        domain: Domain7DBVP,
        center: Tuple[float, float, float],
        width: float = 0.1,
    ) -> np.ndarray:
        """Create a Gaussian source for testing."""
        # Create coordinate arrays
        x = np.linspace(0, domain.L_spatial, domain.N_spatial, endpoint=False)
        y = np.linspace(0, domain.L_spatial, domain.N_spatial, endpoint=False)
        z = np.linspace(0, domain.L_spatial, domain.N_spatial, endpoint=False)
        phi1 = np.linspace(0, 2 * np.pi, domain.N_phase, endpoint=False)
        phi2 = np.linspace(0, 2 * np.pi, domain.N_phase, endpoint=False)
        phi3 = np.linspace(0, 2 * np.pi, domain.N_phase, endpoint=False)
        t = np.linspace(0, domain.T, domain.N_t, endpoint=False)

        # Create meshgrids
        X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(
            x, y, z, phi1, phi2, phi3, t, indexing="ij"
        )

        # Create Gaussian
        cx, cy, cz = center
        source = np.exp(
            -((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * width**2)
        )

        return source

    def test_B01_convergence_test(self, domain_7d):
        """
        Test B0.1: Convergence test with increasing resolution.

        Physical Meaning:
            Tests that the solution converges as the grid resolution increases,
            validating the numerical accuracy of the solver.
        """
        # Test different resolutions
        resolutions = [4, 8, 16]
        solutions = []

        for N in resolutions:
            # Create domain with different resolution
            test_domain = Domain7DBVP(
                L_spatial=1.0, N_spatial=N, N_phase=4, T=1.0, N_t=8
            )
            test_params = Parameters7DBVP(
                mu=1.0,
                beta=1.0,
                lambda_param=0.1,
                precision="float64",
                tolerance=1e-12,
            )
            test_solver = FFTSolver7DBasic(test_domain, test_params)

            # Create Gaussian source
            source = self._create_gaussian_source(
                test_domain, (0.5, 0.5, 0.5), width=0.2
            )

            # Solve
            solution = test_solver.solve_stationary(source)
            solutions.append(solution)

        # Check convergence (solutions should become more similar with higher resolution)
        # Compare solutions at common points
        for i in range(len(solutions) - 1):
            sol1 = solutions[i]
            sol2 = solutions[i + 1]

            # Downsample higher resolution solution to compare
            if sol2.shape[0] > sol1.shape[0]:
                step = sol2.shape[0] // sol1.shape[0]
                sol2_downsampled = sol2[::step, ::step, ::step, :, :, :, :]
                sol2_downsampled = sol2_downsampled[
                    : sol1.shape[0], : sol1.shape[1], : sol1.shape[2], :, :, :, :
                ]
            else:
                sol2_downsampled = sol2

            # Solutions should be similar (converging)
            relative_diff = np.linalg.norm(sol1 - sol2_downsampled) / np.linalg.norm(
                sol1
            )
            assert (
                relative_diff < 0.5
            ), f"Solutions should converge, diff={relative_diff}"

    def test_B02_boundary_conditions(self, solver, domain_7d):
        """
        Test B0.2: Boundary condition handling.

        Physical Meaning:
            Tests that the solver handles boundary conditions correctly,
            ensuring proper behavior at domain boundaries.
        """
        # Create source with non-zero values at boundaries
        source = np.ones(domain_7d.shape)

        # Add boundary effects
        source[0, :, :, :, :, :, :] = 2.0  # Left boundary
        source[-1, :, :, :, :, :, :] = 2.0  # Right boundary
        source[:, 0, :, :, :, :, :] = 2.0  # Bottom boundary
        source[:, -1, :, :, :, :, :] = 2.0  # Top boundary

        # Solve
        solution = solver.solve_stationary(source)

        # Check that solution is finite everywhere
        assert np.all(np.isfinite(solution)), "Solution should be finite everywhere"

        # Check that solution doesn't have extreme values
        max_solution = np.max(np.abs(solution))
        assert (
            max_solution < 100.0
        ), f"Solution should not have extreme values, max={max_solution}"

    def test_B03_numerical_stability(self, solver, domain_7d):
        """
        Test B0.3: Numerical stability test.

        Physical Meaning:
            Tests that the solver is numerically stable for various
            parameter combinations and source configurations.
        """
        # Test with different parameter combinations
        test_cases = [
            {"mu": 1e-6, "beta": 0.5, "lambda_param": 1e-6},  # Very small parameters
            {"mu": 1e6, "beta": 1.5, "lambda_param": 1e6},  # Very large parameters
            {"mu": 1.0, "beta": 0.1, "lambda_param": 0.0},  # Zero damping
            {"mu": 1.0, "beta": 1.9, "lambda_param": 0.1},  # High fractional order
        ]

        for case in test_cases:
            # Create solver with test parameters
            test_params = Parameters7DBVP(
                mu=case["mu"],
                beta=case["beta"],
                lambda_param=case["lambda_param"],
                precision="float64",
                tolerance=1e-12,
            )
            test_solver = FFTSolver7DBasic(domain_7d, test_params)

            # Create test source
            source = self._create_gaussian_source(domain_7d, (0.5, 0.5, 0.5), width=0.3)

            # Solve
            solution = test_solver.solve_stationary(source)

            # Check stability
            assert np.all(
                np.isfinite(solution)
            ), f"Solution should be finite for case {case}"
            assert not np.any(
                np.isnan(solution)
            ), f"Solution should not contain NaN for case {case}"
            assert not np.any(
                np.isinf(solution)
            ), f"Solution should not contain Inf for case {case}"

    def test_B04_precision_validation(self, domain_7d):
        """
        Test B0.4: Precision validation test.

        Physical Meaning:
            Tests that the solver maintains appropriate precision
            for different numerical precision settings.
        """
        # Test different precision settings
        precisions = ["float32", "float64"]
        solutions = []

        for precision in precisions:
            test_params = Parameters7DBVP(
                mu=1.0,
                beta=1.0,
                lambda_param=0.1,
                precision=precision,
                tolerance=1e-12,
            )
            test_solver = FFTSolver7DBasic(domain_7d, test_params)

            # Create test source
            source = self._create_gaussian_source(domain_7d, (0.5, 0.5, 0.5), width=0.2)

            # Solve
            solution = test_solver.solve_stationary(source)
            solutions.append(solution)

        # Check that solutions are reasonable for both precisions
        for i, solution in enumerate(solutions):
            assert np.all(
                np.isfinite(solution)
            ), f"Solution should be finite for precision {precisions[i]}"
            assert (
                np.max(np.abs(solution)) < 100.0
            ), f"Solution should be bounded for precision {precisions[i]}"

    def test_B05_spectral_accuracy(self, solver, domain_7d):
        """
        Test B0.5: Spectral accuracy validation.

        Physical Meaning:
            Tests that the spectral representation is accurate,
            validating the FFT operations and spectral coefficients.
        """
        # Create a simple sinusoidal source
        x = np.linspace(0, domain_7d.L_spatial, domain_7d.N_spatial, endpoint=False)
        y = np.linspace(0, domain_7d.L_spatial, domain_7d.N_spatial, endpoint=False)
        z = np.linspace(0, domain_7d.L_spatial, domain_7d.N_spatial, endpoint=False)
        phi1 = np.linspace(0, 2 * np.pi, domain_7d.N_phase, endpoint=False)
        phi2 = np.linspace(0, 2 * np.pi, domain_7d.N_phase, endpoint=False)
        phi3 = np.linspace(0, 2 * np.pi, domain_7d.N_phase, endpoint=False)
        t = np.linspace(0, domain_7d.T, domain_7d.N_t, endpoint=False)

        X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(
            x, y, z, phi1, phi2, phi3, t, indexing="ij"
        )

        # Create sinusoidal source
        kx, ky, kz = 2, 1, 0
        source = np.sin(2 * np.pi * kx * X / domain_7d.L_spatial) * np.sin(
            2 * np.pi * ky * Y / domain_7d.L_spatial
        )

        # Solve
        solution = solver.solve_stationary(source)

        # Check that solution maintains the sinusoidal structure
        # (should be a scaled version of the source)
        correlation = np.corrcoef(source.flatten(), solution.flatten())[0, 1]
        assert (
            correlation > 0.9
        ), f"Solution should correlate with source, correlation={correlation}"

        # Check that solution is bounded
        max_solution = np.max(np.abs(solution))
        assert max_solution < 10.0, f"Solution should be bounded, max={max_solution}"
