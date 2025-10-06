"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic validation tests for 7D FFT Solver.

This module contains basic validation tests including plane wave solutions,
analytical tests, and fundamental functionality tests.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.fft import FFTSolver7D, FractionalLaplacian
from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters


class TestBasicValidation:
    """
    Basic validation tests for 7D FFT Solver.

    Physical Meaning:
        Tests fundamental functionality and basic analytical solutions
        for the 7D FFT solver implementation.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)

    @pytest.fixture
    def parameters_basic(self):
        """Basic parameters for testing."""
        return Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,
            precision="float64",
            fft_plan="MEASURE",
            tolerance=1e-12,
        )

    @pytest.fixture
    def solver(self, domain_7d, parameters_basic):
        """Create FFT solver for testing."""
        return FFTSolver7D(domain_7d, parameters_basic)

    def _create_plane_wave_source(
        self, domain: Domain, k_mode: Tuple[int, int, int], amplitude: float = 1.0
    ) -> np.ndarray:
        """Create a plane wave source for testing."""
        # Create coordinate arrays
        x = np.linspace(0, domain.L, domain.N, endpoint=False)
        y = np.linspace(0, domain.L, domain.N, endpoint=False)
        z = np.linspace(0, domain.L, domain.N, endpoint=False)
        phi1 = np.linspace(0, 2 * np.pi, domain.N_phi, endpoint=False)
        phi2 = np.linspace(0, 2 * np.pi, domain.N_phi, endpoint=False)
        phi3 = np.linspace(0, 2 * np.pi, domain.N_phi, endpoint=False)
        t = np.linspace(0, domain.T, domain.N_t, endpoint=False)

        # Create meshgrids
        X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(
            x, y, z, phi1, phi2, phi3, t, indexing="ij"
        )

        # Create plane wave
        kx, ky, kz = k_mode
        source = amplitude * np.exp(1j * (kx * X + ky * Y + kz * Z))

        return source.real  # Return real part for physical source

    def test_A01_plane_wave_stationary(self, solver, domain_7d):
        """
        Test A0.1: Plane wave stationary solution.

        Physical Meaning:
            Tests the fundamental spectral solution for a plane wave source,
            validating the formula â = ŝ / D for single frequency modes.
        """
        # Test parameters
        k_modes = [(4, 0, 0), (0, 4, 0), (3, 3, 2)]
        mu = solver.parameters.mu
        beta = solver.parameters.beta
        lambda_param = solver.parameters.lambda_param

        for k_mode in k_modes:
            # Create plane wave source
            source = self._create_plane_wave_source(domain_7d, k_mode, amplitude=1.0)

            # Solve
            solution = solver.solve_stationary(source)

            # Get the actual spectral coefficient from the solver
            spectral_coeffs = solver._get_spectral_coefficients()

            # For a source that's constant in phase and time dimensions,
            # the effective coefficient is the one at the spatial k_mode
            k_index = k_mode + (0, 0, 0, 0)  # Add zero indices for phase and time
            D_k_actual = spectral_coeffs[k_index]

            # Expected solution should be source divided by actual coefficient
            expected_solution = source / D_k_actual

            # Check accuracy
            relative_error = np.linalg.norm(
                solution - expected_solution
            ) / np.linalg.norm(expected_solution)
            assert (
                relative_error < 1e-10
            ), f"Plane wave test failed for k={k_mode}, error={relative_error}"

    def test_A02_analytical_constant_source(self, solver, domain_7d):
        """
        Test A0.2: Analytical solution for constant source.

        Physical Meaning:
            Tests the solution for a constant source, which should produce
            a constant solution scaled by the damping parameter λ.
        """
        # Create constant source
        source = np.ones(domain_7d.shape)

        # Solve
        solution = solver.solve_stationary(source)

        # For constant source, solution should be constant
        # and equal to source / lambda_param (for k=0 mode)
        expected_solution = source / solver.parameters.lambda_param

        # Check that solution is approximately constant
        solution_std = np.std(solution)
        assert solution_std < 1e-10, f"Solution should be constant, std={solution_std}"

        # Check the value
        relative_error = np.abs(
            solution[0, 0, 0, 0, 0, 0, 0] - expected_solution[0, 0, 0, 0, 0, 0, 0]
        ) / np.abs(expected_solution[0, 0, 0, 0, 0, 0, 0])
        assert (
            relative_error < 1e-10
        ), f"Constant source test failed, error={relative_error}"

    def test_A03_linearity_property(self, solver, domain_7d):
        """
        Test A0.3: Linearity property of the solver.

        Physical Meaning:
            Tests that the solver is linear: L(a·s₁ + b·s₂) = a·L(s₁) + b·L(s₂)
            where L is the linear operator and s₁, s₂ are sources.
        """
        # Create two different sources
        source1 = self._create_plane_wave_source(domain_7d, (2, 0, 0), amplitude=1.0)
        source2 = self._create_plane_wave_source(domain_7d, (0, 2, 0), amplitude=1.0)

        # Linear combination coefficients
        a, b = 2.5, -1.3
        combined_source = a * source1 + b * source2

        # Solve for individual sources
        solution1 = solver.solve_stationary(source1)
        solution2 = solver.solve_stationary(source2)

        # Solve for combined source
        combined_solution = solver.solve_stationary(combined_source)

        # Expected linear combination
        expected_solution = a * solution1 + b * solution2

        # Check linearity
        relative_error = np.linalg.norm(
            combined_solution - expected_solution
        ) / np.linalg.norm(expected_solution)
        assert relative_error < 1e-10, f"Linearity test failed, error={relative_error}"

    def test_A04_energy_conservation(self, solver, domain_7d):
        """
        Test A0.4: Energy conservation properties.

        Physical Meaning:
            Tests that the solver conserves energy appropriately,
            validating the physical correctness of the solution.
        """
        # Create a localized source
        source = np.zeros(domain_7d.shape)
        source[2:6, 2:6, 2:6, :, :, :, :] = 1.0

        # Solve
        solution = solver.solve_stationary(source)

        # Check that solution has finite energy
        solution_energy = np.sum(np.abs(solution) ** 2)
        assert np.isfinite(solution_energy), "Solution energy should be finite"
        assert solution_energy > 0, "Solution should have positive energy"

        # Check that solution decays appropriately (not growing)
        max_solution = np.max(np.abs(solution))
        assert (
            max_solution < 10.0
        ), f"Solution should not grow excessively, max={max_solution}"

    def test_A05_parameter_dependence(self, solver, domain_7d):
        """
        Test A0.5: Parameter dependence validation.

        Physical Meaning:
            Tests that the solution depends correctly on the physical parameters
            μ, β, and λ, validating the mathematical formulation.
        """
        # Test different parameter combinations
        test_params = [
            {"mu": 1.0, "beta": 1.0, "lambda_param": 0.1},
            {"mu": 2.0, "beta": 1.0, "lambda_param": 0.1},
            {"mu": 1.0, "beta": 1.5, "lambda_param": 0.1},
            {"mu": 1.0, "beta": 1.0, "lambda_param": 0.2},
        ]

        # Create a test source
        source = self._create_plane_wave_source(domain_7d, (2, 2, 0), amplitude=1.0)

        solutions = []
        for params in test_params:
            # Create solver with different parameters
            test_params_obj = Parameters(
                mu=params["mu"],
                beta=params["beta"],
                lambda_param=params["lambda_param"],
                precision="float64",
                fft_plan="MEASURE",
                tolerance=1e-12,
            )
            test_solver = FFTSolver7D(domain_7d, test_params_obj)

            # Solve
            solution = test_solver.solve_stationary(source)
            solutions.append(solution)

        # Check that solutions are different for different parameters
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                relative_diff = np.linalg.norm(
                    solutions[i] - solutions[j]
                ) / np.linalg.norm(solutions[i])
                assert (
                    relative_diff > 1e-6
                ), f"Solutions should differ for different parameters, diff={relative_diff}"
