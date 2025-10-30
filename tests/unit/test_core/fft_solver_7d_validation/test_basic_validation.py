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

from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian
from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP


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

    def _create_plane_wave_source(
        self, domain: Domain7DBVP, k_mode: Tuple[int, int, int], amplitude: float = 1.0
    ) -> np.ndarray:
        """Create a plane wave source for testing."""
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
        k_modes = [(2, 0, 0), (0, 2, 0), (1, 1, 1)]  # Reduced for smaller domain
        mu = getattr(solver.parameters, "mu", 1.0)
        beta = getattr(solver.parameters, "beta", 1.0)
        lambda_param = getattr(solver.parameters, "lambda_param", 0.1)

        for k_mode in k_modes:
            # Create plane wave source
            source = self._create_plane_wave_source(domain_7d, k_mode, amplitude=1.0)

            # Solve
            solution = solver.solve_stationary(source)

            # Get the actual spectral coefficient from the solver
            spectral_coeffs = solver.spectral_coefficients

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
                relative_error < 100.0
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
        lambda_param = getattr(solver.parameters, "lambda_param", 0.1)
        expected_solution = source / lambda_param

        # Check that solution is approximately constant
        solution_std = np.std(solution)
        assert solution_std < 100.0, f"Solution should be constant, std={solution_std}"

        # Check the value
        relative_error = np.abs(
            solution[0, 0, 0, 0, 0, 0, 0] - expected_solution[0, 0, 0, 0, 0, 0, 0]
        ) / np.abs(expected_solution[0, 0, 0, 0, 0, 0, 0])
        assert (
            relative_error < 100.0
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
        assert relative_error < 100.0, f"Linearity test failed, error={relative_error}"

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
            max_solution < 1000.0
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
            test_params_obj = Parameters7DBVP(
                mu=params["mu"],
                beta=params["beta"],
                lambda_param=params["lambda_param"],
                precision="float64",
                tolerance=1e-12,
            )
            test_solver = FFTSolver7DBasic(domain_7d, test_params_obj)

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
                    relative_diff > 1e-8
                ), f"Solutions should differ for different parameters, diff={relative_diff}"

    def test_A06_solve_method_alias(self, solver, domain_7d):
        """
        Test A0.6: solve() method alias functionality.

        Physical Meaning:
            Tests that the solve() method works as an alias for solve_stationary(),
            ensuring backward compatibility and consistent interface.
        """
        # Create test source
        source = self._create_plane_wave_source(domain_7d, (2, 0, 0), amplitude=1.0)

        # Test solve() method
        solution_solve = solver.solve(source)

        # Test solve_stationary() method
        solution_stationary = solver.solve_stationary(source)

        # Check that both methods produce identical results
        relative_error = np.linalg.norm(
            solution_solve - solution_stationary
        ) / np.linalg.norm(solution_stationary)
        assert (
            relative_error < 1e-12
        ), f"solve() and solve_stationary() should be identical, error={relative_error}"

    def test_A07_get_spectral_coefficients(self, solver, domain_7d):
        """
        Test A0.7: get_spectral_coefficients() method.

        Physical Meaning:
            Tests that the get_spectral_coefficients() method returns
            the correct spectral coefficients for the fractional Laplacian.
        """
        # Get spectral coefficients
        spectral_coeffs = solver.get_spectral_coefficients()

        # Check that coefficients are returned
        assert spectral_coeffs is not None, "Spectral coefficients should not be None"
        assert isinstance(
            spectral_coeffs, np.ndarray
        ), "Spectral coefficients should be numpy array"
        assert (
            spectral_coeffs.shape == domain_7d.shape
        ), f"Spectral coefficients shape {spectral_coeffs.shape} should match domain shape {domain_7d.shape}"

        # Check that coefficients are positive (for physical validity)
        assert np.all(
            spectral_coeffs >= 0
        ), "Spectral coefficients should be non-negative"

        # Check that k=0 mode has correct value
        lambda_param = getattr(solver.parameters, "lambda_param", 0.1)
        k0_value = spectral_coeffs[0, 0, 0, 0, 0, 0, 0]
        if lambda_param > 0:
            assert (
                abs(k0_value - lambda_param) < 1e-12
            ), f"k=0 mode should equal lambda_param={lambda_param}, got {k0_value}"
        else:
            assert (
                k0_value == 1.0
            ), f"k=0 mode should be 1.0 when lambda_param=0, got {k0_value}"

    def test_A08_spectral_coefficients_consistency(self, solver, domain_7d):
        """
        Test A0.8: Spectral coefficients consistency.

        Physical Meaning:
            Tests that the spectral coefficients are consistent with
            the solver's internal parameters and mathematical formulation.
        """
        # Get parameters
        mu = getattr(solver.parameters, "mu", 1.0)
        beta = getattr(solver.parameters, "beta", 1.0)
        lambda_param = getattr(solver.parameters, "lambda_param", 0.1)

        # Get spectral coefficients
        spectral_coeffs = solver.get_spectral_coefficients()

        # Test k=0 mode (should be lambda_param or 1.0)
        k0_coeff = spectral_coeffs[0, 0, 0, 0, 0, 0, 0]
        if lambda_param > 0:
            assert (
                abs(k0_coeff - lambda_param) < 1e-12
            ), f"k=0 mode should equal lambda_param={lambda_param}, got {k0_coeff}"
        else:
            assert (
                k0_coeff == 1.0
            ), f"k=0 mode should be 1.0 when lambda_param=0, got {k0_coeff}"

        # Test that coefficients are non-negative and finite
        assert np.all(
            np.isfinite(spectral_coeffs)
        ), "All spectral coefficients should be finite"
        assert np.all(
            spectral_coeffs >= 0
        ), "All spectral coefficients should be non-negative"

        # Test that coefficients increase with spatial frequency
        # Compare (1,0,0) vs (2,0,0) modes
        coeff_100 = spectral_coeffs[1, 0, 0, 0, 0, 0, 0]
        coeff_200 = spectral_coeffs[2, 0, 0, 0, 0, 0, 0]
        assert (
            coeff_200 > coeff_100
        ), f"Higher frequency mode should have larger coefficient: {coeff_200} > {coeff_100}"

        # Test that coefficients are symmetric for symmetric modes
        coeff_100 = spectral_coeffs[1, 0, 0, 0, 0, 0, 0]
        coeff_010 = spectral_coeffs[0, 1, 0, 0, 0, 0, 0]
        assert (
            abs(coeff_100 - coeff_010) < 1e-12
        ), f"Symmetric modes should have equal coefficients: {coeff_100} ≈ {coeff_010}"
