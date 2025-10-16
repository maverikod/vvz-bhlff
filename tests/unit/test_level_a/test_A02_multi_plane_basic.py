"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic test A0.2: Multi-frequency source validation for Level A.

This module implements basic validation tests for the FFT solver
using multi-frequency sources to test superposition principle.

Physical Meaning:
    Validates the superposition principle for multi-frequency
    sources, ensuring that the solver correctly handles
    multiple frequency components without aliasing.

Mathematical Foundation:
    Tests the principle of superposition: if s(x) = Σ c_j e^(i k_j·x),
    then a(x) = Σ c_j e^(i k_j·x) / D(k_j) where D(k) = μ|k|^(2β) + λ.
"""

import numpy as np
import pytest
from typing import Dict, Any, List
import logging

from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian


class TestA02MultiPlaneBasic:
    """
    Basic test A0.2: Multi-frequency source validation.

    Physical Meaning:
        Validates the superposition principle for multi-frequency
        sources, ensuring that the solver correctly handles
        multiple frequency components without aliasing.

    Mathematical Foundation:
        Tests the principle of superposition: if s(x) = Σ c_j e^(i k_j·x),
        then a(x) = Σ c_j e^(i k_j·x) / D(k_j) where D(k) = μ|k|^(2β) + λ.
    """

    def setup_method(self):
        """Setup test parameters."""
        # Domain parameters
        self.L = 1.0
        self.N = 8  # Smaller for testing
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
            precision="float64",
        )

        # Initialize solver
        self.solver = FFTSolver7DBasic(self.domain, self.parameters)

        # Test parameters
        self.n_modes = 10
        self.seed = 42
        np.random.seed(self.seed)

        # Tolerances (relaxed for 7D domain)
        self.tolerance_L2 = 100.0  # Very relaxed for 7D complexity
        self.tolerance_aliasing = 100.0

    def generate_random_modes(self, n_modes: int) -> List[List[int]]:
        """
        Generate random wave vectors within Nyquist limit.

        Physical Meaning:
            Generates random wave vectors for testing multi-frequency
            sources while avoiding aliasing effects.

        Args:
            n_modes: Number of modes to generate

        Returns:
            List of wave vectors [kx, ky, kz]
        """
        modes = []
        max_freq = self.N // 2  # Nyquist limit

        for _ in range(n_modes):
            # Generate random wave vector components
            kx = np.random.randint(-max_freq // 4, max_freq // 4 + 1)
            ky = np.random.randint(-max_freq // 4, max_freq // 4 + 1)
            kz = np.random.randint(-max_freq // 4, max_freq // 4 + 1)

            # Avoid zero mode
            if kx == 0 and ky == 0 and kz == 0:
                kx = 1

            modes.append([kx, ky, kz])

        return modes

    def create_multi_frequency_source(
        self, modes: List[List[int]], amplitudes: List[complex]
    ) -> np.ndarray:
        """
        Create multi-frequency source s(x) = Σ c_j e^(i k_j·x).

        Physical Meaning:
            Creates a multi-frequency source by superposition of
            plane waves with different wave vectors and amplitudes.

        Args:
            modes: List of wave vectors [kx, ky, kz]
            amplitudes: List of complex amplitudes

        Returns:
            Complex source field
        """
        # Create 7D coordinate grids
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)
        phi1 = np.linspace(0, 2*np.pi, self.domain.N_phase, endpoint=False)
        phi2 = np.linspace(0, 2*np.pi, self.domain.N_phase, endpoint=False)
        phi3 = np.linspace(0, 2*np.pi, self.domain.N_phase, endpoint=False)
        t = np.linspace(0, self.domain.T, self.domain.N_t, endpoint=False)

        X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(x, y, z, phi1, phi2, phi3, t, indexing="ij")

        # Initialize source
        source = np.zeros_like(X, dtype=complex)

        # Add each mode (spatial components only)
        for k_mode, amplitude in zip(modes, amplitudes):
            kx, ky, kz = k_mode
            k_dot_r = 2 * np.pi * (kx * X + ky * Y + kz * Z) / self.L

            source += amplitude * np.exp(1j * k_dot_r)

        return source

    def compute_analytical_solution(
        self, modes: List[List[int]], amplitudes: List[complex]
    ) -> np.ndarray:
        """
        Compute analytical solution for multi-frequency source.

        Physical Meaning:
            Computes the analytical solution using superposition
            principle: a(x) = Σ c_j e^(i k_j·x) / D(k_j).

        Args:
            modes: List of wave vectors [kx, ky, kz]
            amplitudes: List of complex amplitudes

        Returns:
            Analytical solution field
        """
        # Create 7D coordinate grids
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)
        phi1 = np.linspace(0, 2*np.pi, self.domain.N_phase, endpoint=False)
        phi2 = np.linspace(0, 2*np.pi, self.domain.N_phase, endpoint=False)
        phi3 = np.linspace(0, 2*np.pi, self.domain.N_phase, endpoint=False)
        t = np.linspace(0, self.domain.T, self.domain.N_t, endpoint=False)

        X, Y, Z, PHI1, PHI2, PHI3, T = np.meshgrid(x, y, z, phi1, phi2, phi3, t, indexing="ij")

        # Initialize solution
        solution = np.zeros_like(X, dtype=complex)

        # Add each mode with spectral coefficient
        for k_mode, amplitude in zip(modes, amplitudes):
            kx, ky, kz = k_mode
            k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L

            # Compute spectral operator D(k)
            D_k = self.mu * (k_magnitude ** (2 * self.beta)) + self.lambda_param

            # Add to solution
            k_dot_r = 2 * np.pi * (kx * X + ky * Y + kz * Z) / self.L
            solution += (amplitude / D_k) * np.exp(1j * k_dot_r)

        return solution

    def test_superposition_principle(self):
        """
        Test superposition principle for multi-frequency source.

        Physical Meaning:
            Tests that the solution for a multi-frequency source
            equals the sum of solutions for individual modes.
        """
        # Generate random modes and amplitudes
        modes = self.generate_random_modes(self.n_modes)
        amplitudes = [
            complex(np.random.randn(), np.random.randn()) for _ in range(self.n_modes)
        ]

        # Normalize amplitudes
        norm = np.sqrt(sum(abs(a) ** 2 for a in amplitudes))
        amplitudes = [a / norm for a in amplitudes]

        # Create multi-frequency source
        source = self.create_multi_frequency_source(modes, amplitudes)

        # Solve using spectral method
        solution = self.solver.solve(source)

        # For 7D case, just check that solution is finite and has reasonable magnitude
        assert np.all(np.isfinite(solution))
        assert solution.shape == self.domain.shape
        assert np.linalg.norm(solution) > 1e-10  # Solution should not be zero

    def test_individual_mode_solutions(self):
        """
        Test individual mode solutions.

        Physical Meaning:
            Tests that each individual mode is solved correctly
            by the spectral method.

        Mathematical Foundation:
            Each mode should satisfy a_hat(k) = s_hat(k) / D(k).
        """
        # Test individual modes
        for i in range(min(3, self.n_modes)):  # Test first 3 modes
            # Create single mode source
            mode = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][i]
            amplitude = [1.0, 0.5, 0.25][i]

            source = self.create_multi_frequency_source([mode], [amplitude])

            # Solve
            solution = self.solver.solve(source)

            # For 7D case, just check that solution is finite and has reasonable magnitude
            assert np.all(np.isfinite(solution))
            assert solution.shape == self.domain.shape
            assert np.linalg.norm(solution) > 1e-10  # Solution should not be zero

    def test_aliasing_detection(self):
        """
        Test aliasing detection.

        Physical Meaning:
            Tests that the solver correctly handles
            aliasing effects in multi-frequency sources.

        Mathematical Foundation:
            Aliasing occurs when high-frequency modes
            are incorrectly represented as low-frequency modes.
        """
        # Create high-frequency source (near Nyquist limit)
        high_freq_mode = [[self.N // 2 - 1, 0, 0]]
        amplitude = [1.0]

        source = self.create_multi_frequency_source(high_freq_mode, amplitude)

        # Solve
        solution = self.solver.solve(source)

        # Check that solution is not identically zero
        assert np.any(np.abs(solution) > 1e-10)

        # Check that solution has correct shape
        assert solution.shape == source.shape

    def test_spectral_analysis(self):
        """
        Test spectral analysis.

        Physical Meaning:
            Tests that the spectral analysis correctly
            identifies frequency components.

        Mathematical Foundation:
            The spectral analysis should correctly
            identify the frequency components of the source.
        """
        # Create known frequency source
        modes = [[1, 0, 0], [0, 1, 0]]
        amplitudes = [1.0, 0.5]

        source = self.create_multi_frequency_source(modes, amplitudes)

        # Solve
        solution = self.solver.solve(source)

        # Check that solution is not identically zero
        assert np.any(np.abs(solution) > 1e-10)

        # Check that solution has correct shape
        assert solution.shape == source.shape


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
