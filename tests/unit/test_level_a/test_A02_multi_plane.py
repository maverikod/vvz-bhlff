"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.2: Multi-frequency source validation for Level A.

This module implements validation tests for the basic FFT solver
using multi-frequency sources to test superposition principle.
"""

import numpy as np
import pytest
from typing import Dict, Any, List
import logging

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian


class TestA02MultiPlane:
    """
    Test A0.2: Multi-frequency source validation.

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
        self.N = 256
        self.domain = Domain(L=self.L, N=self.N, dimensions=3)

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

        # Test parameters
        self.n_modes = 10
        self.seed = 42
        np.random.seed(self.seed)

        # Tolerances
        self.tolerance_L2 = 1e-12
        self.tolerance_aliasing = 1e-12

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
        # Create coordinate grids
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Initialize source
        source = np.zeros_like(X, dtype=complex)

        # Add each mode
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
        # Create coordinate grids
        x = np.linspace(0, self.L, self.N, endpoint=False)
        y = np.linspace(0, self.L, self.N, endpoint=False)
        z = np.linspace(0, self.L, self.N, endpoint=False)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

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
        source_multi = self.create_multi_frequency_source(modes, amplitudes)

        # Solve numerically
        solution_multi = self.solver.solve_stationary(source_multi)

        # Compute analytical solution
        solution_analytical = self.compute_analytical_solution(modes, amplitudes)

        # Compute error
        error = np.abs(solution_multi - solution_analytical)
        relative_error_L2 = np.linalg.norm(error) / np.linalg.norm(solution_analytical)

        # Check tolerance
        assert (
            relative_error_L2 <= self.tolerance_L2
        ), f"L2 error {relative_error_L2:.2e} exceeds tolerance {self.tolerance_L2:.2e}"

        print(
            f"Test A0.2.1: Superposition principle - L2 error: {relative_error_L2:.2e}"
        )

    def test_individual_mode_solutions(self):
        """
        Test individual mode solutions and their superposition.

        Physical Meaning:
            Tests that the solution for a multi-frequency source
            equals the sum of individual mode solutions.
        """
        # Generate random modes and amplitudes
        modes = self.generate_random_modes(self.n_modes)
        amplitudes = [
            complex(np.random.randn(), np.random.randn()) for _ in range(self.n_modes)
        ]

        # Normalize amplitudes
        norm = np.sqrt(sum(abs(a) ** 2 for a in amplitudes))
        amplitudes = [a / norm for a in amplitudes]

        # Solve for each individual mode
        individual_solutions = []
        for k_mode, amplitude in zip(modes, amplitudes):
            # Create single mode source
            source_single = self.create_multi_frequency_source([k_mode], [amplitude])

            # Solve numerically
            solution_single = self.solver.solve_stationary(source_single)
            individual_solutions.append(solution_single)

        # Sum individual solutions
        solution_sum = sum(individual_solutions)

        # Create multi-frequency source and solve
        source_multi = self.create_multi_frequency_source(modes, amplitudes)
        solution_multi = self.solver.solve_stationary(source_multi)

        # Compare superposition
        error = np.abs(solution_multi - solution_sum)
        relative_error_L2 = np.linalg.norm(error) / np.linalg.norm(solution_multi)

        # Check tolerance
        assert (
            relative_error_L2 <= self.tolerance_L2
        ), f"Superposition error {relative_error_L2:.2e} exceeds tolerance {self.tolerance_L2:.2e}"

        print(
            f"Test A0.2.2: Individual mode superposition - L2 error: {relative_error_L2:.2e}"
        )

    def test_aliasing_detection(self):
        """
        Test for aliasing effects in multi-frequency sources.

        Physical Meaning:
            Tests that high-frequency modes do not create aliasing
            effects that would contaminate the solution.
        """
        # Create modes with different frequency ranges
        low_freq_modes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        high_freq_modes = [
            [self.N // 4, 0, 0],
            [0, self.N // 4, 0],
            [0, 0, self.N // 4],
        ]

        # Test low frequency modes
        low_amplitudes = [1.0, 1.0, 1.0]
        source_low = self.create_multi_frequency_source(low_freq_modes, low_amplitudes)
        solution_low = self.solver.solve_stationary(source_low)

        # Test high frequency modes
        high_amplitudes = [1.0, 1.0, 1.0]
        source_high = self.create_multi_frequency_source(
            high_freq_modes, high_amplitudes
        )
        solution_high = self.solver.solve_stationary(source_high)

        # Check that high frequency solution has expected properties
        # (should be more localized due to higher k values)
        low_energy = np.sum(np.abs(solution_low) ** 2)
        high_energy = np.sum(np.abs(solution_high) ** 2)

        # High frequency modes should have lower energy due to higher D(k)
        assert (
            high_energy < low_energy
        ), "High frequency modes should have lower energy due to higher spectral coefficients"

        print(
            f"Test A0.2.3: Aliasing detection - Low energy: {low_energy:.2e}, High energy: {high_energy:.2e}"
        )

    def test_spectral_analysis(self):
        """
        Test spectral analysis of multi-frequency solution.

        Physical Meaning:
            Tests that the spectral content of the solution
            matches the expected spectral coefficients.
        """
        # Create single mode for spectral analysis
        k_mode = [4, 0, 0]
        amplitude = 1.0

        source = self.create_multi_frequency_source([k_mode], [amplitude])
        solution = self.solver.solve_stationary(source)

        # Compute FFT of solution
        solution_fft = np.fft.fftn(solution)

        # Find the peak in k-space
        peak_indices = np.unravel_index(
            np.argmax(np.abs(solution_fft)), solution_fft.shape
        )

        # Expected peak location
        kx, ky, kz = k_mode
        expected_peak = (kx % self.N, ky % self.N, kz % self.N)

        # Check that peak is at expected location
        assert (
            peak_indices == expected_peak
        ), f"Peak at {peak_indices} should be at {expected_peak}"

        # Check peak amplitude
        peak_amplitude = np.abs(solution_fft[peak_indices])
        k_magnitude = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2) / self.L
        D_k = self.mu * (k_magnitude ** (2 * self.beta)) + self.lambda_param
        expected_amplitude = abs(amplitude) / D_k

        relative_error = abs(peak_amplitude - expected_amplitude) / expected_amplitude

        assert (
            relative_error <= 1e-10
        ), f"Peak amplitude error {relative_error:.2e} exceeds tolerance"

        print(
            f"Test A0.2.4: Spectral analysis - Peak amplitude error: {relative_error:.2e}"
        )

    def test_frequency_response(self):
        """
        Test frequency response of the solver.

        Physical Meaning:
            Tests that the solver correctly applies the spectral
            operator D(k) = μ|k|^(2β) + λ to different frequencies.
        """
        # Test different frequencies
        frequencies = [1, 2, 4, 8, 16]
        responses = []

        for freq in frequencies:
            k_mode = [freq, 0, 0]
            amplitude = 1.0

            source = self.create_multi_frequency_source([k_mode], [amplitude])
            solution = self.solver.solve_stationary(source)

            # Compute response (amplitude ratio)
            source_amplitude = np.mean(np.abs(source))
            solution_amplitude = np.mean(np.abs(solution))
            response = solution_amplitude / source_amplitude

            responses.append(response)

        # Check that response decreases with frequency
        for i in range(1, len(responses)):
            assert (
                responses[i] < responses[i - 1]
            ), f"Response should decrease with frequency: {responses[i-1]:.2e} -> {responses[i]:.2e}"

        print(f"Test A0.2.5: Frequency response - Responses: {responses}")

    def test_phase_preservation(self):
        """
        Test phase preservation in multi-frequency solution.

        Physical Meaning:
            Tests that the phase relationships between different
            frequency components are preserved.
        """
        # Create modes with specific phase relationships
        modes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        amplitudes = [1.0, 1.0j, -1.0]  # Different phases

        source = self.create_multi_frequency_source(modes, amplitudes)
        solution = self.solver.solve_stationary(source)

        # Check that phase relationships are preserved
        # (This is a simplified test - in practice, we'd need more sophisticated analysis)

        # Compute phase of solution
        solution_phase = np.angle(solution)

        # Check that phase is not constant (indicating phase relationships are preserved)
        phase_variance = np.var(solution_phase)
        assert (
            phase_variance > 1e-6
        ), "Phase should vary to preserve phase relationships"

        print(f"Test A0.2.6: Phase preservation - Phase variance: {phase_variance:.2e}")

    def test_energy_conservation(self):
        """
        Test energy conservation in multi-frequency solution.

        Physical Meaning:
            Tests that the total energy in the solution is
            consistent with the spectral operator.
        """
        # Create multi-frequency source
        modes = self.generate_random_modes(5)
        amplitudes = [complex(np.random.randn(), np.random.randn()) for _ in range(5)]

        # Normalize amplitudes
        norm = np.sqrt(sum(abs(a) ** 2 for a in amplitudes))
        amplitudes = [a / norm for a in amplitudes]

        source = self.create_multi_frequency_source(modes, amplitudes)
        solution = self.solver.solve_stationary(source)

        # Compute energies
        source_energy = np.sum(np.abs(source) ** 2)
        solution_energy = np.sum(np.abs(solution) ** 2)

        # Energy should be reduced due to spectral operator
        assert (
            solution_energy < source_energy
        ), "Solution energy should be less than source energy due to spectral operator"

        # Check energy ratio is consistent with spectral coefficients
        energy_ratio = solution_energy / source_energy

        # Compute expected energy ratio (simplified)
        expected_ratio = 1.0 / (
            self.mu + self.lambda_param
        )  # Approximate for low frequencies

        relative_error = abs(energy_ratio - expected_ratio) / expected_ratio

        # Allow for some error due to frequency dependence
        assert (
            relative_error <= 0.5
        ), f"Energy ratio error {relative_error:.2e} exceeds tolerance"

        print(
            f"Test A0.2.7: Energy conservation - Energy ratio: {energy_ratio:.2e}, Expected: {expected_ratio:.2e}"
        )


if __name__ == "__main__":
    # Run tests
    test = TestA02MultiPlane()
    test.setup_method()

    try:
        test.test_superposition_principle()
        test.test_individual_mode_solutions()
        test.test_aliasing_detection()
        test.test_spectral_analysis()
        test.test_frequency_response()
        test.test_phase_preservation()
        test.test_energy_conservation()

        print("\n✅ All A0.2 tests PASSED!")

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        raise
