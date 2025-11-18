"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic test A0.1: Plane wave validation for Level A.

This module validates FFTSolver7DBasic and the fractional Laplacian by using
production-grade generators (BVPSourceGenerators) and FieldArray-based memory
management to ensure CUDA execution with automatic block processing.

Physical Meaning:
    Confirms that monochromatic excitations propagate correctly inside the
    7D phase field solver, preserving isotropy, spectral superposition, and
    grid convergence properties in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Verifies the relation â(k) = ŝ(k) / D(k) with D(k) = μ|k|^{2β} + λ and the
    operator identity (-Δ)^β e^{i k·x} = |k|^{2β} e^{i k·x}.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from bhlff.core.arrays import FieldArray
from bhlff.core.domain import Domain
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian
from bhlff.core.sources.bvp_source_generators import BVPSourceGenerators


def _create_test_domain(L: float = 1.0, N: int = 16) -> Domain:
    """
    Create compact 7D domain for Level A validation.

    Physical Meaning:
        Builds the smallest admissible M₇ lattice (N_φ = N_t = 2) that still
        resolves 3D spatial harmonics for CUDA solvers.

    Mathematical Foundation:
        Provides uniform spacing Δx = L/N for ℝ³ₓ while phase/temporal grids
        stay minimal yet periodic.

    Args:
        L (float): Spatial length scale.
        N (int): Grid points per spatial axis.

    Returns:
        Domain: Configured domain.
    """
    return Domain(L=L, N=N, N_phi=2, N_t=2, T=1.0, dimensions=7)


def _build_solver(domain: Domain, mu: float, beta: float, lambda_param: float) -> FFTSolver7DBasic:
    """
    Build FFTSolver7DBasic with strict CUDA execution.

    Physical Meaning:
        Creates the production spectral solver that enforces μ(-Δ)^β + λ on GPU.

    Mathematical Foundation:
        Initializes orthonormal FFT pipelines shared across all mixins.

    Args:
        domain (Domain): Computational domain.
        mu (float): Diffusion coefficient.
        beta (float): Fractional order.
        lambda_param (float): Damping parameter.

    Returns:
        FFTSolver7DBasic: Solver facade.
    """
    return FFTSolver7DBasic(
        domain,
        {
            "mu": mu,
            "beta": beta,
            "lambda": lambda_param,
            "use_cuda": True,
        },
    )


def _generate_plane_wave_field(domain: Domain, amplitude: float, mode: Tuple[int, int, int]) -> FieldArray:
    """
    Generate plane wave field via BVPSourceGenerators.

    Physical Meaning:
        Uses production GPU kernels to synthesize monochromatic excitations and
        expand them to the full 7D tensor.

    Mathematical Foundation:
        Implements s(x) = A exp(i 2π m·g / N) with integer mode vector m.

    Args:
        domain (Domain): Target domain.
        amplitude (float): Plane wave amplitude.
        mode (Tuple[int, int, int]): Integer wave vector.

    Returns:
        FieldArray: Plane wave field.
    """
    generator = BVPSourceGenerators(
        domain,
        {
            "plane_wave_amplitude": amplitude,
            "plane_wave_mode": list(mode),
            "use_cuda": True,
        },
    )
    return generator.generate_plane_wave_source()


def _extract_spatial_slice(field: FieldArray | np.ndarray) -> np.ndarray:
    """
    Extract spatial 3D slice from 7D field.

    Physical Meaning:
        Keeps only ℝ³ₓ content by fixing phase and temporal indices to zero.

    Mathematical Foundation:
        Evaluates f(x, y, z, φ₁, φ₂, φ₃, t) at (φ, t) = 0.

    Args:
        field (FieldArray | np.ndarray): 7D field.

    Returns:
        np.ndarray: Spatial slice.
    """
    array = field.array if isinstance(field, FieldArray) else field
    if array.ndim != 7:
        raise ValueError(f"Expected 7D array, received shape {array.shape}")
    return array[:, :, :, 0, 0, 0, 0]


def _compute_solver_operator_value(
    mu: float,
    beta: float,
    lambda_param: float,
    mode: Tuple[int, int, int],
    L: float,
) -> float:
    """
    Compute μ|k|^{2β} + λ for the given mode.

    Physical Meaning:
        Represents impedance applied by L_β to a spatial harmonic.

    Mathematical Foundation:
        Uses |k| = (2π/L)‖m‖₂ with integer mode m.

    Args:
        mu (float): Diffusion coefficient.
        beta (float): Fractional order.
        lambda_param (float): Damping parameter.
        mode (Tuple[int, int, int]): Mode indices.
        L (float): Spatial length.

    Returns:
        float: Operator value.
    """
    mode_array = np.asarray(mode, dtype=np.float64)
    k_norm = (2.0 * np.pi / L) * np.linalg.norm(mode_array)
    return mu * (k_norm ** (2.0 * beta)) + lambda_param


def _compute_fractional_multiplier(beta: float, mode: Tuple[int, int, int], L: float) -> float:
    """
    Compute |k|^{2β} multiplier for (-Δ)^β validation.

    Physical Meaning:
        Supplies the analytical gain applied by the fractional Laplacian.

    Mathematical Foundation:
        Relies on |k| = (2π/L)‖m‖₂ for integer mode m.

    Args:
        beta (float): Fractional order.
        mode (Tuple[int, int, int]): Mode indices.
        L (float): Spatial length.

    Returns:
        float: Multiplier |k|^{2β}.
    """
    mode_array = np.asarray(mode, dtype=np.float64)
    k_norm = (2.0 * np.pi / L) * np.linalg.norm(mode_array)
    return k_norm ** (2.0 * beta)


class TestA01PlaneWaveBasic:
    """
    Basic test A0.1: Plane wave validation.

    Physical Meaning:
        Ensures that production solvers reproduce analytical plane wave
        solutions, preserve isotropy, and converge under grid refinement.

    Mathematical Foundation:
        Exercises the identity â(k) = ŝ(k) / (μ|k|^{2β} + λ) and the
        fractional Laplacian response (-Δ)^β e^{ik·x} = |k|^{2β} e^{ik·x}.
    """

    def setup_method(self) -> None:
        """
        Physical Meaning:
            Initializes shared CUDA-ready domain, solver, and fractional operator.
        Mathematical Foundation:
            Keeps μ, β, λ identical across tests for consistent spectral checks.
        """
        self.L = 1.0
        self.N = 16
        self.mu = 1.0
        self.beta = 1.0
        self.lambda_param = 0.1
        self.domain = _create_test_domain(self.L, self.N)
        self.solver = _build_solver(self.domain, self.mu, self.beta, self.lambda_param)
        self.frac_laplacian = FractionalLaplacian(
            self.domain,
            beta=self.beta,
            lambda_param=self.lambda_param,
        )
        self.tolerance = 1e-10

    def test_plane_wave_single_mode_matches_analytical_solution(self) -> None:
        """
        Physical Meaning:
            Ensures longitudinal excitation reproduces analytical impedance.
        Mathematical Foundation:
            Verifies â = ŝ / (μ|k|^{2β} + λ) for mode (1, 0, 0).
        """
        mode = (1, 0, 0)
        source_field = _generate_plane_wave_field(self.domain, amplitude=1.0, mode=mode)
        solution_field = self.solver.solve_stationary(source_field)

        source_spatial = _extract_spatial_slice(source_field)
        solution_spatial = _extract_spatial_slice(solution_field)
        expected = source_spatial / _compute_solver_operator_value(
            self.mu,
            self.beta,
            self.lambda_param,
            mode,
            self.L,
        )

        diff = solution_spatial - expected
        relative_error = np.linalg.norm(diff) / max(np.linalg.norm(expected), np.finfo(float).eps)
        assert relative_error <= self.tolerance, f"Relative error {relative_error:.3e} exceeded tolerance"

    def test_plane_wave_multiple_modes_superposition(self) -> None:
        """
        Physical Meaning:
            Validates linear response for a sum of independent plane waves.
        Mathematical Foundation:
            Checks linearity of â = ŝ / D(k) across multiple k.
        """
        mode_specs: List[Tuple[float, Tuple[int, int, int]]] = [
            (1.0, (1, 0, 0)),
            (0.75, (0, 1, 0)),
            (0.5, (1, 1, 0)),
        ]
        source_components = [
            _generate_plane_wave_field(self.domain, amplitude, mode)
            for amplitude, mode in mode_specs
        ]

        total_array = np.zeros(self.domain.shape, dtype=np.complex128)
        for component in source_components:
            total_array += component.array
        combined_source = FieldArray(array=total_array)

        solution_field = self.solver.solve_stationary(combined_source)
        solution_spatial = _extract_spatial_slice(solution_field)

        expected_spatial = np.zeros_like(solution_spatial)
        for component, (_, mode) in zip(source_components, mode_specs):
            expected_spatial += _extract_spatial_slice(component) / _compute_solver_operator_value(
                self.mu,
                self.beta,
                self.lambda_param,
                mode,
                self.L,
            )

        rel_error = np.linalg.norm(solution_spatial - expected_spatial) / max(
            np.linalg.norm(expected_spatial),
            np.finfo(float).eps,
        )
        assert rel_error <= 5e-10, f"Superposition error {rel_error:.3e} too high"

    def test_anisotropy_invariance_between_axes(self) -> None:
        """
        Physical Meaning:
            Demonstrates isotropy between orthogonal spatial axes.
        Mathematical Foundation:
            Compares |a_x| and |a_y| after axis permutation.
        """
        source_x = _generate_plane_wave_field(self.domain, 1.0, (1, 0, 0))
        source_y = _generate_plane_wave_field(self.domain, 1.0, (0, 1, 0))

        solution_x = _extract_spatial_slice(self.solver.solve_stationary(source_x))
        solution_y = _extract_spatial_slice(self.solver.solve_stationary(source_y))
        solution_y_rotated = np.swapaxes(solution_y, 0, 1)

        vec_x = solution_x.reshape(-1)
        vec_y = solution_y_rotated.reshape(-1)
        correlation = np.vdot(vec_x, vec_y)
        if np.abs(correlation) > np.finfo(float).eps:
            phase = correlation / np.abs(correlation)
            solution_y_rotated *= phase

        np.testing.assert_allclose(
            np.abs(solution_x),
            np.abs(solution_y_rotated),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_grid_convergence_for_plane_wave(self) -> None:
        """
        Physical Meaning:
            Confirms numerical error shrinks when grid resolution increases.
        Mathematical Foundation:
            Tracks ‖a_num - a_exact‖₂ for fixed mode across N.
        """
        grid_sizes = [8, 12, 16]
        errors: List[float] = []
        mode = (1, 1, 0)

        for size in grid_sizes:
            domain = _create_test_domain(self.L, size)
            solver = _build_solver(domain, self.mu, self.beta, self.lambda_param)
            source_field = _generate_plane_wave_field(domain, 1.0, mode)
            solution_field = solver.solve_stationary(source_field)

            source_spatial = _extract_spatial_slice(source_field)
            expected = source_spatial / _compute_solver_operator_value(
                self.mu,
                self.beta,
                self.lambda_param,
                mode,
                self.L,
            )
            numerical = _extract_spatial_slice(solution_field)
            error = np.linalg.norm(numerical - expected) / max(
                np.linalg.norm(expected),
                np.finfo(float).eps,
            )
            errors.append(error)

        for idx in range(1, len(errors)):
            assert errors[idx] <= errors[idx - 1] * 1.05, f"Grid convergence violated: {errors}"

    def test_fractional_laplacian_operator_matches_multiplier(self) -> None:
        """
        Physical Meaning:
            Ensures FractionalLaplacian scales plane waves correctly.
        Mathematical Foundation:
            Checks (-Δ)^β e^{ik·x} = |k|^{2β} e^{ik·x}.
        """
        mode = (1, 2, 0)
        source_field = _generate_plane_wave_field(self.domain, 1.0, mode)
        result = self.frac_laplacian.apply(source_field.array)
        result_spatial = _extract_spatial_slice(result)

        expected_multiplier = _compute_fractional_multiplier(self.beta, mode, self.L)
        expected = _extract_spatial_slice(source_field) * expected_multiplier

        rel_error = np.linalg.norm(result_spatial - expected) / max(
            np.linalg.norm(expected),
            np.finfo(float).eps,
        )
        assert rel_error <= 5e-10, f"Fractional Laplacian mismatch {rel_error:.3e}"

    def test_spectral_coefficients_match_operator_value(self) -> None:
        """
        Physical Meaning:
            Guarantees precomputed impedance aligns with physics inputs.
        Mathematical Foundation:
            Samples μ|k|^{2β} + λ at modes (0,0,0) and (1,0,0).
        """
        coeffs = self.solver.get_spectral_coefficients()
        assert coeffs.shape == self.domain.shape

        zero_coeff = coeffs[(0, 0, 0, 0, 0, 0, 0)]
        np.testing.assert_allclose(zero_coeff, self.lambda_param, rtol=1e-12, atol=1e-12)

        first_mode_coeff = coeffs[(1, 0, 0, 0, 0, 0, 0)]
        expected = _compute_solver_operator_value(
            self.mu,
            self.beta,
            self.lambda_param,
            (1, 0, 0),
            self.L,
        )
        np.testing.assert_allclose(first_mode_coeff, expected, rtol=1e-9, atol=1e-9)

    def test_solver_info_contains_metadata(self) -> None:
        """
        Physical Meaning:
            Provides diagnostic metadata for orchestration layers.
        Mathematical Foundation:
            Not applicable; structural assertion on solver info payload.
        """
        info = self.solver.get_info()
        assert info["solver_type"] == "FFTSolver7DBasic"
        assert tuple(info["domain_shape"]) == self.domain.shape
        assert info["mu"] == self.mu
        assert info["beta"] == self.beta
        assert info["lambda"] == self.lambda_param
