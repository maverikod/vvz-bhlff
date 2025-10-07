"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic FFT solver for fractional Riesz operator in 7D space-time.

This module implements the basic FFT solver functionality for the 7D phase field theory,
providing core solution methods for the fractional Laplacian equation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...solvers.base.abstract_solver import AbstractSolver
    from ..domain import Domain
    from ..domain.parameters import Parameters

from bhlff.core.operators.fractional_laplacian import FractionalLaplacian as FractionalLaplacian
from .spectral_operations import SpectralOperations
from .memory_manager_7d import MemoryManager7D
from .fft_plan_7d import FFTPlan7D
from .spectral_coefficient_cache import SpectralCoefficientCache


class FFTSolver7DBasic:
    """
    Basic FFT solver for fractional Riesz operator in 7D space-time.

    Physical Meaning:
        Solves the fractional Laplacian equation L_β a = s(x) in 7D space-time
        M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, where the evolution is governed by the
        fractional Riesz operator L_β = μ(-Δ)^β + λ.

    Mathematical Foundation:
        Implements the fractional Laplacian equation:
        L_β a = μ(-Δ)^β a + λa = s(x,t)
        where β ∈ (0,2) is the fractional order, μ > 0 is the
        diffusion coefficient, and λ ≥ 0 is the damping parameter.
    """

    def __init__(self, domain: "Domain", parameters: "Parameters"):
        """
        Initialize 7D FFT solver.

        Physical Meaning:
            Sets up the solver with the computational domain and
            physical parameters, pre-computing spectral coefficients
            for efficient solution of the fractional Laplacian equation.

        Args:
            domain (Domain): Computational domain with grid information.
            parameters (Parameters): Dictionary containing:
                - mu (float): Diffusion coefficient μ > 0
                - beta (float): Fractional order β ∈ (0,2)
                - lambda (float): Damping parameter λ ≥ 0
                - precision (str): Numerical precision ('float64')
        """
        self.domain = domain
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

        # Initialize components
        beta = getattr(parameters, "beta", 1.0)
        lambda_param = getattr(parameters, "lambda_param", 0.0)
        self.fractional_laplacian = FractionalLaplacian(domain, beta, lambda_param)
        self.spectral_operations = SpectralOperations(domain, parameters)
        max_memory_gb = getattr(parameters, "max_memory_gb", 8.0)
        self.memory_manager = MemoryManager7D(domain.shape, max_memory_gb)
        self.fft_plan = FFTPlan7D(domain, parameters)
        max_cache_size = getattr(parameters, "max_cache_size", 100)
        self.spectral_cache = SpectralCoefficientCache(max_cache_size)

        # Setup spectral coefficients
        self._setup_spectral_coefficients()

        # Setup FFT plan
        self._setup_fft_plan()

    def solve_stationary(self, source: np.ndarray) -> np.ndarray:
        """
        Solve the stationary fractional Laplacian equation.

        Physical Meaning:
            Solves the stationary fractional Laplacian equation
            L_β a = s in spectral space, representing the steady-state
            solution of the field evolution.

        Mathematical Foundation:
            Solves L_β a = s in spectral space:
            â(k) = ŝ(k) / (μ|k|^(2β) + λ)
            where k is the wave vector and |k| is its magnitude.

        Args:
            source (np.ndarray): Source term s(x) in real space.
                Represents external excitations or initial conditions
                that drive the field evolution.

        Returns:
            np.ndarray: Solution field a(x) in real space.
                Represents the field configuration that
                satisfies the equation and describes the spatial
                distribution of field values.

        Raises:
            ValueError: If source has incompatible shape with domain.
            RuntimeError: If FFT operations fail.
        """
        if source.shape != self.domain.shape:
            raise ValueError(
                f"Source shape {source.shape} incompatible with domain shape {self.domain.shape}"
            )

        self.logger.info("Solving stationary fractional Laplacian equation")

        # Transform to spectral space using unified backend
        from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations
        spectral_ops = UnifiedSpectralOperations(self.domain, precision="float64")
        source_spectral = spectral_ops.forward_fft(source, normalization="physics")

        # Apply spectral operator
        solution_spectral = source_spectral / self.spectral_coefficients

        # Transform back to real space
        solution = spectral_ops.inverse_fft(solution_spectral, normalization="physics")

        self.logger.info("Stationary solution computed")
        return solution.real

    def solve_envelope(self, source: np.ndarray) -> np.ndarray:
        """
        Solve the envelope equation.

        Physical Meaning:
            Solves the envelope equation for the 7D phase field,
            representing the evolution of the field envelope
            in 7D space-time.

        Args:
            source (np.ndarray): Source term for envelope equation.

        Returns:
            np.ndarray: Envelope solution field.
        """
        self.logger.info("Solving envelope equation")

        # Use stationary solver for envelope equation
        solution = self.solve_stationary(source)

        self.logger.info("Envelope solution computed")
        return solution

    def validate_solution(
        self, solution: np.ndarray, source: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate the computed solution.

        Physical Meaning:
            Validates the accuracy of the computed solution by
            checking how well it satisfies the original equation
            and computing quality metrics.

        Args:
            solution (np.ndarray): Computed solution field.
            source (np.ndarray): Original source term.

        Returns:
            Dict[str, Any]: Validation results including:
                - residual: Residual of the equation
                - relative_residual: Relative residual magnitude
                - is_valid: Whether solution is valid
        """
        self.logger.info("Validating solution")

        # Compute residual
        residual = self._compute_residual(solution, source)
        residual_norm = np.linalg.norm(residual)
        source_norm = np.linalg.norm(source)

        # Compute relative residual
        relative_residual = (
            residual_norm / source_norm if source_norm > 0 else float("inf")
        )

        # Determine validity
        is_valid = relative_residual < 1e-6

        validation_results = {
            "residual": residual,
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "is_valid": is_valid,
        }

        self.logger.info(
            f"Solution validation completed: relative_residual = {relative_residual}"
        )
        return validation_results

    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for fractional Laplacian.

        Physical Meaning:
            Pre-computes the spectral representation of the fractional
            Laplacian operator, which is essential for efficient
            solution of the equation in spectral space.
        """
        self.logger.info("Setting up spectral coefficients")

        # Get parameters
        mu = getattr(self.parameters, "mu", 1.0)
        beta = getattr(self.parameters, "beta", 1.0)
        lambda_param = getattr(self.parameters, "lambda_param", 0.0)

        # Compute wave vectors
        wave_vectors = self.spectral_operations._get_wave_vectors()

        # Compute wave vector magnitudes
        k_magnitude_squared = np.zeros(self.domain.shape)
        for i, k_vec in enumerate(wave_vectors):
            # Reshape k_vec for broadcasting across the full domain shape
            reshape_pattern = [1] * len(self.domain.shape)
            reshape_pattern[i] = len(k_vec)
            k_vec_reshaped = k_vec.reshape(reshape_pattern)
            k_magnitude_squared += k_vec_reshaped**2

        # Compute spectral coefficients
        self.spectral_coefficients = mu * (k_magnitude_squared**beta) + lambda_param

        # Handle k=0 mode
        if lambda_param == 0:
            self.spectral_coefficients[0, 0, 0, 0, 0, 0, 0] = (
                1.0  # Avoid division by zero
            )

        self.logger.info("Spectral coefficients computed")

    def _setup_fft_plan(self) -> None:
        """
        Setup FFT plan for efficient computations.

        Physical Meaning:
            Pre-computes FFT plans to optimize the spectral
            transformations required for solving the fractional
            Laplacian equation efficiently.
        """
        self.logger.info("Setting up FFT plan")

        # Setup FFT plan
        self.fft_plan._setup_fft_plans()

        self.logger.info("FFT plan setup completed")

    def _compute_residual(self, solution: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the fractional Laplacian equation.

        Physical Meaning:
            Computes the residual R = L_β a - s, which measures
            how well the solution satisfies the original equation.

        Args:
            solution (np.ndarray): Solution field.
            source (np.ndarray): Source term.

        Returns:
            np.ndarray: Residual field.
        """
        # Apply fractional Laplacian operator
        laplacian_solution = self.fractional_laplacian.apply(solution)

        # Compute residual
        residual = laplacian_solution - source

        return residual

    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get solver information.

        Returns:
            Dict[str, Any]: Solver information.
        """
        return {
            "domain_shape": self.domain.shape,
            "parameters": self.parameters,
            "spectral_coefficients_computed": hasattr(self, "spectral_coefficients"),
            "fft_plan_setup": True,  # Assume setup is always available
            "solver_type": "basic",
        }
