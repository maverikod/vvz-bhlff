"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

High-precision spectral solver for fractional Riesz operator in 7D space-time.

This module implements the core FFT solver for the 7D phase field theory,
providing efficient solution of the fractional Laplacian equation in 7D space-time.

Physical Meaning:
    Solves the fractional Laplacian equation L_β a = s(x) in 7D space-time
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, where the evolution is governed by the
    fractional Riesz operator L_β = μ(-Δ)^β + λ.

Mathematical Foundation:
    Implements the fractional Laplacian equation:
    L_β a = μ(-Δ)^β a + λa = s(x,t)
    where β ∈ (0,2) is the fractional order, μ > 0 is the
    diffusion coefficient, and λ ≥ 0 is the damping parameter.

Example:
    >>> solver = FFTSolver7D(domain, parameters)
    >>> solution = solver.solve_stationary(source_field)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...solvers.base.abstract_solver import AbstractSolver
    from ..domain import Domain
    from ..domain.parameters import Parameters

from .fractional_laplacian import FractionalLaplacian
from .spectral_operations import SpectralOperations
from .memory_manager_7d import MemoryManager7D
from .fft_plan_7d import FFTPlan7D
from .spectral_coefficient_cache import SpectralCoefficientCache
from .fft_solver_time import FFTSolverTimeMethods
from .fft_solver_validation import FFTSolverValidation


class FFTSolver7D:
    """
    High-precision spectral solver for fractional Riesz operator in 7D space-time.

    Physical Meaning:
        Solves the fractional Laplacian equation L_β a = s(x) in 7D space-time
        M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, where the evolution is governed by the
        fractional Riesz operator L_β = μ(-Δ)^β + λ.

    Mathematical Foundation:
        Implements the fractional Laplacian equation:
        L_β a = μ(-Δ)^β a + λa = s(x,t)
        where β ∈ (0,2) is the fractional order, μ > 0 is the
        diffusion coefficient, and λ ≥ 0 is the damping parameter.

    Attributes:
        domain (Domain): Computational domain for the simulation.
        parameters (Parameters): Solver parameters including μ, β, λ.
        _fractional_laplacian (FractionalLaplacian): Fractional Laplacian operator.
        _spectral_ops (SpectralOperations): Spectral operations calculator.
        _memory_manager (MemoryManager7D): Memory management for 7D arrays.
        _fft_plan (FFTPlan7D): Pre-computed FFT plan for efficiency.
        _spectral_cache (SpectralCoefficientCache): Cache for spectral coefficients.
        _time_methods (FFTSolverTimeMethods): Time-dependent methods.
        _validation (FFTSolverValidation): Validation methods.
    """

    def __init__(self, domain: 'Domain', parameters: 'Parameters'):
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
                - lambda_param (float): Damping parameter λ ≥ 0
                - precision (str): Numerical precision ('float64')
        """
        self.domain = domain
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

        # Validate domain dimensions
        if domain.dimensions != 7:
            raise ValueError(f"Dimensions must be 7 for 7D BVP theory, got {domain.dimensions}")

        # Initialize core components
        self._fractional_laplacian = FractionalLaplacian(domain, parameters.beta, parameters.lambda_param)
        self._spectral_ops = SpectralOperations(domain, parameters.precision)
        self._memory_manager = MemoryManager7D(domain.shape, 8.0)  # Default memory limit
        self._fft_plan = FFTPlan7D(domain.shape, parameters.precision)
        self._spectral_cache = SpectralCoefficientCache()
        
        # Pre-compute spectral coefficients
        self._setup_spectral_coefficients()
        
        # Initialize specialized methods
        self._time_methods = FFTSolverTimeMethods(domain, parameters)
        self._validation = FFTSolverValidation(domain, parameters, self._fractional_laplacian)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Mark as initialized
        self._initialized = True

    def solve_stationary(self, source_field: np.ndarray) -> np.ndarray:
        """
        Solve stationary problem L_β a = s(x) using spectral method.

        Physical Meaning:
            Solves the stationary fractional Laplacian equation in 7D space-time,
            representing the equilibrium configuration of the phase field
            under the given source term.

        Mathematical Foundation:
            Solves L_β a = s in spectral space:
            â(k) = ŝ(k) / (μ|k|^(2β) + λ)
            where k is the wave vector and |k| is its magnitude.

        Args:
            source_field (np.ndarray): Source term s(x,φ,t).
                Represents external excitations or initial conditions
                that drive the phase field evolution.

        Returns:
            np.ndarray: Solution field a(x,φ,t).
                Represents the phase field configuration that
                satisfies the equation and describes the spatial
                distribution of phase values.

        Raises:
            ValueError: If source has incompatible shape with domain.
            RuntimeError: If FFT operations fail.
        """
        if not self._initialized:
            raise RuntimeError("Solver not initialized")

        if source_field.shape != self.domain.shape:
            raise ValueError(f"Source shape {source_field.shape} incompatible with domain shape {self.domain.shape}")

        # Validate source for λ=0 case
        self._validate_source_for_lambda_zero(source_field)

        # Transform source to spectral space
        source_spectral = self._spectral_ops.forward_fft(source_field, 'physics')

        # Apply spectral operator
        spectral_coeffs = self._get_spectral_coefficients()
        solution_spectral = source_spectral / spectral_coeffs

        # Transform back to real space
        solution = self._spectral_ops.inverse_fft(solution_spectral, 'physics')

        self.logger.info("Stationary problem solved successfully")
        return solution.real

    def solve_time_dependent(self, initial_field: np.ndarray, source_field: np.ndarray, 
                           time_steps: np.ndarray, method: str = 'exponential') -> np.ndarray:
        """
        Solve time-dependent problem using temporal integrators.
        
        Physical Meaning:
            Solves the dynamic phase field equation:
            ∂a/∂t + ν(-Δ)^β a + λa = s(x,φ,t)
            using high-precision temporal integrators with support for
            memory kernels and quench detection.
            
        Mathematical Foundation:
            Uses either exponential integrator (exact for harmonic sources)
            or Crank-Nicolson integrator (second-order accurate, unconditionally stable).
            
        Args:
            initial_field (np.ndarray): Initial field configuration a(x,φ,0).
            source_field (np.ndarray): Source term s(x,φ,t) over time.
            time_steps (np.ndarray): Time points for integration.
            method (str): Integration method ('exponential' or 'crank_nicolson').
            
        Returns:
            np.ndarray: Field evolution a(x,φ,t) over time.
        """
        return self._time_methods.solve_time_dependent(initial_field, source_field, time_steps, method)

    def set_memory_kernel(self, num_memory_vars: int = 3, 
                         relaxation_times: Optional[List[float]] = None,
                         coupling_strengths: Optional[List[float]] = None) -> None:
        """
        Configure memory kernel for non-local temporal effects.
        
        Physical Meaning:
            Sets up the memory kernel with specified number of memory
            variables, relaxation times, and coupling strengths for
            non-local temporal effects in phase field evolution.
            
        Args:
            num_memory_vars (int): Number of memory variables.
            relaxation_times (Optional[List[float]]): Relaxation times τⱼ.
            coupling_strengths (Optional[List[float]]): Coupling strengths γⱼ.
        """
        self._time_methods.set_memory_kernel(num_memory_vars, relaxation_times, coupling_strengths)

    def set_quench_detector(self, energy_threshold: float = 1e-3,
                           rate_threshold: float = 1e-2,
                           magnitude_threshold: float = 10.0) -> None:
        """
        Configure quench detection system.
        
        Physical Meaning:
            Sets up the quench detection system with specified thresholds
            for monitoring energy dumping events during integration.
            
        Args:
            energy_threshold (float): Energy change threshold for quench detection.
            rate_threshold (float): Rate of change threshold.
            magnitude_threshold (float): Field magnitude threshold.
        """
        self._time_methods.set_quench_detector(energy_threshold, rate_threshold, magnitude_threshold)

    def get_quench_history(self) -> List[Dict]:
        """
        Get quench detection history.
        
        Returns:
            List[Dict]: History of detected quench events.
        """
        return self._time_methods.get_quench_history()

    def get_memory_contribution(self) -> np.ndarray:
        """
        Get current memory kernel contribution.
        
        Returns:
            np.ndarray: Current memory contribution to field.
        """
        return self._time_methods.get_memory_contribution()

    def validate_solution(self, solution: np.ndarray, source: np.ndarray,
                         tolerance: float = 1e-12) -> Dict[str, Any]:
        """
        Validate solution to fractional Laplacian equation.
        
        Physical Meaning:
            Validates the solution by computing the residual of the equation
            L_β a = s and checking that it satisfies the equation within
            the specified tolerance.
            
        Mathematical Foundation:
            Computes residual: r = L_β a - s = μ(-Δ)^β a + λa - s
            and checks that ||r|| < tolerance.
            
        Args:
            solution (np.ndarray): Solution field a(x,φ,t).
            source (np.ndarray): Source field s(x,φ,t).
            tolerance (float): Tolerance for residual validation.
            
        Returns:
            Dict[str, Any]: Validation results including residual norm and status.
        """
        return self._validation.validate_solution(solution, source, tolerance)

    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for fractional Laplacian.

        Physical Meaning:
            Pre-computes the spectral representation of the fractional
            Laplacian operator, which is essential for efficient
            solution of the equation in spectral space.
        """
        # Get cached coefficients or compute new ones
        self._spectral_coeffs = self._spectral_cache.get_coefficients(
            mu=self.parameters.mu,
            beta=self.parameters.beta,
            lambda_param=self.parameters.lambda_param,
            domain_shape=self.domain.shape
        )

    def _get_spectral_coefficients(self) -> np.ndarray:
        """
        Get spectral coefficients for fractional Laplacian.

        Returns:
            np.ndarray: Spectral coefficients.
        """
        return self._spectral_coeffs

    def _validate_source_for_lambda_zero(self, source: np.ndarray) -> None:
        """
        Validate source field for λ=0 case.

        Physical Meaning:
            For λ=0, the k=0 mode requires special handling to avoid
            division by zero. The source must have zero mean.

        Mathematical Foundation:
            For λ=0, the k=0 mode gives D(0) = 0, so we require ŝ(0) = 0,
            which means the source must have zero mean.

        Args:
            source (np.ndarray): Source field to validate.

        Raises:
            ValueError: If λ=0 and source has non-zero mean.
        """
        if self.parameters.lambda_param == 0:
            # Check that source has zero mean
            zero_mode = np.mean(source)
            if abs(zero_mode) > 1e-12:
                raise ValueError(
                    f"lambda=0 requires mean(source)=0, but got {zero_mode}. "
                    "Remove constant component from source field."
                )

    def __repr__(self) -> str:
        """String representation of solver."""
        return (f"FFTSolver7D("
                f"domain={self.domain.shape}, "
                f"mu={self.parameters.mu}, "
                f"beta={self.parameters.beta}, "
                f"lambda={self.parameters.lambda_param})")
