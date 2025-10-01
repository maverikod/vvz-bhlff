"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

7D FFT Solver for Fractional Riesz Operator in BHLFF Framework.

This module implements the core FFT solver for solving the fractional Riesz operator
equation in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, representing the evolution of
phase field configurations with U(1)³ phase structure.

Theoretical Background:
    The solver implements the 7D spectral solution for the fractional Laplacian
    equation: L_β a = μ(-Δ)^β a + λa = s(x,φ,t), where the solution in spectral
    space is: â(k_x, k_φ, k_t) = ŝ(k_x, k_φ, k_t) / (μ|k|^(2β) + λ).

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
from ..time import BVPExponentialIntegrator, CrankNicolsonIntegrator, MemoryKernel, QuenchDetector


class FFTSolver7D:
    """
    High-precision spectral solver for fractional Riesz operator in 7D space-time.
    
    Physical Meaning:
        Solves the fractional Laplacian equation L_β a = μ(-Δ)^β a + λa = s(x,φ,t)
        in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, representing the evolution of
        phase field configurations with U(1)³ phase structure.
        
    Mathematical Foundation:
        Implements 7D spectral solution: â(k_x, k_φ, k_t) = ŝ(k_x, k_φ, k_t) / (μ|k|^(2β) + λ)
        where |k|² = |k_x|² + |k_φ|² + k_t² is the 7D wave vector magnitude.
        
    Attributes:
        domain (Domain): Computational domain for the simulation.
        parameters (Dict[str, Any]): Solver parameters including
            μ, β, λ, and numerical settings.
        _fractional_laplacian (FractionalLaplacian): Fractional Laplacian operator.
        _spectral_ops (SpectralOperations): Spectral operations handler.
        _memory_manager (MemoryManager7D): Memory management for 7D fields.
        _fft_plan (FFTPlan7D): Optimized FFT plans.
        _spectral_cache (SpectralCoefficientCache): Cache for spectral coefficients.
        _spectral_coeffs (np.ndarray): Pre-computed spectral coefficients.
    """
    
    def __init__(self, domain: 'Domain', parameters: 'Parameters'):
        """
        Initialize FFT solver with domain and physics parameters.
        
        Physical Meaning:
            Sets up the solver with the computational domain and
            physical parameters, pre-computing spectral coefficients
            for efficient solution of the fractional Laplacian equation.
            
        Args:
            domain (Domain): Computational domain with grid information.
            parameters (Dict[str, Any]): Dictionary containing:
                - mu (float): Diffusion coefficient μ > 0
                - beta (float): Fractional order β ∈ (0,2)
                - lambda (float): Damping parameter λ ≥ 0
                - precision (str): Numerical precision ('float64')
                - fft_plan (str): FFT plan type ('MEASURE')
                - tolerance (float): Convergence tolerance (1e-12)
        """
        # Convert Dict to Parameters if needed
        if isinstance(parameters, dict):
            from ..domain.parameters import Parameters
            parameters = Parameters(**parameters)
        
        self.domain = domain
        self.parameters = parameters
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize components
        self._fractional_laplacian = FractionalLaplacian(domain, parameters.beta, parameters.lambda_param)
        self._spectral_ops = SpectralOperations(domain, parameters.precision)
        self._memory_manager = MemoryManager7D(domain.shape, 8.0)  # Default memory limit
        self._fft_plan = FFTPlan7D(domain.shape, parameters.precision)
        self._spectral_cache = SpectralCoefficientCache()
        
        # Pre-compute spectral coefficients
        self._setup_spectral_coefficients()
        
        # Initialize time integrators
        self._exponential_integrator = None
        self._crank_nicolson_integrator = None
        self._memory_kernel = None
        self._quench_detector = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Mark as initialized
        self._initialized = True
    
    def solve_stationary(self, source_field: np.ndarray) -> np.ndarray:
        """
        Solve stationary problem L_β a = s(x) using spectral method.
        
        Physical Meaning:
            Computes the phase field configuration that satisfies
            the fractional Laplacian equation with the given source
            term, representing the response of the phase field to
            external excitations or initial conditions.
            
        Mathematical Foundation:
            Solves L_β a = s in spectral space:
            â(k) = ŝ(k) / (μ|k|^(2β) + λ)
            where k is the wave vector and |k| is its magnitude.
            
        Args:
            source_field (np.ndarray): Source term s(x) in real space.
                Represents external excitations or initial conditions
                that drive the phase field evolution.
                
        Returns:
            np.ndarray: Solution field a(x) in real space.
                Represents the phase field configuration that
                satisfies the equation and describes the spatial
                distribution of phase values.
                
        Raises:
            ValueError: If source has incompatible shape with domain.
            RuntimeError: If FFT operations fail.
        """
        if source_field.shape != self.domain.shape:
            raise ValueError(f"Source shape {source_field.shape} incompatible with domain shape {self.domain.shape}")
        
        # Check for zero mode compatibility
        self._check_zero_mode_compatibility(source_field)
        
        # Transform to spectral space
        source_spectral = self._spectral_ops.forward_fft(source_field)
        
        # Apply spectral operator
        solution_spectral = source_spectral / self._spectral_coeffs
        
        # Transform back to real space
        solution = self._spectral_ops.inverse_fft(solution_spectral)
        
        return solution.real
    
    def solve_time_dependent(self, source_field: np.ndarray, 
                           time_params: Dict[str, Any]) -> np.ndarray:
        """
        Solve time-dependent problem with temporal integration.
        
        Physical Meaning:
            Integrates the time evolution of phase field configurations
            with memory kernel effects, maintaining energy conservation
            and numerical stability.
            
        Mathematical Foundation:
            Solves ∂_t a + ν(-Δ)^β a + λa = s with exponential integrator:
            â^{n+1}(k) = e^{-α_k Δt} â^n(k) + ∫_0^{Δt} e^{-α_k(Δt-τ)} ŝ(k,τ) dτ
            where α_k = ν|k|^(2β) + λ.
            
        Args:
            source_field (np.ndarray): Source term s(x,t) in real space.
            time_params (Dict[str, Any]): Time integration parameters:
                - t_final (float): Final time
                - dt (float): Time step
                - scheme (str): Integration scheme ('exponential' or 'crank_nicolson')
                
        Returns:
            np.ndarray: Time evolution of the field a(x,t).
        """
        # TODO: Implement time integrator
        # For now, return the stationary solution
        self.logger.warning("Time-dependent solver not yet implemented, returning stationary solution")
        return self.solve_stationary(source_field)
    
    def get_spectral_coefficients(self) -> np.ndarray:
        """
        Get precomputed spectral coefficients D(k) = μ|k|^(2β) + λ.
        
        Physical Meaning:
            Returns the spectral representation of the fractional
            Laplacian operator, which is essential for efficient
            solution of the equation in spectral space.
            
        Returns:
            np.ndarray: Spectral coefficients D(k) for all wave vectors.
        """
        return self._spectral_coeffs.copy()
    
    def validate_solution(self, solution: np.ndarray, 
                         source: np.ndarray) -> Dict[str, float]:
        """
        Validate solution quality and compute residuals.
        
        Physical Meaning:
            Computes validation metrics to ensure the solution
            satisfies the fractional Laplacian equation with
            sufficient accuracy and maintains physical properties.
            
        Mathematical Foundation:
            Computes residual r = L_β a - s and validates:
            - Relative residual: ||r||₂ / ||s||₂ ≤ 10⁻¹²
            - Orthogonality: Re(Σ â*(k) r̂(k)) ≈ 0
            - Energy balance: |E_out - E_in| / E_in ≤ 3%
            
        Args:
            solution (np.ndarray): Computed solution a(x).
            source (np.ndarray): Source term s(x).
            
        Returns:
            Dict[str, float]: Validation metrics including:
                - residual_norm: Relative residual norm
                - orthogonality: Orthogonality condition
                - energy_balance: Energy conservation
                - max_error: Maximum pointwise error
        """
        # Compute residual
        residual = self._compute_residual(solution, source)
        
        # Compute metrics
        residual_norm = np.linalg.norm(residual) / np.linalg.norm(source)
        
        # Orthogonality check
        solution_spectral = self._spectral_ops.forward_fft(solution)
        residual_spectral = self._spectral_ops.forward_fft(residual)
        orthogonality = np.real(np.sum(np.conj(solution_spectral) * residual_spectral))
        orthogonality /= (np.linalg.norm(solution_spectral) * np.linalg.norm(residual_spectral))
        
        # Energy balance
        energy_in = np.sum(np.abs(source)**2)
        energy_out = np.sum(np.abs(solution)**2)
        energy_balance = abs(energy_out - energy_in) / energy_in
        
        # Maximum error
        max_error = np.max(np.abs(residual))
        
        return {
            'residual_norm': residual_norm,
            'orthogonality': abs(orthogonality),
            'energy_balance': energy_balance,
            'max_error': max_error
        }
    
    def _validate_parameters(self) -> None:
        """
        Validate solver parameters.
        
        Physical Meaning:
            Ensures all physical parameters are within valid ranges
            for the fractional Laplacian equation.
        """
        mu = self.parameters.mu
        beta = self.parameters.beta
        lambda_param = self.parameters.lambda_param
        
        if mu <= 0:
            raise ValueError(f"Diffusion coefficient μ must be positive, got {mu}")
        
        if not (0 < beta < 2):
            raise ValueError(f"Fractional order β must be in (0,2), got {beta}")
        
        if lambda_param < 0:
            raise ValueError(f"Damping parameter λ must be non-negative, got {lambda_param}")
    
    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for fractional Laplacian.
        
        Physical Meaning:
            Pre-computes the spectral representation of the fractional
            Laplacian operator, which is essential for efficient
            solution of the equation in spectral space.
        """
        mu = self.parameters.mu
        beta = self.parameters.beta
        lambda_param = self.parameters.lambda_param
        
        # Get coefficients from cache or compute new ones
        self._spectral_coeffs = self._spectral_cache.get_coefficients(
            mu, beta, lambda_param, self.domain.shape
        )
    
    def _check_zero_mode_compatibility(self, source_field: np.ndarray) -> None:
        """
        Check zero mode compatibility for λ=0 case.
        
        Physical Meaning:
            Ensures that when λ=0, the source field has zero mean
            to avoid singularity in the spectral solution.
        """
        lambda_param = self.parameters.lambda_param
        
        if lambda_param == 0:
            source_spectral = self._spectral_ops.forward_fft(source_field)
            zero_mode = source_spectral[tuple([0] * len(self.domain.shape))]
            
            if abs(zero_mode) > 1e-12:
                raise ValueError(
                    f"lambda=0 requires mean(source)=0, but got {zero_mode}. "
                    "Remove constant component from source field."
                )
    
    def _compute_residual(self, solution: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual r = L_β a - s.
        
        Physical Meaning:
            Computes the residual of the fractional Laplacian equation
            to validate the solution quality.
        """
        # Apply fractional Laplacian to solution
        laplacian_solution = self._fractional_laplacian.apply(solution)
        
        # Compute residual
        residual = self.parameters['mu'] * laplacian_solution + self.parameters['lambda'] * solution - source
        
        return residual
    
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
        if not hasattr(self, '_initialized') or not self._initialized:
            raise RuntimeError("Solver not initialized")
        
        # Validate inputs
        if initial_field.shape != self.domain.shape:
            raise ValueError(f"Initial field shape {initial_field.shape} incompatible with domain {self.domain.shape}")
        
        if source_field.shape != (len(time_steps),) + self.domain.shape:
            raise ValueError(f"Source field shape {source_field.shape} incompatible with time steps and domain")
        
        # Get or create integrator
        if method == 'exponential':
            if self._exponential_integrator is None:
                self._exponential_integrator = BVPExponentialIntegrator(self.domain, self.parameters)
                self._setup_integrator_components(self._exponential_integrator)
            integrator = self._exponential_integrator
        elif method == 'crank_nicolson':
            if self._crank_nicolson_integrator is None:
                self._crank_nicolson_integrator = CrankNicolsonIntegrator(self.domain, self.parameters)
                self._setup_integrator_components(self._crank_nicolson_integrator)
            integrator = self._crank_nicolson_integrator
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        # Integrate
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        self.logger.info(f"Time-dependent integration completed using {method} method")
        return result
    
    def _setup_integrator_components(self, integrator) -> None:
        """
        Setup memory kernel and quench detector for integrator.
        
        Physical Meaning:
            Configures the integrator with memory kernel for non-local
            temporal effects and quench detector for monitoring energy
            dumping events.
        """
        # Setup memory kernel
        if self._memory_kernel is None:
            self._memory_kernel = MemoryKernel(self.domain, num_memory_vars=3)
        integrator.set_memory_kernel(self._memory_kernel)
        
        # Setup quench detector
        if self._quench_detector is None:
            self._quench_detector = QuenchDetector(self.domain)
        integrator.set_quench_detector(self._quench_detector)
    
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
        self._memory_kernel = MemoryKernel(self.domain, num_memory_vars)
        
        if relaxation_times is not None:
            self._memory_kernel.set_relaxation_times(relaxation_times)
        
        if coupling_strengths is not None:
            self._memory_kernel.set_coupling_strengths(coupling_strengths)
        
        # Update existing integrators
        if self._exponential_integrator is not None:
            self._exponential_integrator.set_memory_kernel(self._memory_kernel)
        if self._crank_nicolson_integrator is not None:
            self._crank_nicolson_integrator.set_memory_kernel(self._memory_kernel)
        
        self.logger.info(f"Memory kernel configured with {num_memory_vars} variables")
    
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
        self._quench_detector = QuenchDetector(
            self.domain, 
            energy_threshold=energy_threshold,
            rate_threshold=rate_threshold,
            magnitude_threshold=magnitude_threshold
        )
        
        # Update existing integrators
        if self._exponential_integrator is not None:
            self._exponential_integrator.set_quench_detector(self._quench_detector)
        if self._crank_nicolson_integrator is not None:
            self._crank_nicolson_integrator.set_quench_detector(self._quench_detector)
        
        self.logger.info(f"Quench detector configured with thresholds: "
                        f"energy={energy_threshold}, rate={rate_threshold}, "
                        f"magnitude={magnitude_threshold}")
    
    def get_quench_history(self) -> List[Dict]:
        """
        Get quench detection history.
        
        Returns:
            List[Dict]: History of detected quench events.
        """
        if self._quench_detector is None:
            return []
        return self._quench_detector.get_quench_history()
    
    def get_memory_contribution(self) -> np.ndarray:
        """
        Get current memory kernel contribution.
        
        Returns:
            np.ndarray: Current memory contribution to field.
        """
        if self._memory_kernel is None:
            return np.zeros(self.domain.shape, dtype=np.complex128)
        return self._memory_kernel.get_memory_contribution()
