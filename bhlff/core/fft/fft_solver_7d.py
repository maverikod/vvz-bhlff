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
from typing import Dict, Any, Optional, Tuple
import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..base.abstract_solver import AbstractSolver
    from ..domain import Domain

from .fractional_laplacian import FractionalLaplacian
from .spectral_operations import SpectralOperations
from .memory_manager_7d import MemoryManager7D
from .fft_plan_7d import FFTPlan7D
from .spectral_coefficient_cache import SpectralCoefficientCache


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
    
    def __init__(self, domain: 'Domain', parameters: Dict[str, Any]):
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
        self.domain = domain
        self.parameters = parameters
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize components
        self._fractional_laplacian = FractionalLaplacian(domain, parameters['beta'])
        self._spectral_ops = SpectralOperations(domain, parameters.get('precision', 'float64'))
        self._memory_manager = MemoryManager7D(domain.shape, parameters.get('max_memory_gb', 8.0))
        self._fft_plan = FFTPlan7D(domain.shape, parameters.get('precision', 'float64'))
        self._spectral_cache = SpectralCoefficientCache()
        
        # Pre-compute spectral coefficients
        self._setup_spectral_coefficients()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
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
        mu = self.parameters.get('mu', 1.0)
        beta = self.parameters.get('beta', 1.0)
        lambda_param = self.parameters.get('lambda', 0.0)
        
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
        mu = self.parameters['mu']
        beta = self.parameters['beta']
        lambda_param = self.parameters['lambda']
        
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
        lambda_param = self.parameters.get('lambda', 0.0)
        
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
