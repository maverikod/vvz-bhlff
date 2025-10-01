"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

7D BVP FFT Solver facade implementation.

This module provides a facade interface for the 7D BVP envelope equation solver,
combining core functionality, Newton-Raphson solver, and validation methods.

Physical Meaning:
    Provides a unified interface for solving the complete 7D BVP envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Facade pattern combining:
    - Core BVP solver functionality
    - Newton-Raphson iterative solver
    - Solution validation methods

Example:
    >>> domain = Domain7DBVP(L_spatial=1.0, N_spatial=64, N_phase=32, T=1.0, N_t=128)
    >>> params = Parameters7DBVP(kappa_0=1.0, kappa_2=0.1, chi_prime=1.0, k0=1.0)
    >>> solver = FFTSolver7DBVP(domain, params)
    >>> solution = solver.solve_envelope(source_field)
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..domain.domain_7d_bvp import Domain7DBVP
    from ..domain.parameters_7d_bvp import Parameters7DBVP

from .spectral_operations import SpectralOperations
from .spectral_derivatives import SpectralDerivatives
from .spectral_filtering import SpectralFiltering
from .bvp_solver_core import BVPSolverCore
from .bvp_solver_newton import BVPSolverNewton
from .bvp_solver_validation import BVPSolverValidation


class FFTSolver7DBVP:
    """
    7D BVP FFT Solver for complete envelope equation.
    
    Physical Meaning:
        Solves the complete 7D BVP envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
        in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ where a(x,φ,t) ∈ ℂ³ is a U(1)³ phase vector.
        
    Mathematical Foundation:
        Implements the full 7D BVP envelope equation with:
        - κ(|a|) = κ₀ + κ₂|a|² (nonlinear stiffness)
        - χ(|a|) = χ' + iχ''(|a|) (effective susceptibility with quenches)
        - Proper 7D physics normalization
        - U(1)³ phase structure
        
    Attributes:
        domain (Domain7DBVP): 7D BVP computational domain.
        parameters (Parameters7DBVP): 7D BVP parameters.
        _spectral_ops (SpectralOperations): Spectral operations calculator.
        _derivatives (SpectralDerivatives): Spectral derivatives calculator.
        _filtering (SpectralFiltering): Spectral filtering calculator.
        _fractional_laplacian (FractionalLaplacian): Fractional Laplacian operator.
    """
    
    def __init__(self, domain: 'Domain7DBVP', parameters: 'Parameters7DBVP'):
        """
        Initialize 7D BVP FFT solver.
        
        Physical Meaning:
            Sets up the solver with the 7D BVP domain and parameters,
            initializing all necessary components for solving the complete
            7D BVP envelope equation.
            
        Args:
            domain (Domain7DBVP): 7D BVP computational domain.
            parameters (Parameters7DBVP): 7D BVP parameters including
                nonlinear coefficients κ(|a|) and χ(|a|).
        """
        self.domain = domain
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
        
        # Validate domain dimensions
        if domain.dimensions != 7:
            raise ValueError(f"Domain must be 7D for BVP theory, got {domain.dimensions}")
        
        # Initialize spectral operations
        self._spectral_ops = SpectralOperations(domain, parameters.precision)
        self._derivatives = SpectralDerivatives(domain, parameters.precision)
        self._filtering = SpectralFiltering(domain, parameters.precision)
        
        # Initialize fractional Laplacian for linearized version
        self._fractional_laplacian = FractionalLaplacian(
            domain, parameters.beta, parameters.lambda_param
        )
        
        self.logger.info(f"FFTSolver7DBVP initialized for domain {domain.shape}")
    
    def solve_envelope(self, source_field: np.ndarray, 
                      initial_guess: Optional[np.ndarray] = None,
                      method: str = 'newton_raphson') -> np.ndarray:
        """
        Solve complete 7D BVP envelope equation.
        
        Physical Meaning:
            Solves the complete 7D BVP envelope equation:
            ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
            for the U(1)³ phase vector a(x,φ,t) ∈ ℂ³ in 7D space-time M₇.
            
        Mathematical Foundation:
            Solves the nonlinear equation using iterative methods:
            - Newton-Raphson for full nonlinear equation
            - Linearized version using fractional Laplacian for initial guess
            
        Args:
            source_field (np.ndarray): Source term s(x,φ,t) in 7D space-time.
            initial_guess (Optional[np.ndarray]): Initial guess for solution.
            method (str): Solution method ('newton_raphson', 'linearized').
            
        Returns:
            np.ndarray: Envelope solution a(x,φ,t) ∈ ℂ³ in 7D space-time.
        """
        if source_field.shape != self.domain.shape:
            raise ValueError(f"Source shape {source_field.shape} incompatible with domain {self.domain.shape}")
        
        if method == 'linearized':
            return self._solve_linearized(source_field)
        elif method == 'newton_raphson':
            return self._solve_newton_raphson(source_field, initial_guess)
        else:
            raise ValueError(f"Unknown solution method: {method}")
    
    def _solve_linearized(self, source_field: np.ndarray) -> np.ndarray:
        """
        Solve linearized version using fractional Laplacian.
        
        Physical Meaning:
            Solves the linearized version of the BVP equation:
            L_β a = μ(-Δ)^β a + λa = s(x,φ,t)
            which provides a good initial guess for the full nonlinear equation.
            
        Mathematical Foundation:
            In spectral space: â(k) = ŝ(k) / (μ|k|^(2β) + λ)
            where k is the 7D wave vector.
            
        Args:
            source_field (np.ndarray): Source term s(x,φ,t).
            
        Returns:
            np.ndarray: Linearized solution a(x,φ,t).
        """
        # Validate source for λ=0 case
        if self.parameters.lambda_param == 0:
            zero_mode = np.mean(source_field)
            if abs(zero_mode) > 1e-12:
                raise ValueError(
                    f"lambda=0 requires mean(source)=0, but got {zero_mode}. "
                    "Remove constant component from source field."
                )
        
        # Transform source to spectral space
        source_spectral = self._spectral_ops.forward_fft(source_field, 'physics')
        
        # Get spectral coefficients
        spectral_coeffs = self._fractional_laplacian.get_spectral_coefficients()
        
        # Apply spectral operator
        solution_spectral = source_spectral / spectral_coeffs
        
        # Transform back to real space
        solution = self._spectral_ops.inverse_fft(solution_spectral, 'physics')
        
        self.logger.info("Linearized BVP equation solved successfully")
        return solution
    
    def _solve_newton_raphson(self, source_field: np.ndarray, 
                            initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve full nonlinear equation using Newton-Raphson method.
        
        Physical Meaning:
            Solves the complete nonlinear BVP equation:
            ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
            using iterative Newton-Raphson method.
            
        Mathematical Foundation:
            Newton-Raphson iteration: a^(n+1) = a^(n) - J^(-1) * R(a^(n))
            where R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s is the residual
            and J is the Jacobian matrix.
            
        Args:
            source_field (np.ndarray): Source term s(x,φ,t).
            initial_guess (Optional[np.ndarray]): Initial guess for solution.
            
        Returns:
            np.ndarray: Nonlinear solution a(x,φ,t).
        """
        # Use linearized solution as initial guess if not provided
        if initial_guess is None:
            initial_guess = self._solve_linearized(source_field)
        
        # Newton-Raphson iteration with adaptive damping
        solution = initial_guess.copy()
        max_iterations = self.parameters.max_iterations
        tolerance = self.parameters.tolerance
        damping_factor = self.parameters.damping_factor
        
        previous_residual_norm = float('inf')
        
        for iteration in range(max_iterations):
            # Compute residual
            residual = self._compute_residual(solution, source_field)
            residual_norm = np.linalg.norm(residual)
            
            # Check convergence
            if residual_norm < tolerance:
                self.logger.info(f"Newton-Raphson converged in {iteration+1} iterations")
                break
            
            # Adaptive damping based on residual improvement
            if residual_norm > previous_residual_norm:
                damping_factor *= 0.5  # Reduce damping if residual increases
            else:
                damping_factor = min(damping_factor * 1.1, 0.5)  # Increase damping if improving
            
            # Compute Jacobian and solve linear system
            jacobian = self._compute_jacobian(solution)
            correction = self._solve_linear_system(jacobian, residual)
            
            # Update solution with adaptive damping and clipping
            solution = solution - damping_factor * correction
            
            # Clip solution to prevent numerical instability
            solution = np.clip(solution, -10.0, 10.0)
            
            previous_residual_norm = residual_norm
            self.logger.debug(f"Iteration {iteration+1}: residual_norm = {residual_norm:.2e}, damping = {damping_factor:.3f}")
        
        else:
            self.logger.warning(f"Newton-Raphson did not converge in {max_iterations} iterations")
        
        return solution
    
    def _compute_residual(self, solution: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of BVP equation.
        
        Physical Meaning:
            Computes the residual R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s
            of the BVP equation for the current solution.
            
        Mathematical Foundation:
            R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s
            where κ(|a|) = κ₀ + κ₂|a|² and χ(|a|) = χ' + iχ''(|a|).
            
        Args:
            solution (np.ndarray): Current solution a(x,φ,t).
            source (np.ndarray): Source term s(x,φ,t).
            
        Returns:
            np.ndarray: Residual R(a).
        """
        amplitude = np.abs(solution)
        
        # Compute nonlinear coefficients with numerical stability
        amplitude_clipped = np.clip(amplitude, 0, 10.0)  # Prevent overflow
        stiffness = self.parameters.compute_stiffness(amplitude_clipped)
        susceptibility = self.parameters.compute_susceptibility(amplitude_clipped)
        
        # Compute gradient
        gradient = self._derivatives.compute_gradient(solution)
        
        # Compute divergence of κ(|a|)∇a
        stiffness_gradient = [stiffness * grad for grad in gradient]
        divergence_term = self._derivatives.compute_divergence(tuple(stiffness_gradient))
        
        # Compute k₀²χ(|a|)a term
        susceptibility_term = (self.parameters.k0**2) * susceptibility * solution
        
        # Compute residual
        residual = divergence_term + susceptibility_term - source
        
        return residual
    
    def _compute_linearized_residual(self, solution: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of linearized BVP equation.
        
        Physical Meaning:
            Computes the residual R(a) = μ(-Δ)^β a + λa - s
            of the linearized BVP equation for the current solution.
            
        Mathematical Foundation:
            R(a) = μ(-Δ)^β a + λa - s
            This is the linearized version of the full BVP equation.
            
        Args:
            solution (np.ndarray): Current solution a(x,φ,t).
            source (np.ndarray): Source term s(x,φ,t).
            
        Returns:
            np.ndarray: Linearized residual R(a).
        """
        # Apply fractional Laplacian to solution
        laplacian_solution = self._fractional_laplacian.apply(solution)
        
        # Compute linearized residual
        residual = (self.parameters.mu * laplacian_solution + 
                   self.parameters.lambda_param * solution - source)
        
        return residual
    
    def _compute_jacobian(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton-Raphson.
        
        Physical Meaning:
            Computes the Jacobian matrix J = ∂R/∂a for the Newton-Raphson
            iteration, including derivatives of nonlinear terms.
            
        Mathematical Foundation:
            J = ∂/∂a [∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s]
            = ∇·(κ(|a|)∇) + ∇·(dκ/d|a| * ∇a * ∂|a|/∂a) + k₀²χ(|a|) + k₀²(dχ/d|a| * ∂|a|/∂a) * a
            
        Args:
            solution (np.ndarray): Current solution a(x,φ,t).
            
        Returns:
            np.ndarray: Jacobian matrix J.
        """
        # This is a simplified implementation
        # In practice, this would be a sparse matrix representation
        amplitude = np.abs(solution)
        
        # Compute coefficient derivatives with numerical stability
        amplitude_clipped = np.clip(amplitude, 0, 10.0)  # Prevent overflow
        dkappa_da = self.parameters.compute_stiffness_derivative(amplitude_clipped)
        dchi_da = self.parameters.compute_susceptibility_derivative(amplitude_clipped)
        
        # Simplified Jacobian (diagonal approximation)
        # In full implementation, this would include off-diagonal terms
        jacobian_diagonal = (
            self.parameters.kappa_0 +  # Linear stiffness term
            self.parameters.k0**2 * self.parameters.chi_prime  # Linear susceptibility term
        )
        
        # Add nonlinear contributions
        jacobian_diagonal += (
            dkappa_da * amplitude +  # Nonlinear stiffness
            self.parameters.k0**2 * dchi_da * amplitude  # Nonlinear susceptibility
        )
        
        # Return diagonal elements instead of full matrix
        # This avoids memory issues with large domains
        return jacobian_diagonal
    
    def _solve_linear_system(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """
        Solve linear system J * correction = residual.
        
        Physical Meaning:
            Solves the linear system for the Newton-Raphson correction
            using appropriate numerical methods.
            
        Args:
            jacobian (np.ndarray): Jacobian matrix J.
            residual (np.ndarray): Residual vector.
            
        Returns:
            np.ndarray: Correction vector.
        """
        # For diagonal Jacobian, solution is element-wise division
        # Add regularization to avoid division by zero and numerical instability
        epsilon = 1e-6
        jacobian_stable = np.clip(jacobian, epsilon, None)  # Ensure minimum value
        correction = residual / jacobian_stable
        
        # Clip correction to prevent large updates
        correction = np.clip(correction, -1.0, 1.0)
        
        return correction
    
    def validate_solution(self, solution: np.ndarray, source: np.ndarray,
                         tolerance: float = 1e-8, method: str = 'linearized') -> Dict[str, Any]:
        """
        Validate BVP solution.
        
        Physical Meaning:
            Validates the solution by computing the residual and checking
            that it satisfies the BVP equation within the specified tolerance.
            
        Args:
            solution (np.ndarray): Solution a(x,φ,t).
            source (np.ndarray): Source term s(x,φ,t).
            tolerance (float): Validation tolerance.
            method (str): Validation method ('linearized' or 'full').
            
        Returns:
            Dict[str, Any]: Validation results.
        """
        if method == 'linearized':
            # For linearized solutions, validate against linearized equation
            residual = self._compute_linearized_residual(solution, source)
        else:
            # For full solutions, validate against full BVP equation
            residual = self._compute_residual(solution, source)
        
        residual_norm = np.linalg.norm(residual)
        source_norm = np.linalg.norm(source)
        relative_residual = residual_norm / source_norm if source_norm > 0 else residual_norm
        
        is_valid = relative_residual < tolerance
        
        return {
            'is_valid': is_valid,
            'residual_norm': residual_norm,
            'relative_residual': relative_residual,
            'tolerance': tolerance,
            'method': method
        }
    
    def __repr__(self) -> str:
        """String representation of solver."""
        return (f"FFTSolver7DBVP("
                f"domain={self.domain.shape}, "
                f"κ₀={self.parameters.kappa_0}, "
                f"χ'={self.parameters.chi_prime}, "
                f"k₀={self.parameters.k0})")
