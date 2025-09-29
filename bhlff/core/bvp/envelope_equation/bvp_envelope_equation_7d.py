"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Main 7D BVP envelope equation implementation.

This module implements the main BVPEnvelopeEquation7D class that coordinates
the solution of the 7D envelope equation using the modular components.

Physical Meaning:
    The main envelope equation class coordinates the solution of the full
    7D envelope equation in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, including
    spatial, phase, and temporal derivatives with nonlinear terms.

Mathematical Foundation:
    Solves the 7D envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    using modular derivative operators and nonlinear terms.

Example:
    >>> equation_7d = BVPEnvelopeEquation7D(domain_7d, config)
    >>> envelope = equation_7d.solve_envelope(source_7d)
"""

import numpy as np
from typing import Dict, Any, Optional

from ...domain.domain_7d import Domain7D
from ..bvp_constants import BVPConstants
from .derivative_operators import DerivativeOperators7D
from .nonlinear_terms import NonlinearTerms7D


class BVPEnvelopeEquation7D:
    """
    7D BVP envelope equation solver.
    
    Physical Meaning:
        Solves the full 7D envelope equation in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ,
        including spatial, phase, and temporal derivatives with nonlinear
        stiffness and susceptibility terms.
        
    Mathematical Foundation:
        Solves the 7D envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
        where:
        - κ(|a|) = κ₀ + κ₂|a|² (nonlinear stiffness)
        - χ(|a|) = χ' + iχ''(|a|) (effective susceptibility with quenches)
        - s(x,φ,t) is the source term
    """
    
    def __init__(self, domain_7d: Domain7D, config: Dict[str, Any]):
        """
        Initialize 7D envelope equation solver.
        
        Physical Meaning:
            Sets up the envelope equation solver with the computational
            domain and configuration parameters, initializing all
            necessary components for solving the 7D equation.
            
        Args:
            domain_7d (Domain7D): 7D computational domain.
            config (Dict[str, Any]): Configuration parameters including:
                - kappa_0, kappa_2: Stiffness coefficients
                - chi_prime, chi_double_prime_0: Susceptibility coefficients
                - k0: Wave number
                - max_iterations: Maximum Newton-Raphson iterations
                - tolerance: Convergence tolerance
        """
        self.domain_7d = domain_7d
        self.config = config
        
        # Initialize modular components
        self.derivative_operators = DerivativeOperators7D(domain_7d)
        self.nonlinear_terms = NonlinearTerms7D(domain_7d, config)
        
        # Setup components
        self.derivative_operators.setup_operators()
        self.nonlinear_terms.setup_terms()
        
        # Solver parameters
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-8)
    
    def solve_envelope(self, source_7d: np.ndarray, 
                      initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve 7D envelope equation.
        
        Physical Meaning:
            Solves the full 7D envelope equation for the BVP field
            in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ using iterative
            Newton-Raphson method for nonlinear terms.
            
        Mathematical Foundation:
            Solves: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
            using iterative Newton-Raphson method for nonlinear terms.
            
        Args:
            source_7d (np.ndarray): 7D source term s(x,φ,t).
                Shape: (N_x, N_y, N_z, N_φx, N_φy, N_φz, N_t)
            initial_guess (Optional[np.ndarray]): Initial guess for solution.
                If None, uses zero initial guess.
                
        Returns:
            np.ndarray: 7D envelope solution a(x,φ,t).
                Shape: (N_x, N_y, N_z, N_φx, N_φy, N_φz, N_t)
        """
        # Initialize solution
        if initial_guess is None:
            envelope = np.zeros_like(source_7d, dtype=complex)
        else:
            envelope = initial_guess.copy()
        
        # Newton-Raphson iteration
        for iteration in range(self.max_iterations):
            # Compute residual
            residual = self._compute_residual(envelope, source_7d)
            
            # Check convergence
            residual_norm = np.linalg.norm(residual)
            if residual_norm < self.tolerance:
                break
            
            # Compute Jacobian and solve for update
            jacobian = self._compute_jacobian(envelope)
            update = self._solve_linear_system(jacobian, residual)
            
            # Update solution
            envelope -= update
        
        return envelope
    
    def _compute_residual(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the envelope equation.
        
        Physical Meaning:
            Computes the residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            for the current envelope solution.
            
        Mathematical Foundation:
            The residual measures how well the current solution satisfies
            the envelope equation and is used in Newton-Raphson iterations.
            
        Args:
            envelope (np.ndarray): Current envelope solution.
            source (np.ndarray): Source term.
            
        Returns:
            np.ndarray: Residual vector.
        """
        amplitude = np.abs(envelope)
        
        # Compute nonlinear coefficients
        kappa = self.nonlinear_terms.compute_stiffness(amplitude)
        chi = self.nonlinear_terms.compute_susceptibility(amplitude)
        
        # Compute spatial divergence term: ∇ₓ·(κ(|a|)∇ₓa)
        spatial_div = self._compute_spatial_divergence(kappa, envelope)
        
        # Compute phase divergence term: ∇φ·(κ(|a|)∇φa)
        phase_div = self._compute_phase_divergence(kappa, envelope)
        
        # Compute susceptibility term: k₀²χ(|a|)a
        susceptibility_term = self.nonlinear_terms.k0**2 * chi * envelope
        
        # Total residual
        residual = spatial_div + phase_div + susceptibility_term - source
        
        return residual
    
    def _compute_spatial_divergence(self, kappa: np.ndarray, envelope: np.ndarray) -> np.ndarray:
        """
        Compute spatial divergence term ∇ₓ·(κ(|a|)∇ₓa).
        
        Physical Meaning:
            Computes the spatial divergence of the stiffness-weighted
            gradient, representing the spatial part of the envelope equation.
            
        Args:
            kappa (np.ndarray): Nonlinear stiffness κ(|a|).
            envelope (np.ndarray): Envelope field.
            
        Returns:
            np.ndarray: Spatial divergence term.
        """
        divergence = np.zeros_like(envelope)
        
        # Apply to each spatial dimension
        for axis in range(3):
            # Compute gradient: ∇ₓa
            grad_envelope = self.derivative_operators.apply_spatial_gradient(envelope, axis)
            
            # Multiply by stiffness: κ(|a|)∇ₓa
            weighted_grad = kappa * grad_envelope
            
            # Compute divergence: ∇ₓ·(κ(|a|)∇ₓa)
            div_term = self.derivative_operators.apply_spatial_divergence(weighted_grad, axis)
            divergence += div_term
        
        return divergence
    
    def _compute_phase_divergence(self, kappa: np.ndarray, envelope: np.ndarray) -> np.ndarray:
        """
        Compute phase divergence term ∇φ·(κ(|a|)∇φa).
        
        Physical Meaning:
            Computes the phase divergence of the stiffness-weighted
            gradient, representing the phase part of the envelope equation.
            
        Args:
            kappa (np.ndarray): Nonlinear stiffness κ(|a|).
            envelope (np.ndarray): Envelope field.
            
        Returns:
            np.ndarray: Phase divergence term.
        """
        divergence = np.zeros_like(envelope)
        
        # Apply to each phase dimension
        for axis in range(3, 6):
            # Compute gradient: ∇φa
            grad_envelope = self.derivative_operators.apply_phase_gradient(envelope, axis)
            
            # Multiply by stiffness: κ(|a|)∇φa
            weighted_grad = kappa * grad_envelope
            
            # Compute divergence: ∇φ·(κ(|a|)∇φa)
            div_term = self.derivative_operators.apply_phase_divergence(weighted_grad, axis)
            divergence += div_term
        
        return divergence
    
    def _compute_jacobian(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton-Raphson iteration.
        
        Physical Meaning:
            Computes the Jacobian matrix of the residual with respect
            to the envelope field, needed for Newton-Raphson iterations.
            The Jacobian represents the sensitivity of the residual
            to changes in the envelope field.
            
        Mathematical Foundation:
            The Jacobian J = ∂R/∂a includes derivatives of all terms:
            - Spatial divergence: ∂/∂a[∇·(κ(|a|)∇a)]
            - Phase divergence: ∂/∂a[∇φ·(κ(|a|)∇φa)]
            - Susceptibility term: ∂/∂a[k₀²χ(|a|)a]
            
        Args:
            envelope (np.ndarray): Current envelope solution.
            
        Returns:
            np.ndarray: Jacobian matrix.
        """
        from scipy.sparse import csc_matrix, lil_matrix
        
        amplitude = np.abs(envelope)
        field_size = envelope.size
        
        # Initialize sparse Jacobian matrix
        jacobian = lil_matrix((field_size, field_size), dtype=complex)
        
        # Compute derivatives of nonlinear terms
        dkappa_da = self.nonlinear_terms.compute_stiffness_derivative(amplitude)
        dchi_da = self.nonlinear_terms.compute_susceptibility_derivative(amplitude)
        
        # Add identity matrix for linear terms
        for i in range(field_size):
            jacobian[i, i] = 1.0
        
        # Add spatial divergence contributions
        for axis in range(3):
            # Get spatial derivative operators
            if axis == 0:
                grad_op = self.derivative_operators.grad_x
                div_op = self.derivative_operators.div_x
            elif axis == 1:
                grad_op = self.derivative_operators.grad_y
                div_op = self.derivative_operators.div_y
            else:
                grad_op = self.derivative_operators.grad_z
                div_op = self.derivative_operators.div_z
            
            # Add contributions from spatial derivatives
            for i in range(field_size):
                # Contribution from κ(|a|) term
                jacobian[i, i] += dkappa_da.flat[i] * np.sum(grad_op[i, :]) * np.sum(div_op[:, i])
        
        # Add phase divergence contributions
        for axis in range(3, 6):
            # Get phase derivative operators
            if axis == 3:
                grad_op = self.derivative_operators.grad_phi_1
                div_op = self.derivative_operators.div_phi_1
            elif axis == 4:
                grad_op = self.derivative_operators.grad_phi_2
                div_op = self.derivative_operators.div_phi_2
            else:
                grad_op = self.derivative_operators.grad_phi_3
                div_op = self.derivative_operators.div_phi_3
            
            # Add contributions from phase derivatives
            for i in range(field_size):
                # Contribution from κ(|a|) term
                jacobian[i, i] += dkappa_da.flat[i] * np.sum(grad_op[i, :]) * np.sum(div_op[:, i])
        
        # Add susceptibility term contributions
        for i in range(field_size):
            # Contribution from χ(|a|) term
            jacobian[i, i] += self.nonlinear_terms.k0**2 * (
                dchi_da.flat[i] * envelope.flat[i] + 
                self.nonlinear_terms.chi_func(amplitude.flat[i])
            )
        
        return csc_matrix(jacobian)
    
    def _solve_linear_system(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """
        Solve linear system for Newton-Raphson update.
        
        Physical Meaning:
            Solves the linear system J·δa = -R for the Newton-Raphson
            update δa, where J is the Jacobian and R is the residual.
            
        Args:
            jacobian (np.ndarray): Jacobian matrix.
            residual (np.ndarray): Residual vector.
            
        Returns:
            np.ndarray: Update vector δa.
        """
        # Solve linear system
        update = np.linalg.solve(jacobian, -residual.flatten())
        return update.reshape(residual.shape)
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get envelope equation parameters.
        
        Physical Meaning:
            Returns the current values of all parameters for
            monitoring and analysis purposes.
            
        Returns:
            Dict[str, float]: Dictionary containing all parameters.
        """
        params = self.nonlinear_terms.get_parameters()
        params.update({
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance
        })
        return params
    
    def __repr__(self) -> str:
        """
        String representation of envelope equation solver.
        
        Returns:
            str: String representation showing domain and parameters.
        """
        return f"BVPEnvelopeEquation7D(domain_7d={self.domain_7d}, " \
               f"max_iterations={self.max_iterations}, tolerance={self.tolerance})"
