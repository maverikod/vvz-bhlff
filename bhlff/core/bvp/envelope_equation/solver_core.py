"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Solver core for 7D BVP envelope equation.

This module implements the core solving algorithms for the 7D BVP envelope
equation, including Newton-Raphson iterations and linear system solving.

Physical Meaning:
    The solver core implements the iterative solution of the 7D envelope
    equation using Newton-Raphson method for nonlinear terms, representing
    the numerical solution of the field evolution in 7D space-time.

Mathematical Foundation:
    Implements Newton-Raphson iteration for solving:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    using iterative linearization and sparse linear system solving.

Example:
    >>> solver_core = EnvelopeSolverCore7D(domain_7d, config)
    >>> solution = solver_core.solve_envelope(source_7d)
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.sparse import csc_matrix, lil_matrix

from ...domain.domain_7d import Domain7D


class EnvelopeSolverCore7D:
    """
    Core solver for 7D BVP envelope equation.
    
    Physical Meaning:
        Implements the core solving algorithms for the 7D envelope equation
        using Newton-Raphson iterations for nonlinear terms and sparse
        linear system solving for the linearized equations.
        
    Mathematical Foundation:
        Solves the 7D envelope equation using Newton-Raphson method:
        1. Compute residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s
        2. Compute Jacobian J = ∂R/∂a
        3. Solve J·δa = -R for update δa
        4. Update solution a ← a - δa
        5. Repeat until convergence
    """
    
    def __init__(self, domain_7d: Domain7D, config: Dict[str, Any]):
        """
        Initialize envelope solver core.
        
        Physical Meaning:
            Sets up the solver core with the computational domain and
            configuration parameters for solving the 7D envelope equation.
            
        Args:
            domain_7d (Domain7D): 7D computational domain.
            config (Dict[str, Any]): Configuration parameters including:
                - max_iterations: Maximum Newton-Raphson iterations
                - tolerance: Convergence tolerance
        """
        self.domain_7d = domain_7d
        self.config = config
        
        # Solver parameters
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-8)
    
    def solve_envelope(self, source_7d: np.ndarray, 
                      initial_guess: Optional[np.ndarray] = None,
                      residual_func: callable = None,
                      jacobian_func: callable = None) -> np.ndarray:
        """
        Solve 7D envelope equation using Newton-Raphson method.
        
        Physical Meaning:
            Solves the full 7D envelope equation for the BVP field
            in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ using iterative
            Newton-Raphson method for nonlinear terms.
            
        Mathematical Foundation:
            Implements Newton-Raphson iteration:
            1. Compute residual R = F(a) - s
            2. Compute Jacobian J = ∂F/∂a
            3. Solve J·δa = -R
            4. Update a ← a - δa
            5. Repeat until ||R|| < tolerance
            
        Args:
            source_7d (np.ndarray): 7D source term s(x,φ,t).
            initial_guess (Optional[np.ndarray]): Initial guess for solution.
            residual_func (callable): Function to compute residual.
            jacobian_func (callable): Function to compute Jacobian.
                
        Returns:
            np.ndarray: 7D envelope solution a(x,φ,t).
        """
        # Initialize solution
        if initial_guess is None:
            envelope = np.zeros_like(source_7d, dtype=complex)
        else:
            envelope = initial_guess.copy()
        
        # Newton-Raphson iteration
        for iteration in range(self.max_iterations):
            # Compute residual
            residual = residual_func(envelope, source_7d)
            
            # Check convergence
            residual_norm = np.linalg.norm(residual)
            if residual_norm < self.tolerance:
                break
            
            # Compute Jacobian and solve for update
            jacobian = jacobian_func(envelope)
            update = self._solve_linear_system(jacobian, residual)
            
            # Update solution
            envelope -= update
        
        return envelope
    
    def _solve_linear_system(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """
        Solve linear system for Newton-Raphson update.
        
        Physical Meaning:
            Solves the linear system J·δa = -R for the Newton-Raphson
            update δa, where J is the Jacobian and R is the residual.
            
        Mathematical Foundation:
            Solves the linearized system using sparse linear algebra
            methods appropriate for the large sparse Jacobian matrix.
            
        Args:
            jacobian (np.ndarray): Jacobian matrix.
            residual (np.ndarray): Residual vector.
            
        Returns:
            np.ndarray: Update vector δa.
        """
        # Solve linear system
        update = np.linalg.solve(jacobian, -residual.flatten())
        return update.reshape(residual.shape)
    
    def compute_jacobian_sparse(self, envelope: np.ndarray, 
                               dkappa_da: np.ndarray, 
                               dchi_da: np.ndarray,
                               derivative_operators: object,
                               nonlinear_terms: object) -> csc_matrix:
        """
        Compute sparse Jacobian matrix for Newton-Raphson iteration.
        
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
            dkappa_da (np.ndarray): Derivative of stiffness with respect to amplitude.
            dchi_da (np.ndarray): Derivative of susceptibility with respect to amplitude.
            derivative_operators: Derivative operators object.
            nonlinear_terms: Nonlinear terms object.
            
        Returns:
            csc_matrix: Sparse Jacobian matrix.
        """
        field_size = envelope.size
        
        # Initialize sparse Jacobian matrix
        jacobian = lil_matrix((field_size, field_size), dtype=complex)
        
        # Add identity matrix for linear terms
        for i in range(field_size):
            jacobian[i, i] = 1.0
        
        # Add spatial divergence contributions
        for axis in range(3):
            # Get spatial derivative operators
            if axis == 0:
                grad_op = derivative_operators.spatial.grad_x
                div_op = derivative_operators.spatial.div_x
            elif axis == 1:
                grad_op = derivative_operators.spatial.grad_y
                div_op = derivative_operators.spatial.div_y
            else:
                grad_op = derivative_operators.spatial.grad_z
                div_op = derivative_operators.spatial.div_z
            
            # Add contributions from spatial derivatives
            for i in range(field_size):
                # Contribution from κ(|a|) term
                jacobian[i, i] += dkappa_da.flat[i] * np.sum(grad_op[i, :]) * np.sum(div_op[:, i])
        
        # Add phase divergence contributions
        for axis in range(3, 6):
            # Get phase derivative operators
            if axis == 3:
                grad_op = derivative_operators.phase.grad_phi_1
                div_op = derivative_operators.phase.div_phi_1
            elif axis == 4:
                grad_op = derivative_operators.phase.grad_phi_2
                div_op = derivative_operators.phase.div_phi_2
            else:
                grad_op = derivative_operators.phase.grad_phi_3
                div_op = derivative_operators.phase.div_phi_3
            
            # Add contributions from phase derivatives
            for i in range(field_size):
                # Contribution from κ(|a|) term
                jacobian[i, i] += dkappa_da.flat[i] * np.sum(grad_op[i, :]) * np.sum(div_op[:, i])
        
        # Add susceptibility term contributions
        for i in range(field_size):
            # Contribution from χ(|a|) term
            jacobian[i, i] += nonlinear_terms.k0**2 * (
                dchi_da.flat[i] * envelope.flat[i] + 
                nonlinear_terms.chi_func(np.abs(envelope).flat[i])
            )
        
        return csc_matrix(jacobian)
    
    def get_solver_parameters(self) -> Dict[str, Any]:
        """
        Get solver parameters.
        
        Physical Meaning:
            Returns the current values of all solver parameters for
            monitoring and analysis purposes.
            
        Returns:
            Dict[str, Any]: Dictionary containing solver parameters.
        """
        return {
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'domain_shape': self.domain_7d.get_spatial_shape(),
            'field_size': np.prod(self.domain_7d.get_spatial_shape())
        }
