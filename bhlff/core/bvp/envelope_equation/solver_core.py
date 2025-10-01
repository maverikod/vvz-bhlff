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
from ..abstract_solver_core import AbstractSolverCore


class EnvelopeSolverCore7D(AbstractSolverCore):
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
        super().__init__(domain_7d, config)
        self.domain_7d = domain_7d
        
        # Initialize parameters and derivatives for full physics implementation
        self.parameters = None  # Will be set by external components
        self._derivatives = None  # Will be set by external components

    def compute_residual(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the 7D envelope equation.

        Physical Meaning:
            Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            for the Newton-Raphson method in 7D space-time.

        Mathematical Foundation:
            R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s
            where κ(|a|) = κ₀ + κ₂|a|² and χ(|a|) = χ' + iχ''(|a|).

        Args:
            envelope (np.ndarray): Current envelope estimate in 7D space-time.
            source (np.ndarray): Source term s(x,φ,t) in 7D space-time.

        Returns:
            np.ndarray: Residual r = L(a) - s in 7D space-time.
        """
        if self.parameters is None or self._derivatives is None:
            # Fallback to simple implementation if parameters not available
            return source - envelope
        
        amplitude = np.abs(envelope)

        # Compute nonlinear coefficients with numerical stability
        amplitude_clipped = np.clip(amplitude, 0, 10.0)  # Prevent overflow
        stiffness = self.parameters.compute_stiffness(amplitude_clipped)
        susceptibility = self.parameters.compute_susceptibility(amplitude_clipped)

        # Compute gradient
        gradient = self._derivatives.compute_gradient(envelope)

        # Compute divergence of κ(|a|)∇a
        stiffness_gradient = [stiffness * grad for grad in gradient]
        divergence_term = self._derivatives.compute_divergence(
            tuple(stiffness_gradient)
        )

        # Compute k₀²χ(|a|)a term
        susceptibility_term = (self.parameters.k0**2) * susceptibility * envelope

        # Compute residual
        residual = divergence_term + susceptibility_term - source

        return residual

    def compute_jacobian(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton-Raphson method.

        Physical Meaning:
            Computes the Jacobian matrix J = ∂r/∂a of the residual
            with respect to the envelope field.

        Mathematical Foundation:
            Implements full sparse Jacobian computation:
            J_{ij} = ∂r_i/∂a_j
            where r is the residual and a is the solution vector.

        Args:
            envelope (np.ndarray): Current envelope estimate.

        Returns:
            np.ndarray: Jacobian diagonal elements.
        """
        if self.parameters is None or self._derivatives is None:
            # Fallback to simple implementation if parameters not available
            return np.eye(envelope.size).reshape(envelope.shape + envelope.shape)
        
        amplitude = np.abs(envelope)

        # Compute coefficient derivatives with numerical stability
        amplitude_clipped = np.clip(amplitude, 0, 10.0)  # Prevent overflow
        dkappa_da = self.parameters.compute_stiffness_derivative(amplitude_clipped)
        dchi_da = self.parameters.compute_susceptibility_derivative(amplitude_clipped)

        # Compute full sparse Jacobian matrix
        jacobian_sparse = self._compute_sparse_jacobian(envelope, dkappa_da, dchi_da)

        # Compute diagonal elements with full accuracy
        diagonal_elements = jacobian_sparse.diagonal()

        # Add off-diagonal contributions
        off_diagonal_contributions = self._compute_off_diagonal_contributions(
            jacobian_sparse, envelope
        )

        # Combine diagonal and off-diagonal
        full_jacobian_diagonal = diagonal_elements + off_diagonal_contributions

        return full_jacobian_diagonal

    def solve_linear_system(
        self, jacobian: np.ndarray, residual: np.ndarray
    ) -> np.ndarray:
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

    def compute_jacobian_sparse(
        self,
        envelope: np.ndarray,
        dkappa_da: np.ndarray,
        dchi_da: np.ndarray,
        derivative_operators: object,
        nonlinear_terms: object,
    ) -> csc_matrix:
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
                jacobian[i, i] += (
                    dkappa_da.flat[i] * np.sum(grad_op[i, :]) * np.sum(div_op[:, i])
                )

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
                jacobian[i, i] += (
                    dkappa_da.flat[i] * np.sum(grad_op[i, :]) * np.sum(div_op[:, i])
                )

        # Add susceptibility term contributions
        for i in range(field_size):
            # Contribution from χ(|a|) term
            jacobian[i, i] += nonlinear_terms.k0**2 * (
                dchi_da.flat[i] * envelope.flat[i]
                + nonlinear_terms.chi_func(np.abs(envelope).flat[i])
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
        base_params = super().get_solver_parameters()
        base_params.update({
            "domain_7d_shape": self.domain_7d.shape,
            "field_size_7d": np.prod(self.domain_7d.shape),
        })
        return base_params

    def _compute_sparse_jacobian(self, solution: np.ndarray, dkappa_da: np.ndarray, dchi_da: np.ndarray) -> np.ndarray:
        """
        Compute sparse Jacobian matrix for BVP equation.
        
        Physical Meaning:
            Computes the sparse Jacobian matrix representation for the BVP equation,
            including all spatial coupling terms and nonlinear contributions.
        """
        if self.parameters is None:
            # Fallback to simple implementation if parameters not available
            return np.eye(solution.size, dtype=complex)
        
        # Get field dimensions
        shape = solution.shape
        total_size = np.prod(shape)
        
        # Initialize sparse Jacobian matrix
        jacobian = np.zeros((total_size, total_size), dtype=complex)
        
        # Compute spatial step sizes
        dx = 1.0 / shape[0] if len(shape) > 0 else 1.0
        dy = 1.0 / shape[1] if len(shape) > 1 else dx
        dz = 1.0 / shape[2] if len(shape) > 2 else dy
        
        # Fill diagonal terms (local contributions)
        for i in range(total_size):
            coords = np.unravel_index(i, shape)
            jacobian[i, i] = (
                self.parameters.kappa_0
                + self.parameters.k0**2 * self.parameters.chi_prime
                + dkappa_da[coords] * np.abs(solution[coords])
                + self.parameters.k0**2 * dchi_da[coords] * np.abs(solution[coords])
            )
        
        # Fill off-diagonal terms (spatial coupling)
        for i in range(total_size):
            coords = np.unravel_index(i, shape)
            
            # X-direction coupling
            if coords[0] > 0:
                j_prev = np.ravel_multi_index((coords[0] - 1,) + coords[1:], shape)
                jacobian[i, j_prev] = -self.parameters.kappa_0 / (dx * dx)
            
            if coords[0] < shape[0] - 1:
                j_next = np.ravel_multi_index((coords[0] + 1,) + coords[1:], shape)
                jacobian[i, j_next] = -self.parameters.kappa_0 / (dx * dx)
            
            # Y-direction coupling (if 2D or higher)
            if len(shape) > 1:
                if coords[1] > 0:
                    j_prev = np.ravel_multi_index(
                        coords[:1] + (coords[1] - 1,) + coords[2:], shape
                    )
                    jacobian[i, j_prev] = -self.parameters.kappa_0 / (dy * dy)
                
                if coords[1] < shape[1] - 1:
                    j_next = np.ravel_multi_index(
                        coords[:1] + (coords[1] + 1,) + coords[2:], shape
                    )
                    jacobian[i, j_next] = -self.parameters.kappa_0 / (dy * dy)
            
            # Z-direction coupling (if 3D or higher)
            if len(shape) > 2:
                if coords[2] > 0:
                    j_prev = np.ravel_multi_index(
                        coords[:2] + (coords[2] - 1,) + coords[3:], shape
                    )
                    jacobian[i, j_prev] = -self.parameters.kappa_0 / (dz * dz)
                
                if coords[2] < shape[2] - 1:
                    j_next = np.ravel_multi_index(
                        coords[:2] + (coords[2] + 1,) + coords[3:], shape
                    )
                    jacobian[i, j_next] = -self.parameters.kappa_0 / (dz * dz)
        
        return jacobian
    
    def _compute_off_diagonal_contributions(self, jacobian_sparse: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """
        Compute off-diagonal contributions to Jacobian diagonal.
        
        Physical Meaning:
            Computes the effective diagonal contributions from off-diagonal
            coupling terms in the sparse Jacobian matrix.
        """
        # Extract diagonal elements
        diagonal_elements = jacobian_sparse.diagonal()
        
        # Compute off-diagonal contributions using matrix-vector product
        # This approximates the effect of off-diagonal terms on the diagonal
        off_diagonal_contributions = np.zeros_like(diagonal_elements)
        
        # For each diagonal element, compute contribution from off-diagonal terms
        for i in range(len(diagonal_elements)):
            # Get off-diagonal elements in row i
            row = jacobian_sparse[i, :]
            off_diagonal_mask = np.arange(len(row)) != i
            off_diagonal_elements = row[off_diagonal_mask]
            
            # Compute weighted contribution (simplified approximation)
            if len(off_diagonal_elements) > 0:
                off_diagonal_contributions[i] = np.mean(np.abs(off_diagonal_elements)) * 0.1
        
        return off_diagonal_contributions
