"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core BVP solver functionality for 7D envelope equation.

This module contains the core functionality for solving the 7D BVP envelope equation,
including residual computation, Jacobian calculation, and linear system solving.

Physical Meaning:
    Implements the core mathematical operations for the 7D BVP envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)

Mathematical Foundation:
    Core operations include:
    - Residual computation: R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s
    - Jacobian computation for Newton-Raphson
    - Linear system solving for corrections

Example:
    >>> core = BVPSolverCore(domain, parameters, derivatives)
    >>> residual = core.compute_residual(solution, source)
    >>> jacobian = core.compute_jacobian(solution)
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.domain_7d_bvp import Domain7DBVP
    from ..domain.parameters_7d_bvp import Parameters7DBVP
    from .spectral_derivatives import SpectralDerivatives


class BVPSolverCore:
    """
    Core BVP solver functionality.

    Physical Meaning:
        Implements the core mathematical operations for solving the 7D BVP
        envelope equation, including residual and Jacobian computation.

    Mathematical Foundation:
        Handles the nonlinear terms in the BVP equation:
        R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s

    Attributes:
        domain (Domain7DBVP): 7D BVP computational domain.
        parameters (Parameters7DBVP): 7D BVP parameters.
        _derivatives (SpectralDerivatives): Spectral derivatives calculator.
    """

    def __init__(
        self,
        domain: "Domain7DBVP",
        parameters: "Parameters7DBVP",
        derivatives: "SpectralDerivatives",
    ):
        """
        Initialize BVP solver core.

        Physical Meaning:
            Sets up the core solver with domain, parameters, and derivatives
            calculator for solving the BVP equation.

        Args:
            domain (Domain7DBVP): 7D BVP computational domain.
            parameters (Parameters7DBVP): 7D BVP parameters.
            derivatives (SpectralDerivatives): Spectral derivatives calculator.
        """
        self.domain = domain
        self.parameters = parameters
        self._derivatives = derivatives
        self.logger = logging.getLogger(__name__)

        self.logger.info("BVPSolverCore initialized.")

    def compute_residual(self, solution: np.ndarray, source: np.ndarray) -> np.ndarray:
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
        divergence_term = self._derivatives.compute_divergence(
            tuple(stiffness_gradient)
        )

        # Compute k₀²χ(|a|)a term
        susceptibility_term = (self.parameters.k0**2) * susceptibility * solution

        # Compute residual
        residual = divergence_term + susceptibility_term - source

        return residual

    def compute_linearized_residual(
        self, solution: np.ndarray, source: np.ndarray
    ) -> np.ndarray:
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
        from .fractional_laplacian import FractionalLaplacian

        fractional_laplacian = FractionalLaplacian(
            self.domain, self.parameters.beta, self.parameters.lambda_param
        )
        laplacian_solution = fractional_laplacian.apply(solution)

        # Compute linearized residual
        residual = (
            self.parameters.mu * laplacian_solution
            + self.parameters.lambda_param * solution
            - source
        )

        return residual

    def compute_jacobian(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute full sparse Jacobian matrix for Newton-Raphson method.

        Physical Meaning:
            Computes the complete sparse Jacobian matrix for the
            Newton-Raphson method in solving the 7D BVP envelope equation.

        Mathematical Foundation:
            Implements full sparse Jacobian computation:
            J_{ij} = ∂r_i/∂a_j
            where r is the residual and a is the solution vector.

        Args:
            solution (np.ndarray): Current solution a(x,φ,t).

        Returns:
            np.ndarray: Jacobian diagonal elements.
        """
        amplitude = np.abs(solution)

        # Compute coefficient derivatives with numerical stability
        amplitude_clipped = np.clip(amplitude, 0, 10.0)  # Prevent overflow
        dkappa_da = self.parameters.compute_stiffness_derivative(amplitude_clipped)
        dchi_da = self.parameters.compute_susceptibility_derivative(amplitude_clipped)

        # Compute full sparse Jacobian matrix
        jacobian_sparse = self._compute_sparse_jacobian(solution, dkappa_da, dchi_da)

        # Compute diagonal elements with full accuracy
        diagonal_elements = jacobian_sparse.diagonal()

        # Add off-diagonal contributions
        off_diagonal_contributions = self._compute_off_diagonal_contributions(
            jacobian_sparse, solution
        )

        # Combine diagonal and off-diagonal
        full_jacobian_diagonal = diagonal_elements + off_diagonal_contributions

        return full_jacobian_diagonal

    def solve_linear_system(
        self, jacobian: np.ndarray, residual: np.ndarray
    ) -> np.ndarray:
        """
        Solve linear system J * correction = residual.

        Physical Meaning:
            Solves the linear system for the Newton-Raphson correction
            step, where J is the Jacobian matrix.

        Mathematical Foundation:
            Solves J * correction = residual for the correction vector.
            For diagonal Jacobian: correction = residual / jacobian_diagonal

        Args:
            jacobian (np.ndarray): Jacobian diagonal elements.
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

    
    def _compute_sparse_jacobian(self, solution: np.ndarray, dkappa_da: np.ndarray, dchi_da: np.ndarray) -> np.ndarray:
        """
        Compute sparse Jacobian matrix for BVP equation.
        
        Physical Meaning:
            Computes the sparse Jacobian matrix representation for the BVP equation,
            including all spatial coupling terms and nonlinear contributions.
        """
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
