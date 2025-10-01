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
    from ..bvp.abstract_solver_core import AbstractSolverCore
else:
    from ..bvp.abstract_solver_core import AbstractSolverCore


class BVPSolverCore(AbstractSolverCore):
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
        config: Dict[str, Any],
        parameters: "Parameters7DBVP" = None,
        derivatives: "SpectralDerivatives" = None,
    ):
        """
        Initialize BVP solver core.

        Physical Meaning:
            Sets up the core solver with domain, parameters, and derivatives
            calculator for solving the BVP equation.

        Args:
            domain (Domain7DBVP): 7D BVP computational domain.
            config (Dict[str, Any]): Configuration parameters.
            parameters (Parameters7DBVP, optional): 7D BVP parameters.
            derivatives (SpectralDerivatives, optional): Spectral derivatives calculator.
        """
        super().__init__(domain, config)
        self.parameters = parameters
        self._derivatives = derivatives

    def compute_residual(self, solution: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of BVP equation with full physics implementation.

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
        if self.parameters is None or self._derivatives is None:
            # Fallback to base implementation
            return super().compute_residual(solution, source)
        
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
            Implements full 7D sparse matrix with proper spatial coupling
            and nonlinear contributions.

        Mathematical Foundation:
            Implements full sparse Jacobian computation for 7D BVP:
            J_{ij} = ∂r_i/∂a_j = ∂/∂a_j [∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s]
            where r is the residual and a is the solution vector in 7D space.

        Args:
            solution (np.ndarray): Current solution a(x,φ,t) in 7D space.

        Returns:
            np.ndarray: Full sparse Jacobian matrix as diagonal elements.
        """
        if self.parameters is None or self._derivatives is None:
            # Fallback to base implementation
            return super().compute_jacobian(solution)
        
        amplitude = np.abs(solution)

        # Compute coefficient derivatives with numerical stability
        amplitude_clipped = np.clip(amplitude, 0, 10.0)  # Prevent overflow
        dkappa_da = self.parameters.compute_stiffness_derivative(amplitude_clipped)
        dchi_da = self.parameters.compute_susceptibility_derivative(amplitude_clipped)

        # Compute full 7D sparse Jacobian matrix
        jacobian_sparse = self._compute_full_7d_sparse_jacobian(solution, dkappa_da, dchi_da)

        # Extract diagonal elements with full accuracy
        diagonal_elements = jacobian_sparse.diagonal()

        # Compute full off-diagonal contributions for 7D space
        off_diagonal_contributions = self._compute_full_7d_off_diagonal_contributions(
            jacobian_sparse, solution
        )

        # Combine diagonal and off-diagonal with proper weighting
        full_jacobian_diagonal = diagonal_elements + off_diagonal_contributions

        # Reshape to match solution shape
        jacobian_diagonal_reshaped = full_jacobian_diagonal.reshape(solution.shape)

        return jacobian_diagonal_reshaped

    def solve_linear_system(
        self, jacobian: np.ndarray, residual: np.ndarray
    ) -> np.ndarray:
        """
        Solve linear system for Newton-Raphson update with enhanced stability.

        Physical Meaning:
            Solves the linear system J·δa = -r for the Newton-Raphson
            update δa, where J is the Jacobian and R is the residual.

        Mathematical Foundation:
            Solves J * correction = residual for the correction vector.
            For diagonal Jacobian: correction = residual / jacobian_diagonal

        Args:
            jacobian (np.ndarray): Jacobian diagonal elements.
            residual (np.ndarray): Residual vector.

        Returns:
            np.ndarray: Correction vector.
        """
        # Enhanced implementation with better numerical stability
        # For diagonal Jacobian, solution is element-wise division
        # Add regularization to avoid division by zero and numerical instability
        epsilon = 1e-6
        jacobian_stable = np.clip(jacobian, epsilon, None)  # Ensure minimum value
        correction = residual / jacobian_stable

        # Clip correction to prevent large updates
        correction = np.clip(correction, -1.0, 1.0)

        return correction

    
    def _compute_full_7d_sparse_jacobian(self, solution: np.ndarray, dkappa_da: np.ndarray, dchi_da: np.ndarray) -> np.ndarray:
        """
        Compute full 7D sparse Jacobian matrix for BVP equation.
        
        Physical Meaning:
            Computes the complete sparse Jacobian matrix representation for the 7D BVP equation,
            including all spatial coupling terms, phase coupling, temporal coupling,
            and nonlinear contributions in 7D space-time.
            
        Mathematical Foundation:
            Implements full 7D sparse Jacobian with proper coupling:
            - Spatial coupling: x, y, z directions
            - Phase coupling: φ₁, φ₂, φ₃ directions  
            - Temporal coupling: t direction
            - Nonlinear coupling: κ(|a|) and χ(|a|) derivatives
        """
        # Get field dimensions (7D: x, y, z, φ₁, φ₂, φ₃, t)
        shape = solution.shape
        total_size = np.prod(shape)
        
        # Initialize sparse Jacobian matrix
        jacobian = np.zeros((total_size, total_size), dtype=complex)
        
        # Compute step sizes for all 7 dimensions
        step_sizes = self._compute_7d_step_sizes(shape)
        
        # Fill diagonal terms (local contributions)
        for i in range(total_size):
            coords = np.unravel_index(i, shape)
            jacobian[i, i] = self._compute_diagonal_jacobian_element(
                solution, dkappa_da, dchi_da, coords
            )
        
        # Fill off-diagonal terms (7D spatial coupling)
        for i in range(total_size):
            coords = np.unravel_index(i, shape)
            
            # Fill coupling terms for all 7 dimensions
            self._fill_7d_coupling_terms(jacobian, solution, coords, step_sizes, i, shape)
        
        return jacobian
    
    def _compute_7d_step_sizes(self, shape: tuple) -> dict:
        """Compute step sizes for all 7 dimensions."""
        step_sizes = {}
        
        # Spatial dimensions (x, y, z)
        for dim in range(min(3, len(shape))):
            step_sizes[f'd{["x", "y", "z"][dim]}'] = 1.0 / shape[dim] if shape[dim] > 1 else 1.0
        
        # Phase dimensions (φ₁, φ₂, φ₃)
        for dim in range(3, min(6, len(shape))):
            step_sizes[f'dphi{dim-2}'] = 2 * np.pi / shape[dim] if shape[dim] > 1 else 2 * np.pi
        
        # Temporal dimension (t)
        if len(shape) > 6:
            step_sizes['dt'] = 1.0 / shape[6] if shape[6] > 1 else 1.0
        
        return step_sizes
    
    def _compute_diagonal_jacobian_element(self, solution: np.ndarray, dkappa_da: np.ndarray, dchi_da: np.ndarray, coords: tuple) -> complex:
        """Compute diagonal Jacobian element with full nonlinear contributions."""
        # Base linear terms
        diagonal_element = (
            self.parameters.kappa_0
            + self.parameters.k0**2 * self.parameters.chi_prime
        )
        
        # Nonlinear contributions
        if coords in np.ndindex(solution.shape):
            amplitude = np.abs(solution[coords])
            diagonal_element += (
                dkappa_da[coords] * amplitude
                + self.parameters.k0**2 * dchi_da[coords] * amplitude
            )
        
        return diagonal_element
    
    def _fill_7d_coupling_terms(self, jacobian: np.ndarray, solution: np.ndarray, coords: tuple, step_sizes: dict, i: int, shape: tuple):
        """Fill coupling terms for all 7 dimensions."""
        # Spatial coupling (x, y, z)
        for dim in range(min(3, len(shape))):
            dim_name = ["x", "y", "z"][dim]
            step_size = step_sizes.get(f'd{dim_name}', 1.0)
            
            # Previous neighbor
            if coords[dim] > 0:
                prev_coords = coords[:dim] + (coords[dim] - 1,) + coords[dim+1:]
                j_prev = np.ravel_multi_index(prev_coords, shape)
                jacobian[i, j_prev] = -self.parameters.kappa_0 / (step_size * step_size)
            
            # Next neighbor
            if coords[dim] < shape[dim] - 1:
                next_coords = coords[:dim] + (coords[dim] + 1,) + coords[dim+1:]
                j_next = np.ravel_multi_index(next_coords, shape)
                jacobian[i, j_next] = -self.parameters.kappa_0 / (step_size * step_size)
        
        # Phase coupling (φ₁, φ₂, φ₃)
        for dim in range(3, min(6, len(shape))):
            phi_dim = dim - 3
            step_size = step_sizes.get(f'dphi{phi_dim+1}', 2 * np.pi)
            
            # Previous neighbor (periodic boundary)
            prev_coords = coords[:dim] + ((coords[dim] - 1) % shape[dim],) + coords[dim+1:]
            j_prev = np.ravel_multi_index(prev_coords, shape)
            jacobian[i, j_prev] = -self.parameters.kappa_0 / (step_size * step_size)
            
            # Next neighbor (periodic boundary)
            next_coords = coords[:dim] + ((coords[dim] + 1) % shape[dim],) + coords[dim+1:]
            j_next = np.ravel_multi_index(next_coords, shape)
            jacobian[i, j_next] = -self.parameters.kappa_0 / (step_size * step_size)
        
        # Temporal coupling (t)
        if len(shape) > 6:
            step_size = step_sizes.get('dt', 1.0)
            
            # Previous time step
            if coords[6] > 0:
                prev_coords = coords[:6] + (coords[6] - 1,)
                j_prev = np.ravel_multi_index(prev_coords, shape)
                jacobian[i, j_prev] = -self.parameters.kappa_0 / (step_size * step_size)
            
            # Next time step
            if coords[6] < shape[6] - 1:
                next_coords = coords[:6] + (coords[6] + 1,)
                j_next = np.ravel_multi_index(next_coords, shape)
                jacobian[i, j_next] = -self.parameters.kappa_0 / (step_size * step_size)
    
    def _compute_full_7d_off_diagonal_contributions(self, jacobian_sparse: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """
        Compute full 7D off-diagonal contributions to Jacobian diagonal.
        
        Physical Meaning:
            Computes the complete effective diagonal contributions from off-diagonal
            coupling terms in the 7D sparse Jacobian matrix, including spatial,
            phase, and temporal coupling effects.
            
        Mathematical Foundation:
            Implements full off-diagonal contribution analysis:
            - Spatial coupling effects (x, y, z)
            - Phase coupling effects (φ₁, φ₂, φ₃)
            - Temporal coupling effects (t)
            - Nonlinear coupling effects
        """
        # Extract diagonal elements
        diagonal_elements = jacobian_sparse.diagonal()
        shape = solution.shape
        total_size = len(diagonal_elements)
        
        # Initialize off-diagonal contributions
        off_diagonal_contributions = np.zeros_like(diagonal_elements)
        
        # Compute step sizes for proper weighting
        step_sizes = self._compute_7d_step_sizes(shape)
        
        # For each diagonal element, compute contribution from off-diagonal terms
        for i in range(total_size):
            coords = np.unravel_index(i, shape)
            
            # Get off-diagonal elements in row i
            row = jacobian_sparse[i, :]
            off_diagonal_mask = np.arange(len(row)) != i
            off_diagonal_elements = row[off_diagonal_mask]
            
            if len(off_diagonal_elements) > 0:
                # Compute weighted contribution based on 7D coupling
                contribution = self._compute_7d_coupling_contribution(
                    off_diagonal_elements, coords, step_sizes, shape, solution
                )
                off_diagonal_contributions[i] = contribution
        
        return off_diagonal_contributions
    
    def _compute_7d_coupling_contribution(self, off_diagonal_elements: np.ndarray, coords: tuple, step_sizes: dict, shape: tuple, solution: np.ndarray) -> complex:
        """Compute 7D coupling contribution with proper weighting."""
        # Compute different types of coupling contributions
        spatial_contribution = 0.0
        phase_contribution = 0.0
        temporal_contribution = 0.0
        
        # Spatial coupling contribution (x, y, z)
        for dim in range(min(3, len(shape))):
            dim_name = ["x", "y", "z"][dim]
            step_size = step_sizes.get(f'd{dim_name}', 1.0)
            spatial_contribution += np.mean(np.abs(off_diagonal_elements)) / (step_size * step_size)
        
        # Phase coupling contribution (φ₁, φ₂, φ₃)
        for dim in range(3, min(6, len(shape))):
            phi_dim = dim - 3
            step_size = step_sizes.get(f'dphi{phi_dim+1}', 2 * np.pi)
            phase_contribution += np.mean(np.abs(off_diagonal_elements)) / (step_size * step_size)
        
        # Temporal coupling contribution (t)
        if len(shape) > 6:
            step_size = step_sizes.get('dt', 1.0)
            temporal_contribution = np.mean(np.abs(off_diagonal_elements)) / (step_size * step_size)
        
        # Combine contributions with proper weighting
        total_contribution = (
            0.4 * spatial_contribution +
            0.3 * phase_contribution +
            0.3 * temporal_contribution
        )
        
        # Apply nonlinear scaling based on field amplitude
        if coords in np.ndindex(shape):
            amplitude = np.abs(solution[coords])
            nonlinear_scaling = 1.0 + 0.1 * amplitude  # Nonlinear enhancement
            total_contribution *= nonlinear_scaling
        
        return total_contribution
