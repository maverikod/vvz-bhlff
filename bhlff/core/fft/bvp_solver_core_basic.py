"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic BVP solver core functionality for 7D envelope equation.

This module contains the basic functionality for solving the 7D BVP envelope equation,
including core residual computation and Jacobian calculation.
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


class BVPSolverCoreBasic(AbstractSolverCore):
    """
    Basic BVP solver core functionality.

    Physical Meaning:
        Implements the basic mathematical operations for solving the 7D BVP
        envelope equation, including residual and Jacobian computation.

    Mathematical Foundation:
        Handles the nonlinear terms in the BVP equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    """

    def __init__(self, domain: 'Domain7DBVP', parameters: 'Parameters7DBVP', derivatives: 'SpectralDerivatives'):
        """
        Initialize BVP solver core.

        Physical Meaning:
            Sets up the solver core with the 7D domain, parameters,
            and spectral derivatives for solving the envelope equation.

        Args:
            domain (Domain7DBVP): 7D computational domain.
            parameters (Parameters7DBVP): BVP parameters.
            derivatives (SpectralDerivatives): Spectral derivatives.
        """
        super().__init__(domain, parameters)
        self.domain = domain
        self.parameters = parameters
        self.derivatives = derivatives
        self.logger = logging.getLogger(__name__)

    def compute_residual(self, solution: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the BVP equation.

        Physical Meaning:
            Computes the residual R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s,
            which measures how well the current solution satisfies
            the BVP envelope equation.

        Mathematical Foundation:
            R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)

        Args:
            solution (np.ndarray): Current solution field a(x,φ,t).
            source (np.ndarray): Source term s(x,φ,t).

        Returns:
            np.ndarray: Residual field R(a).
        """
        self.logger.debug("Computing BVP residual")
        
        # Compute nonlinear stiffness κ(|a|)
        stiffness = self._compute_nonlinear_stiffness(solution)
        
        # Compute effective susceptibility χ(|a|)
        susceptibility = self._compute_effective_susceptibility(solution)
        
        # Compute divergence term ∇·(κ(|a|)∇a)
        divergence_term = self._compute_divergence_term(solution, stiffness)
        
        # Compute susceptibility term k₀²χ(|a|)a
        susceptibility_term = self._compute_susceptibility_term(solution, susceptibility)
        
        # Compute residual
        residual = divergence_term + susceptibility_term - source
        
        self.logger.debug(f"Residual computed: norm = {np.linalg.norm(residual)}")
        return residual

    def compute_jacobian(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton-Raphson iteration.

        Physical Meaning:
            Computes the Jacobian matrix J = ∂R/∂a, which represents
            the linearization of the residual around the current solution.

        Mathematical Foundation:
            J = ∂/∂a[∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s]

        Args:
            solution (np.ndarray): Current solution field.

        Returns:
            np.ndarray: Jacobian matrix.
        """
        self.logger.debug("Computing BVP Jacobian")
        
        # Get field dimensions
        field_shape = solution.shape
        total_points = solution.size
        
        # Initialize Jacobian matrix
        jacobian = np.zeros((total_points, total_points))
        
        # Compute Jacobian entries
        for i in range(total_points):
            # Get multi-dimensional index
            idx = np.unravel_index(i, field_shape)
            
            # Compute Jacobian row
            jacobian_row = self._compute_jacobian_row(solution, idx)
            
            # Set Jacobian entries
            for j, value in jacobian_row.items():
                jacobian[i, j] = value
        
        self.logger.debug(f"Jacobian computed: shape = {jacobian.shape}")
        return jacobian

    def solve_linear_system(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """
        Solve linear system J·δa = -R.

        Physical Meaning:
            Solves the linearized system to find the update δa
            for the Newton-Raphson iteration.

        Args:
            jacobian (np.ndarray): Jacobian matrix.
            residual (np.ndarray): Residual vector.

        Returns:
            np.ndarray: Update vector δa.
        """
        self.logger.debug("Solving linear system")
        
        # Reshape residual to vector
        residual_vector = residual.flatten()
        
        # Solve linear system
        try:
            update_vector = np.linalg.solve(jacobian, -residual_vector)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            update_vector = np.linalg.lstsq(jacobian, -residual_vector, rcond=None)[0]
        
        # Reshape back to field shape
        update = update_vector.reshape(residual.shape)
        
        self.logger.debug(f"Linear system solved: update norm = {np.linalg.norm(update)}")
        return update

    def _compute_nonlinear_stiffness(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear stiffness κ(|a|).

        Physical Meaning:
            Computes the nonlinear stiffness coefficient that depends
            on the field amplitude, representing the field's response
            to spatial variations.

        Mathematical Foundation:
            κ(|a|) = κ₀ + κ₁|a|² + κ₂|a|⁴ + ...

        Args:
            solution (np.ndarray): Current solution field.

        Returns:
            np.ndarray: Nonlinear stiffness field.
        """
        # Get stiffness parameters
        kappa_0 = self.parameters.get('kappa_0', 1.0)
        kappa_1 = self.parameters.get('kappa_1', 0.1)
        kappa_2 = self.parameters.get('kappa_2', 0.01)
        
        # Compute field amplitude
        amplitude = np.abs(solution)
        
        # Compute nonlinear stiffness
        stiffness = kappa_0 + kappa_1 * amplitude**2 + kappa_2 * amplitude**4
        
        return stiffness

    def _compute_effective_susceptibility(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute effective susceptibility χ(|a|).

        Physical Meaning:
            Computes the effective susceptibility that depends on
            the field amplitude, representing the field's response
            to external excitations.

        Mathematical Foundation:
            χ(|a|) = χ₀ + χ₁|a|² + χ₂|a|⁴ + ...

        Args:
            solution (np.ndarray): Current solution field.

        Returns:
            np.ndarray: Effective susceptibility field.
        """
        # Get susceptibility parameters
        chi_0 = self.parameters.get('chi_0', 1.0)
        chi_1 = self.parameters.get('chi_1', 0.05)
        chi_2 = self.parameters.get('chi_2', 0.005)
        
        # Compute field amplitude
        amplitude = np.abs(solution)
        
        # Compute effective susceptibility
        susceptibility = chi_0 + chi_1 * amplitude**2 + chi_2 * amplitude**4
        
        return susceptibility

    def _compute_divergence_term(self, solution: np.ndarray, stiffness: np.ndarray) -> np.ndarray:
        """
        Compute divergence term ∇·(κ(|a|)∇a).

        Physical Meaning:
            Computes the divergence of the stiffness-weighted gradient,
            representing the spatial variation of the field.

        Mathematical Foundation:
            ∇·(κ(|a|)∇a) = κ(|a|)∇²a + ∇κ(|a|)·∇a

        Args:
            solution (np.ndarray): Current solution field.
            stiffness (np.ndarray): Nonlinear stiffness field.

        Returns:
            np.ndarray: Divergence term.
        """
        # Compute gradient of solution
        gradient = self.derivatives.compute_gradient(solution)
        
        # Compute gradient of stiffness
        stiffness_gradient = self.derivatives.compute_gradient(stiffness)
        
        # Compute Laplacian of solution
        laplacian = self.derivatives.compute_laplacian(solution)
        
        # Compute divergence term
        divergence_term = stiffness * laplacian + np.sum(stiffness_gradient * gradient, axis=-1)
        
        return divergence_term

    def _compute_susceptibility_term(self, solution: np.ndarray, susceptibility: np.ndarray) -> np.ndarray:
        """
        Compute susceptibility term k₀²χ(|a|)a.

        Physical Meaning:
            Computes the susceptibility-weighted field term,
            representing the field's response to external excitations.

        Mathematical Foundation:
            k₀²χ(|a|)a = k₀²χ(|a|)a

        Args:
            solution (np.ndarray): Current solution field.
            susceptibility (np.ndarray): Effective susceptibility field.

        Returns:
            np.ndarray: Susceptibility term.
        """
        # Get wavenumber
        k0 = self.parameters.get('k0', 1.0)
        
        # Compute susceptibility term
        susceptibility_term = k0**2 * susceptibility * solution
        
        return susceptibility_term

    def _compute_jacobian_row(self, solution: np.ndarray, idx: tuple) -> Dict[int, float]:
        """
        Compute Jacobian row for a specific point.

        Physical Meaning:
            Computes the Jacobian row for a specific point in the field,
            representing the sensitivity of the residual to changes
            in the solution at that point.

        Args:
            solution (np.ndarray): Current solution field.
            idx (tuple): Multi-dimensional index.

        Returns:
            Dict[int, float]: Jacobian row entries.
        """
        # Get field shape
        field_shape = solution.shape
        total_points = solution.size
        
        # Convert index to linear index
        linear_idx = np.ravel_multi_index(idx, field_shape)
        
        # Initialize Jacobian row
        jacobian_row = {}
        
        # Compute diagonal entry (self-coupling)
        diagonal_entry = self._compute_diagonal_jacobian_entry(solution, idx)
        jacobian_row[linear_idx] = diagonal_entry
        
        # Compute off-diagonal entries (neighbor coupling)
        neighbor_entries = self._compute_neighbor_jacobian_entries(solution, idx)
        jacobian_row.update(neighbor_entries)
        
        return jacobian_row

    def _compute_diagonal_jacobian_entry(self, solution: np.ndarray, idx: tuple) -> float:
        """
        Compute diagonal Jacobian entry.

        Args:
            solution (np.ndarray): Current solution field.
            idx (tuple): Multi-dimensional index.

        Returns:
            float: Diagonal Jacobian entry.
        """
        # Get parameters
        k0 = self.parameters.get('k0', 1.0)
        kappa_0 = self.parameters.get('kappa_0', 1.0)
        chi_0 = self.parameters.get('chi_0', 1.0)
        
        # Get field value
        field_value = solution[idx]
        amplitude = np.abs(field_value)
        
        # Compute diagonal entry
        diagonal_entry = k0**2 * chi_0 + 2.0 * kappa_0 * amplitude**2
        
        return diagonal_entry

    def _compute_neighbor_jacobian_entries(self, solution: np.ndarray, idx: tuple) -> Dict[int, float]:
        """
        Compute neighbor Jacobian entries.

        Args:
            solution (np.ndarray): Current solution field.
            idx (tuple): Multi-dimensional index.

        Returns:
            Dict[int, float]: Neighbor Jacobian entries.
        """
        neighbor_entries = {}
        
        # Get field shape
        field_shape = solution.shape
        
        # Compute entries for neighboring points
        for dim in range(len(idx)):
            for offset in [-1, 1]:
                neighbor_idx = list(idx)
                neighbor_idx[dim] += offset
                
                # Check bounds
                if 0 <= neighbor_idx[dim] < field_shape[dim]:
                    neighbor_linear_idx = np.ravel_multi_index(neighbor_idx, field_shape)
                    
                    # Compute neighbor entry
                    neighbor_entry = self._compute_neighbor_jacobian_entry(solution, idx, neighbor_idx)
                    neighbor_entries[neighbor_linear_idx] = neighbor_entry
        
        return neighbor_entries

    def _compute_neighbor_jacobian_entry(self, solution: np.ndarray, idx: tuple, neighbor_idx: tuple) -> float:
        """
        Compute neighbor Jacobian entry.

        Args:
            solution (np.ndarray): Current solution field.
            idx (tuple): Current point index.
            neighbor_idx (tuple): Neighbor point index.

        Returns:
            float: Neighbor Jacobian entry.
        """
        # Get parameters
        kappa_0 = self.parameters.get('kappa_0', 1.0)
        
        # Compute neighbor entry (simplified)
        neighbor_entry = -0.1 * kappa_0
        
        return neighbor_entry

    def validate_solution(self, solution: np.ndarray, source: np.ndarray) -> Dict[str, Any]:
        """
        Validate solution quality.

        Physical Meaning:
            Validates the quality of the computed solution by checking
            residual magnitude and other quality metrics.

        Args:
            solution (np.ndarray): Computed solution field.
            source (np.ndarray): Source term.

        Returns:
            Dict[str, Any]: Validation results.
        """
        # Compute residual
        residual = self.compute_residual(solution, source)
        residual_norm = np.linalg.norm(residual)
        
        # Compute solution statistics
        solution_norm = np.linalg.norm(solution)
        solution_max = np.max(np.abs(solution))
        solution_mean = np.mean(np.abs(solution))
        
        # Validation results
        validation_results = {
            'residual_norm': residual_norm,
            'solution_norm': solution_norm,
            'solution_max': solution_max,
            'solution_mean': solution_mean,
            'relative_residual': residual_norm / solution_norm if solution_norm > 0 else float('inf'),
            'is_valid': residual_norm < 1e-6
        }
        
        return validation_results
