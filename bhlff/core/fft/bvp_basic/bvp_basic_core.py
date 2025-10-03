"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core basic BVP solver for 7D envelope equation.

This module implements the core basic BVP solving functionality
for the 7D envelope equation.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.domain_7d_bvp import Domain7DBVP
    from ..domain.parameters_7d_bvp import Parameters7DBVP
    from .spectral_derivatives import SpectralDerivatives
    from ...bvp.abstract_solver_core import AbstractSolverCore
else:
    from ...bvp.abstract_solver_core import AbstractSolverCore

from .bvp_residual import BVPResidual
from .bvp_jacobian import BVPJacobian
from .bvp_linear_solver import BVPLinearSolver


class BVBBasicCore(AbstractSolverCore):
    """
    Core basic BVP solver functionality.

    Physical Meaning:
        Implements core basic mathematical operations for solving the 7D BVP
        envelope equation, coordinating specialized modules for different aspects
        of basic solving.

    Mathematical Foundation:
        Handles the nonlinear terms in the BVP equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    """

    def __init__(self, domain: 'Domain7DBVP', parameters: 'Parameters7DBVP', derivatives: 'SpectralDerivatives'):
        """
        Initialize BVP solver core.

        Physical Meaning:
            Sets up the basic BVP solver with all necessary components
            for core solving capabilities.

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
        
        # Basic solver parameters
        self.max_iterations = parameters.get('max_iterations', 100)
        self.tolerance = parameters.get('tolerance', 1e-6)
        
        # Initialize specialized modules
        self.residual = BVPResidual(domain, parameters, derivatives)
        self.jacobian = BVPJacobian(domain, parameters, derivatives)
        self.linear_solver = BVPLinearSolver(domain, parameters, derivatives)

    def compute_residual(self, solution: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the BVP equation.

        Physical Meaning:
            Computes the residual of the BVP envelope equation
            to measure how well the current solution satisfies the equation.

        Mathematical Foundation:
            Residual = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            where κ(|a|) is nonlinear stiffness and χ(|a|) is effective susceptibility.

        Args:
            solution (np.ndarray): Current solution field.
            source (np.ndarray): Source term.

        Returns:
            np.ndarray: Residual field.

        Raises:
            ValueError: If solution or source have incompatible shapes.
        """
        if solution.shape != self.domain.shape or source.shape != self.domain.shape:
            raise ValueError("Solution and source must have compatible shapes with domain")

        return self.residual.compute_residual(solution, source)

    def compute_jacobian(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix of the BVP equation.

        Physical Meaning:
            Computes the Jacobian matrix for the BVP envelope equation
            to enable Newton-Raphson iteration.

        Mathematical Foundation:
            Jacobian = ∂F/∂a where F is the BVP equation residual.

        Args:
            solution (np.ndarray): Current solution field.

        Returns:
            np.ndarray: Jacobian matrix.

        Raises:
            ValueError: If solution has incompatible shape.
        """
        if solution.shape != self.domain.shape:
            raise ValueError("Solution must have compatible shape with domain")

        return self.jacobian.compute_jacobian(solution)

    def solve_linear_system(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """
        Solve linear system for Newton-Raphson update.

        Physical Meaning:
            Solves the linear system J·δa = -r for the Newton-Raphson update,
            where J is the Jacobian and r is the residual.

        Mathematical Foundation:
            Solves J·δa = -r using appropriate linear algebra methods.

        Args:
            jacobian (np.ndarray): Jacobian matrix.
            residual (np.ndarray): Residual vector.

        Returns:
            np.ndarray: Solution update vector.

        Raises:
            ValueError: If jacobian and residual have incompatible shapes.
        """
        if jacobian.shape[0] != residual.size:
            raise ValueError("Jacobian and residual must have compatible shapes")

        return self.linear_solver.solve_linear_system(jacobian, residual)

    def validate_solution(self, solution: np.ndarray, source: np.ndarray) -> Dict[str, Any]:
        """
        Validate solution quality.

        Physical Meaning:
            Validates the quality of the solution by checking
            residual norms and other quality metrics.

        Args:
            solution (np.ndarray): Solution field to validate.
            source (np.ndarray): Original source term.

        Returns:
            Dict[str, Any]: Validation results.

        Raises:
            ValueError: If solution or source have incompatible shapes.
        """
        if solution.shape != self.domain.shape or source.shape != self.domain.shape:
            raise ValueError("Solution and source must have compatible shapes with domain")

        # Compute residual
        residual = self.compute_residual(solution, source)
        residual_norm = np.linalg.norm(residual)
        solution_norm = np.linalg.norm(solution)
        
        return {
            'residual_norm': float(residual_norm),
            'solution_norm': float(solution_norm),
            'relative_residual': float(residual_norm / solution_norm) if solution_norm > 0 else 0.0,
            'validation_passed': residual_norm < self.tolerance
        }
