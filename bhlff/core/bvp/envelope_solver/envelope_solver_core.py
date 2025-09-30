"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core envelope solver facade for BVP envelope equation.

This module provides the main facade class for the envelope solver,
coordinating all components for solving the 7D BVP envelope equation.

Physical Meaning:
    Provides the main interface for solving the nonlinear 7D envelope
    equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t) using advanced
    numerical methods in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Coordinates Newton-Raphson method with line search and regularization
    for robust solution of nonlinear 7D envelope equations.

Example:
    >>> core = EnvelopeSolverCore(domain, config)
    >>> residual = core.compute_residual(envelope, source)
    >>> jacobian = core.compute_jacobian(envelope)
"""

import numpy as np
from typing import Dict, Any

from ...domain import Domain
from ..bvp_constants import BVPConstants
from .residual_computer import ResidualComputer
from .jacobian_computer import JacobianComputer
from .newton_solver import NewtonSolver
from .gradient_computer import GradientComputer


class EnvelopeSolverCore:
    """
    Core facade for 7D BVP envelope equation solver.

    Physical Meaning:
        Coordinates all components for solving the nonlinear 7D envelope
        equation using advanced numerical methods in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

    Mathematical Foundation:
        Provides unified interface for Newton-Raphson method with line search
        and regularization for robust solution of nonlinear 7D envelope equations.

    Attributes:
        domain (Domain): 7D computational domain.
        constants (BVPConstants): BVP constants instance.
        residual_computer (ResidualComputer): Residual computation component.
        jacobian_computer (JacobianComputer): Jacobian computation component.
        newton_solver (NewtonSolver): Newton-Raphson solver component.
        gradient_computer (GradientComputer): Gradient computation component.
    """

    def __init__(
        self, domain: Domain, config: Dict[str, Any], constants: BVPConstants = None
    ) -> None:
        """
        Initialize envelope solver core.

        Physical Meaning:
            Sets up the core mathematical operations with parameters
            for the nonlinear envelope equation.

        Args:
            domain (Domain): Computational domain.
            config (Dict[str, Any]): Configuration parameters.
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.domain = domain
        self.constants = constants or BVPConstants(config)
        
        # Initialize components
        self.residual_computer = ResidualComputer(domain, self.constants)
        self.jacobian_computer = JacobianComputer(domain, self.constants)
        self.newton_solver = NewtonSolver(domain, self.constants)
        self.gradient_computer = GradientComputer(domain, self.constants)

    def compute_residual(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the 7D envelope equation.

        Physical Meaning:
            Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            for the Newton-Raphson method in 7D space-time.

        Args:
            envelope (np.ndarray): Current envelope estimate in 7D space-time.
            source (np.ndarray): Source term s(x,φ,t) in 7D space-time.

        Returns:
            np.ndarray: Residual r = L(a) - s in 7D space-time.
        """
        return self.residual_computer.compute_residual(envelope, source)

    def compute_jacobian(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton-Raphson method.

        Physical Meaning:
            Computes the Jacobian matrix J = ∂r/∂a of the residual
            with respect to the envelope field.

        Args:
            envelope (np.ndarray): Current envelope estimate.

        Returns:
            np.ndarray: Jacobian matrix J.
        """
        return self.jacobian_computer.compute_jacobian(envelope)

    def solve_newton_system(
        self, jacobian: np.ndarray, residual: np.ndarray
    ) -> np.ndarray:
        """
        Solve Newton system J * δa = -r.

        Physical Meaning:
            Solves the linear system for the Newton update step
            using advanced numerical methods.

        Args:
            jacobian (np.ndarray): Jacobian matrix J.
            residual (np.ndarray): Residual vector r.

        Returns:
            np.ndarray: Newton update step δa.
        """
        return self.newton_solver.solve_newton_system(jacobian, residual)

    def compute_gradient(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute gradient for fallback gradient descent.

        Physical Meaning:
            Computes the gradient of the residual norm for use
            in gradient descent when Newton method fails.

        Args:
            envelope (np.ndarray): Current envelope estimate.
            source (np.ndarray): Source term.

        Returns:
            np.ndarray: Gradient vector.
        """
        return self.gradient_computer.compute_gradient(envelope, source)

    def __repr__(self) -> str:
        """String representation of envelope solver core."""
        return (
            f"EnvelopeSolverCore(domain={self.domain}, "
            f"constants={self.constants})"
        )
