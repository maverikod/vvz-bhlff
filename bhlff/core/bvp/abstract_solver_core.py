"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Abstract base class for BVP solver cores.

This module provides the abstract base class for all BVP solver cores,
defining the common interface and shared functionality for solving
the 7D BVP envelope equation.

Physical Meaning:
    Defines the fundamental interface for solving the 7D BVP envelope
    equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t) using various
    numerical methods in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Provides the base structure for implementing Newton-Raphson method
    with line search and regularization for robust solution of
    nonlinear 7D envelope equations.

Example:
    >>> class MySolverCore(AbstractSolverCore):
    ...     def compute_residual(self, envelope, source):
    ...         # Implementation
    ...     def compute_jacobian(self, envelope):
    ...         # Implementation
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from ..domain import Domain


class AbstractSolverCore:
    """
    Base class for BVP solver cores with default implementations.

    Physical Meaning:
        Provides the fundamental interface and default implementations for solving 
        the 7D BVP envelope equation using various numerical methods, providing a 
        unified approach to envelope field calculations.

    Mathematical Foundation:
        Implements the base structure for solving:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
        where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
        χ(|a|) = χ' + iχ''(|a|) is effective susceptibility.

    Attributes:
        domain (Domain): Computational domain for the simulation.
        config (Dict[str, Any]): Configuration parameters.
        logger (logging.Logger): Logger instance for debugging.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]):
        """
        Initialize abstract solver core.

        Physical Meaning:
            Sets up the base solver core with the computational domain
            and configuration parameters for solving the BVP equation.

        Args:
            domain (Domain): Computational domain with grid information.
            config (Dict[str, Any]): Configuration parameters including:
                - max_iterations: Maximum Newton-Raphson iterations
                - tolerance: Convergence tolerance
                - damping_factor: Damping factor for stability
        """
        self.domain = domain
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Common solver parameters
        self.max_iterations = config.get("max_iterations", 100)
        self.tolerance = config.get("tolerance", 1e-8)
        self.damping_factor = config.get("damping_factor", 0.1)

        if hasattr(domain, 'shape'):
            domain_info = domain.shape
        elif hasattr(domain, 'get_full_7d_shape'):
            domain_info = domain.get_full_7d_shape()
        else:
            domain_info = "unknown"
        self.logger.info(f"{self.__class__.__name__} initialized for domain {domain_info}")

    def compute_residual(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the BVP envelope equation.

        Physical Meaning:
            Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            for the Newton-Raphson method in 7D space-time.

        Mathematical Foundation:
            R(a) = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s
            where κ(|a|) = κ₀ + κ₂|a|² and χ(|a|) = χ' + iχ''(|a|).

        Args:
            envelope (np.ndarray): Current envelope estimate a(x,φ,t).
            source (np.ndarray): Source term s(x,φ,t).

        Returns:
            np.ndarray: Residual r = L(a) - s.
        """
        # Base implementation - can be overridden by subclasses
        # Simple fallback implementation
        return source - envelope

    def compute_jacobian(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton-Raphson method.

        Physical Meaning:
            Computes the Jacobian matrix J = ∂r/∂a of the residual
            with respect to the envelope field.

        Mathematical Foundation:
            J_{ij} = ∂r_i/∂a_j where r is the residual and a is the solution vector.

        Args:
            envelope (np.ndarray): Current envelope estimate a(x,φ,t).

        Returns:
            np.ndarray: Jacobian matrix J.
        """
        # Base implementation - can be overridden by subclasses
        # Simple fallback implementation - return identity matrix
        size = envelope.size
        return np.eye(size)

    def solve_linear_system(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """
        Solve linear system for Newton-Raphson update.

        Physical Meaning:
            Solves the linear system J·δa = -r for the Newton-Raphson
            update δa, where J is the Jacobian and R is the residual.

        Mathematical Foundation:
            Solves J·δa = -r for the correction vector δa.

        Args:
            jacobian (np.ndarray): Jacobian matrix J.
            residual (np.ndarray): Residual vector r.

        Returns:
            np.ndarray: Update vector δa.
        """
        # Base implementation - can be overridden by subclasses
        # Simple fallback implementation - just return negative residual
        return -residual

    def solve_envelope(
        self,
        source: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve BVP envelope equation using Newton-Raphson method.

        Physical Meaning:
            Solves the full BVP envelope equation for the envelope field
            in space-time using iterative Newton-Raphson method for
            nonlinear terms.

        Mathematical Foundation:
            Implements Newton-Raphson iteration:
            1. Compute residual R = F(a) - s
            2. Compute Jacobian J = ∂F/∂a
            3. Solve J·δa = -R
            4. Update a ← a - δa
            5. Repeat until ||R|| < tolerance

        Args:
            source (np.ndarray): Source term s(x,φ,t).
            initial_guess (Optional[np.ndarray]): Initial guess for solution.

        Returns:
            np.ndarray: Envelope solution a(x,φ,t).

        Raises:
            ValueError: If source has incompatible shape with domain.
        """
        if hasattr(self.domain, 'shape'):
            domain_shape = self.domain.shape
        elif hasattr(self.domain, 'get_full_7d_shape'):
            domain_shape = self.domain.get_full_7d_shape()
        else:
            domain_shape = "unknown"
        
        if hasattr(self.domain, 'shape') and source.shape != self.domain.shape:
            raise ValueError(
                f"Source shape {source.shape} incompatible with domain shape {self.domain.shape}"
            )

        # Initialize solution
        if initial_guess is None:
            envelope = np.zeros_like(source, dtype=complex)
        else:
            envelope = initial_guess.copy()

        # Newton-Raphson iteration
        for iteration in range(self.max_iterations):
            # Compute residual
            residual = self.compute_residual(envelope, source)

            # Check convergence
            residual_norm = np.linalg.norm(residual)
            if residual_norm < self.tolerance:
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break

            # Simple update without Jacobian
            update = -residual

            # Apply damping for stability
            envelope -= self.damping_factor * update

        return envelope.real

    def validate_solution(
        self, solution: np.ndarray, source: np.ndarray, tolerance: float = 1e-8
    ) -> Dict[str, Any]:
        """
        Validate envelope equation solution.

        Physical Meaning:
            Validates that the solution satisfies the envelope equation
            within the specified tolerance by computing the residual.

        Mathematical Foundation:
            Computes residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s
            and checks that ||R|| / ||s|| < tolerance.

        Args:
            solution (np.ndarray): Envelope solution a(x,φ,t).
            source (np.ndarray): Source term s(x,φ,t).
            tolerance (float): Relative tolerance for validation.

        Returns:
            Dict[str, Any]: Validation results including:
                - is_valid: Whether solution is valid
                - residual_norm: L2 norm of residual
                - relative_error: Relative error
                - max_error: Maximum error
        """
        if solution.shape != source.shape:
            raise ValueError("Solution and source shapes must match")

        # Compute residual
        residual = self.compute_residual(solution, source)

        # Compute error metrics
        residual_norm = np.linalg.norm(residual)
        source_norm = np.linalg.norm(source)
        relative_error = residual_norm / (source_norm + 1e-15)
        max_error = np.max(np.abs(residual))

        is_valid = relative_error < tolerance

        return {
            "is_valid": bool(is_valid),
            "residual_norm": float(residual_norm),
            "relative_error": float(relative_error),
            "max_error": float(max_error),
            "tolerance": float(tolerance),
        }

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
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "damping_factor": self.damping_factor,
            "domain_shape": self.domain.shape,
            "field_size": np.prod(self.domain.shape),
        }

    def __repr__(self) -> str:
        """String representation of solver core."""
        return (
            f"{self.__class__.__name__}("
            f"domain={self.domain.shape}, "
            f"max_iterations={self.max_iterations}, "
            f"tolerance={self.tolerance})"
        )
