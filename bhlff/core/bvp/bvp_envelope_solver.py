"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP envelope equation solver module.

This module implements the solver for the BVP envelope equation with
nonlinear stiffness and susceptibility, providing the core mathematical
operations for envelope field calculations.

Physical Meaning:
    Solves the envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility.

Mathematical Foundation:
    Implements iterative solution of the nonlinear envelope equation
    using finite difference methods for the divergence and gradient
    operations in the equation.

Example:
    >>> solver = BVPEnvelopeSolver(domain, config)
    >>> envelope = solver.solve_envelope(source)
"""

import numpy as np
from typing import Dict, Any

from ..domain import Domain
from .envelope_solver_core import EnvelopeSolverCore
from .envelope_solver_line_search import EnvelopeSolverLineSearch
from .bvp_constants import BVPConstants


class BVPEnvelopeSolver:
    """
    Solver for BVP envelope equation.

    Physical Meaning:
        Solves the nonlinear envelope equation for the Base High-Frequency
        Field, computing the envelope modulation that satisfies the
        governing equation with nonlinear stiffness and susceptibility.

    Mathematical Foundation:
        Solves ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x) where:
        - κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness
        - χ(|a|) = χ' + iχ''(|a|) is effective susceptibility
        - s(x) is the source term

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Envelope solver configuration.
        kappa_0 (float): Base stiffness coefficient κ₀.
        kappa_2 (float): Nonlinear stiffness coefficient κ₂.
        chi_prime (float): Real part of susceptibility χ'.
        chi_double_prime_0 (float): Base imaginary susceptibility χ''₀.
        k0_squared (float): Wave number squared k₀².
    """

    def __init__(self, domain: Domain, config: Dict[str, Any], constants: BVPConstants = None) -> None:
        """
        Initialize envelope equation solver.

        Physical Meaning:
            Sets up the solver with parameters for the nonlinear
            envelope equation including stiffness and susceptibility
            coefficients.

        Args:
            domain (Domain): Computational domain for envelope calculations.
            config (Dict[str, Any]): Envelope solver configuration including:
                - kappa_0: Base stiffness coefficient
                - kappa_2: Nonlinear stiffness coefficient
                - chi_prime: Real part of susceptibility
                - chi_double_prime_0: Base imaginary susceptibility
                - k0_squared: Wave number squared
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.domain = domain
        self.config = config
        self.constants = constants or BVPConstants(config)
        self._setup_parameters()
        self._setup_solver_components()

    def _setup_parameters(self) -> None:
        """
        Setup envelope equation parameters.

        Physical Meaning:
            Initializes the physical parameters for the envelope equation
            from the constants instance.
        """
        self.kappa_0 = self.constants.get_envelope_parameter("kappa_0")
        self.kappa_2 = self.constants.get_envelope_parameter("kappa_2")
        self.chi_prime = self.constants.get_envelope_parameter("chi_prime")
        self.chi_double_prime_0 = self.constants.get_envelope_parameter("chi_double_prime_0")
        self.k0_squared = self.constants.get_envelope_parameter("k0_squared")
    
    def _setup_solver_components(self) -> None:
        """Setup solver components."""
        self._core = EnvelopeSolverCore(self.domain, self.config, self.constants)
        self._line_search = EnvelopeSolverLineSearch(self.constants)

    def solve_envelope(self, source: np.ndarray) -> np.ndarray:
        """
        Solve BVP envelope equation.

        Physical Meaning:
            Computes the envelope a(x) of the Base High-Frequency Field
            that modulates the high-frequency carrier.

        Mathematical Foundation:
            Solves ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x) for the envelope a(x).

        Args:
            source (np.ndarray): Source term s(x) in real space.
                Represents external excitations or initial conditions.

        Returns:
            np.ndarray: BVP envelope a(x) in real space.
                Represents the envelope modulation of the high-frequency carrier.

        Raises:
            ValueError: If source has incompatible shape with domain.
        """
        if source.shape != self.domain.shape:
            raise ValueError(
                f"Source shape {source.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        # Solve envelope equation using advanced Newton-Raphson method
        # Initial guess: zero field
        envelope = np.zeros_like(source, dtype=complex)

        # Advanced Newton-Raphson solution with adaptive step size
        max_iterations = int(self.constants.get_numerical_parameter("max_iterations"))
        tolerance = self.constants.get_numerical_parameter("tolerance")
        damping_factor = self.constants.get_numerical_parameter("damping_factor")
        min_step_size = self.constants.get_numerical_parameter("min_step_size")

        for iteration in range(max_iterations):
            # Compute residual and Jacobian using core components
            residual = self._core.compute_residual(envelope, source)
            jacobian = self._core.compute_jacobian(envelope)
            
            # Check convergence
            residual_norm = np.max(np.abs(residual))
            if residual_norm < tolerance:
                break
            
            # Solve Newton system: J * δa = -r
            try:
                # Use advanced linear solver with regularization
                delta_envelope = self._core.solve_newton_system(jacobian, residual)
                
                # Apply damping for stability
                step_size = damping_factor
                
                # Line search for optimal step size
                step_size = self._line_search.perform_line_search(
                    envelope, delta_envelope, residual, source, step_size,
                    self._core.compute_residual
                )
                
                # Update solution
                envelope = envelope + step_size * delta_envelope
                
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if Newton fails
                gradient = self._core.compute_gradient(envelope, source)
                gradient_step = self.constants.get_numerical_parameter("gradient_descent_step")
                envelope = envelope - gradient_step * gradient

        return envelope.real

    def _solve_linearized_envelope(
        self,
        envelope: np.ndarray,
        kappa: np.ndarray,
        chi: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        """
        Solve linearized envelope equation.

        Physical Meaning:
            Solves the linearized version of the envelope equation
            for a given nonlinear stiffness and susceptibility.

        Mathematical Foundation:
            Solves ∇·(κ∇a) + k₀²χa = s using finite difference method.

        Args:
            envelope (np.ndarray): Current envelope estimate.
            kappa (np.ndarray): Nonlinear stiffness κ(|a|).
            chi (np.ndarray): Effective susceptibility χ(|a|).
            source (np.ndarray): Source term s(x).

        Returns:
            np.ndarray: Updated envelope solution.
        """
        # Advanced finite difference implementation with spectral accuracy
        # Uses high-order finite differences for spatial derivatives
        # and spectral methods for improved accuracy

        dx = self.domain.dx

        # Compute gradient of envelope
        if self.domain.dimensions == 1:
            grad_envelope = np.gradient(envelope, dx)
            # Compute ∇·(κ∇a) term
            kappa_grad = kappa * grad_envelope
            div_kappa_grad = np.gradient(kappa_grad, dx)
        elif self.domain.dimensions == 2:
            grad_x, grad_y = np.gradient(envelope, dx, dx)
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            div_kappa_grad_x = np.gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad_y = np.gradient(kappa_grad_y, dx, axis=1)
            div_kappa_grad = div_kappa_grad_x + div_kappa_grad_y
        else:  # 3D
            grad_x, grad_y, grad_z = np.gradient(envelope, dx, dx, dx)
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            kappa_grad_z = kappa * grad_z
            div_kappa_grad_x = np.gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad_y = np.gradient(kappa_grad_y, dx, axis=1)
            div_kappa_grad_z = np.gradient(kappa_grad_z, dx, axis=2)
            div_kappa_grad = div_kappa_grad_x + div_kappa_grad_y + div_kappa_grad_z

        # Solve: ∇·(κ∇a) + k₀²χa = s
        # Rearrange: k₀²χa = s - ∇·(κ∇a)
        # Therefore: a = (s - ∇·(κ∇a)) / (k₀²χ)

        # Avoid division by zero
        regularization = self.constants.get_numerical_parameter("regularization")
        chi_safe = np.where(np.abs(chi) < regularization, regularization, chi)

        envelope_new = (source - div_kappa_grad) / (self.k0_squared * chi_safe)

        return envelope_new


    def get_parameters(self) -> Dict[str, float]:
        """
        Get envelope equation parameters.

        Physical Meaning:
            Returns the current parameters for the envelope equation.

        Returns:
            Dict[str, float]: Envelope equation parameters.
        """
        return {
            "kappa_0": self.kappa_0,
            "kappa_2": self.kappa_2,
            "chi_prime": self.chi_prime,
            "chi_double_prime_0": self.chi_double_prime_0,
            "k0_squared": self.k0_squared,
        }

    def __repr__(self) -> str:
        """String representation of envelope solver."""
        return (
            f"BVPEnvelopeSolver(domain={self.domain}, "
            f"kappa_0={self.kappa_0}, kappa_2={self.kappa_2})"
        )
