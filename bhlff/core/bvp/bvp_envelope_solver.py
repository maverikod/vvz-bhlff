"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

7D BVP envelope equation solver module.

This module implements the solver for the 7D BVP envelope equation with
nonlinear stiffness and susceptibility, providing the core mathematical
operations for envelope field calculations in 7D space-time.

Physical Meaning:
    Solves the 7D envelope equation in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility.

Mathematical Foundation:
    Implements iterative solution of the nonlinear 7D envelope equation
    using finite difference methods for the divergence and gradient
    operations in all 7 dimensions (3 spatial + 3 phase + 1 temporal).

Example:
    >>> solver = BVPEnvelopeSolver(domain, config)
    >>> envelope = solver.solve_envelope(source)
"""

import numpy as np
from typing import Dict, Any

from ..domain import Domain
from .envelope_solver.envelope_solver_core import EnvelopeSolverCore
from .envelope_solver_line_search import EnvelopeSolverLineSearch
from .bvp_constants import BVPConstants
from .envelope_nonlinear_coefficients import EnvelopeNonlinearCoefficients
from .envelope_linear_solver import EnvelopeLinearSolver
from .memory_protection import MemoryProtector
from .memory_decorator import memory_protected_class_method


class BVPEnvelopeSolver:
    """
    Solver for 7D BVP envelope equation.

    Physical Meaning:
        Solves the nonlinear 7D envelope equation for the Base High-Frequency
        Field in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, computing the envelope modulation
        that satisfies the governing equation with nonlinear stiffness and susceptibility.

    Mathematical Foundation:
        Solves ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t) where:
        - κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness
        - χ(|a|) = χ' + iχ''(|a|) is effective susceptibility
        - s(x,φ,t) is the source term in 7D space-time
        - ∇ includes gradients in all 7 dimensions

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Envelope solver configuration.
        kappa_0 (float): Base stiffness coefficient κ₀.
        kappa_2 (float): Nonlinear stiffness coefficient κ₂.
        chi_prime (float): Real part of susceptibility χ'.
        chi_double_prime_0 (float): Base imaginary susceptibility χ''₀.
        k0_squared (float): Wave number squared k₀².
    """

    def __init__(
        self, domain: Domain, config: Dict[str, Any], constants: BVPConstants = None
    ) -> None:
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
            Initializes the base physical parameters for the envelope equation
            from the constants instance. These are used as base values for
            computing nonlinear coefficients that depend on field amplitude.
        """
        # Base parameters for nonlinear coefficient computation
        self.kappa_0 = self.constants.get_envelope_parameter("kappa_0")
        self.kappa_2 = self.constants.get_envelope_parameter("kappa_2")
        self.chi_prime = self.constants.get_envelope_parameter("chi_prime")
        self.chi_double_prime_0 = self.constants.get_envelope_parameter(
            "chi_double_prime_0"
        )
        self.k0_squared = self.constants.get_envelope_parameter("k0_squared")

        # Initialize nonlinear coefficients computer and linear solver
        self.nonlinear_coeffs = EnvelopeNonlinearCoefficients(self.constants)
        self.linear_solver = EnvelopeLinearSolver(self.domain, self.constants)

        # Initialize memory protection
        try:
            memory_threshold = self.constants.get_numerical_parameter(
                "memory_threshold"
            )
        except KeyError:
            memory_threshold = 0.8
        self.memory_protector = MemoryProtector(memory_threshold)

    def _setup_solver_components(self) -> None:
        """Setup solver components."""
        self._core = EnvelopeSolverCore(self.domain, self.config, self.constants)
        self._line_search = EnvelopeSolverLineSearch(self.constants)

    @memory_protected_class_method(
        memory_threshold=0.8, shape_param="source", dtype_param="source"
    )
    def solve_envelope(self, source: np.ndarray) -> np.ndarray:
        """
        Solve 7D BVP envelope equation.

        Physical Meaning:
            Computes the envelope a(x,φ,t) of the Base High-Frequency Field
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ that modulates the high-frequency carrier.

        Mathematical Foundation:
            Solves ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t) for the envelope a(x,φ,t)
            where ∇ includes gradients in all 7 dimensions (3 spatial + 3 phase + 1 temporal).

        Args:
            source (np.ndarray): Source term s(x,φ,t) in 7D space-time.
                Represents external excitations or initial conditions in M₇.

        Returns:
            np.ndarray: BVP envelope a(x,φ,t) in 7D space-time.
                Represents the envelope modulation of the high-frequency carrier.

        Raises:
            ValueError: If source has incompatible shape with 7D domain.
        """
        if source.shape != self.domain.shape:
            raise ValueError(
                f"Source shape {source.shape} incompatible with "
                f"7D domain shape {self.domain.shape}"
            )

        # Check memory usage before starting calculation
        try:
            self.memory_protector.check_memory_usage(source.shape, source.dtype)
        except MemoryError as e:
            raise MemoryError(
                f"Memory protection triggered: {e}. "
                f"Consider reducing domain size or using lower precision."
            )

        # Solve envelope equation using advanced Newton-Raphson method
        # Initial guess: zero field
        envelope = np.zeros_like(source, dtype=complex)

        # Advanced Newton-Raphson solution with adaptive step size
        max_iterations = int(self.constants.get_numerical_parameter("max_iterations"))
        tolerance = self.constants.get_numerical_parameter("tolerance")
        damping_factor = self.constants.get_numerical_parameter("damping_factor")

        for iteration in range(max_iterations):
            # Compute nonlinear coefficients based on current envelope
            nonlinear_coeffs = self.nonlinear_coeffs.compute_coefficients(envelope)

            # Compute residual and Jacobian using nonlinear coefficients
            residual = self._core.compute_residual_with_coefficients(
                envelope, source, nonlinear_coeffs
            )
            jacobian = self._core.compute_jacobian_with_coefficients(
                envelope, nonlinear_coeffs
            )

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
                    envelope,
                    delta_envelope,
                    residual,
                    source,
                    step_size,
                    self._core.compute_residual,
                )

                # Update solution
                envelope = envelope + step_size * delta_envelope

            except np.linalg.LinAlgError:
                # Fallback to gradient descent if Newton fails
                gradient = self._core.compute_gradient(envelope, source)
                gradient_step = self.constants.get_numerical_parameter(
                    "gradient_descent_step"
                )
                envelope = envelope - gradient_step * gradient

        return envelope.real

    @memory_protected_class_method(
        memory_threshold=0.8, shape_param="source", dtype_param="source"
    )
    def solve_envelope_linearized(self, source: np.ndarray) -> np.ndarray:
        """
        Solve linearized 7D BVP envelope equation.

        Physical Meaning:
            Solves the linearized version of the envelope equation
            ∇·(κ₀∇a) + k₀²χ'a = s(x,φ,t) for initial guess generation.

        Mathematical Foundation:
            Solves the linearized equation using spectral methods
            for efficient computation of initial guess.

        Args:
            source (np.ndarray): Source term s(x,φ,t) in 7D space-time.

        Returns:
            np.ndarray: Linearized envelope solution a(x,φ,t).
        """
        return self.linear_solver.solve_linearized(source)

    def get_parameters(self) -> Dict[str, float]:
        """
        Get envelope equation parameters.

        Physical Meaning:
            Returns the current base parameters for the envelope equation.
            Note: Actual coefficients are computed dynamically as functions
            of field amplitude.

        Returns:
            Dict[str, float]: Base envelope equation parameters.
        """
        return {
            "kappa_0": self.kappa_0,
            "kappa_2": self.kappa_2,
            "chi_prime": self.chi_prime,
            "chi_double_prime_0": self.chi_double_prime_0,
            "k0_squared": self.k0_squared,
        }

    def get_nonlinear_coefficients(self, envelope: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get nonlinear coefficients for given envelope.

        Physical Meaning:
            Computes and returns the nonlinear coefficients κ(|a|) and χ(|a|)
            for the given envelope field, showing how they depend on
            local field amplitude.

        Mathematical Foundation:
            Returns κ(|a|) = κ₀ + κ₂|a|² and χ(|a|) = χ' + iχ''(|a|)
            computed from the current envelope field.

        Args:
            envelope (np.ndarray): Envelope field a(x,φ,t).

        Returns:
            Dict[str, np.ndarray]: Nonlinear coefficients:
                - kappa: Nonlinear stiffness κ(|a|)
                - chi_real: Real part of susceptibility χ'(|a|)
                - chi_imag: Imaginary part of susceptibility χ''(|a|)
        """
        return self.nonlinear_coeffs.compute_coefficients(envelope)

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information and usage statistics.

        Physical Meaning:
            Returns current memory usage statistics for monitoring
            and debugging purposes.

        Returns:
            Dict[str, Any]: Memory information including:
                - current_usage: Current memory usage
                - domain_estimate: Estimated memory for current domain
                - threshold: Memory threshold setting
                - is_safe: Whether current usage is safe
        """
        memory_info = self.memory_protector.get_memory_info()
        domain_estimate = self.memory_protector.estimate_memory_requirement(
            self.domain.shape, np.float64
        )

        return {
            "current_usage": memory_info,
            "domain_estimate": domain_estimate,
            "threshold": self.memory_protector.memory_threshold,
            "is_safe": self.memory_protector.check_and_warn(
                self.domain.shape, np.float64
            ),
        }

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

        # Compute nonlinear coefficients for validation
        nonlinear_coeffs = self.nonlinear_coeffs.compute_coefficients(solution)

        # Compute residual: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s
        residual = self._core.compute_residual_with_coefficients(
            solution, source, nonlinear_coeffs
        )

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

    def __repr__(self) -> str:
        """String representation of envelope solver."""
        return (
            f"BVPEnvelopeSolver(domain={self.domain}, "
            f"kappa_0={self.kappa_0}, kappa_2={self.kappa_2})"
        )
