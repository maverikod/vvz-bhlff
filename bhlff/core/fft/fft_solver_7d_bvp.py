"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

7D BVP FFT Solver facade implementation.

This module provides a facade interface for the 7D BVP envelope equation solver,
combining core functionality, Newton-Raphson solver, and validation methods.

Physical Meaning:
    Provides a unified interface for solving the complete 7D BVP envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Facade pattern combining:
    - Core BVP solver functionality
    - Newton-Raphson iterative solver
    - Solution validation methods

Example:
    >>> domain = Domain7DBVP(L_spatial=1.0, N_spatial=64, N_phase=32, T=1.0, N_t=128)
    >>> params = Parameters7DBVP(kappa_0=1.0, kappa_2=0.1, chi_prime=1.0, k0=1.0)
    >>> solver = FFTSolver7DBVP(domain, params)
    >>> solution = solver.solve_envelope(source_field)
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.domain_7d_bvp import Domain7DBVP
    from ..domain.parameters_7d_bvp import Parameters7DBVP

from .spectral_operations import SpectralOperations
from .spectral_derivatives import SpectralDerivatives
from .spectral_filtering import SpectralFiltering
from ..bvp.bvp_envelope_solver import BVPEnvelopeSolver
from ..bvp.bvp_constants import BVPConstants


class FFTSolver7DBVP:
    """
    7D BVP FFT Solver facade for complete envelope equation.

    Physical Meaning:
        Provides a unified interface for solving the complete 7D BVP envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
        in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

    Mathematical Foundation:
        Facade pattern combining:
        - Core BVP solver functionality
        - Newton-Raphson iterative solver
        - Solution validation methods

    Attributes:
        domain (Domain7DBVP): 7D BVP computational domain.
        parameters (Parameters7DBVP): 7D BVP parameters.
        _core (BVPSolverCore): Core BVP solver functionality.
        _newton_solver (BVPSolverNewton): Newton-Raphson solver.
        _validator (BVPSolverValidation): Solution validator.
    """

    def __init__(self, domain: "Domain7DBVP", parameters: "Parameters7DBVP"):
        """
        Initialize 7D BVP FFT solver facade.

        Physical Meaning:
            Sets up the solver facade with the 7D BVP domain and parameters,
            initializing all necessary components for solving the complete
            7D BVP envelope equation.

        Args:
            domain (Domain7DBVP): 7D BVP computational domain.
            parameters (Parameters7DBVP): 7D BVP parameters including
                nonlinear coefficients κ(|a|) and χ(|a|).
        """
        self.domain = domain
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

        # Validate domain dimensions
        if domain.dimensions != 7:
            raise ValueError(
                f"Domain must be 7D for BVP theory, got {domain.dimensions}"
            )

        # Initialize spectral operations
        self._spectral_ops = SpectralOperations(domain, parameters.precision)
        self._derivatives = SpectralDerivatives(domain, parameters.precision)
        self._filtering = SpectralFiltering(domain, parameters.precision)

        # Initialize BVP envelope solver integration
        self._bvp_constants = BVPConstants(parameters.to_dict())
        self._envelope_solver = BVPEnvelopeSolver(
            domain, parameters.to_dict(), self._bvp_constants
        )

        self._initialized = True
        self.logger.info("FFTSolver7DBVP facade initialized.")

    def solve_envelope(
        self,
        source_field: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        method: str = "newton_raphson",
    ) -> np.ndarray:
        """
        Solve complete 7D BVP envelope equation.

        Physical Meaning:
            Solves the complete 7D BVP envelope equation:
            ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
            for the U(1)³ phase vector a(x,φ,t) ∈ ℂ³ in 7D space-time M₇.

        Mathematical Foundation:
            Solves the nonlinear equation using iterative methods:
            - Newton-Raphson for full nonlinear equation
            - Linearized version using fractional Laplacian for initial guess

        Args:
            source_field (np.ndarray): Source term s(x,φ,t) in 7D space-time.
            initial_guess (Optional[np.ndarray]): Initial guess for solution.
            method (str): Solution method ('newton_raphson', 'linearized').

        Returns:
            np.ndarray: Envelope solution a(x,φ,t) ∈ ℂ³ in 7D space-time.
        """
        if source_field.shape != self.domain.shape:
            raise ValueError(
                f"Source shape {source_field.shape} incompatible with domain {self.domain.shape}"
            )

        # Use integrated BVP envelope solver
        if method == "linearized":
            # Use linearized version for initial guess
            return self._envelope_solver.solve_envelope_linearized(source_field)
        elif method == "newton_raphson":
            # Use full nonlinear solver
            return self._envelope_solver.solve_envelope(source_field, initial_guess)
        else:
            raise ValueError(f"Unknown solution method: {method}")

    def validate_solution(
        self,
        solution: np.ndarray,
        source: np.ndarray,
        tolerance: float = 1e-8,
        method: str = "linearized",
    ) -> Dict[str, Any]:
        """
        Validate BVP solution.

        Physical Meaning:
            Validates the solution by computing the residual and checking
            that it satisfies the BVP equation within the specified tolerance.

        Args:
            solution (np.ndarray): Solution a(x,φ,t).
            source (np.ndarray): Source term s(x,φ,t).
            tolerance (float): Validation tolerance.
            method (str): Validation method ('linearized' or 'full').

        Returns:
            Dict[str, Any]: Validation results.
        """
        # Use integrated BVP envelope solver validation
        return self._envelope_solver.validate_solution(solution, source, tolerance)

    def check_energy_conservation(
        self,
        field: np.ndarray,
        expected_energy: Optional[float] = None,
        tolerance: float = 1e-10,
    ) -> Dict[str, Any]:
        """
        Check energy conservation for the field.

        Physical Meaning:
            Verifies that the field satisfies energy conservation principles
            within the specified tolerance.

        Args:
            field (np.ndarray): Field to check.
            expected_energy (Optional[float]): Expected energy value.
            tolerance (float): Energy conservation tolerance.

        Returns:
            Dict[str, Any]: Energy conservation results.
        """
        return self._validator.check_energy_conservation(
            field, expected_energy, tolerance
        )

    def check_boundary_conditions(
        self, field: np.ndarray, boundary_type: str = "periodic"
    ) -> Dict[str, Any]:
        """
        Check boundary conditions for the field.

        Physical Meaning:
            Verifies that the field satisfies the specified boundary conditions
            at the domain boundaries.

        Args:
            field (np.ndarray): Field to check.
            boundary_type (str): Type of boundary conditions.

        Returns:
            Dict[str, Any]: Boundary condition results.
        """
        return self._validator.check_boundary_conditions(field, boundary_type)

    def __repr__(self) -> str:
        """String representation of solver."""
        return (
            f"FFTSolver7DBVP("
            f"domain={self.domain.shape}, "
            f"κ₀={self.parameters.kappa_0}, "
            f"χ'={self.parameters.chi_prime}, "
            f"k₀={self.parameters.k0})"
        )
