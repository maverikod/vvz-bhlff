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

from .domain import Domain


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

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
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
        """
        self.domain = domain
        self.config = config
        self._setup_parameters()

    def _setup_parameters(self) -> None:
        """
        Setup envelope equation parameters.

        Physical Meaning:
            Initializes the physical parameters for the envelope equation
            from the configuration dictionary.
        """
        envelope_config = self.config.get("envelope_equation", {})
        self.kappa_0 = envelope_config.get("kappa_0", 1.0)
        self.kappa_2 = envelope_config.get("kappa_2", 0.1)
        self.chi_prime = envelope_config.get("chi_prime", 1.0)
        self.chi_double_prime_0 = envelope_config.get("chi_double_prime_0", 0.01)
        self.k0_squared = envelope_config.get("k0_squared", 1.0)

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

        # Solve envelope equation using iterative method
        # Initial guess: zero field
        envelope = np.zeros_like(source, dtype=complex)

        # Simple iterative solution (can be improved with more sophisticated
        # methods)
        max_iterations = 100
        tolerance = 1e-6

        for iteration in range(max_iterations):
            # Compute nonlinear stiffness κ(|a|) = κ₀ + κ₂|a|²
            amplitude_squared = np.abs(envelope) ** 2
            kappa = self.kappa_0 + self.kappa_2 * amplitude_squared

            # Compute effective susceptibility χ(|a|) = χ' + iχ''(|a|)
            chi_double_prime = self.chi_double_prime_0 * amplitude_squared
            chi = self.chi_prime + 1j * chi_double_prime

            # Solve linearized equation: ∇·(κ∇a) + k₀²χa = s
            # For simplicity, use finite difference approximation
            envelope_new = self._solve_linearized_envelope(envelope, kappa, chi, source)

            # Check convergence
            residual = np.max(np.abs(envelope_new - envelope))
            if residual < tolerance:
                break

            envelope = envelope_new

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
        # Simple finite difference implementation
        # This is a simplified version - in practice, more sophisticated
        # methods like spectral methods or advanced finite differences would
        # be used

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
        chi_safe = np.where(np.abs(chi) < 1e-12, 1e-12, chi)

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
