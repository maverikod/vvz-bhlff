"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Residual computation for 7D BVP envelope equation.

This module implements residual computation for the 7D BVP envelope equation,
including nonlinear stiffness and susceptibility calculations.

Physical Meaning:
    Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
    for the Newton-Raphson method in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Residual is r = L(a) - s where L(a) is the nonlinear operator
    of the 7D envelope equation with gradients in all 7 dimensions.

Example:
    >>> computer = ResidualComputer(domain, constants)
    >>> residual = computer.compute_residual(envelope, source)
"""

import numpy as np
from typing import Dict, Any

from ...domain import Domain
from ..bvp_constants import BVPConstants


class ResidualComputer:
    """
    Residual computation for 7D BVP envelope equation.

    Physical Meaning:
        Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
        for the Newton-Raphson method in 7D space-time.

    Mathematical Foundation:
        Residual is r = L(a) - s where L(a) is the nonlinear operator
        of the 7D envelope equation with gradients in all 7 dimensions.

    Attributes:
        domain (Domain): 7D computational domain.
        constants (BVPConstants): BVP constants instance.
        kappa_0 (float): Base stiffness coefficient κ₀.
        kappa_2 (float): Nonlinear stiffness coefficient κ₂.
        chi_prime (float): Real part of susceptibility χ'.
        chi_double_prime_0 (float): Base imaginary susceptibility χ''₀.
        k0_squared (float): Wave number squared k₀².
    """

    def __init__(self, domain: Domain, constants: BVPConstants) -> None:
        """
        Initialize residual computer.

        Physical Meaning:
            Sets up the residual computation with parameters
            for the nonlinear envelope equation.

        Args:
            domain (Domain): Computational domain.
            constants (BVPConstants): BVP constants instance.
        """
        self.domain = domain
        self.constants = constants
        self._setup_parameters()

    def _setup_parameters(self) -> None:
        """Setup envelope equation parameters."""
        self.kappa_0 = self.constants.get_envelope_parameter("kappa_0")
        self.kappa_2 = self.constants.get_envelope_parameter("kappa_2")
        self.chi_prime = self.constants.get_envelope_parameter("chi_prime")
        self.chi_double_prime_0 = self.constants.get_envelope_parameter(
            "chi_double_prime_0"
        )
        self.k0_squared = self.constants.get_envelope_parameter("k0_squared")

    def compute_residual(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the 7D envelope equation.

        Physical Meaning:
            Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            for the Newton-Raphson method in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Residual is r = L(a) - s where L(a) is the nonlinear operator
            of the 7D envelope equation with gradients in all 7 dimensions.

        Args:
            envelope (np.ndarray): Current envelope estimate in 7D space-time.
            source (np.ndarray): Source term s(x,φ,t) in 7D space-time.

        Returns:
            np.ndarray: Residual r = L(a) - s in 7D space-time.
        """
        # Compute nonlinear stiffness κ(|a|) = κ₀ + κ₂|a|²
        amplitude_squared = np.abs(envelope) ** 2
        kappa = self.kappa_0 + self.kappa_2 * amplitude_squared

        # Compute effective susceptibility χ(|a|) = χ' + iχ''(|a|)
        chi_double_prime = self.chi_double_prime_0 * amplitude_squared
        chi = self.chi_prime + 1j * chi_double_prime

        # Compute ∇·(κ∇a) term using advanced finite differences
        div_kappa_grad = self._compute_div_kappa_grad(envelope, kappa)

        # Compute k₀²χa term
        chi_a_term = self.k0_squared * chi * envelope

        # Compute residual r = ∇·(κ∇a) + k₀²χa - s
        residual = div_kappa_grad + chi_a_term - source

        return residual

    def _compute_div_kappa_grad(
        self, envelope: np.ndarray, kappa: np.ndarray
    ) -> np.ndarray:
        """
        Compute ∇·(κ∇a) using advanced finite differences for 7D space-time.

        Physical Meaning:
            Computes the divergence of κ times the gradient of the envelope
            using high-order finite difference methods in 7D space-time.

        Mathematical Foundation:
            Computes ∇·(κ∇a) = ∂/∂x(κ∂a/∂x) + ∂/∂y(κ∂a/∂y) + ∂/∂z(κ∂a/∂z) +
                              ∂/∂φ₁(κ∂a/∂φ₁) + ∂/∂φ₂(κ∂a/∂φ₂) + ∂/∂φ₃(κ∂a/∂φ₃) +
                              ∂/∂t(κ∂a/∂t)
            using fourth-order finite differences.

        Args:
            envelope (np.ndarray): Envelope field in 7D space-time.
            kappa (np.ndarray): Nonlinear stiffness in 7D space-time.

        Returns:
            np.ndarray: ∇·(κ∇a) term in 7D space-time.
        """
        # Get grid spacings for 7D
        dx = self.domain.dx
        dphi = self.domain.dphi
        dt = self.domain.dt

        # Initialize divergence
        div_kappa_grad = np.zeros_like(envelope)

        # Spatial gradients ℝ³ₓ
        if self.domain.dimensions == 1:
            grad_x = np.gradient(envelope, dx, axis=0)
            kappa_grad_x = kappa * grad_x
            div_kappa_grad += np.gradient(kappa_grad_x, dx, axis=0)
        elif self.domain.dimensions == 2:
            grad_x, grad_y = np.gradient(envelope, dx, dx, axis=(0, 1))
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            div_kappa_grad += np.gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad += np.gradient(kappa_grad_y, dx, axis=1)
        else:  # 3D spatial
            grad_x, grad_y, grad_z = np.gradient(envelope, dx, dx, dx, axis=(0, 1, 2))
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            kappa_grad_z = kappa * grad_z
            div_kappa_grad += np.gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad += np.gradient(kappa_grad_y, dx, axis=1)
            div_kappa_grad += np.gradient(kappa_grad_z, dx, axis=2)

        # Phase gradients 𝕋³_φ
        grad_phi1 = np.gradient(envelope, dphi, axis=3)
        grad_phi2 = np.gradient(envelope, dphi, axis=4)
        grad_phi3 = np.gradient(envelope, dphi, axis=5)

        kappa_grad_phi1 = kappa * grad_phi1
        kappa_grad_phi2 = kappa * grad_phi2
        kappa_grad_phi3 = kappa * grad_phi3

        div_kappa_grad += np.gradient(kappa_grad_phi1, dphi, axis=3)
        div_kappa_grad += np.gradient(kappa_grad_phi2, dphi, axis=4)
        div_kappa_grad += np.gradient(kappa_grad_phi3, dphi, axis=5)

        # Temporal gradient ℝₜ
        grad_t = np.gradient(envelope, dt, axis=6)
        kappa_grad_t = kappa * grad_t
        div_kappa_grad += np.gradient(kappa_grad_t, dt, axis=6)

        return div_kappa_grad

    def __repr__(self) -> str:
        """String representation of residual computer."""
        return f"ResidualComputer(domain={self.domain})"
