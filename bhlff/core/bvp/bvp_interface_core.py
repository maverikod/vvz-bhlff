"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core BVP interface operations for system component integration.

This module implements the core mathematical operations for the BVP interface,
including field gradient computation and basic interface functions.

Physical Meaning:
    Provides the fundamental mathematical operations for interfacing BVP
    envelope with other system components, including gradient computation
    and basic field analysis.

Mathematical Foundation:
    Implements core interface functions that transform BVP envelope data
    into appropriate formats for different system components.

Example:
    >>> interface_core = BVPInterfaceCore(domain, constants)
    >>> gradient = interface_core.compute_field_gradient(envelope)
"""

import numpy as np
from typing import TYPE_CHECKING, Optional

from .bvp_constants import BVPConstants

if TYPE_CHECKING:
    from ..domain import Domain


class BVPInterfaceCore:
    """
    Core mathematical operations for BVP interface.

    Physical Meaning:
        Implements the core mathematical operations for interfacing BVP
        envelope with other system components.

    Mathematical Foundation:
        Provides fundamental interface functions including field gradient
        computation and basic field analysis operations.

    Attributes:
        domain (Domain): Computational domain.
        constants (BVPConstants): BVP constants instance.
    """

    def __init__(
        self, domain: "Domain", constants: Optional[BVPConstants] = None
    ) -> None:
        """
        Initialize BVP interface core.

        Physical Meaning:
            Sets up the core mathematical operations for BVP interface
            with the computational domain and constants.

        Args:
            domain (Domain): Computational domain.
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.domain = domain
        self.constants = constants or BVPConstants()

    def compute_field_gradient(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute field gradient in 7D space-time.

        Physical Meaning:
            Computes the gradient of the field envelope in 7D space-time
            M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ using high-order finite difference methods.

        Mathematical Foundation:
            Computes ∇A using finite difference approximation in 7D:
            ∇A = (∂A/∂x, ∂A/∂y, ∂A/∂z, ∂A/∂φ₁, ∂A/∂φ₂, ∂A/∂φ₃, ∂A/∂t)

        Args:
            envelope (np.ndarray): Field envelope in 7D space-time.

        Returns:
            np.ndarray: Field gradient components in 7D.
        """
        dx = self.domain.dx
        dphi = self.domain.dphi
        dt = self.domain.dt

        # Compute gradients in all 7 dimensions
        gradients = []

        # Spatial gradients ℝ³ₓ
        if self.domain.dimensions >= 1:
            gradients.append(np.gradient(envelope, dx, axis=0))
        if self.domain.dimensions >= 2:
            gradients.append(np.gradient(envelope, dx, axis=1))
        if self.domain.dimensions >= 3:
            gradients.append(np.gradient(envelope, dx, axis=2))

        # Phase gradients 𝕋³_φ
        if envelope.ndim > 3:
            gradients.append(np.gradient(envelope, dphi, axis=3))  # φ₁
        if envelope.ndim > 4:
            gradients.append(np.gradient(envelope, dphi, axis=4))  # φ₂
        if envelope.ndim > 5:
            gradients.append(np.gradient(envelope, dphi, axis=5))  # φ₃

        # Temporal gradient ℝₜ
        if envelope.ndim > 6:
            gradients.append(np.gradient(envelope, dt, axis=6))  # t

        return gradients

    def compute_field_amplitude(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute field amplitude.

        Physical Meaning:
            Computes the magnitude of the complex field envelope.

        Mathematical Foundation:
            Computes |A| = √(Re(A)² + Im(A)²)

        Args:
            envelope (np.ndarray): Field envelope.

        Returns:
            np.ndarray: Field amplitude |A|.
        """
        return np.abs(envelope)

    def compute_field_phase(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute field phase.

        Physical Meaning:
            Computes the phase of the complex field envelope.

        Mathematical Foundation:
            Computes φ = arg(A) = atan2(Im(A), Re(A))

        Args:
            envelope (np.ndarray): Field envelope.

        Returns:
            np.ndarray: Field phase φ.
        """
        return np.angle(envelope)

    def compute_gradient_magnitude_squared(
        self, field_gradient: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient magnitude squared.

        Physical Meaning:
            Computes the squared magnitude of the field gradient.

        Mathematical Foundation:
            Computes |∇A|² = |∂A/∂x|² + |∂A/∂y|² + |∂A/∂z|²

        Args:
            field_gradient (np.ndarray): Field gradient components.

        Returns:
            np.ndarray: Gradient magnitude squared |∇A|².
        """
        if isinstance(field_gradient, tuple):
            # Multi-dimensional gradient
            return np.sum([np.abs(g) ** 2 for g in field_gradient], axis=0)
        else:
            # One-dimensional gradient
            return np.abs(field_gradient) ** 2

    def __repr__(self) -> str:
        """String representation of BVP interface core."""
        return f"BVPInterfaceCore(domain={self.domain})"
