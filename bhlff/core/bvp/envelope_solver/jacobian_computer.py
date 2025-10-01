"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Jacobian computation for 7D BVP envelope equation.

This module implements Jacobian matrix computation for the Newton-Raphson
method in solving the 7D BVP envelope equation.

Physical Meaning:
    Computes the Jacobian matrix J = ∂r/∂a of the residual
    with respect to the envelope field for Newton-Raphson iterations.

Mathematical Foundation:
    Jacobian is J = ∂/∂a[∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s]
    computed using finite difference approximation.

Example:
    >>> computer = JacobianComputer(domain, constants)
    >>> jacobian = computer.compute_jacobian(envelope)
"""

import numpy as np
from typing import Dict, Any

from ...domain import Domain
from ..bvp_constants import BVPConstants
from ..residual_computer import ResidualComputer


class JacobianComputer:
    """
    Jacobian computation for 7D BVP envelope equation.

    Physical Meaning:
        Computes the Jacobian matrix J = ∂r/∂a of the residual
        with respect to the envelope field for Newton-Raphson iterations.

    Mathematical Foundation:
        Jacobian is J = ∂/∂a[∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s]
        computed using finite difference approximation.

    Attributes:
        domain (Domain): 7D computational domain.
        constants (BVPConstants): BVP constants instance.
        residual_computer (ResidualComputer): Residual computation component.
    """

    def __init__(self, domain: Domain, constants: BVPConstants) -> None:
        """
        Initialize Jacobian computer.

        Physical Meaning:
            Sets up the Jacobian computation with parameters
            for the nonlinear envelope equation.

        Args:
            domain (Domain): Computational domain.
            constants (BVPConstants): BVP constants instance.
        """
        self.domain = domain
        self.constants = constants
        self.residual_computer = ResidualComputer(domain, constants)

    def compute_jacobian(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton-Raphson method.

        Physical Meaning:
            Computes the Jacobian matrix J = ∂r/∂a of the residual
            with respect to the envelope field.

        Mathematical Foundation:
            Jacobian is J = ∂/∂a[∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s]
            computed using finite difference approximation.

        Args:
            envelope (np.ndarray): Current envelope estimate.

        Returns:
            np.ndarray: Jacobian matrix J.
        """
        # For efficiency, use finite difference approximation of Jacobian
        # In practice, could use automatic differentiation or analytical derivatives

        n = envelope.size
        jacobian = np.zeros((n, n), dtype=complex)

        # Compute base residual
        base_residual = self.residual_computer.compute_residual(
            envelope, np.zeros_like(envelope)
        )

        # Finite difference step
        h = self.constants.get_numerical_parameter("finite_diff_step")

        for i in range(n):
            # Create perturbation
            envelope_pert = envelope.flatten().copy()
            envelope_pert[i] += h
            envelope_pert = envelope_pert.reshape(envelope.shape)

            # Compute perturbed residual
            pert_residual = self.residual_computer.compute_residual(
                envelope_pert, np.zeros_like(envelope)
            )

            # Compute Jacobian column
            jacobian[:, i] = (pert_residual.flatten() - base_residual.flatten()) / h

        return jacobian

    def __repr__(self) -> str:
        """String representation of Jacobian computer."""
        return f"JacobianComputer(domain={self.domain})"
