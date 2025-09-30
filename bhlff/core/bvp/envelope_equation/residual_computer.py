"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Residual computer for 7D BVP envelope equation.

This module implements computation of residuals for the 7D BVP envelope
equation, including spatial and phase divergence terms with nonlinear
stiffness and susceptibility.

Physical Meaning:
    The residual computer implements the computation of the residual
    R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t) for the 7D envelope
    equation, representing how well the current solution satisfies
    the equation.

Mathematical Foundation:
    Computes the residual of the 7D envelope equation:
    R = ∇ₓ·(κ(|a|)∇ₓa) + ∇φ·(κ(|a|)∇φa) + k₀²χ(|a|)a - s(x,φ,t)
    where κ(|a|) and χ(|a|) are nonlinear coefficients.

Example:
    >>> residual_comp = ResidualComputer(domain_7d, config)
    >>> residual = residual_comp.compute_residual(envelope, source)
"""

import numpy as np
from typing import Dict, Any

from ...domain.domain_7d import Domain7D


class ResidualComputer:
    """
    Residual computer for 7D BVP envelope equation.

    Physical Meaning:
        Computes the residual of the 7D envelope equation, representing
        how well the current solution satisfies the equation and is used
        in Newton-Raphson iterations.

    Mathematical Foundation:
        Computes the residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
        including spatial and phase divergence terms with nonlinear
        stiffness and susceptibility coefficients.
    """

    def __init__(self, domain_7d: Domain7D, config: Dict[str, Any]):
        """
        Initialize residual computer.

        Physical Meaning:
            Sets up the residual computer with the computational domain
            and configuration parameters for computing residuals of the
            7D envelope equation.

        Args:
            domain_7d (Domain7D): 7D computational domain.
            config (Dict[str, Any]): Configuration parameters.
        """
        self.domain_7d = domain_7d
        self.config = config

    def compute_residual(
        self,
        envelope: np.ndarray,
        source: np.ndarray,
        derivative_operators: object,
        nonlinear_terms: object,
    ) -> np.ndarray:
        """
        Compute residual of the envelope equation.

        Physical Meaning:
            Computes the residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            for the current envelope solution.

        Mathematical Foundation:
            The residual measures how well the current solution satisfies
            the envelope equation and is used in Newton-Raphson iterations.

        Args:
            envelope (np.ndarray): Current envelope solution.
            source (np.ndarray): Source term.
            derivative_operators: Derivative operators object.
            nonlinear_terms: Nonlinear terms object.

        Returns:
            np.ndarray: Residual vector.
        """
        amplitude = np.abs(envelope)

        # Compute nonlinear coefficients
        kappa = nonlinear_terms.compute_stiffness(amplitude)
        chi = nonlinear_terms.compute_susceptibility(amplitude)

        # Compute spatial divergence term: ∇ₓ·(κ(|a|)∇ₓa)
        spatial_div = self._compute_spatial_divergence(
            kappa, envelope, derivative_operators
        )

        # Compute phase divergence term: ∇φ·(κ(|a|)∇φa)
        phase_div = self._compute_phase_divergence(
            kappa, envelope, derivative_operators
        )

        # Compute susceptibility term: k₀²χ(|a|)a
        susceptibility_term = nonlinear_terms.k0**2 * chi * envelope

        # Total residual
        residual = spatial_div + phase_div + susceptibility_term - source

        return residual

    def _compute_spatial_divergence(
        self, kappa: np.ndarray, envelope: np.ndarray, derivative_operators: object
    ) -> np.ndarray:
        """
        Compute spatial divergence term ∇ₓ·(κ(|a|)∇ₓa).

        Physical Meaning:
            Computes the spatial divergence of the stiffness-weighted
            gradient, representing the spatial part of the envelope equation.

        Mathematical Foundation:
            Computes ∇ₓ·(κ(|a|)∇ₓa) = Σᵢ ∂/∂xᵢ(κ(|a|)∂a/∂xᵢ)
            for i = x, y, z spatial coordinates.

        Args:
            kappa (np.ndarray): Nonlinear stiffness κ(|a|).
            envelope (np.ndarray): Envelope field.
            derivative_operators: Derivative operators object.

        Returns:
            np.ndarray: Spatial divergence term.
        """
        divergence = np.zeros_like(envelope)

        # Apply to each spatial dimension
        for axis in range(3):
            # Compute gradient: ∇ₓa
            grad_envelope = derivative_operators.apply_spatial_gradient(envelope, axis)

            # Multiply by stiffness: κ(|a|)∇ₓa
            weighted_grad = kappa * grad_envelope

            # Compute divergence: ∇ₓ·(κ(|a|)∇ₓa)
            div_term = derivative_operators.apply_spatial_divergence(
                weighted_grad, axis
            )
            divergence += div_term

        return divergence

    def _compute_phase_divergence(
        self, kappa: np.ndarray, envelope: np.ndarray, derivative_operators: object
    ) -> np.ndarray:
        """
        Compute phase divergence term ∇φ·(κ(|a|)∇φa).

        Physical Meaning:
            Computes the phase divergence of the stiffness-weighted
            gradient, representing the phase part of the envelope equation.

        Mathematical Foundation:
            Computes ∇φ·(κ(|a|)∇φa) = Σᵢ ∂/∂φᵢ(κ(|a|)∂a/∂φᵢ)
            for i = φ₁, φ₂, φ₃ phase coordinates.

        Args:
            kappa (np.ndarray): Nonlinear stiffness κ(|a|).
            envelope (np.ndarray): Envelope field.
            derivative_operators: Derivative operators object.

        Returns:
            np.ndarray: Phase divergence term.
        """
        divergence = np.zeros_like(envelope)

        # Apply to each phase dimension
        for axis in range(3, 6):
            # Compute gradient: ∇φa
            grad_envelope = derivative_operators.apply_phase_gradient(envelope, axis)

            # Multiply by stiffness: κ(|a|)∇φa
            weighted_grad = kappa * grad_envelope

            # Compute divergence: ∇φ·(κ(|a|)∇φa)
            div_term = derivative_operators.apply_phase_divergence(weighted_grad, axis)
            divergence += div_term

        return divergence

    def compute_residual_norm(self, residual: np.ndarray) -> float:
        """
        Compute norm of residual for convergence checking.

        Physical Meaning:
            Computes the L2 norm of the residual vector for monitoring
            convergence of the Newton-Raphson iterations.

        Args:
            residual (np.ndarray): Residual vector.

        Returns:
            float: L2 norm of the residual.
        """
        return float(np.linalg.norm(residual))

    def analyze_residual_components(
        self,
        envelope: np.ndarray,
        source: np.ndarray,
        derivative_operators: object,
        nonlinear_terms: object,
    ) -> Dict[str, Any]:
        """
        Analyze components of the residual.

        Physical Meaning:
            Analyzes the individual components of the residual to understand
            the relative contributions of different terms in the equation.

        Args:
            envelope (np.ndarray): Current envelope solution.
            source (np.ndarray): Source term.
            derivative_operators: Derivative operators object.
            nonlinear_terms: Nonlinear terms object.

        Returns:
            Dict[str, Any]: Dictionary containing residual component analysis.
        """
        amplitude = np.abs(envelope)

        # Compute nonlinear coefficients
        kappa = nonlinear_terms.compute_stiffness(amplitude)
        chi = nonlinear_terms.compute_susceptibility(amplitude)

        # Compute individual components
        spatial_div = self._compute_spatial_divergence(
            kappa, envelope, derivative_operators
        )
        phase_div = self._compute_phase_divergence(
            kappa, envelope, derivative_operators
        )
        susceptibility_term = nonlinear_terms.k0**2 * chi * envelope

        # Compute norms
        spatial_norm = np.linalg.norm(spatial_div)
        phase_norm = np.linalg.norm(phase_div)
        susceptibility_norm = np.linalg.norm(susceptibility_term)
        source_norm = np.linalg.norm(source)

        # Total residual
        total_residual = spatial_div + phase_div + susceptibility_term - source
        total_norm = np.linalg.norm(total_residual)

        return {
            "spatial_divergence_norm": float(spatial_norm),
            "phase_divergence_norm": float(phase_norm),
            "susceptibility_norm": float(susceptibility_norm),
            "source_norm": float(source_norm),
            "total_residual_norm": float(total_norm),
            "component_ratios": {
                "spatial_ratio": (
                    float(spatial_norm / total_norm) if total_norm > 0 else 0.0
                ),
                "phase_ratio": (
                    float(phase_norm / total_norm) if total_norm > 0 else 0.0
                ),
                "susceptibility_ratio": (
                    float(susceptibility_norm / total_norm) if total_norm > 0 else 0.0
                ),
                "source_ratio": (
                    float(source_norm / total_norm) if total_norm > 0 else 0.0
                ),
            },
        }
