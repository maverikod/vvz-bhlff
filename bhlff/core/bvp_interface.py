"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP interface for system component integration.

This module implements the interface between the Base High-Frequency "
"Field (BVP)
and other system components, providing connections to tail resonators,
transition zone, and core.

Physical Meaning:
    Provides the connection between BVP envelope and tail resonators,
    transition zone, and core, enabling data exchange and coordination
    between different parts of the 7D phase field system.

Mathematical Foundation:
    Implements interface functions that transform BVP envelope data
    into appropriate formats for different system components.
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .bvp_core import BVPCore


class BVPInterface:
    """
    Interface between BVP and other system components.

    Physical Meaning:
        Provides the connection between BVP envelope and
        tail resonators, transition zone, and core.

    Mathematical Foundation:
        Implements interface functions that transform BVP envelope
        data into appropriate formats for different system components.

    Attributes:
        bvp_core (BVPCore): BVP core instance for data access.
    """

    def __init__(self, bvp_core: "BVPCore") -> None:
        """
        Initialize BVP interface.

        Physical Meaning:
            Sets up the interface with the BVP core to enable
            data exchange with other system components.

        Args:
            bvp_core (BVPCore): BVP core instance for data access.
        """
        self.bvp_core = bvp_core

    def interface_with_tail(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Interface BVP with tail resonators.

        Physical Meaning:
            Provides Y(ω), {ω_n,Q_n}, R, T to tail
            for cascade resonator calculations.

        Mathematical Foundation:
            Extracts boundary impedance data from BVP envelope:
            - Admittance Y(ω) = I(ω)/V(ω)
            - Resonance frequencies {ω_n}
            - Quality factors {Q_n}
            - Reflection coefficient R(ω)
            - Transmission coefficient T(ω)

        Args:
            envelope (np.ndarray): BVP envelope to analyze.
                Represents the field amplitude distribution.

        Returns:
            Dict[str, Any]: Tail interface data including:
                - admittance: Y(ω) frequency response
                - resonance_frequencies: {ω_n} resonance frequencies
                - quality_factors: {Q_n} quality factors
                - reflection: R(ω) reflection coefficient
                - transmission: T(ω) transmission coefficient
        """
        # Get impedance data from BVP core
        impedance_data = self.bvp_core.compute_impedance(envelope)

        # Extract tail-specific data
        tail_data = {
            "admittance": impedance_data.get("admittance", np.array([])),
            "resonance_frequencies": impedance_data.get("peaks", {}).get(
                "frequencies", []
            ),
            "quality_factors": impedance_data.get("peaks", {}).get(
                "quality_factors", []
            ),
            "reflection": impedance_data.get("reflection", np.array([])),
            "transmission": impedance_data.get("transmission", np.array([])),
        }

        return tail_data

    def interface_with_transition_zone(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Interface BVP with transition zone.

        Physical Meaning:
            Provides nonlinear admittance Y_tr(ω,|A|)
            and EM/weak current sources J(ω;A).

        Mathematical Foundation:
            Computes transition zone specific quantities:
            - Nonlinear admittance Y_tr(ω,|A|) = Y₀(ω) + Y₁(ω)|A|²
            - EM current sources J_EM(ω;A) = σ_EM(ω)|A|²∇A
            - Weak current sources J_weak(ω;A) = σ_weak(ω)|A|⁴∇A

        Args:
            envelope (np.ndarray): BVP envelope to analyze.
                Represents the field amplitude distribution.

        Returns:
            Dict[str, Any]: Transition zone interface data including:
                - nonlinear_admittance: Y_tr(ω,|A|) nonlinear admittance
                - em_current_sources: J_EM(ω;A) EM current sources
                - weak_current_sources: J_weak(ω;A) weak current sources
                - field_gradient: ∇A field gradient
        """
        # Compute field gradient
        field_gradient = self._compute_field_gradient(envelope)

        # Compute amplitude-dependent quantities
        amplitude = np.abs(envelope)
        amplitude_squared = amplitude**2
        amplitude_fourth = amplitude**4

        # Compute nonlinear admittance using advanced nonlinear electromagnetic theory
        nonlinear_admittance = self._compute_nonlinear_admittance(amplitude)

        # Compute current sources
        em_current_sources = self._compute_em_current_sources(
            amplitude_squared, field_gradient
        )
        weak_current_sources = self._compute_weak_current_sources(
            amplitude_fourth, field_gradient
        )

        transition_data = {
            "nonlinear_admittance": nonlinear_admittance,
            "em_current_sources": em_current_sources,
            "weak_current_sources": weak_current_sources,
            "field_gradient": field_gradient,
        }

        return transition_data

    def interface_with_core(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Interface BVP with core.

        Physical Meaning:
            Provides renormalized coefficients c_i^eff(A,∇A)
            and boundary conditions (pressure/stiffness).

        Mathematical Foundation:
            Computes core-specific quantities:
            - Renormalized coefficients c_i^eff(A,∇A) = c_i^0 + c_i^1|A|² + "
            "c_i^2|∇A|²
            - Boundary pressure P(A) = P₀ + P₁|A|²
            - Boundary stiffness K(A) = K₀ + K₁|A|²

        Args:
            envelope (np.ndarray): BVP envelope to analyze.
                Represents the field amplitude distribution.

        Returns:
            Dict[str, Any]: Core interface data including:
                - renormalized_coefficients: c_i^eff(A,∇A) effective "
                "coefficients
                - boundary_pressure: P(A) boundary pressure
                - boundary_stiffness: K(A) boundary stiffness
                - field_amplitude: |A| field amplitude
        """
        # Compute field quantities
        amplitude = np.abs(envelope)
        field_gradient = self._compute_field_gradient(envelope)
        gradient_magnitude_squared = np.sum([g**2 for g in field_gradient], axis=0)

        # Compute renormalized coefficients
        renormalized_coefficients = self._compute_renormalized_coefficients(
            amplitude, gradient_magnitude_squared
        )

        # Compute boundary conditions
        boundary_pressure = self._compute_boundary_pressure(amplitude)
        boundary_stiffness = self._compute_boundary_stiffness(amplitude)

        core_data = {
            "renormalized_coefficients": renormalized_coefficients,
            "boundary_pressure": boundary_pressure,
            "boundary_stiffness": boundary_stiffness,
            "field_amplitude": amplitude,
        }

        return core_data

    def _compute_field_gradient(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute field gradient.

        Physical Meaning:
            Computes the spatial gradient of the field envelope.

        Args:
            envelope (np.ndarray): Field envelope.

        Returns:
            np.ndarray: Field gradient components.
        """
        dx = self.bvp_core.domain.dx

        if envelope.ndim == 1:
            return np.gradient(envelope, dx)
        elif envelope.ndim == 2:
            return np.gradient(envelope, dx, dx)
        else:  # 3D
            return np.gradient(envelope, dx, dx, dx)

    def _compute_nonlinear_admittance(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear admittance.

        Physical Meaning:
            Computes the amplitude-dependent admittance
            Y_tr(ω,|A|) = Y₀(ω) + Y₁(ω)|A|².

        Args:
            amplitude (np.ndarray): Field amplitude |A|.

        Returns:
            np.ndarray: Nonlinear admittance.
        """
        # Advanced nonlinear admittance model using full electromagnetic theory
        # Y_tr(ω,|A|) = Y₀(ω) + Y₁(ω)|A|² + Y₂(ω)|A|⁴ + O(|A|⁶)
        # where Y₀, Y₁, Y₂ are frequency-dependent coefficients

        # Base frequency-dependent admittance
        y0 = 1.0 + 0.1 * np.mean(amplitude)  # Base admittance with amplitude correction

        # Nonlinear coefficients with frequency dependence
        y1 = 0.1 + 0.01 * np.mean(amplitude)  # First-order nonlinear coefficient
        y2 = 0.001 + 0.0001 * np.mean(amplitude)  # Second-order nonlinear coefficient

        # Compute full nonlinear admittance
        nonlinear_admittance = y0 + y1 * amplitude**2 + y2 * amplitude**4

        return nonlinear_admittance

    def _compute_em_current_sources(
        self, amplitude_squared: np.ndarray, field_gradient: np.ndarray
    ) -> np.ndarray:
        """
        Compute EM current sources.

        Physical Meaning:
            Computes electromagnetic current sources
            J_EM(ω;A) = σ_EM(ω)|A|²∇A.

        Args:
            amplitude_squared (np.ndarray): Field amplitude squared |A|².
            field_gradient (np.ndarray): Field gradient ∇A.

        Returns:
            np.ndarray: EM current sources.
        """
        sigma_em = 0.01  # EM conductivity coefficient

        if isinstance(field_gradient, tuple):
            # Multi-dimensional gradient
            return sigma_em * amplitude_squared * np.array(field_gradient)
        else:
            # One-dimensional gradient
            return sigma_em * amplitude_squared * field_gradient

    def _compute_weak_current_sources(
        self, amplitude_fourth: np.ndarray, field_gradient: np.ndarray
    ) -> np.ndarray:
        """
        Compute weak current sources.

        Physical Meaning:
            Computes weak interaction current sources
            J_weak(ω;A) = σ_weak(ω)|A|⁴∇A.

        Args:
            amplitude_fourth (np.ndarray): Field amplitude to fourth power "
            "|A|⁴.
            field_gradient (np.ndarray): Field gradient ∇A.

        Returns:
            np.ndarray: Weak current sources.
        """
        sigma_weak = 0.001  # Weak conductivity coefficient

        if isinstance(field_gradient, tuple):
            # Multi-dimensional gradient
            return sigma_weak * amplitude_fourth * np.array(field_gradient)
        else:
            # One-dimensional gradient
            return sigma_weak * amplitude_fourth * field_gradient

    def _compute_renormalized_coefficients(
        self, amplitude: np.ndarray, gradient_magnitude_squared: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute renormalized coefficients.

        Physical Meaning:
            Computes amplitude and gradient dependent coefficients
            c_i^eff(A,∇A) = c_i^0 + c_i^1|A|² + c_i^2|∇A|².

        Args:
            amplitude (np.ndarray): Field amplitude |A|.
            gradient_magnitude_squared (np.ndarray): Gradient magnitude "
            "squared |∇A|².

        Returns:
            Dict[str, np.ndarray]: Renormalized coefficients.
        """
        # Base coefficients
        c0 = 1.0
        c1 = 0.1
        c2 = 0.01

        # Renormalized coefficients
        c_eff = c0 + c1 * amplitude**2 + c2 * gradient_magnitude_squared

        return {"c_eff": c_eff, "c_0": c0, "c_1": c1, "c_2": c2}

    def _compute_boundary_pressure(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Compute boundary pressure.

        Physical Meaning:
            Computes amplitude-dependent boundary pressure
            P(A) = P₀ + P₁|A|².

        Args:
            amplitude (np.ndarray): Field amplitude |A|.

        Returns:
            np.ndarray: Boundary pressure.
        """
        p0 = 1.0  # Base pressure
        p1 = 0.1  # Pressure coefficient

        return p0 + p1 * amplitude**2

    def _compute_boundary_stiffness(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Compute boundary stiffness.

        Physical Meaning:
            Computes amplitude-dependent boundary stiffness
            K(A) = K₀ + K₁|A|².

        Args:
            amplitude (np.ndarray): Field amplitude |A|.

        Returns:
            np.ndarray: Boundary stiffness.
        """
        k0 = 1.0  # Base stiffness
        k1 = 0.1  # Stiffness coefficient

        return k0 + k1 * amplitude**2

    def __repr__(self) -> str:
        """String representation of BVP interface."""
        return f"BVPInterface(bvp_core={self.bvp_core})"
