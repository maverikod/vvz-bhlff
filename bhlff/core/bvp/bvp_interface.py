"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP interface facade for system component integration.

This module provides a unified interface between the Base High-Frequency
Field (BVP) and other system components by combining core and advanced
interface operations.

Physical Meaning:
    Provides the connection between BVP envelope and tail resonators,
    transition zone, and core, enabling data exchange and coordination
    between different parts of the 7D phase field system.

Mathematical Foundation:
    Combines core and advanced interface functions that transform BVP
    envelope data into appropriate formats for different system components.

Example:
    >>> interface = BVPInterface(bvp_core)
    >>> tail_data = interface.interface_with_tail(envelope)
    >>> transition_data = interface.interface_with_transition_zone(envelope)
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING, Optional

from .bvp_constants import BVPConstants
from .bvp_interface_core import BVPInterfaceCore
from .bvp_interface_advanced import BVPInterfaceAdvanced

if TYPE_CHECKING:
    from .bvp_core import BVPCore


class BVPInterface:
    """
    Unified interface between BVP and other system components.

    Physical Meaning:
        Provides the connection between BVP envelope and
        tail resonators, transition zone, and core by combining
        core and advanced interface operations.

    Mathematical Foundation:
        Combines core and advanced interface functions that transform BVP
        envelope data into appropriate formats for different system components.

    Attributes:
        bvp_core (BVPCore): BVP core instance for data access.
        constants (BVPConstants): BVP constants instance.
        _interface_core (BVPInterfaceCore): Core interface operations.
        _interface_advanced (BVPInterfaceAdvanced): Advanced interface operations.
    """

    def __init__(
        self, bvp_core: "BVPCore", constants: Optional[BVPConstants] = None
    ) -> None:
        """
        Initialize BVP interface.

        Physical Meaning:
            Sets up the interface with the BVP core and initializes
            core and advanced interface components.

        Args:
            bvp_core (BVPCore): BVP core instance for data access.
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.bvp_core = bvp_core
        self.constants = constants or bvp_core.constants

        # Initialize interface components
        self._interface_core = BVPInterfaceCore(bvp_core.domain, self.constants)
        self._interface_advanced = BVPInterfaceAdvanced(self.constants)

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
        # Compute field gradient using core interface
        field_gradient = self._interface_core.compute_field_gradient(envelope)

        # Compute amplitude-dependent quantities
        amplitude = self._interface_core.compute_field_amplitude(envelope)
        amplitude_squared = amplitude**2
        amplitude_fourth = amplitude**4

        # Compute nonlinear admittance using advanced interface
        nonlinear_admittance = self._interface_advanced.compute_nonlinear_admittance(
            amplitude
        )

        # Compute current sources using advanced interface
        em_current_sources = self._interface_advanced.compute_em_current_sources(
            amplitude_squared, field_gradient
        )
        weak_current_sources = self._interface_advanced.compute_weak_current_sources(
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
            - Renormalized coefficients c_i^eff(A,∇A) = c_i^0 + c_i^1|A|² + c_i^2|∇A|²
            - Boundary pressure P(A) = P₀ + P₁|A|²
            - Boundary stiffness K(A) = K₀ + K₁|A|²

        Args:
            envelope (np.ndarray): BVP envelope to analyze.
                Represents the field amplitude distribution.

        Returns:
            Dict[str, Any]: Core interface data including:
                - renormalized_coefficients: c_i^eff(A,∇A) effective coefficients
                - boundary_pressure: P(A) boundary pressure
                - boundary_stiffness: K(A) boundary stiffness
                - field_amplitude: |A| field amplitude
        """
        # Compute field quantities using core interface
        amplitude = self._interface_core.compute_field_amplitude(envelope)
        field_gradient = self._interface_core.compute_field_gradient(envelope)
        gradient_magnitude_squared = (
            self._interface_core.compute_gradient_magnitude_squared(field_gradient)
        )

        # Compute renormalized coefficients using advanced interface
        renormalized_coefficients = (
            self._interface_advanced.compute_renormalized_coefficients(
                amplitude, gradient_magnitude_squared
            )
        )

        # Compute boundary conditions using advanced interface
        boundary_pressure = self._interface_advanced.compute_boundary_pressure(
            amplitude
        )
        boundary_stiffness = self._interface_advanced.compute_boundary_stiffness(
            amplitude
        )

        core_data = {
            "renormalized_coefficients": renormalized_coefficients,
            "boundary_pressure": boundary_pressure,
            "boundary_stiffness": boundary_stiffness,
            "field_amplitude": amplitude,
        }

        return core_data

    def __repr__(self) -> str:
        """String representation of BVP interface."""
        return f"BVPInterface(bvp_core={self.bvp_core})"
