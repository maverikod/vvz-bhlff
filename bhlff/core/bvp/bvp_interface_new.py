"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Interface implementation according to step 00 specification.

This module implements the interface between BVP and other system
components, providing connections to tail resonators, transition zone,
and core according to the 7D phase field theory.

Theoretical Background:
    The BVP interface serves as the connection point between the BVP
    envelope and other system components. It provides the necessary
    data transformations and interface functions for:
    - Tail resonators: Y(ω), {ω_n,Q_n}, R, T
    - Transition zone: Y_tr(ω,|A|), J_EM(ω;A)
    - Core: c_i^eff(A,∇A), boundary conditions

Example:
    >>> interface = BVPInterface(bvp_core)
    >>> tail_data = interface.interface_with_tail(envelope)
    >>> transition_data = interface.interface_with_transition_zone(envelope)
    >>> core_data = interface.interface_with_core(envelope)
"""

import numpy as np
from typing import Dict, Any, List

from .bvp_core_new import BVPCore
from ..domain.domain_7d import Domain7D


class BVPInterface:
    """
    Interface between BVP and other system components.

    Physical Meaning:
        Provides the connection between BVP envelope and
        tail resonators, transition zone, and core. This interface
        implements the data transformations required for integrating
        BVP with other system components according to the 7D theory.

    Mathematical Foundation:
        Implements interface functions for:
        1. Tail interface: Provides Y(ω), {ω_n,Q_n}, R, T
        2. Transition zone interface: Provides Y_tr(ω,|A|), J_EM(ω;A)
        3. Core interface: Provides c_i^eff(A,∇A), boundary conditions
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize BVP interface.

        Physical Meaning:
            Sets up the interface with the BVP core module,
            establishing connections to all system components.

        Args:
            bvp_core (BVPCore): BVP core module instance.
        """
        self.bvp_core = bvp_core
        self.domain_7d = bvp_core.domain_7d
        self.config = bvp_core.config

    def interface_with_tail(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Interface BVP with tail resonators.

        Physical Meaning:
            Provides the necessary data for tail resonator calculations:
            - Admittance Y(ω) for cascade resonator calculations
            - Resonance peaks {ω_n,Q_n} for resonator chain analysis
            - Reflection R(ω) and transmission T(ω) coefficients
            - Spectral data S(ω) inherited from BVP

        Mathematical Foundation:
            Computes boundary functions from BVP envelope:
            - Y(ω) = I(ω)/V(ω) - admittance response
            - {ω_n, Q_n} - resonance frequencies and quality factors
            - R(ω), T(ω) - reflection and transmission coefficients
            - S(ω) - spectral data for cascade calculations

        Args:
            envelope (np.ndarray): 7D envelope field at boundaries.

        Returns:
            Dict[str, Any]: Tail interface data including:
                - admittance (np.ndarray): Y(ω) frequency response
                - resonance_peaks (List[Dict]): {ω_n, Q_n} resonance data
                - reflection_coefficient (np.ndarray): R(ω) reflection
                - transmission_coefficient (np.ndarray): T(ω) transmission
                - spectral_data (np.ndarray): S(ω) spectral information
        """
        # Compute impedance data from BVP envelope
        impedance_data = self.bvp_core.compute_impedance(envelope)

        # Extract tail-specific data
        tail_data = {
            "admittance": impedance_data["admittance"],
            "resonance_peaks": impedance_data["resonance_peaks"],
            "reflection_coefficient": impedance_data["reflection_coefficient"],
            "transmission_coefficient": impedance_data["transmission_coefficient"],
            "spectral_data": self._compute_spectral_data(envelope),
        }

        return tail_data

    def interface_with_transition_zone(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Interface BVP with transition zone.

        Physical Meaning:
            Provides the necessary data for transition zone calculations:
            - Nonlinear admittance Y_tr(ω,|A|) for transition zone analysis
            - EM/weak current sources J_EM(ω;A) generated from envelope
            - Loss map χ''(|A|) for quench analysis
            - Input admittance Y_in from tail interface

        Mathematical Foundation:
            Computes transition zone interface functions:
            - Y_tr(ω,|A|) = Y_0(ω) + Y_nl(|A|) - nonlinear admittance
            - J_EM(ω;A) = f(A,∇A) - EM current sources from envelope
            - χ''(|A|) = χ''_0 + χ''_nl(|A|) - loss map with quenches

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            Dict[str, Any]: Transition zone interface data including:
                - nonlinear_admittance (np.ndarray): Y_tr(ω,|A|) response
                - em_current_sources (np.ndarray): J_EM(ω;A) current sources
                - loss_map (np.ndarray): χ''(|A|) loss distribution
                - input_admittance (np.ndarray): Y_in from tail
        """
        # Compute nonlinear admittance
        nonlinear_admittance = self._compute_nonlinear_admittance(envelope)

        # Compute EM current sources
        em_current_sources = self._compute_em_current_sources(envelope)

        # Compute loss map
        loss_map = self._compute_loss_map(envelope)

        # Get input admittance from tail interface
        tail_data = self.interface_with_tail(envelope)
        input_admittance = tail_data["admittance"]

        transition_data = {
            "nonlinear_admittance": nonlinear_admittance,
            "em_current_sources": em_current_sources,
            "loss_map": loss_map,
            "input_admittance": input_admittance,
        }

        return transition_data

    def interface_with_core(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Interface BVP with core.

        Physical Meaning:
            Provides the necessary data for core calculations:
            - Renormalized coefficients c_i^eff(A,∇A) from BVP averaging
            - Boundary conditions (pressure/stiffness) from BVP field
            - Core energy density and gradients
            - Effective parameters for core evolution

        Mathematical Foundation:
            Computes core interface functions through BVP averaging:
            - c_i^eff = c_i + α_i|A|² + β_i|∇A|²/ω₀² + ...
            - Boundary pressure: P_boundary = f(|A|,|∇A|)
            - Core stiffness: K_core = f(|A|,|∇A|)

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            Dict[str, Any]: Core interface data including:
                - renormalized_coefficients (Dict): c_i^eff(A,∇A) coefficients
                - boundary_pressure (np.ndarray): P_boundary pressure
                - core_stiffness (np.ndarray): K_core stiffness
                - energy_density (np.ndarray): Core energy density
                - effective_parameters (Dict): Effective core parameters
        """
        # Compute renormalized coefficients
        renormalized_coefficients = self._compute_renormalized_coefficients(envelope)

        # Compute boundary conditions
        boundary_pressure = self._compute_boundary_pressure(envelope)
        core_stiffness = self._compute_core_stiffness(envelope)

        # Compute energy density
        energy_density = self._compute_core_energy_density(envelope)

        # Compute effective parameters
        effective_parameters = self._compute_effective_parameters(envelope)

        core_data = {
            "renormalized_coefficients": renormalized_coefficients,
            "boundary_pressure": boundary_pressure,
            "core_stiffness": core_stiffness,
            "energy_density": energy_density,
            "effective_parameters": effective_parameters,
        }

        return core_data

    def _compute_spectral_data(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute spectral data S(ω) from BVP envelope.

        Physical Meaning:
            Computes the spectral data S(ω) that represents the
            frequency content of the BVP envelope for cascade
            resonator calculations.

        Returns:
            np.ndarray: Spectral data S(ω).
        """
        # Compute FFT of envelope in time dimension
        spectral_data = np.fft.fft(envelope, axis=-1)
        return spectral_data

    def _compute_nonlinear_admittance(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear admittance Y_tr(ω,|A|).

        Physical Meaning:
            Computes the nonlinear admittance that depends on both
            frequency and envelope amplitude, representing the
            transition zone response.

        Returns:
            np.ndarray: Nonlinear admittance Y_tr(ω,|A|).
        """
        # Base admittance
        base_admittance = np.ones_like(envelope)

        # Nonlinear correction based on amplitude
        amplitude = np.abs(envelope)
        nonlinear_correction = 1.0 + 0.1 * amplitude**2

        return base_admittance * nonlinear_correction

    def _compute_em_current_sources(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute EM current sources J_EM(ω;A).

        Physical Meaning:
            Computes the electromagnetic current sources generated
            from the BVP envelope, representing the coupling
            between BVP and EM fields.

        Returns:
            np.ndarray: EM current sources J_EM(ω;A).
        """
        # Current sources proportional to envelope amplitude and gradient
        amplitude = np.abs(envelope)

        # Compute gradient magnitude
        differentials = self.domain_7d.get_differentials()
        grad_x = np.gradient(envelope, differentials["dx"], axis=0)
        grad_y = np.gradient(envelope, differentials["dy"], axis=1)
        grad_z = np.gradient(envelope, differentials["dz"], axis=2)

        grad_magnitude = np.sqrt(
            np.abs(grad_x) ** 2 + np.abs(grad_y) ** 2 + np.abs(grad_z) ** 2
        )

        # Current sources
        current_sources = amplitude * grad_magnitude

        return current_sources

    def _compute_loss_map(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute loss map χ''(|A|).

        Physical Meaning:
            Computes the loss map that shows how losses depend
            on the envelope amplitude, including quench effects.

        Returns:
            np.ndarray: Loss map χ''(|A|).
        """
        # Base losses
        base_losses = 0.01 * np.ones_like(envelope)

        # Nonlinear losses (quenches)
        amplitude = np.abs(envelope)
        nonlinear_losses = 0.1 * amplitude**2

        return base_losses + nonlinear_losses

    def _compute_renormalized_coefficients(
        self, envelope: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute renormalized coefficients c_i^eff(A,∇A).

        Physical Meaning:
            Computes the renormalized coefficients that result
            from BVP averaging over the high-frequency carrier.

        Returns:
            Dict[str, float]: Renormalized coefficients.
        """
        amplitude = np.abs(envelope)
        mean_amplitude = np.mean(amplitude)

        # Renormalized coefficients
        coefficients = {
            "c2_eff": 1.0 + 0.1 * mean_amplitude**2,
            "c4_eff": 0.1 + 0.01 * mean_amplitude**2,
            "c6_eff": 0.01 + 0.001 * mean_amplitude**2,
        }

        return coefficients

    def _compute_boundary_pressure(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute boundary pressure P_boundary.

        Physical Meaning:
            Computes the boundary pressure that results from
            the BVP field at the boundaries.

        Returns:
            np.ndarray: Boundary pressure.
        """
        amplitude = np.abs(envelope)
        return 0.5 * amplitude**2

    def _compute_core_stiffness(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute core stiffness K_core.

        Physical Meaning:
            Computes the core stiffness that results from
            the BVP field interaction with the core.

        Returns:
            np.ndarray: Core stiffness.
        """
        amplitude = np.abs(envelope)
        return 1.0 + 0.2 * amplitude**2

    def _compute_core_energy_density(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute core energy density.

        Physical Meaning:
            Computes the energy density in the core region
            resulting from the BVP field.

        Returns:
            np.ndarray: Core energy density.
        """
        amplitude = np.abs(envelope)
        return 0.5 * amplitude**2

    def _compute_effective_parameters(self, envelope: np.ndarray) -> Dict[str, float]:
        """
        Compute effective core parameters.

        Physical Meaning:
            Computes the effective parameters for core evolution
            that result from BVP averaging.

        Returns:
            Dict[str, float]: Effective parameters.
        """
        amplitude = np.abs(envelope)
        mean_amplitude = np.mean(amplitude)

        parameters = {
            "effective_mass": 1.0 + 0.1 * mean_amplitude**2,
            "effective_damping": 0.01 + 0.001 * mean_amplitude**2,
            "effective_coupling": 0.1 + 0.01 * mean_amplitude**2,
        }

        return parameters
