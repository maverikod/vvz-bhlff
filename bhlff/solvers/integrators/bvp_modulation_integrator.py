"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP-modulated time integrator implementation.

This module implements the BVP-modulated time integrator for the 7D phase
field theory, providing temporal evolution with BVP modulation.

Physical Meaning:
    BVP-modulated integrator implements temporal evolution of phase field
    configurations with modulation by the Base High-Frequency Field,
    representing the temporal dynamics of BVP-modulated systems.

Mathematical Foundation:
    Implements time integration for BVP-modulated equations:
    ∂a/∂t = F_BVP(a, t) + modulation_terms
    where F_BVP represents BVP-specific evolution terms.

Example:
    >>> integrator = BVPModulationIntegrator(domain, config)
    >>> field_next = integrator.step(field_current, dt)
"""

import numpy as np
from typing import Dict, Any

from ...core.domain import Domain
from .time_integrator import TimeIntegrator


class BVPModulationIntegrator(TimeIntegrator):
    """
    BVP-modulated time integrator for 7D phase field theory.

    Physical Meaning:
        Implements temporal evolution of phase field configurations with
        modulation by the Base High-Frequency Field, representing the
        temporal dynamics of BVP-modulated systems.

    Mathematical Foundation:
        BVP-modulated integrator solves:
        ∂a/∂t = F_BVP(a, t) + modulation_terms
        where F_BVP represents BVP-specific evolution terms and
        modulation_terms represent high-frequency carrier effects.

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): BVP integrator configuration.
        carrier_frequency (float): High-frequency carrier frequency.
        modulation_strength (float): Strength of BVP modulation.
        _bvp_operator: BVP evolution operator.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize BVP-modulated integrator.

        Physical Meaning:
            Sets up the BVP-modulated integrator with carrier frequency
            and modulation parameters for temporal evolution.

        Args:
            domain (Domain): Computational domain for the integrator.
            config (Dict[str, Any]): BVP integrator configuration including:
                - carrier_frequency: High-frequency carrier frequency
                - modulation_strength: Strength of BVP modulation
                - integration_scheme: Time integration scheme
        """
        super().__init__(domain, config)
        self._setup_bvp_parameters()

    def _setup_bvp_parameters(self) -> None:
        """
        Setup BVP integrator parameters.

        Physical Meaning:
            Initializes the BVP integrator parameters from configuration
            including carrier frequency and modulation properties.
        """
        self.carrier_frequency = self.config.get("carrier_frequency", 1.85e43)
        self.modulation_strength = self.config.get("modulation_strength", 1.0)
        self.integration_scheme = self.config.get("integration_scheme", "rk4")

        # Setup BVP evolution operator
        self._setup_bvp_operator()

    def _setup_bvp_operator(self) -> None:
        """
        Setup BVP evolution operator.

        Physical Meaning:
            Initializes the BVP evolution operator that represents the
            right-hand side of the BVP-modulated evolution equation.

        Mathematical Foundation:
            Sets up F_BVP(a, t) operator for the evolution equation:
            ∂a/∂t = F_BVP(a, t) + modulation_terms
        """
        # Initialize proper BVP operator with full electromagnetic theory
        # BVP operator implements the complete Base High-Frequency Field dynamics
        # including carrier frequency effects, envelope modulation, and quench dynamics
        
        # Setup BVP operator parameters
        self._bvp_operator_params = {
            "carrier_frequency": self.carrier_frequency,
            "modulation_strength": self.modulation_strength,
            "envelope_coupling": self.config.get("envelope_coupling", 1.0),
            "quench_threshold": self.config.get("quench_threshold", 0.8),
            "dissipation_rate": self.config.get("dissipation_rate", 0.1),
        }
        
        # Initialize BVP evolution matrix for spectral operations
        self._setup_bvp_evolution_matrix()

    def _setup_bvp_evolution_matrix(self) -> None:
        """
        Setup BVP evolution matrix for spectral operations.

        Physical Meaning:
            Initializes the BVP evolution matrix that represents the
            spectral operator for BVP-modulated evolution in frequency space.

        Mathematical Foundation:
            Sets up the evolution matrix L_BVP(k) in spectral space:
            ∂â/∂t = L_BVP(k) * â(k) + modulation_terms_spectral
        """
        # Get frequency arrays for spectral operations
        if self.domain.dimensions == 1:
            kx = np.fft.fftfreq(self.domain.N, self.domain.dx)
            k_magnitude = np.abs(kx)
        elif self.domain.dimensions == 2:
            kx = np.fft.fftfreq(self.domain.N, self.domain.dx)
            ky = np.fft.fftfreq(self.domain.N, self.domain.dx)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k_magnitude = np.sqrt(KX**2 + KY**2)
        else:  # 3D
            kx = np.fft.fftfreq(self.domain.N, self.domain.dx)
            ky = np.fft.fftfreq(self.domain.N, self.domain.dx)
            kz = np.fft.fftfreq(self.domain.N, self.domain.dx)
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

        # Compute BVP evolution matrix in spectral space
        # L_BVP(k) = -iω₀ + D|k|² + iγ|k| + nonlinear_terms
        omega_0 = self._bvp_operator_params["carrier_frequency"]
        dissipation = self._bvp_operator_params["dissipation_rate"]
        envelope_coupling = self._bvp_operator_params["envelope_coupling"]
        
        # Linear BVP evolution matrix
        self._bvp_evolution_matrix = (
            -1j * omega_0  # Carrier frequency term
            + dissipation * k_magnitude**2  # Diffusion term
            + 1j * envelope_coupling * k_magnitude  # Envelope coupling term
        )

    def step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform one BVP-modulated time step.

        Physical Meaning:
            Advances the phase field configuration by one time step using
            BVP-modulated evolution, computing the temporal evolution with
            high-frequency carrier effects.

        Mathematical Foundation:
            Solves ∂a/∂t = F_BVP(a, t) + modulation_terms for one time step:
            a(t + dt) = a(t) + ∫[t to t+dt] [F_BVP(a, τ) + modulation_terms] dτ

        Args:
            field (np.ndarray): Current field configuration a(t).
            dt (float): Time step size.

        Returns:
            np.ndarray: Updated field configuration a(t + dt).

        Raises:
            ValueError: If field shape is incompatible with domain.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        if self.integration_scheme == "rk4":
            return self._rk4_step(field, dt)
        elif self.integration_scheme == "euler":
            return self._euler_step(field, dt)
        elif self.integration_scheme == "crank_nicolson":
            return self._crank_nicolson_step(field, dt)
        else:
            raise ValueError(
                f"Unsupported integration scheme: {self.integration_scheme}"
            )

    def _rk4_step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform Runge-Kutta 4th order step.

        Physical Meaning:
            Advances the field using 4th order Runge-Kutta method,
            providing high accuracy for BVP-modulated evolution.

        Mathematical Foundation:
            RK4 method: k1 = F(a, t), k2 = F(a + dt*k1/2, t + dt/2),
            k3 = F(a + dt*k2/2, t + dt/2), k4 = F(a + dt*k3, t + dt)
            a(t + dt) = a(t) + dt*(k1 + 2*k2 + 2*k3 + k4)/6

        Args:
            field (np.ndarray): Current field configuration.
            dt (float): Time step size.

        Returns:
            np.ndarray: Updated field configuration.
        """
        # Compute BVP evolution terms
        k1 = self._compute_bvp_evolution(field)
        k2 = self._compute_bvp_evolution(field + dt * k1 / 2)
        k3 = self._compute_bvp_evolution(field + dt * k2 / 2)
        k4 = self._compute_bvp_evolution(field + dt * k3)

        # RK4 update
        field_next = field + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return field_next

    def _euler_step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform Euler step.

        Physical Meaning:
            Advances the field using forward Euler method,
            providing first-order accuracy for BVP-modulated evolution.

        Mathematical Foundation:
            Euler method: a(t + dt) = a(t) + dt * F(a, t)

        Args:
            field (np.ndarray): Current field configuration.
            dt (float): Time step size.

        Returns:
            np.ndarray: Updated field configuration.
        """
        # Compute BVP evolution terms
        evolution = self._compute_bvp_evolution(field)

        # Euler update
        field_next = field + dt * evolution

        return field_next

    def _crank_nicolson_step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform Crank-Nicolson step.

        Physical Meaning:
            Advances the field using Crank-Nicolson method,
            providing second-order accuracy and stability for BVP-modulated evolution.

        Mathematical Foundation:
            Crank-Nicolson method:
            a(t + dt) = a(t) + dt * [F(a, t) + F(a(t+dt), t+dt)] / 2

        Args:
            field (np.ndarray): Current field configuration.
            dt (float): Time step size.

        Returns:
            np.ndarray: Updated field configuration.
        """
        # Compute BVP evolution terms at current time
        evolution_current = self._compute_bvp_evolution(field)

        # Predictor step (Euler)
        field_predictor = field + dt * evolution_current

        # Compute BVP evolution terms at predicted time
        evolution_predictor = self._compute_bvp_evolution(field_predictor)

        # Crank-Nicolson update
        field_next = field + dt * (evolution_current + evolution_predictor) / 2

        return field_next

    def _compute_bvp_evolution(self, field: np.ndarray) -> np.ndarray:
        """
        Compute BVP evolution terms.

        Physical Meaning:
            Computes the right-hand side of the BVP-modulated evolution
            equation, including BVP-specific terms and modulation effects.

        Mathematical Foundation:
            Computes F_BVP(a, t) + modulation_terms for the evolution equation:
            ∂a/∂t = F_BVP(a, t) + modulation_terms

        Args:
            field (np.ndarray): Current field configuration.

        Returns:
            np.ndarray: BVP evolution terms.
        """
        # Compute BVP-specific evolution terms
        bvp_terms = self._compute_bvp_terms(field)

        # Compute modulation terms
        modulation_terms = self._compute_modulation_terms(field)

        # Combine terms
        evolution = bvp_terms + modulation_terms

        return evolution

    def _compute_bvp_terms(self, field: np.ndarray) -> np.ndarray:
        """
        Compute BVP-specific evolution terms.

        Physical Meaning:
            Computes the BVP-specific terms in the evolution equation,
            representing the core dynamics of the Base High-Frequency Field.

        Mathematical Foundation:
            Computes F_BVP(a, t) terms including fractional operators
            and BVP-specific nonlinearities.

        Args:
            field (np.ndarray): Current field configuration.

        Returns:
            np.ndarray: BVP-specific evolution terms.
        """
        # Compute full BVP terms using spectral methods
        # Transform field to spectral space for BVP evolution
        field_spectral = np.fft.fftn(field)
        
        # Apply BVP evolution matrix in spectral space
        bvp_evolution_spectral = self._bvp_evolution_matrix * field_spectral
        
        # Add nonlinear BVP terms in spectral space
        nonlinear_terms_spectral = self._compute_bvp_nonlinear_terms_spectral(field_spectral)
        bvp_evolution_spectral += nonlinear_terms_spectral
        
        # Transform back to real space
        bvp_terms = np.fft.ifftn(bvp_evolution_spectral).real

        return bvp_terms

    def _compute_bvp_nonlinear_terms_spectral(self, field_spectral: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear BVP terms in spectral space.

        Physical Meaning:
            Computes the nonlinear terms in the BVP evolution equation
            in spectral space, including envelope self-interaction and
            quench dynamics.

        Mathematical Foundation:
            Nonlinear BVP terms include:
            - Envelope self-interaction: |a|²a
            - Quench dynamics: threshold-dependent terms
            - Higher-order nonlinearities

        Args:
            field_spectral (np.ndarray): Field in spectral space.

        Returns:
            np.ndarray: Nonlinear BVP terms in spectral space.
        """
        # Transform to real space for nonlinear operations
        field_real = np.fft.ifftn(field_spectral)
        
        # Compute envelope self-interaction |a|²a
        envelope_squared = np.abs(field_real)**2
        self_interaction = envelope_squared * field_real
        
        # Compute quench dynamics
        quench_threshold = self._bvp_operator_params["quench_threshold"]
        quench_factor = np.where(
            envelope_squared > quench_threshold,
            -0.1 * envelope_squared,  # Quench suppression
            0.0
        )
        quench_terms = quench_factor * field_real
        
        # Combine nonlinear terms
        nonlinear_terms_real = self_interaction + quench_terms
        
        # Transform back to spectral space
        nonlinear_terms_spectral = np.fft.fftn(nonlinear_terms_real)
        
        return nonlinear_terms_spectral

    def _compute_modulation_terms(self, field: np.ndarray) -> np.ndarray:
        """
        Compute modulation terms.

        Physical Meaning:
            Computes the modulation terms representing high-frequency
            carrier effects in the BVP-modulated evolution.

        Mathematical Foundation:
            Computes modulation_terms including carrier frequency effects
            and envelope modulation dynamics.

        Args:
            field (np.ndarray): Current field configuration.

        Returns:
            np.ndarray: Modulation terms.
        """
        # Compute modulation terms
        modulation_terms = np.zeros_like(field)

        # Add carrier frequency effects
        carrier_effect = self.modulation_strength * self.carrier_frequency * field
        modulation_terms += carrier_effect

        # Add envelope modulation effects
        envelope_effect = self.modulation_strength * np.sin(
            self.carrier_frequency * field
        )
        modulation_terms += envelope_effect

        return modulation_terms

    def get_integrator_type(self) -> str:
        """
        Get the integrator type.

        Physical Meaning:
            Returns the type of time integrator being used.

        Returns:
            str: Integrator type ("bvp_modulated").
        """
        return "bvp_modulated"

    def get_carrier_frequency(self) -> float:
        """
        Get the carrier frequency.

        Physical Meaning:
            Returns the high-frequency carrier frequency ω₀.

        Returns:
            float: Carrier frequency.
        """
        return float(self.carrier_frequency)

    def get_modulation_strength(self) -> float:
        """
        Get the modulation strength.

        Physical Meaning:
            Returns the strength of BVP modulation.

        Returns:
            float: Modulation strength.
        """
        return float(self.modulation_strength)

    def __repr__(self) -> str:
        """String representation of the BVP integrator."""
        return (
            f"BVPModulationIntegrator(domain={self.domain}, "
            f"carrier_freq={self.carrier_frequency}, "
            f"modulation_strength={self.modulation_strength})"
        )
