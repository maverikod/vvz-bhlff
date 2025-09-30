"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Main BVP core implementation.

This module implements the main BVPCore class that coordinates all
BVP operations and provides the central interface to the BVP framework.

Physical Meaning:
    The BVP core serves as the central backbone of the entire system, where
    all observed particles and fields are manifestations of envelope
    modulations and beatings of the high-frequency carrier field.

Mathematical Foundation:
    BVP implements the envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.

Example:
    >>> bvp_core = BVPCore(domain, config)
    >>> envelope = bvp_core.solve_envelope(source)
    >>> quenches = bvp_core.detect_quenches(envelope)
"""

import numpy as np
from typing import Dict, Any, Optional

from ...domain import Domain
from ...domain.domain_7d import Domain7D
from ..bvp_constants import BVPConstants
from .bvp_operations import BVPCoreOperations
from .bvp_7d_interface import BVPCore7DInterface


class BVPCore:
    """
    Base High-Frequency Field (BVP) core module.

    Physical Meaning:
        Implements the central framework of the 7D theory where
        all observed "modes" are envelope modulations and beatings
        of the Base High-Frequency Field (BVP). Serves as the central
        backbone of the entire system.

    Mathematical Foundation:
        BVP implements the envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x)
        where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
        χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.
    """

    def __init__(
        self, domain: Domain, config: Dict[str, Any], domain_7d: Domain7D = None
    ) -> None:
        """
        Initialize BVP core framework.

        Physical Meaning:
            Sets up the central BVP framework with computational domains
            and configuration parameters. Initializes all necessary
            components for BVP operations including envelope solving,
            quench detection, impedance computation, and 7D operations.

        Args:
            domain (Domain): Standard computational domain for BVP operations.
            config (Dict[str, Any]): Configuration parameters including:
                - carrier_frequency: BVP carrier frequency
                - kappa_0, kappa_2: Stiffness coefficients
                - chi_prime, chi_double_prime_0: Susceptibility coefficients
                - k0: Wave number
            domain_7d (Domain7D, optional): 7D computational domain for
                full space-time operations.
        """
        self.domain = domain
        self.config = config
        self.domain_7d = domain_7d

        # Initialize BVP constants
        self._bvp_constants = BVPConstants(config)

        # Initialize core operations
        self._operations = BVPCoreOperations(domain, config, domain_7d)

        # Initialize 7D interface if 7D domain is available
        self._7d_interface = None
        if domain_7d is not None:
            self._7d_interface = BVPCore7DInterface(domain_7d, config)

    def solve_envelope(self, source: np.ndarray) -> np.ndarray:
        """
        Solve BVP envelope equation for U(1)³ phase structure.

        Physical Meaning:
            Computes the envelope a(x,φ,t) of the Base High-Frequency Field
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ that modulates the high-frequency carrier.
            The envelope is a vector of three U(1) phase components Θ_a (a=1..3).

        Mathematical Foundation:
            Solves ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t) for the envelope a(x,φ,t)
            where a is a vector of three U(1) phase components in 7D space-time.

        Args:
            source (np.ndarray): Source term s(x,φ,t) in 7D space-time.
                Represents external excitations or initial conditions in M₇.

        Returns:
            np.ndarray: BVP envelope a(x,φ,t) in 7D space-time.
                Represents the envelope modulation of the high-frequency carrier
                as a vector of three U(1) phase components.

        Raises:
            ValueError: If source has incompatible shape with 7D domain.
        """
        return self._operations.solve_envelope(source)

    def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quench events when local thresholds are reached.

        Physical Meaning:
            Identifies when BVP dissipatively "dumps" energy into
            the medium at local thresholds (amplitude/detuning/gradient).

        Mathematical Foundation:
            Applies three threshold criteria:
            - amplitude: |A| > |A_q|
            - detuning: |ω - ω_0| > Δω_q
            - gradient: |∇A| > |∇A_q|

        Args:
            envelope (np.ndarray): BVP envelope a(x) to analyze.

        Returns:
            Dict[str, Any]: Quench detection results including:
                - quench_locations: Spatial locations of quenches
                - quench_types: Types of quenches detected
                - energy_dumped: Energy dumped at each quench
        """
        return self._operations.detect_quenches(envelope)

    def compute_impedance(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Compute impedance/admittance from BVP envelope.

        Physical Meaning:
            Calculates Y(ω), R(ω), T(ω), and peaks {ω_n,Q_n}
            from the BVP envelope at boundaries.

        Mathematical Foundation:
            Computes boundary functions from envelope:
            - Admittance Y(ω) = I(ω)/V(ω)
            - Reflection coefficient R(ω)
            - Transmission coefficient T(ω)
            - Resonance peaks {ω_n,Q_n}

        Args:
            envelope (np.ndarray): BVP envelope a(x) to analyze.

        Returns:
            Dict[str, Any]: Impedance analysis results including:
                - admittance: Admittance Y(ω)
                - reflection: Reflection coefficient R(ω)
                - transmission: Transmission coefficient T(ω)
                - resonance_peaks: Resonance peaks {ω_n,Q_n}
        """
        return self._operations.compute_impedance(envelope)

    def get_carrier_frequency(self) -> float:
        """
        Get BVP carrier frequency.

        Physical Meaning:
            Returns the high-frequency carrier frequency ω₀ of the BVP field,
            which is the fundamental frequency that all envelope modulations
            and beatings are based upon.

        Returns:
            float: BVP carrier frequency ω₀.
        """
        return self._bvp_constants.get_carrier_frequency()

    def solve_envelope_7d(self, source_7d: np.ndarray) -> np.ndarray:
        """
        Solve 7D BVP envelope equation.

        Physical Meaning:
            Solves the full 7D envelope equation in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
            using the 7D envelope equation solver.

        Args:
            source_7d (np.ndarray): 7D source term s(x,φ,t).

        Returns:
            np.ndarray: 7D envelope solution a(x,φ,t).

        Raises:
            RuntimeError: If 7D domain is not available.
        """
        if self._7d_interface is None:
            raise RuntimeError("7D domain not available for 7D envelope equation")

        return self._7d_interface.solve_envelope_7d(source_7d)

    def validate_postulates_7d(self, envelope_7d: np.ndarray) -> Dict[str, Any]:
        """
        Validate all 9 BVP postulates for 7D field.

        Physical Meaning:
            Validates all 9 BVP postulates to ensure the 7D field
            satisfies the fundamental properties of the BVP framework.

        Args:
            envelope_7d (np.ndarray): 7D BVP envelope field.

        Returns:
            Dict[str, Any]: Validation results from all postulates.

        Raises:
            RuntimeError: If 7D postulates are not available.
        """
        if self._7d_interface is None:
            raise RuntimeError("7D postulates not available")

        return self._7d_interface.validate_postulates_7d(envelope_7d)

    def get_7d_domain(self) -> Optional[Domain7D]:
        """
        Get the 7D domain.

        Returns:
            Optional[Domain7D]: The 7D space-time domain if available.
        """
        if self._7d_interface is None:
            return None
        return self._7d_interface.get_7d_domain()

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get BVP core parameters.

        Physical Meaning:
            Returns the current values of all BVP parameters for
            monitoring and analysis purposes.

        Returns:
            Dict[str, Any]: Dictionary containing all parameters.
        """
        params = self._bvp_constants.get_all_constants()

        if self._7d_interface is not None:
            params["7d_interface"] = self._7d_interface.get_7d_parameters()

        return params

    def __repr__(self) -> str:
        """
        String representation of BVP core.

        Returns:
            str: String representation showing domain and carrier frequency.
        """
        return (
            f"BVPCore(domain={self.domain}, "
            f"carrier_freq={self.get_carrier_frequency()}, "
            f"7d_available={self._7d_interface is not None})"
        )
