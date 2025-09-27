"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP (Base High-Frequency Field) core module.

This module implements the central framework of the 7D theory where all
observed "modes" are envelope modulations and beatings of the Base
High-Frequency Field (BVP).

Physical Meaning:
    BVP serves as the central backbone of the entire system, where all
    observed particles and fields are manifestations of envelope modulations
    and beatings of the high-frequency carrier field.

Mathematical Foundation:
    BVP implements the envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.

Example:
    >>> bvp_core = BVPCore(domain, config)
    >>> envelope = bvp_core.solve_envelope(source)
    >>> quenches = bvp_core.detect_quenches(envelope)
    >>> impedance = bvp_core.compute_impedance(envelope)
"""

import numpy as np
from typing import Dict, Any

from ..domain import Domain
from .quench_detector import QuenchDetector
from .bvp_envelope_solver import BVPEnvelopeSolver
from .bvp_impedance_calculator import BVPImpedanceCalculator


class BVPCore:
    """
    Base High-Frequency Field (BVP) core module.

    Physical Meaning:
        Implements the central framework of the 7D theory where
        all observed "modes" are envelope modulations and beatings
        of the Base High-Frequency Field (BVP).

    Mathematical Foundation:
        Solves the envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x)
        where κ(|a|) = κ₀ + κ₂|a|² and χ(|a|) = χ' + iχ''(|a|).

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): BVP configuration parameters.
        _envelope_solver (BVPEnvelopeSolver): Envelope equation solver.
        _quench_detector (QuenchDetector): Quench event detector.
        _impedance_calculator (BVPImpedanceCalculator): Impedance calculator.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize BVP core with configuration.

        Physical Meaning:
            Sets up the high-frequency carrier with envelope
            modulation capabilities and quench detection.

        Args:
            domain (Domain): Computational domain for BVP calculations.
            config (Dict[str, Any]): BVP configuration including:
                - carrier_frequency: High-frequency carrier frequency
                - envelope_equation: Parameters for envelope equation
                - quench_detection: Quench detection thresholds
                - impedance_calculation: Impedance calculation settings
        """
        self.domain = domain
        self.config = config
        self._setup_envelope_solver()
        self._setup_quench_detector()
        self._setup_impedance_calculator()

    def _setup_envelope_solver(self) -> None:
        """
        Setup envelope equation solver.

        Physical Meaning:
            Initializes the solver for the BVP envelope equation
            with nonlinear stiffness and susceptibility.
        """
        self._envelope_solver = BVPEnvelopeSolver(self.domain, self.config)

    def _setup_quench_detector(self) -> None:
        """
        Setup quench event detector.

        Physical Meaning:
            Initializes the detector for quench events when
            local thresholds are reached.
        """
        quench_config = self.config.get("quench_detection", {})
        self._quench_detector = QuenchDetector(quench_config)

    def _setup_impedance_calculator(self) -> None:
        """
        Setup impedance/admittance calculator.

        Physical Meaning:
            Initializes the calculator for Y(ω), R(ω), T(ω)
            and peaks {ω_n,Q_n} from BVP envelope.
        """
        self._impedance_calculator = BVPImpedanceCalculator(self.domain, self.config)

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
        return self._envelope_solver.solve_envelope(source)

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
        return self._quench_detector.detect_quenches(envelope)

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
                - admittance: Y(ω) frequency response
                - reflection: R(ω) reflection coefficient
                - transmission: T(ω) transmission coefficient
                - peaks: {ω_n,Q_n} resonance peaks
        """
        return self._impedance_calculator.compute_impedance(envelope)

    def get_carrier_frequency(self) -> float:
        """
        Get the high-frequency carrier frequency.

        Physical Meaning:
            Returns the frequency ω₀ of the high-frequency carrier
            that is modulated by the envelope.

        Returns:
            float: Carrier frequency ω₀.
        """
        return float(self.config.get("carrier_frequency", 1.85e43))

    def get_envelope_parameters(self) -> Dict[str, float]:
        """
        Get envelope equation parameters.

        Physical Meaning:
            Returns the parameters κ₀, κ₂, χ', χ'' for the
            envelope equation.

        Returns:
            Dict[str, float]: Envelope equation parameters.
        """
        return self._envelope_solver.get_parameters()

    def get_quench_thresholds(self) -> Dict[str, float]:
        """
        Get quench detection thresholds.

        Physical Meaning:
            Returns the current threshold values used for quench detection.

        Returns:
            Dict[str, float]: Quench detection thresholds.
        """
        return self._quench_detector.get_thresholds()

    def set_quench_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set new quench detection thresholds.

        Physical Meaning:
            Updates the threshold values used for quench detection.

        Args:
            thresholds (Dict[str, float]): New threshold values.
        """
        self._quench_detector.set_thresholds(thresholds)

    def get_impedance_parameters(self) -> Dict[str, Any]:
        """
        Get impedance calculation parameters.

        Physical Meaning:
            Returns the current parameters for impedance calculation.

        Returns:
            Dict[str, Any]: Impedance calculation parameters.
        """
        return self._impedance_calculator.get_parameters()

    def __repr__(self) -> str:
        """String representation of BVP core."""
        return (
            f"BVPCore(domain={self.domain}, "
            f"carrier_freq={self.get_carrier_frequency()})"
        )
