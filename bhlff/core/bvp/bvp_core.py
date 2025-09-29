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
from typing import Dict, Any, List

from ..domain import Domain
from .quench_detector import QuenchDetector
from .bvp_envelope_solver import BVPEnvelopeSolver
from .bvp_impedance_calculator import BVPImpedanceCalculator
from .bvp_constants import BVPConstants
from .phase_vector import PhaseVector


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
        _phase_vector (PhaseVector): U(1)³ phase vector structure.
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
        self.constants = BVPConstants(config)
        self._setup_phase_vector()
        self._setup_envelope_solver()
        self._setup_quench_detector()
        self._setup_impedance_calculator()

    def _setup_phase_vector(self) -> None:
        """
        Setup U(1)³ phase vector structure.
        
        Physical Meaning:
            Initializes the three-component phase vector Θ_a (a=1..3)
            that represents the fundamental phase structure of the BVP field.
        """
        self._phase_vector = PhaseVector(self.domain, self.config, self.constants)

    def _setup_envelope_solver(self) -> None:
        """
        Setup envelope equation solver.

        Physical Meaning:
            Initializes the solver for the BVP envelope equation
            with nonlinear stiffness and susceptibility.
        """
        self._envelope_solver = BVPEnvelopeSolver(self.domain, self.config, self.constants)

    def _setup_quench_detector(self) -> None:
        """
        Setup quench event detector.

        Physical Meaning:
            Initializes the detector for quench events when
            local thresholds are reached.
        """
        quench_config = self.config.get("quench_detection", {})
        self._quench_detector = QuenchDetector(quench_config, self.constants)

    def _setup_impedance_calculator(self) -> None:
        """
        Setup impedance/admittance calculator.

        Physical Meaning:
            Initializes the calculator for Y(ω), R(ω), T(ω)
            and peaks {ω_n,Q_n} from BVP envelope.
        """
        self._impedance_calculator = BVPImpedanceCalculator(self.domain, self.config, self.constants)

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
        if source.shape != self.domain.shape:
            raise ValueError(f"Source shape {source.shape} incompatible with 7D domain shape {self.domain.shape}")
        
        # Solve envelope equation for U(1)³ phase structure
        envelope = self._envelope_solver.solve_envelope(source)
        
        # Update phase vector with solved envelope
        self._phase_vector.update_phase_components(envelope)
        
        return envelope

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
        return self.constants.get_envelope_parameter("carrier_frequency")

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

    def get_phase_vector(self) -> PhaseVector:
        """
        Get the U(1)³ phase vector structure.
        
        Physical Meaning:
            Returns the phase vector structure containing the three
            U(1) phase components Θ_a (a=1..3).
            
        Returns:
            PhaseVector: The U(1)³ phase vector structure.
        """
        return self._phase_vector

    def get_phase_components(self) -> List[np.ndarray]:
        """
        Get the three U(1) phase components Θ_a (a=1..3).
        
        Physical Meaning:
            Returns the three independent U(1) phase components
            that form the U(1)³ structure of the BVP field.
            
        Returns:
            List[np.ndarray]: List of three phase components Θ_a.
        """
        return self._phase_vector.get_phase_components()

    def get_total_phase(self) -> np.ndarray:
        """
        Get the total phase from U(1)³ structure.
        
        Physical Meaning:
            Computes the total phase by combining the three
            U(1) components with proper SU(2) coupling.
            
        Returns:
            np.ndarray: Total phase field.
        """
        return self._phase_vector.get_total_phase()

    def compute_electroweak_currents(self, envelope: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute electroweak currents as functionals of the envelope.
        
        Physical Meaning:
            Computes electromagnetic and weak currents that are
            generated as functionals of the BVP envelope through
            the U(1)³ phase structure.
            
        Args:
            envelope (np.ndarray): BVP envelope |A|.
            
        Returns:
            Dict[str, np.ndarray]: Electroweak currents including:
                - em_current: Electromagnetic current
                - weak_current: Weak interaction current
                - mixed_current: Mixed electroweak current
        """
        return self._phase_vector.compute_electroweak_currents(envelope)

    def compute_phase_coherence(self) -> np.ndarray:
        """
        Compute phase coherence measure.
        
        Physical Meaning:
            Computes a measure of phase coherence across the
            U(1)³ structure, indicating the degree of
            synchronization between the three phase components.
            
        Returns:
            np.ndarray: Phase coherence measure.
        """
        return self._phase_vector.compute_phase_coherence()

    def get_su2_coupling_strength(self) -> float:
        """
        Get the SU(2) coupling strength.
        
        Physical Meaning:
            Returns the strength of the weak hierarchical
            coupling to SU(2)/core.
            
        Returns:
            float: SU(2) coupling strength.
        """
        return self._phase_vector.get_su2_coupling_strength()

    def set_su2_coupling_strength(self, strength: float) -> None:
        """
        Set the SU(2) coupling strength.
        
        Physical Meaning:
            Updates the strength of the weak hierarchical
            coupling to SU(2)/core.
            
        Args:
            strength (float): New SU(2) coupling strength.
        """
        self._phase_vector.set_su2_coupling_strength(strength)

    def __repr__(self) -> str:
        """String representation of BVP core."""
        return (
            f"BVPCore(domain={self.domain}, "
            f"carrier_freq={self.get_carrier_frequency()})"
        )
