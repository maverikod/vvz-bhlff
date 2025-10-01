"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP core operations module.

This module implements the core operations of the BVP framework,
including envelope solving, quench detection, and impedance computation.

Physical Meaning:
    Implements the fundamental operations of the BVP framework that
    work with the envelope modulations and beatings of the high-frequency
    carrier field, including solving, analysis, and characterization.

Mathematical Foundation:
    Provides operations for:
    - Solving the envelope equation: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x)
    - Detecting quench events at local thresholds
    - Computing impedance and admittance characteristics

Example:
    >>> operations = BVPCoreOperations(domain, config)
    >>> envelope = operations.solve_envelope(source)
    >>> quenches = operations.detect_quenches(envelope)
"""

import numpy as np
from typing import Dict, Any

from ...domain import Domain
from ...domain.domain_7d import Domain7D
from ..quench_detector import QuenchDetector
from ..bvp_envelope_solver import BVPEnvelopeSolver
from ..bvp_impedance_calculator import BVPImpedanceCalculator
from ..phase_vector import PhaseVector
from ..bvp_phase_operations import BVPPhaseOperations
from ..bvp_parameter_access import BVPParameterAccess


class BVPCoreOperations:
    """
    BVP core operations for envelope solving and analysis.

    Physical Meaning:
        Implements the core operations of the BVP framework including
        envelope solving, quench detection, impedance computation,
        and phase operations for the 7D space-time theory.

    Mathematical Foundation:
        Provides operations for solving and analyzing the BVP envelope
        equation and its physical consequences in 7D space-time.
    """

    def __init__(
        self, domain: Domain, config: Dict[str, Any], domain_7d: Domain7D = None
    ):
        """
        Initialize BVP core operations.

        Physical Meaning:
            Sets up the core operations with the computational domains
            and configuration parameters, initializing all necessary
            components for BVP operations.

        Args:
            domain (Domain): Standard computational domain.
            config (Dict[str, Any]): Configuration parameters.
            domain_7d (Domain7D, optional): 7D computational domain.
        """
        self.domain = domain
        self.config = config
        self.domain_7d = domain_7d

        # Initialize components
        self._setup_phase_vector()
        self._setup_envelope_solver()
        self._setup_quench_detector()
        self._setup_impedance_calculator()
        self._setup_phase_operations()
        # Parameter access needs to be initialized after all other components
        self._setup_parameter_access()

    def _setup_phase_vector(self) -> None:
        """Setup phase vector for U(1)³ phase structure."""
        self._phase_vector = PhaseVector(self.domain, self.config)

    def _setup_envelope_solver(self) -> None:
        """Setup envelope solver for BVP equation."""
        self._envelope_solver = BVPEnvelopeSolver(self.domain, self.config)

    def _setup_quench_detector(self) -> None:
        """Setup quench detector for threshold events."""
        from ..bvp_constants import BVPConstants
        constants = BVPConstants(self.config)
        self._quench_detector = QuenchDetector(self.domain, constants)

    def _setup_impedance_calculator(self) -> None:
        """Setup impedance calculator for boundary analysis."""
        self._impedance_calculator = BVPImpedanceCalculator(self.domain, self.config)

    def _setup_phase_operations(self) -> None:
        """Setup phase operations for U(1)³ structure."""
        self._phase_operations = BVPPhaseOperations(self._phase_vector)

    def _setup_parameter_access(self) -> None:
        """Setup parameter access for configuration management."""
        from ..bvp_constants import BVPConstants
        constants = BVPConstants(self.config)
        self._parameter_access = BVPParameterAccess(
            constants, 
            self._envelope_solver, 
            self._quench_detector, 
            self._impedance_calculator
        )

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
            raise ValueError(
                f"Source shape {source.shape} incompatible with 7D domain shape {self.domain.shape}"
            )

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
                - admittance: Admittance Y(ω)
                - reflection: Reflection coefficient R(ω)
                - transmission: Transmission coefficient T(ω)
                - resonance_peaks: Resonance peaks {ω_n,Q_n}
        """
        return self._impedance_calculator.compute_impedance(envelope)

    def get_phase_vector(self) -> PhaseVector:
        """
        Get phase vector for U(1)³ phase structure.

        Physical Meaning:
            Returns the phase vector containing the three U(1) phase
            components Θ_a (a=1..3) that represent the BVP field
            structure in 7D space-time.

        Returns:
            PhaseVector: Phase vector with U(1)³ structure.
        """
        return self._phase_vector

    def get_phase_operations(self) -> BVPPhaseOperations:
        """
        Get phase operations for U(1)³ structure.

        Physical Meaning:
            Returns the phase operations object for working with
            the U(1)³ phase structure of the BVP field.

        Returns:
            BVPPhaseOperations: Phase operations object.
        """
        return self._phase_operations

    def get_parameter_access(self) -> BVPParameterAccess:
        """
        Get parameter access for configuration management.

        Physical Meaning:
            Returns the parameter access object for managing
            BVP configuration parameters and settings.

        Returns:
            BVPParameterAccess: Parameter access object.
        """
        return self._parameter_access
