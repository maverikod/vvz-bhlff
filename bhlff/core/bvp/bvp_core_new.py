"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Core module implementation according to step 00 specification.

This module implements the central framework of the 7D theory where
all observed "modes" are envelope modulations and beatings of the
Base High-Frequency Field (BVP).

Theoretical Background:
    The BVP core serves as the central backbone of the 7D phase field
    theory, implementing the high-frequency carrier with envelope
    modulation capabilities and quench detection according to the
    7D space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Example:
    >>> bvp_core = BVPCore(config)
    >>> envelope = bvp_core.solve_envelope(source_field)
    >>> quenches = bvp_core.detect_quenches(envelope)
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ..domain.domain_7d import Domain7D
from .quench_detector import QuenchDetector
from .bvp_impedance_calculator import BVPImpedanceCalculator
from .envelope_solver_core import EnvelopeSolverCore


class BVPCore:
    """
    Base High-Frequency Field (BVP) core module.

    Physical Meaning:
        Implements the central framework of the 7D theory where
        all observed "modes" are envelope modulations and beatings
        of the Base High-Frequency Field (BVP). This is the core
        module that serves as the backbone for all other system
        components.

    Mathematical Foundation:
        Implements the 7D envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
        where κ(|a|) = κ₀ + κ₂|a|² is the nonlinear BVP stiffness,
        χ(|a|) = χ' + iχ''(|a|) is the effective susceptibility,
        and s(x,φ,t) are the sources/seeds.

    Attributes:
        config (Dict[str, Any]): Configuration parameters for BVP core.
        domain_7d (Domain7D): 7D computational domain.
        envelope_solver (EnvelopeSolverCore): Core envelope equation solver.
        quench_detector (QuenchDetector): Quench event detector.
        impedance_calculator (BVPImpedanceCalculator): Impedance calculator.
    """

    def __init__(self, config: Dict[str, Any], domain_7d: Optional[Domain7D] = None):
        """
        Initialize BVP core with configuration.

        Physical Meaning:
            Sets up the high-frequency carrier with envelope
            modulation capabilities and quench detection according
            to the 7D phase field theory.

        Args:
            config (Dict[str, Any]): Configuration parameters including:
                - carrier_frequency (float): BVP carrier frequency ω₀
                - envelope_equation (Dict): Envelope equation parameters
                - quench_detection (Dict): Quench detection parameters
                - impedance_calculation (Dict): Impedance calculation parameters
            domain_7d (Domain7D, optional): 7D computational domain.
                If None, creates default domain from config.
        """
        self.config = config
        self.domain_7d = domain_7d or self._create_default_domain()

        # Initialize core components
        self._setup_envelope_solver()
        self._setup_quench_detector()
        self._setup_impedance_calculator()

    def solve_envelope(self, source: np.ndarray) -> np.ndarray:
        """
        Solve BVP envelope equation.

        Physical Meaning:
            Computes the envelope a(x,φ,t) of the Base High-Frequency Field
            that modulates the high-frequency carrier. The envelope represents
            the slow modulation of the fast BVP carrier according to the
            scale separation postulate.

        Mathematical Foundation:
            Solves the 7D envelope equation:
            ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
            where the solution a(x,φ,t) represents the envelope modulation
            of the BVP carrier field.

        Args:
            source (np.ndarray): Source term s(x,φ,t) with shape
                (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t)

        Returns:
            np.ndarray: Envelope field a(x,φ,t) with same shape as source.
                Represents the BVP envelope modulation.

        Raises:
            ValueError: If source shape is incompatible with domain.
            RuntimeError: If envelope equation solution fails.
        """
        if source.shape != self.domain_7d.shape:
            raise ValueError(
                f"Source shape {source.shape} incompatible with domain shape {self.domain_7d.shape}"
            )

        # Solve envelope equation using core solver
        envelope = self.envelope_solver.solve(source)

        return envelope

    def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quench events when local thresholds are reached.

        Physical Meaning:
            Identifies when BVP dissipatively "dumps" energy into
            the medium at local thresholds (amplitude/detuning/gradient).
            Quenches represent threshold events where the BVP field
            undergoes a local regime transition.

        Mathematical Foundation:
            Applies three threshold criteria:
            - amplitude: |A| > |A_q|
            - detuning: |ω - ω_0| > Δω_q
            - gradient: |∇A| > |∇A_q|
            where A_q, Δω_q, and ∇A_q are the quench thresholds.

        Args:
            envelope (np.ndarray): 7D envelope field to analyze.

        Returns:
            Dict[str, Any]: Quench detection results including:
                - quenches_detected (bool): Whether any quenches were found
                - quench_locations (List[Tuple]): Coordinates of quench events
                - quench_types (List[str]): Types of quenches (amplitude/detuning/gradient)
                - quench_strengths (List[float]): Strength of each quench
        """
        return self.quench_detector.detect_quenches(envelope)

    def compute_impedance(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Compute impedance/admittance from BVP envelope.

        Physical Meaning:
            Calculates Y(ω), R(ω), T(ω), and peaks {ω_n,Q_n}
            from the BVP envelope at boundaries. This provides
            the interface between BVP and other system components
            (tail resonators, transition zone, core).

        Mathematical Foundation:
            Computes boundary functions from BVP envelope:
            - Admittance: Y(ω) = I(ω)/V(ω)
            - Reflection coefficient: R(ω)
            - Transmission coefficient: T(ω)
            - Resonance peaks: {ω_n, Q_n}

        Args:
            envelope (np.ndarray): 7D envelope field at boundaries.

        Returns:
            Dict[str, Any]: Impedance calculation results including:
                - admittance (np.ndarray): Y(ω) frequency response
                - reflection_coefficient (np.ndarray): R(ω) reflection
                - transmission_coefficient (np.ndarray): T(ω) transmission
                - resonance_peaks (List[Dict]): {ω_n, Q_n} resonance data
        """
        return self.impedance_calculator.compute_impedance(envelope)

    def _create_default_domain(self) -> Domain7D:
        """
        Create default 7D domain from configuration.

        Physical Meaning:
            Creates a default 7D computational domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
            based on configuration parameters.

        Returns:
            Domain7D: Default 7D computational domain.
        """
        domain_config = self.config.get("domain", {})
        return Domain7D(
            L=domain_config.get("L", 1.0),
            N=domain_config.get("N", 64),
            dimensions=7,
            phase_periods=domain_config.get(
                "phase_periods", [2 * np.pi, 2 * np.pi, 2 * np.pi]
            ),
            temporal_config=domain_config.get("temporal", {}),
        )

    def _setup_envelope_solver(self) -> None:
        """
        Setup envelope equation solver.

        Physical Meaning:
            Initializes the core envelope equation solver with
            the 7D envelope equation parameters from configuration.
        """
        envelope_config = self.config.get("envelope_equation", {})
        self.envelope_solver = EnvelopeSolverCore(
            domain_7d=self.domain_7d, config=envelope_config
        )

    def _setup_quench_detector(self) -> None:
        """
        Setup quench detector.

        Physical Meaning:
            Initializes the quench detector with threshold parameters
            for amplitude, detuning, and gradient quenches.
        """
        quench_config = self.config.get("quench_detection", {})
        self.quench_detector = QuenchDetector(
            domain_7d=self.domain_7d, config=quench_config
        )

    def _setup_impedance_calculator(self) -> None:
        """
        Setup impedance calculator.

        Physical Meaning:
            Initializes the impedance calculator for computing
            boundary functions from BVP envelope.
        """
        impedance_config = self.config.get("impedance_calculation", {})
        self.impedance_calculator = BVPImpedanceCalculator(
            domain_7d=self.domain_7d, config=impedance_config
        )
