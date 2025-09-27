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
"""

import numpy as np
from typing import Dict, Any

from .domain import Domain
from .quench_detector import QuenchDetector


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
        _envelope_solver: Internal envelope equation solver.
        _quench_detector: Quench event detector.
        _impedance_calculator: Impedance/admittance calculator.
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
        # Initialize envelope solver with configuration
        envelope_config = self.config.get("envelope_equation", {})
        self._envelope_solver = {
            "kappa_0": envelope_config.get("kappa_0", 1.0),
            "kappa_2": envelope_config.get("kappa_2", 0.1),
            "chi_prime": envelope_config.get("chi_prime", 1.0),
            "chi_double_prime_0": envelope_config.get(
                "chi_double_prime_0", 0.01
            ),
            "k0_squared": envelope_config.get("k0_squared", 1.0),
        }

    def _setup_quench_detector(self) -> None:
        """
        Setup quench event detector.

        Physical Meaning:
            Initializes the detector for quench events when
            local thresholds are reached.
        """
        # Initialize quench detector with configuration
        quench_config = self.config.get("quench_detection", {})
        self._quench_detector = QuenchDetector(
            {
                "amplitude_threshold": quench_config.get(
                    "amplitude_threshold", 0.8
                ),
                "detuning_threshold": quench_config.get(
                    "detuning_threshold", 0.1
                ),
                "gradient_threshold": quench_config.get(
                    "gradient_threshold", 0.5
                ),
            }
        )

    def _setup_impedance_calculator(self) -> None:
        """
        Setup impedance/admittance calculator.

        Physical Meaning:
            Initializes the calculator for Y(ω), R(ω), T(ω)
            and peaks {ω_n,Q_n} from BVP envelope.
        """
        # Initialize impedance calculator with configuration
        impedance_config = self.config.get("impedance_calculation", {})
        self._impedance_calculator = {
            "frequency_range": impedance_config.get(
                "frequency_range", (0.1, 10.0)
            ),
            "frequency_points": impedance_config.get("frequency_points", 1000),
            "boundary_conditions": impedance_config.get(
                "boundary_conditions", "periodic"
            ),
            "quality_factor_threshold": impedance_config.get(
                "quality_factor_threshold", 0.1
            ),
        }

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
                Represents the envelope modulation of the high-frequency "
                "carrier.

        Raises:
            ValueError: If source has incompatible shape with domain.
        """
        if source.shape != self.domain.shape:
            raise ValueError(
                f"Source shape {source.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        # Solve envelope equation using iterative method
        # Initial guess: zero field
        envelope = np.zeros_like(source, dtype=complex)

        # Get envelope parameters
        kappa_0 = self._envelope_solver["kappa_0"]
        kappa_2 = self._envelope_solver["kappa_2"]
        chi_prime = self._envelope_solver["chi_prime"]
        k0_squared = self._envelope_solver["k0_squared"]

        # Simple iterative solution (can be improved with more sophisticated
        # methods)
        max_iterations = 100
        tolerance = 1e-6

        for iteration in range(max_iterations):
            # Compute nonlinear stiffness κ(|a|) = κ₀ + κ₂|a|²
            amplitude_squared = np.abs(envelope) ** 2
            kappa = kappa_0 + kappa_2 * amplitude_squared

            # Compute effective susceptibility χ(|a|) = χ' + iχ''(|a|)
            chi_double_prime = (
                self._envelope_solver["chi_double_prime_0"] * amplitude_squared
            )
            chi = chi_prime + 1j * chi_double_prime

            # Solve linearized equation: ∇·(κ∇a) + k₀²χa = s
            # For simplicity, use finite difference approximation
            envelope_new = self._solve_linearized_envelope(
                envelope, kappa, chi, k0_squared, source
            )

            # Check convergence
            residual = np.max(np.abs(envelope_new - envelope))
            if residual < tolerance:
                break

            envelope = envelope_new

        return envelope.real

    def _solve_linearized_envelope(
        self,
        envelope: np.ndarray,
        kappa: np.ndarray,
        chi: np.ndarray,
        k0_squared: float,
        source: np.ndarray,
    ) -> np.ndarray:
        """
        Solve linearized envelope equation.

        Physical Meaning:
            Solves the linearized version of the envelope equation
            for a given nonlinear stiffness and susceptibility.

        Mathematical Foundation:
            Solves ∇·(κ∇a) + k₀²χa = s using finite difference method.

        Args:
            envelope (np.ndarray): Current envelope estimate.
            kappa (np.ndarray): Nonlinear stiffness κ(|a|).
            chi (np.ndarray): Effective susceptibility χ(|a|).
            k0_squared (float): Wave number squared k₀².
            source (np.ndarray): Source term s(x).

        Returns:
            np.ndarray: Updated envelope solution.
        """
        # Simple finite difference implementation
        # This is a simplified version - in practice, more sophisticated
        # methods like spectral methods or advanced finite differences would
        # be used

        dx = self.domain.dx

        # Compute gradient of envelope
        if self.domain.dimensions == 1:
            grad_envelope = np.gradient(envelope, dx)
            # Compute ∇·(κ∇a) term
            kappa_grad = kappa * grad_envelope
            div_kappa_grad = np.gradient(kappa_grad, dx)
        elif self.domain.dimensions == 2:
            grad_x, grad_y = np.gradient(envelope, dx, dx)
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            div_kappa_grad_x = np.gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad_y = np.gradient(kappa_grad_y, dx, axis=1)
            div_kappa_grad = div_kappa_grad_x + div_kappa_grad_y
        else:  # 3D
            grad_x, grad_y, grad_z = np.gradient(envelope, dx, dx, dx)
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            kappa_grad_z = kappa * grad_z
            div_kappa_grad_x = np.gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad_y = np.gradient(kappa_grad_y, dx, axis=1)
            div_kappa_grad_z = np.gradient(kappa_grad_z, dx, axis=2)
            div_kappa_grad = (
                div_kappa_grad_x + div_kappa_grad_y + div_kappa_grad_z
            )

        # Solve: ∇·(κ∇a) + k₀²χa = s
        # Rearrange: k₀²χa = s - ∇·(κ∇a)
        # Therefore: a = (s - ∇·(κ∇a)) / (k₀²χ)

        # Avoid division by zero
        chi_safe = np.where(np.abs(chi) < 1e-12, 1e-12, chi)

        envelope_new = (source - div_kappa_grad) / (k0_squared * chi_safe)

        return envelope_new

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
        # Use the quench detector to find quench events
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
        # Compute impedance from envelope using configuration
        freq_min, freq_max = self._impedance_calculator["frequency_range"]
        freq_points = self._impedance_calculator["frequency_points"]

        # Create frequency array
        frequencies = np.linspace(freq_min, freq_max, freq_points)

        # Compute admittance Y(ω) from envelope
        # This is a simplified calculation - in practice, more sophisticated
        # boundary analysis would be performed
        admittance = self._compute_admittance_from_envelope(
            envelope, frequencies
        )

        # Compute reflection and transmission coefficients
        reflection = self._compute_reflection_coefficient(admittance)
        transmission = self._compute_transmission_coefficient(admittance)

        # Find resonance peaks
        peaks = self._find_resonance_peaks(frequencies, admittance)

        return {
            "admittance": admittance,
            "reflection": reflection,
            "transmission": transmission,
            "peaks": peaks,
        }

    def _compute_admittance_from_envelope(
        self, envelope: np.ndarray, frequencies: np.ndarray
    ) -> np.ndarray:
        """
        Compute admittance from envelope.

        Physical Meaning:
            Computes the frequency-dependent admittance Y(ω)
            from the BVP envelope using boundary analysis.

        Args:
            envelope (np.ndarray): BVP envelope.
            frequencies (np.ndarray): Frequency array.

        Returns:
            np.ndarray: Admittance Y(ω).
        """
        # Simplified calculation - in practice, more sophisticated
        # boundary analysis would be performed

        # Compute admittance as function of frequency
        # This is a simplified model
        admittance = np.zeros_like(frequencies, dtype=complex)

        for i, freq in enumerate(frequencies):
            # Simplified admittance calculation
            # In practice, this would involve proper boundary analysis
            admittance[i] = 1.0 / (1.0 + 1j * freq)

        return admittance

    def _compute_reflection_coefficient(
        self, admittance: np.ndarray
    ) -> np.ndarray:
        """
        Compute reflection coefficient from admittance.

        Physical Meaning:
            Computes the reflection coefficient R(ω) from
            the admittance Y(ω).

        Args:
            admittance (np.ndarray): Admittance Y(ω).

        Returns:
            np.ndarray: Reflection coefficient R(ω).
        """
        # Simplified calculation: R = (1 - Y) / (1 + Y)
        # In practice, this would involve proper boundary analysis
        reflection = (1.0 - admittance) / (1.0 + admittance)
        return reflection

    def _compute_transmission_coefficient(
        self, admittance: np.ndarray
    ) -> np.ndarray:
        """
        Compute transmission coefficient from admittance.

        Physical Meaning:
            Computes the transmission coefficient T(ω) from
            the admittance Y(ω).

        Args:
            admittance (np.ndarray): Admittance Y(ω).

        Returns:
            np.ndarray: Transmission coefficient T(ω).
        """
        # Simplified calculation: T = 2 / (1 + Y)
        # In practice, this would involve proper boundary analysis
        transmission = 2.0 / (1.0 + admittance)
        return transmission

    def _find_resonance_peaks(
        self, frequencies: np.ndarray, admittance: np.ndarray
    ) -> Dict[str, list]:
        """
        Find resonance peaks in admittance.

        Physical Meaning:
            Identifies resonance frequencies and quality factors
            from the admittance spectrum.

        Args:
            frequencies (np.ndarray): Frequency array.
            admittance (np.ndarray): Admittance Y(ω).

        Returns:
            Dict[str, list]: Resonance peaks including frequencies and "
            "quality factors.
        """
        # Find peaks in admittance magnitude
        admittance_magnitude = np.abs(admittance)

        # Simple peak finding (in practice, more sophisticated methods would
        # be used)
        peaks = []
        quality_factors = []

        # Find local maxima
        for i in range(1, len(admittance_magnitude) - 1):
            if (
                admittance_magnitude[i] > admittance_magnitude[i - 1]
                and admittance_magnitude[i] > admittance_magnitude[i + 1]
            ):
                peaks.append(frequencies[i])

                # Estimate quality factor (simplified)
                q_factor = admittance_magnitude[i] / np.mean(
                    admittance_magnitude
                )
                quality_factors.append(q_factor)

        return {"frequencies": peaks, "quality_factors": quality_factors}

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
        return dict(self.config.get(
            "envelope_equation",
            {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
            },
        ))

    def __repr__(self) -> str:
        """String representation of BVP core."""
        return (
            f"BVPCore(domain={self.domain}, "
            f"carrier_freq={self.get_carrier_frequency()})"
        )
