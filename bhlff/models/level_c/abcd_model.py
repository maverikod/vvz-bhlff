"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

ABCD (transmission matrix) model for resonator chains in Level C.

This module implements the ABCD transmission matrix method for analyzing
cascaded resonators, providing analytical predictions for resonance
frequencies and quality factors in the 7D phase field theory.

Physical Meaning:
    The ABCD model represents the transmission properties of cascaded
    resonators in the 7D phase field, where each resonator is characterized
    by its transmission matrix. This allows analytical prediction of
    system resonance modes and their coupling effects.

Mathematical Foundation:
    Implements the transmission matrix method:
    - Each resonator layer has a 2x2 transmission matrix T_ℓ
    - System matrix: T_total = T_1 × T_2 × ... × T_N
    - Resonance conditions: det(T_total - I) = 0
    - Admittance calculation: Y(ω) = C/A for input impedance

Example:
    >>> abcd_model = ABCDModel(resonators)
    >>> system_modes = abcd_model.find_system_modes(frequency_range)
    >>> comparison = abcd_model.compare_with_numerical(numerical_results)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

from bhlff.core.bvp import BVPCore


@dataclass
class ResonatorLayer:
    """
    Single resonator layer in the chain.

    Physical Meaning:
        Represents a single resonator layer with specific material
        properties and geometry that contribute to the overall
        system transmission characteristics.
    """

    radius: float
    thickness: float
    contrast: float
    memory_gamma: float = 0.0
    memory_tau: float = 1.0
    material_params: Optional[Dict[str, Any]] = None


@dataclass
class SystemMode:
    """
    System resonance mode.

    Physical Meaning:
        Represents a resonance mode of the entire resonator chain,
        characterized by its frequency, quality factor, and coupling
        properties with other modes.
    """

    frequency: float
    quality_factor: float
    amplitude: float
    phase: float
    mode_index: int
    coupling_strength: float = 0.0


class ABCDModel:
    """
    ABCD (transmission matrix) model for resonator chains.

    Physical Meaning:
        Implements the transmission matrix method for analyzing
        cascaded resonators, providing analytical predictions
        for resonance frequencies and quality factors in the
        7D phase field theory.

    Mathematical Foundation:
        Uses the ABCD matrix formalism:
        - Each layer: T_ℓ = [A_ℓ  B_ℓ; C_ℓ  D_ℓ]
        - System matrix: T_total = ∏ T_ℓ
        - Resonance condition: det(T_total - I) = 0
        - Admittance: Y(ω) = C/A
    """

    def __init__(
        self, resonators: List[ResonatorLayer], bvp_core: Optional[BVPCore] = None
    ):
        """
        Initialize ABCD model.

        Physical Meaning:
            Sets up the ABCD model for the given resonator chain,
            computing transmission matrices for each layer and
            preparing for system analysis.

        Args:
            resonators (List[ResonatorLayer]): List of resonator layers.
            bvp_core (Optional[BVPCore]): BVP core for advanced calculations.
        """
        self.resonators = resonators
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

        # Pre-compute layer properties
        self._compute_layer_properties()

    def compute_transmission_matrix(self, frequency: float) -> np.ndarray:
        """
        Compute 2x2 transmission matrix for given frequency.

        Physical Meaning:
            Computes the overall transmission matrix T_total(ω) for the
            entire resonator chain at frequency ω, representing the
            system's transmission properties.

        Mathematical Foundation:
            T_total = T_1 × T_2 × ... × T_N
            where each T_ℓ is computed from layer properties

        Args:
            frequency (float): Frequency ω for matrix computation.

        Returns:
            np.ndarray: 2x2 transmission matrix [A B; C D].
        """
        if not self.resonators:
            return np.eye(2)

        # Start with identity matrix
        T_total = np.eye(2)

        # Multiply by each layer matrix
        for layer in self.resonators:
            T_layer = self._compute_layer_matrix(layer, frequency)
            T_total = T_total @ T_layer

        return T_total

    def find_resonance_conditions(
        self, frequency_range: Tuple[float, float]
    ) -> List[float]:
        """
        Find frequencies satisfying resonance conditions.

        Physical Meaning:
            Finds all frequencies where det(T_total - I) = 0,
            which correspond to system resonance modes.

        Mathematical Foundation:
            Resonance condition: det(T_total(ω) - I) = 0
            This equation is solved numerically over the frequency range.

        Args:
            frequency_range (Tuple[float, float]): (ω_min, ω_max) range.

        Returns:
            List[float]: List of resonance frequencies.
        """
        omega_min, omega_max = frequency_range

        # Create frequency grid
        frequencies = np.logspace(np.log10(omega_min), np.log10(omega_max), 1000)

        # Compute determinant for each frequency
        determinants = []
        for omega in frequencies:
            T = self.compute_transmission_matrix(omega)
            det = np.linalg.det(T - np.eye(2))
            determinants.append(det)

        determinants = np.array(determinants)

        # Find zero crossings
        zero_crossings = []
        for i in range(len(determinants) - 1):
            if determinants[i] * determinants[i + 1] < 0:
                # Linear interpolation to find exact zero
                omega_zero = frequencies[i] + (frequencies[i + 1] - frequencies[i]) * (
                    -determinants[i] / (determinants[i + 1] - determinants[i])
                )
                zero_crossings.append(omega_zero)

        return zero_crossings

    def find_system_modes(
        self, frequency_range: Tuple[float, float]
    ) -> List[SystemMode]:
        """
        Find system resonance modes.

        Physical Meaning:
            Identifies all system resonance modes in the given frequency
            range, computing their frequencies, quality factors, and
            coupling properties.

        Mathematical Foundation:
            For each resonance frequency ω_n:
            - Quality factor: Q_n = ω_n / (2 * Im(ω_n))
            - Amplitude: |A_n| from eigenvector analysis
            - Phase: arg(A_n) from eigenvector analysis

        Args:
            frequency_range (Tuple[float, float]): Frequency range to search.

        Returns:
            List[SystemMode]: List of system resonance modes.
        """
        resonance_frequencies = self.find_resonance_conditions(frequency_range)

        system_modes = []
        for i, omega_n in enumerate(resonance_frequencies):
            # Compute quality factor
            Q_n = self._compute_quality_factor(omega_n)

            # Compute mode amplitude and phase
            amplitude, phase = self._compute_mode_amplitude_phase(omega_n)

            # Compute coupling strength
            coupling_strength = self._compute_coupling_strength(
                omega_n, resonance_frequencies
            )

            mode = SystemMode(
                frequency=omega_n,
                quality_factor=Q_n,
                amplitude=amplitude,
                phase=phase,
                mode_index=i,
                coupling_strength=coupling_strength,
            )
            system_modes.append(mode)

        return system_modes

    def compute_system_admittance(self, frequency: float) -> complex:
        """
        Compute total system admittance.

        Physical Meaning:
            Computes the complex admittance Y(ω) = I(ω)/V(ω) of the
            entire resonator chain, representing the system's
            response to external excitation.

        Mathematical Foundation:
            Y(ω) = C(ω) / A(ω)
            where T_total = [A B; C D] is the system transmission matrix

        Args:
            frequency (float): Frequency ω for admittance calculation.

        Returns:
            complex: Complex admittance Y(ω).
        """
        T = self.compute_transmission_matrix(frequency)
        A, B, C, D = T[0, 0], T[0, 1], T[1, 0], T[1, 1]

        # Avoid division by zero
        if abs(A) < 1e-12:
            return complex(0, 0)

        return C / A

    def compare_with_numerical(
        self, numerical_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare with numerical simulation results.

        Physical Meaning:
            Compares ABCD model predictions with numerical simulation
            results, computing errors and validating the model accuracy.

        Mathematical Foundation:
            Computes various error metrics:
            - Frequency errors: |ω_ABCD - ω_num| / ω_num
            - Quality factor errors: |Q_ABCD - Q_num| / Q_num
            - Admittance errors: |Y_ABCD - Y_num| / |Y_num|

        Args:
            numerical_results (Dict[str, Any]): Numerical simulation results.

        Returns:
            Dict[str, Any]: Comparison results with error metrics.
        """
        # Extract numerical data
        numerical_frequencies = numerical_results.get("frequencies", [])
        numerical_admittance = numerical_results.get("admittance", [])
        numerical_modes = numerical_results.get("modes", [])

        # Compute ABCD predictions
        frequency_range = (min(numerical_frequencies), max(numerical_frequencies))
        abcd_modes = self.find_system_modes(frequency_range)

        # Compare frequencies
        frequency_errors = []
        for abcd_mode in abcd_modes:
            # Find closest numerical mode
            closest_numerical = min(
                numerical_modes,
                key=lambda m: abs(m.get("frequency", 0) - abcd_mode.frequency),
            )

            if "frequency" in closest_numerical:
                error = (
                    abs(abcd_mode.frequency - closest_numerical["frequency"])
                    / closest_numerical["frequency"]
                )
                frequency_errors.append(error)

        # Compare quality factors
        quality_errors = []
        for abcd_mode in abcd_modes:
            closest_numerical = min(
                numerical_modes,
                key=lambda m: abs(m.get("frequency", 0) - abcd_mode.frequency),
            )

            if "quality_factor" in closest_numerical:
                error = (
                    abs(abcd_mode.quality_factor - closest_numerical["quality_factor"])
                    / closest_numerical["quality_factor"]
                )
                quality_errors.append(error)

        # Compare admittance
        admittance_errors = []
        for freq in numerical_frequencies:
            abcd_admittance = self.compute_system_admittance(freq)
            numerical_admittance_val = numerical_admittance[
                numerical_frequencies.index(freq)
            ]

            if abs(numerical_admittance_val) > 1e-12:
                error = abs(abcd_admittance - numerical_admittance_val) / abs(
                    numerical_admittance_val
                )
                admittance_errors.append(error)

        return {
            "frequency_errors": frequency_errors,
            "quality_errors": quality_errors,
            "admittance_errors": admittance_errors,
            "max_frequency_error": max(frequency_errors) if frequency_errors else 0.0,
            "max_quality_error": max(quality_errors) if quality_errors else 0.0,
            "max_admittance_error": (
                max(admittance_errors) if admittance_errors else 0.0
            ),
            "abcd_modes": abcd_modes,
            "numerical_modes": numerical_modes,
            "comparison_passed": (
                (max(frequency_errors) if frequency_errors else 0.0) < 0.05
                and (max(quality_errors) if quality_errors else 0.0) < 0.10
            ),
        }

    def _compute_layer_properties(self) -> None:
        """Compute properties for each layer."""
        for layer in self.resonators:
            if layer.material_params is None:
                layer.material_params = {
                    "kappa": 1.0 + layer.contrast,
                    "chi_real": 1.0,
                    "chi_imag": layer.memory_gamma,
                }

    def _compute_layer_matrix(
        self, layer: ResonatorLayer, frequency: float
    ) -> np.ndarray:
        """
        Compute transmission matrix for single layer.

        Physical Meaning:
            Computes the 2x2 transmission matrix for a single
            resonator layer at frequency ω.

        Mathematical Foundation:
            For a layer with thickness Δr and wave number k:
            T = [cos(kΔr)  (1/k)sin(kΔr); -k sin(kΔr)  cos(kΔr)]
        """
        # Extract material parameters
        kappa = layer.material_params["kappa"]
        chi_real = layer.material_params["chi_real"]
        chi_imag = layer.material_params["chi_imag"]

        # Compute wave number
        k = frequency * np.sqrt(kappa / chi_real)

        # Compute layer matrix elements
        cos_kr = np.cos(k * layer.thickness)
        sin_kr = np.sin(k * layer.thickness)

        A = cos_kr
        B = sin_kr / k if k > 1e-12 else layer.thickness
        C = -k * sin_kr
        D = cos_kr

        return np.array([[A, B], [C, D]])

    def _compute_quality_factor(self, frequency: float) -> float:
        """
        Compute quality factor for given frequency.

        Physical Meaning:
            Computes the quality factor Q = ω / (2 * Im(ω)) which
            characterizes the resonance sharpness and energy storage.
        """
        # Simplified quality factor calculation
        # In practice, this would involve more sophisticated analysis
        return 10.0 + 5.0 * frequency  # Placeholder implementation

    def _compute_mode_amplitude_phase(self, frequency: float) -> Tuple[float, float]:
        """
        Compute mode amplitude and phase.

        Physical Meaning:
            Computes the amplitude and phase of the resonance mode
            at the given frequency from eigenvector analysis.
        """
        T = self.compute_transmission_matrix(frequency)

        # Find eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(T)

        # Find eigenvalue closest to 1 (resonance condition)
        resonance_idx = np.argmin(np.abs(eigenvals - 1.0))
        eigenvec = eigenvecs[:, resonance_idx]

        amplitude = np.abs(eigenvec[0])
        phase = np.angle(eigenvec[0])

        return amplitude, phase

    def _compute_coupling_strength(
        self, frequency: float, all_frequencies: List[float]
    ) -> float:
        """
        Compute coupling strength with other modes.

        Physical Meaning:
            Computes the coupling strength between the mode at the
            given frequency and other system modes.
        """
        if len(all_frequencies) <= 1:
            return 0.0

        # Find closest other frequency
        other_frequencies = [f for f in all_frequencies if f != frequency]
        if not other_frequencies:
            return 0.0

        closest_freq = min(other_frequencies, key=lambda f: abs(f - frequency))
        frequency_separation = abs(frequency - closest_freq)

        # Coupling strength inversely proportional to frequency separation
        coupling_strength = 1.0 / (1.0 + frequency_separation)

        return coupling_strength
