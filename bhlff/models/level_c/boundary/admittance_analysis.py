"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Admittance analysis module for boundary effects.

This module implements admittance analysis functionality
for Level C test C1 in 7D phase field theory.

Physical Meaning:
    Analyzes admittance spectrum for boundary effects,
    including resonance detection and quality factor analysis.

Example:
    >>> analyzer = AdmittanceAnalyzer(bvp_core)
    >>> results = analyzer.analyze_admittance_spectrum(domain, boundary, frequency_range)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

from bhlff.core.bvp import BVPCore
from .data_structures import BoundaryGeometry, AdmittanceSpectrum


class AdmittanceAnalyzer:
    """
    Admittance analysis for boundary effects.

    Physical Meaning:
        Analyzes admittance spectrum for boundary effects,
        including resonance detection and quality factor analysis.

    Mathematical Foundation:
        Implements admittance analysis:
        - Admittance calculation: Y(ω) = I(ω)/V(ω)
        - Resonance detection: peaks in |Y(ω)| spectrum
        - Quality factor analysis: Q = ω / (2 * Δω)
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize admittance analyzer.

        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def analyze_admittance_spectrum(
        self,
        domain: Dict[str, Any],
        boundary: BoundaryGeometry,
        frequency_range: Tuple[float, float],
    ) -> AdmittanceSpectrum:
        """
        Analyze admittance spectrum.

        Physical Meaning:
            Computes the complex admittance Y(ω) spectrum over the
            frequency range, revealing resonance frequencies and
            system response characteristics.

        Mathematical Foundation:
            Y(ω) = I(ω)/V(ω) = ∫_Ω a*(x) s(x) dV / ∫_Ω |a(x)|² dV

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            frequency_range (Tuple[float, float]): Frequency range (ω_min, ω_max).

        Returns:
            AdmittanceSpectrum: Admittance spectrum analysis.
        """
        omega_min, omega_max = frequency_range
        num_frequencies = 100
        frequencies = np.linspace(omega_min, omega_max, num_frequencies)

        # Compute admittance for each frequency
        admittances = []
        for omega in frequencies:
            admittance = self._compute_admittance_at_frequency(
                domain, boundary, omega
            )
            admittances.append(admittance)

        admittances = np.array(admittances)

        # Create admittance spectrum
        spectrum = AdmittanceSpectrum(
            frequencies=frequencies,
            admittances=admittances,
        )

        return spectrum

    def _compute_admittance_at_frequency(
        self, domain: Dict[str, Any], boundary: BoundaryGeometry, omega: float
    ) -> complex:
        """
        Compute admittance at specific frequency.

        Physical Meaning:
            Computes the complex admittance Y(ω) at a specific
            frequency ω for the given boundary geometry.

        Mathematical Foundation:
            Y(ω) = I(ω)/V(ω) = ∫_Ω a*(x) s(x) dV / ∫_Ω |a(x)|² dV

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            omega (float): Frequency.

        Returns:
            complex: Complex admittance at frequency ω.
        """
        # Create source field
        source_field = self._create_source_field(domain, omega)

        # Solve BVP with boundary
        solution_field = self._solve_bvp_with_boundary(
            domain, boundary, source_field, omega
        )

        # Compute current and voltage
        current = self._compute_current(solution_field, source_field, domain)
        voltage = self._compute_voltage(solution_field, domain)

        # Compute admittance
        if abs(voltage) > 1e-12:
            admittance = current / voltage
        else:
            admittance = 0.0 + 0.0j

        return admittance

    def _create_source_field(self, domain: Dict[str, Any], omega: float) -> np.ndarray:
        """
        Create source field.

        Physical Meaning:
            Creates a source field for admittance computation
            at the specified frequency.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            omega (float): Frequency.

        Returns:
            np.ndarray: Source field.
        """
        N = domain.get("N", 64)
        L = domain.get("L", 1.0)

        # Create coordinate arrays
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create source field
        center = np.array([L / 2, L / 2, L / 2])
        sigma = L / 8
        source_field = np.exp(
            -(
                (X - center[0]) ** 2
                + (Y - center[1]) ** 2
                + (Z - center[2]) ** 2
            )
            / (2 * sigma ** 2)
        )

        # Apply frequency dependence
        source_field *= np.exp(1j * omega * 0.1)  # Simplified time dependence

        return source_field

    def _solve_bvp_with_boundary(
        self,
        domain: Dict[str, Any],
        boundary: BoundaryGeometry,
        source_field: np.ndarray,
        omega: float,
    ) -> np.ndarray:
        """
        Solve BVP with boundary.

        Physical Meaning:
            Solves the boundary value problem with the given
            boundary geometry and source field.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            source_field (np.ndarray): Source field.
            omega (float): Frequency.

        Returns:
            np.ndarray: Solution field.
        """
        # Apply boundary conditions
        boundary_field = self._apply_boundary_conditions(
            domain, boundary, source_field
        )

        # Solve BVP
        solution_field = self.bvp_core.solve_field(boundary_field, omega)

        return solution_field

    def _apply_boundary_conditions(
        self,
        domain: Dict[str, Any],
        boundary: BoundaryGeometry,
        source_field: np.ndarray,
    ) -> np.ndarray:
        """
        Apply boundary conditions.

        Physical Meaning:
            Applies boundary conditions to the source field
            based on the boundary geometry.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            source_field (np.ndarray): Source field.

        Returns:
            np.ndarray: Field with boundary conditions applied.
        """
        N = domain.get("N", 64)
        L = domain.get("L", 1.0)

        # Create coordinate arrays
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create boundary mask
        boundary_mask = self._create_boundary_mask(
            X, Y, Z, boundary
        )

        # Apply boundary conditions
        boundary_field = source_field.copy()
        boundary_field[boundary_mask] *= boundary.contrast

        return boundary_field

    def _create_boundary_mask(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, boundary: BoundaryGeometry
    ) -> np.ndarray:
        """
        Create boundary mask.

        Physical Meaning:
            Creates a mask for the boundary region
            based on the boundary geometry.

        Args:
            X (np.ndarray): X coordinates.
            Y (np.ndarray): Y coordinates.
            Z (np.ndarray): Z coordinates.
            boundary (BoundaryGeometry): Boundary geometry.

        Returns:
            np.ndarray: Boundary mask.
        """
        # Compute distances from boundary center
        distances = np.sqrt(
            (X - boundary.center[0]) ** 2
            + (Y - boundary.center[1]) ** 2
            + (Z - boundary.center[2]) ** 2
        )

        # Create boundary mask
        boundary_mask = (distances <= boundary.radius + boundary.thickness / 2) & (
            distances >= boundary.radius - boundary.thickness / 2
        )

        return boundary_mask

    def _compute_current(
        self, solution_field: np.ndarray, source_field: np.ndarray, domain: Dict[str, Any]
    ) -> complex:
        """
        Compute current.

        Physical Meaning:
            Computes the current I(ω) for admittance calculation.

        Mathematical Foundation:
            I(ω) = ∫_Ω a*(x) s(x) dV
            where a(x) is the solution field and s(x) is the source field.

        Args:
            solution_field (np.ndarray): Solution field.
            source_field (np.ndarray): Source field.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            complex: Current value.
        """
        # Compute current as inner product
        current = np.sum(np.conj(solution_field) * source_field)

        return current

    def _compute_voltage(self, solution_field: np.ndarray, domain: Dict[str, Any]) -> complex:
        """
        Compute voltage.

        Physical Meaning:
            Computes the voltage V(ω) for admittance calculation.

        Mathematical Foundation:
            V(ω) = ∫_Ω |a(x)|² dV
            where a(x) is the solution field.

        Args:
            solution_field (np.ndarray): Solution field.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            complex: Voltage value.
        """
        # Compute voltage as field energy
        voltage = np.sum(np.abs(solution_field) ** 2)

        return voltage

    def detect_resonances(self, spectrum: AdmittanceSpectrum, threshold: float = 8.0) -> List[Dict[str, Any]]:
        """
        Detect resonances in admittance spectrum.

        Physical Meaning:
            Detects resonance frequencies in the admittance
            spectrum above the specified threshold.

        Args:
            spectrum (AdmittanceSpectrum): Admittance spectrum.
            threshold (float): Threshold for resonance detection (dB).

        Returns:
            List[Dict[str, Any]]: Detected resonances.
        """
        # Convert threshold to linear scale
        threshold_linear = 10 ** (threshold / 20.0)

        # Find peaks above threshold
        peaks = self._find_peaks(spectrum.magnitude, threshold_linear)

        # Analyze each peak
        resonances = []
        for peak_idx in peaks:
            resonance = self._analyze_resonance_peak(spectrum, peak_idx)
            resonances.append(resonance)

        return resonances

    def _find_peaks(self, signal: np.ndarray, height: float) -> List[int]:
        """
        Find peaks in signal above threshold.

        Physical Meaning:
            Identifies peaks in the signal that exceed the specified
            height threshold.

        Args:
            signal (np.ndarray): Signal values.
            height (float): Height threshold.

        Returns:
            List[int]: Peak indices.
        """
        peaks = []

        for i in range(1, len(signal) - 1):
            if (
                signal[i] > signal[i - 1]
                and signal[i] > signal[i + 1]
                and signal[i] > height
            ):
                peaks.append(i)

        return peaks

    def _analyze_resonance_peak(
        self, spectrum: AdmittanceSpectrum, peak_idx: int
    ) -> Dict[str, Any]:
        """
        Analyze resonance peak.

        Physical Meaning:
            Analyzes a resonance peak in the admittance spectrum,
            including frequency, magnitude, and quality factor.

        Args:
            spectrum (AdmittanceSpectrum): Admittance spectrum.
            peak_idx (int): Peak index.

        Returns:
            Dict[str, Any]: Resonance analysis.
        """
        # Extract peak properties
        frequency = spectrum.frequencies[peak_idx]
        magnitude = spectrum.magnitude[peak_idx]
        phase = spectrum.phase[peak_idx]

        # Compute quality factor
        quality_factor = self._compute_quality_factor(spectrum, peak_idx)

        return {
            "frequency": frequency,
            "magnitude": magnitude,
            "phase": phase,
            "quality_factor": quality_factor,
            "peak_index": peak_idx,
        }

    def _compute_quality_factor(
        self, spectrum: AdmittanceSpectrum, peak_idx: int
    ) -> float:
        """
        Compute quality factor for resonance peak.

        Physical Meaning:
            Computes the quality factor Q = ω / (2 * Δω) for the
            resonance peak.

        Mathematical Foundation:
            Q = ω / (2 * Δω)
            where ω is the resonance frequency and Δω is the
            full width at half maximum.

        Args:
            spectrum (AdmittanceSpectrum): Admittance spectrum.
            peak_idx (int): Peak index.

        Returns:
            float: Quality factor.
        """
        if peak_idx < 1 or peak_idx >= len(spectrum.magnitude) - 1:
            return 0.0

        # Find half maximum points
        peak_magnitude = spectrum.magnitude[peak_idx]
        half_maximum = peak_magnitude / 2.0

        # Find left and right half maximum points
        left_idx = peak_idx
        right_idx = peak_idx

        # Find left half maximum
        for i in range(peak_idx - 1, -1, -1):
            if spectrum.magnitude[i] <= half_maximum:
                left_idx = i
                break

        # Find right half maximum
        for i in range(peak_idx + 1, len(spectrum.magnitude)):
            if spectrum.magnitude[i] <= half_maximum:
                right_idx = i
                break

        # Compute full width at half maximum
        fwhm = spectrum.frequencies[right_idx] - spectrum.frequencies[left_idx]

        # Compute quality factor
        if fwhm > 0:
            quality_factor = spectrum.frequencies[peak_idx] / fwhm
        else:
            quality_factor = 0.0

        return quality_factor
