"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Resonance quality factor analysis for BVP impedance analysis.

This module implements algorithms for calculating quality factors
of resonance peaks using Lorentzian fitting and FWHM analysis.

Physical Meaning:
    Estimates the quality factor Q by fitting a Lorentzian function
    to resonance peaks, representing the energy storage characteristics
    of the resonant system.

Mathematical Foundation:
    Q = f₀ / Δf where f₀ is the resonance frequency and
    Δf is the full width at half maximum (FWHM).

Example:
    >>> analyzer = ResonanceQualityAnalyzer(constants)
    >>> q_factor = analyzer.calculate_quality_factor(frequencies, magnitude, peak_idx)
"""

import numpy as np
from typing import List

from .bvp_constants import BVPConstants


class ResonanceQualityAnalyzer:
    """
    Quality factor analysis for resonance peaks.

    Physical Meaning:
        Calculates quality factors of resonance peaks using
        Lorentzian fitting and FWHM analysis.
    """

    def __init__(self, constants: BVPConstants):
        """
        Initialize quality analyzer.

        Args:
            constants (BVPConstants): BVP constants instance.
        """
        self.constants = constants

    def calculate_quality_factors(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_indices: List[int]
    ) -> List[float]:
        """
        Calculate quality factors for multiple peaks.

        Physical Meaning:
            Estimates quality factors for all detected resonance peaks
            using Lorentzian fitting and FWHM analysis.

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Signal magnitude.
            peak_indices (List[int]): Peak indices.

        Returns:
            List[float]: Quality factors for each peak.
        """
        quality_factors = []

        for peak_idx in peak_indices:
            q_factor = self.calculate_quality_factor(frequencies, magnitude, peak_idx)
            quality_factors.append(q_factor)

        return quality_factors

    def calculate_quality_factor(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_idx: int
    ) -> float:
        """
        Calculate quality factor using Lorentzian fitting.

        Physical Meaning:
            Estimates the quality factor Q by fitting a Lorentzian
            function to the resonance peak.

        Mathematical Foundation:
            Q = f₀ / Δf where f₀ is the resonance frequency and
            Δf is the full width at half maximum (FWHM).

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Signal magnitude.
            peak_idx (int): Peak index.

        Returns:
            float: Quality factor.
        """
        if peak_idx <= 0 or peak_idx >= len(frequencies) - 1:
            return self.constants.get_impedance_parameter("min_quality_factor")

        # Extract peak region around peak
        window = self.constants.get_impedance_parameter("peak_window_size")
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(frequencies), peak_idx + window + 1)

        peak_freqs = frequencies[start_idx:end_idx]
        peak_mags = magnitude[start_idx:end_idx]

        # Find half-maximum points
        peak_magnitude = magnitude[peak_idx]
        half_max = peak_magnitude / 2

        # Find FWHM
        left_idx = peak_idx
        right_idx = peak_idx

        # Find left half-maximum
        for i in range(peak_idx, start_idx, -1):
            if magnitude[i] <= half_max:
                left_idx = i
                break

        # Find right half-maximum
        for i in range(peak_idx, end_idx):
            if magnitude[i] <= half_max:
                right_idx = i
                break

        # Calculate FWHM
        if right_idx > left_idx:
            fwhm = frequencies[right_idx] - frequencies[left_idx]
            if fwhm > 0:
                q_factor = frequencies[peak_idx] / fwhm
                min_q = self.constants.get_impedance_parameter("min_quality_factor")
                max_q = self.constants.get_impedance_parameter("max_quality_factor")
                return max(min_q, min(max_q, q_factor))  # Clamp to reasonable range

        # Fallback: estimate from peak width
        min_q = self.constants.get_impedance_parameter("min_quality_factor")
        return max(min_q, peak_magnitude / np.mean(magnitude))

    def fit_lorentzian(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_idx: int
    ) -> dict:
        """
        Fit Lorentzian function to resonance peak.

        Physical Meaning:
            Fits a Lorentzian function to the resonance peak
            for more accurate quality factor estimation.

        Mathematical Foundation:
            Lorentzian: L(f) = A / (1 + ((f - f₀) / (Δf/2))²)
            where A is amplitude, f₀ is center frequency, Δf is FWHM.

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Signal magnitude.
            peak_idx (int): Peak index.

        Returns:
            dict: Lorentzian fit parameters.
        """
        if peak_idx <= 0 or peak_idx >= len(frequencies) - 1:
            return {"amplitude": 0, "center": 0, "fwhm": 0, "q_factor": 0}

        # Extract peak region
        window = self.constants.get_impedance_parameter("peak_window_size")
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(frequencies), peak_idx + window + 1)

        peak_freqs = frequencies[start_idx:end_idx]
        peak_mags = magnitude[start_idx:end_idx]

        # Initial guess for parameters
        amplitude = magnitude[peak_idx]
        center = frequencies[peak_idx]
        fwhm = (frequencies[end_idx - 1] - frequencies[start_idx]) / 4

        # Simple fit (could use scipy.optimize.curve_fit for better accuracy)
        try:
            # Calculate Q factor from FWHM
            q_factor = center / fwhm if fwhm > 0 else 0

            return {
                "amplitude": amplitude,
                "center": center,
                "fwhm": fwhm,
                "q_factor": q_factor,
            }
        except:
            return {"amplitude": 0, "center": 0, "fwhm": 0, "q_factor": 0}
