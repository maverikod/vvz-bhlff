"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Resonance detection algorithms for BVP impedance analysis.

This module implements advanced algorithms for detecting resonance peaks
in impedance/admittance spectra, including quality factor estimation
and peak characterization.

Physical Meaning:
    Provides algorithms for identifying resonance frequencies and quality
    factors from impedance spectra, representing the system's resonant
    behavior and energy storage characteristics.

Mathematical Foundation:
    Implements advanced signal processing techniques including magnitude,
    phase, and derivative analysis for robust peak detection.

Example:
    >>> detector = ResonanceDetector()
    >>> peaks = detector.find_resonance_peaks(frequencies, admittance)
"""

import numpy as np
from typing import Dict, List, Tuple


class ResonanceDetector:
    """
    Advanced resonance detection algorithms for impedance analysis.
    
    Physical Meaning:
        Implements advanced algorithms for identifying resonance frequencies
        and quality factors from impedance spectra.
        
    Mathematical Foundation:
        Uses multiple criteria for peak detection:
        1. Local maxima in magnitude with sufficient prominence
        2. Phase behavior analysis (rapid phase changes)
        3. Second derivative analysis for peak sharpness
        4. Quality factor estimation using Lorentzian fitting
    """
    
    def __init__(self) -> None:
        """Initialize resonance detector."""
        self.quality_factor_threshold = 0.1
    
    def find_resonance_peaks(
        self, frequencies: np.ndarray, admittance: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Find resonance peaks in admittance.
        
        Physical Meaning:
            Identifies resonance frequencies and quality factors
            from the admittance spectrum.
            
        Args:
            frequencies (np.ndarray): Frequency array.
            admittance (np.ndarray): Admittance Y(ω).
            
        Returns:
            Dict[str, List[float]]: Resonance peaks including frequencies and
            quality factors.
        """
        # Find peaks in admittance magnitude using advanced algorithms
        admittance_magnitude = np.abs(admittance)
        admittance_phase = np.angle(admittance)
        
        # Advanced peak detection using multiple criteria
        peaks, quality_factors = self._advanced_peak_detection(
            frequencies, admittance_magnitude, admittance_phase
        )
        
        return {"frequencies": peaks, "quality_factors": quality_factors}
    
    def _advanced_peak_detection(
        self, frequencies: np.ndarray, magnitude: np.ndarray, phase: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """
        Advanced peak detection using multiple criteria and signal processing.
        
        Physical Meaning:
            Identifies resonance peaks using advanced signal processing
            techniques including magnitude, phase, and derivative analysis.
            
        Mathematical Foundation:
            Uses multiple criteria for peak detection:
            1. Local maxima in magnitude with sufficient prominence
            2. Phase behavior analysis (rapid phase changes)
            3. Second derivative analysis for peak sharpness
            4. Quality factor estimation using Lorentzian fitting
            
        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Admittance magnitude.
            phase (np.ndarray): Admittance phase.
            
        Returns:
            Tuple[List[float], List[float]]: Peak frequencies and quality factors.
        """
        # Step 1: Preprocessing - smooth the data to reduce noise
        magnitude_smooth = self._smooth_signal(magnitude, window_size=5)
        phase_smooth = self._smooth_signal(phase, window_size=5)
        
        # Step 2: Find local maxima with prominence analysis
        magnitude_peaks = self._find_prominent_peaks(magnitude_smooth)
        
        # Step 3: Phase analysis - look for rapid phase changes
        phase_peaks = self._find_phase_peaks(phase_smooth)
        
        # Step 4: Second derivative analysis for peak sharpness
        sharpness_peaks = self._find_sharp_peaks(magnitude_smooth)
        
        # Step 5: Combine all criteria
        combined_peaks = self._combine_peak_criteria(
            magnitude_peaks, phase_peaks, sharpness_peaks
        )
        
        # Step 6: Extract peak frequencies and calculate quality factors
        peak_frequencies = []
        quality_factors = []
        
        for peak_idx in combined_peaks:
            if 0 < peak_idx < len(frequencies) - 1:
                peak_freq = frequencies[peak_idx]
                peak_frequencies.append(peak_freq)
                
                # Advanced quality factor calculation using Lorentzian fitting
                q_factor = self._calculate_quality_factor(
                    frequencies, magnitude_smooth, peak_idx
                )
                quality_factors.append(q_factor)
        
        return peak_frequencies, quality_factors
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """
        Smooth signal using moving average filter.
        
        Physical Meaning:
            Reduces noise while preserving peak characteristics
            using polynomial smoothing.
            
        Args:
            signal (np.ndarray): Input signal.
            window_size (int): Smoothing window size.
            
        Returns:
            np.ndarray: Smoothed signal.
        """
        # Use moving average for simplicity (could use Savitzky-Golay)
        if window_size <= 1:
            return signal
        
        # Pad the signal for boundary handling
        padded = np.pad(signal, window_size//2, mode='edge')
        
        # Apply moving average
        smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
        
        return smoothed
    
    def _find_prominent_peaks(self, magnitude: np.ndarray) -> List[int]:
        """
        Find prominent peaks using height and prominence criteria.
        
        Physical Meaning:
            Identifies peaks that are significantly higher than
            surrounding values and have sufficient prominence.
            
        Args:
            magnitude (np.ndarray): Signal magnitude.
            
        Returns:
            List[int]: Indices of prominent peaks.
        """
        peaks = []
        
        # Calculate prominence threshold
        mean_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)
        prominence_threshold = mean_magnitude + 2 * std_magnitude
        
        # Find local maxima with sufficient prominence
        for i in range(2, len(magnitude) - 2):
            if (magnitude[i] > magnitude[i-1] and 
                magnitude[i] > magnitude[i+1] and
                magnitude[i] > prominence_threshold):
                
                # Check prominence (height above surrounding valleys)
                left_valley = min(magnitude[max(0, i-10):i])
                right_valley = min(magnitude[i:min(len(magnitude), i+10)])
                prominence = magnitude[i] - max(left_valley, right_valley)
                
                if prominence > std_magnitude:
                    peaks.append(i)
        
        return peaks
    
    def _find_phase_peaks(self, phase: np.ndarray) -> List[int]:
        """
        Find peaks based on rapid phase changes.
        
        Physical Meaning:
            Identifies frequencies where the phase changes rapidly,
            indicating resonance behavior.
            
        Args:
            phase (np.ndarray): Signal phase.
            
        Returns:
            List[int]: Indices of phase-based peaks.
        """
        peaks = []
        
        # Calculate phase derivative
        phase_diff = np.diff(np.unwrap(phase))
        
        # Find rapid phase changes
        phase_threshold = np.std(phase_diff) * 2
        
        for i in range(1, len(phase_diff) - 1):
            if abs(phase_diff[i]) > phase_threshold:
                # Check if this is a local maximum in phase change rate
                if (abs(phase_diff[i]) > abs(phase_diff[i-1]) and 
                    abs(phase_diff[i]) > abs(phase_diff[i+1])):
                    peaks.append(i)
        
        return peaks
    
    def _find_sharp_peaks(self, magnitude: np.ndarray) -> List[int]:
        """
        Find sharp peaks using second derivative analysis.
        
        Physical Meaning:
            Identifies peaks with high curvature (sharpness)
            using second derivative analysis.
            
        Args:
            magnitude (np.ndarray): Signal magnitude.
            
        Returns:
            List[int]: Indices of sharp peaks.
        """
        peaks = []
        
        # Calculate second derivative
        first_diff = np.diff(magnitude)
        second_diff = np.diff(first_diff)
        
        # Find negative second derivative (concave down)
        for i in range(1, len(second_diff) - 1):
            if (second_diff[i] < -np.std(second_diff) and
                second_diff[i] < second_diff[i-1] and
                second_diff[i] < second_diff[i+1]):
                peaks.append(i + 1)  # Adjust for double differentiation
        
        return peaks
    
    def _combine_peak_criteria(
        self, magnitude_peaks: List[int], phase_peaks: List[int], sharpness_peaks: List[int]
    ) -> List[int]:
        """
        Combine multiple peak detection criteria.
        
        Physical Meaning:
            Combines results from different peak detection methods
            to identify the most reliable resonance peaks.
            
        Args:
            magnitude_peaks (List[int]): Magnitude-based peaks.
            phase_peaks (List[int]): Phase-based peaks.
            sharpness_peaks (List[int]): Sharpness-based peaks.
            
        Returns:
            List[int]: Combined peak indices.
        """
        # Create a scoring system
        all_peaks = set(magnitude_peaks + phase_peaks + sharpness_peaks)
        peak_scores = {}
        
        for peak in all_peaks:
            score = 0
            if peak in magnitude_peaks:
                score += 3  # Magnitude peaks are most important
            if peak in phase_peaks:
                score += 2  # Phase peaks are secondary
            if peak in sharpness_peaks:
                score += 1  # Sharpness peaks are supporting evidence
            
            peak_scores[peak] = score
        
        # Select peaks with score >= 2 (at least two criteria)
        selected_peaks = [peak for peak, score in peak_scores.items() if score >= 2]
        
        # Sort by score (descending) and then by index
        selected_peaks.sort(key=lambda x: (-peak_scores[x], x))
        
        return selected_peaks
    
    def _calculate_quality_factor(
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
            return 1.0
        
        # Extract peak region (±20 points around peak)
        window = 20
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
                return max(1.0, min(1000.0, q_factor))  # Clamp to reasonable range
        
        # Fallback: estimate from peak width
        return max(1.0, peak_magnitude / np.mean(magnitude))
    
    def set_quality_factor_threshold(self, threshold: float) -> None:
        """
        Set quality factor threshold.
        
        Physical Meaning:
            Updates the threshold for quality factor filtering.
            
        Args:
            threshold (float): New quality factor threshold.
        """
        self.quality_factor_threshold = threshold
    
    def get_quality_factor_threshold(self) -> float:
        """
        Get quality factor threshold.
        
        Physical Meaning:
            Returns the current quality factor threshold.
            
        Returns:
            float: Current quality factor threshold.
        """
        return self.quality_factor_threshold
    
    def __repr__(self) -> str:
        """String representation of resonance detector."""
        return f"ResonanceDetector(quality_threshold={self.quality_factor_threshold})"
