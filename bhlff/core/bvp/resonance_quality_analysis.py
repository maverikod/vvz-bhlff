"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced resonance quality factor analysis for BVP impedance analysis.

This module implements advanced algorithms for resonance quality analysis,
including statistical analysis, quality factor optimization, and
resonance characterization.
"""

import numpy as np
from typing import List, Dict, Tuple

from .bvp_constants import BVPConstants


class ResonanceQualityAnalysis:
    """
    Advanced resonance quality factor analysis.

    Physical Meaning:
        Provides advanced analysis of resonance quality factors,
        including statistical analysis, optimization, and
        resonance characterization.
    """

    def __init__(self, constants: BVPConstants):
        """
        Initialize advanced quality analyzer.

        Args:
            constants (BVPConstants): BVP constants instance.
        """
        self.constants = constants

    def optimize_quality_factors(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_indices: List[int]
    ) -> List[float]:
        """
        Optimize quality factors using advanced fitting techniques.

        Physical Meaning:
            Optimizes quality factors using advanced fitting techniques
            to improve accuracy and reliability of resonance analysis.

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Magnitude response array.
            peak_indices (List[int]): List of peak indices.

        Returns:
            List[float]: Optimized quality factors.
        """
        optimized_quality_factors = []
        
        for peak_idx in peak_indices:
            # Extract peak region
            peak_region = self._extract_peak_region(frequencies, magnitude, peak_idx)
            
            # Perform advanced fitting
            optimized_params = self._advanced_lorentzian_fitting(peak_region)
            
            # Calculate optimized quality factor
            quality_factor = self._calculate_optimized_quality_factor(optimized_params)
            optimized_quality_factors.append(quality_factor)
        
        return optimized_quality_factors

    def analyze_resonance_characteristics(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_indices: List[int]
    ) -> Dict[str, any]:
        """
        Analyze comprehensive resonance characteristics.

        Physical Meaning:
            Performs comprehensive analysis of resonance characteristics,
            including quality factors, resonance shapes, and
            frequency domain properties.

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Magnitude response array.
            peak_indices (List[int]): List of peak indices.

        Returns:
            Dict[str, any]: Comprehensive resonance characteristics.
        """
        characteristics = {
            'quality_factors': [],
            'resonance_shapes': [],
            'frequency_properties': [],
            'amplitude_properties': [],
            'resonance_types': []
        }
        
        for peak_idx in peak_indices:
            # Extract peak region
            peak_region = self._extract_peak_region(frequencies, magnitude, peak_idx)
            
            # Analyze resonance shape
            resonance_shape = self._analyze_resonance_shape(peak_region)
            characteristics['resonance_shapes'].append(resonance_shape)
            
            # Analyze frequency properties
            frequency_properties = self._analyze_frequency_properties(peak_region)
            characteristics['frequency_properties'].append(frequency_properties)
            
            # Analyze amplitude properties
            amplitude_properties = self._analyze_amplitude_properties(peak_region)
            characteristics['amplitude_properties'].append(amplitude_properties)
            
            # Classify resonance type
            resonance_type = self._classify_resonance_type(peak_region)
            characteristics['resonance_types'].append(resonance_type)
            
            # Calculate quality factor
            quality_factor = self._calculate_quality_factor_from_characteristics(
                resonance_shape, frequency_properties
            )
            characteristics['quality_factors'].append(quality_factor)
        
        return characteristics

    def compare_resonance_quality(
        self, quality_factors_1: List[float], quality_factors_2: List[float]
    ) -> Dict[str, float]:
        """
        Compare quality factors between two sets of resonances.

        Physical Meaning:
            Compares quality factors between two sets of resonances
            to analyze differences in resonance characteristics.

        Args:
            quality_factors_1 (List[float]): First set of quality factors.
            quality_factors_2 (List[float]): Second set of quality factors.

        Returns:
            Dict[str, float]: Comparison results.
        """
        if not quality_factors_1 or not quality_factors_2:
            return {
                'mean_difference': 0.0,
                'std_difference': 0.0,
                'correlation': 0.0,
                'significance': 0.0
            }
        
        # Calculate statistics
        mean_1 = np.mean(quality_factors_1)
        mean_2 = np.mean(quality_factors_2)
        std_1 = np.std(quality_factors_1)
        std_2 = np.std(quality_factors_2)
        
        # Calculate differences
        mean_difference = mean_1 - mean_2
        std_difference = np.sqrt(std_1**2 + std_2**2)
        
        # Calculate correlation
        if len(quality_factors_1) == len(quality_factors_2):
            correlation = np.corrcoef(quality_factors_1, quality_factors_2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Calculate significance (simplified)
        significance = abs(mean_difference) / std_difference if std_difference > 0 else 0.0
        
        return {
            'mean_difference': mean_difference,
            'std_difference': std_difference,
            'correlation': correlation,
            'significance': significance
        }

    def _extract_peak_region(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_idx: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract region around a resonance peak.

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Magnitude response array.
            peak_idx (int): Index of the resonance peak.

        Returns:
            Dict[str, np.ndarray]: Peak region data.
        """
        # Define region width (adjustable parameter)
        region_width = 20  # Number of points around peak
        
        # Calculate region bounds
        start_idx = max(0, peak_idx - region_width // 2)
        end_idx = min(len(frequencies), peak_idx + region_width // 2 + 1)
        
        # Extract region
        region_frequencies = frequencies[start_idx:end_idx]
        region_magnitude = magnitude[start_idx:end_idx]
        
        return {
            'frequencies': region_frequencies,
            'magnitude': region_magnitude,
            'peak_idx': peak_idx - start_idx
        }

    def _advanced_lorentzian_fitting(self, peak_region: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform advanced Lorentzian fitting.

        Args:
            peak_region (Dict[str, np.ndarray]): Peak region data.

        Returns:
            Dict[str, float]: Advanced fitting parameters.
        """
        frequencies = peak_region['frequencies']
        magnitude = peak_region['magnitude']
        peak_idx = peak_region['peak_idx']
        
        # Initial parameter estimates
        amplitude = magnitude[peak_idx]
        center = frequencies[peak_idx]
        
        # Estimate FWHM using more sophisticated method
        half_max = amplitude / 2.0
        fwhm_indices = np.where(magnitude >= half_max)[0]
        
        if len(fwhm_indices) > 1:
            fwhm = frequencies[fwhm_indices[-1]] - frequencies[fwhm_indices[0]]
        else:
            fwhm = (frequencies[-1] - frequencies[0]) / 10.0  # Fallback estimate
        
        # Additional parameters for advanced fitting
        baseline = np.min(magnitude)
        noise_level = np.std(magnitude)
        
        return {
            'amplitude': amplitude,
            'center': center,
            'fwhm': fwhm,
            'baseline': baseline,
            'noise_level': noise_level
        }

    def _calculate_optimized_quality_factor(self, params: Dict[str, float]) -> float:
        """
        Calculate optimized quality factor.

        Args:
            params (Dict[str, float]): Advanced fitting parameters.

        Returns:
            float: Optimized quality factor.
        """
        center = params['center']
        fwhm = params['fwhm']
        
        # Apply optimization corrections
        optimized_fwhm = fwhm * (1.0 + params['noise_level'] / params['amplitude'])
        
        quality_factor = center / optimized_fwhm if optimized_fwhm > 0 else 0.0
        return quality_factor

    def _analyze_resonance_shape(self, peak_region: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze resonance shape characteristics.

        Args:
            peak_region (Dict[str, np.ndarray]): Peak region data.

        Returns:
            Dict[str, float]: Resonance shape characteristics.
        """
        magnitude = peak_region['magnitude']
        peak_idx = peak_region['peak_idx']
        
        # Calculate shape metrics
        peak_amplitude = magnitude[peak_idx]
        peak_width = self._calculate_peak_width(magnitude, peak_idx)
        peak_symmetry = self._calculate_peak_symmetry(magnitude, peak_idx)
        
        return {
            'amplitude': peak_amplitude,
            'width': peak_width,
            'symmetry': peak_symmetry
        }

    def _analyze_frequency_properties(self, peak_region: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze frequency properties.

        Args:
            peak_region (Dict[str, np.ndarray]): Peak region data.

        Returns:
            Dict[str, float]: Frequency properties.
        """
        frequencies = peak_region['frequencies']
        magnitude = peak_region['magnitude']
        peak_idx = peak_region['peak_idx']
        
        # Calculate frequency metrics
        center_frequency = frequencies[peak_idx]
        frequency_span = frequencies[-1] - frequencies[0]
        frequency_resolution = frequency_span / len(frequencies)
        
        return {
            'center_frequency': center_frequency,
            'frequency_span': frequency_span,
            'frequency_resolution': frequency_resolution
        }

    def _analyze_amplitude_properties(self, peak_region: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze amplitude properties.

        Args:
            peak_region (Dict[str, np.ndarray]): Peak region data.

        Returns:
            Dict[str, float]: Amplitude properties.
        """
        magnitude = peak_region['magnitude']
        
        # Calculate amplitude metrics
        max_amplitude = np.max(magnitude)
        min_amplitude = np.min(magnitude)
        mean_amplitude = np.mean(magnitude)
        std_amplitude = np.std(magnitude)
        
        return {
            'max_amplitude': max_amplitude,
            'min_amplitude': min_amplitude,
            'mean_amplitude': mean_amplitude,
            'std_amplitude': std_amplitude
        }

    def _classify_resonance_type(self, peak_region: Dict[str, np.ndarray]) -> str:
        """
        Classify resonance type.

        Args:
            peak_region (Dict[str, np.ndarray]): Peak region data.

        Returns:
            str: Resonance type classification.
        """
        magnitude = peak_region['magnitude']
        peak_idx = peak_region['peak_idx']
        
        # Simple classification based on shape
        peak_amplitude = magnitude[peak_idx]
        mean_amplitude = np.mean(magnitude)
        
        if peak_amplitude > 2.0 * mean_amplitude:
            return 'strong'
        elif peak_amplitude > 1.5 * mean_amplitude:
            return 'moderate'
        else:
            return 'weak'

    def _calculate_quality_factor_from_characteristics(
        self, resonance_shape: Dict[str, float], frequency_properties: Dict[str, float]
    ) -> float:
        """
        Calculate quality factor from resonance characteristics.

        Args:
            resonance_shape (Dict[str, float]): Resonance shape characteristics.
            frequency_properties (Dict[str, float]): Frequency properties.

        Returns:
            float: Quality factor.
        """
        center_frequency = frequency_properties['center_frequency']
        peak_width = resonance_shape['width']
        
        quality_factor = center_frequency / peak_width if peak_width > 0 else 0.0
        return quality_factor

    def _calculate_peak_width(self, magnitude: np.ndarray, peak_idx: int) -> float:
        """
        Calculate peak width.

        Args:
            magnitude (np.ndarray): Magnitude array.
            peak_idx (int): Peak index.

        Returns:
            float: Peak width.
        """
        peak_amplitude = magnitude[peak_idx]
        half_max = peak_amplitude / 2.0
        
        # Find indices where magnitude is above half maximum
        above_half_max = np.where(magnitude >= half_max)[0]
        
        if len(above_half_max) > 1:
            width = above_half_max[-1] - above_half_max[0]
        else:
            width = 1.0  # Fallback
        
        return float(width)

    def _calculate_peak_symmetry(self, magnitude: np.ndarray, peak_idx: int) -> float:
        """
        Calculate peak symmetry.

        Args:
            magnitude (np.ndarray): Magnitude array.
            peak_idx (int): Peak index.

        Returns:
            float: Peak symmetry metric.
        """
        # Calculate symmetry by comparing left and right sides
        left_side = magnitude[:peak_idx]
        right_side = magnitude[peak_idx+1:]
        
        if len(left_side) == 0 or len(right_side) == 0:
            return 1.0  # Perfect symmetry if no sides
        
        # Calculate mean difference
        mean_left = np.mean(left_side)
        mean_right = np.mean(right_side)
        
        # Symmetry metric (1.0 = perfect symmetry)
        symmetry = 1.0 - abs(mean_left - mean_right) / (mean_left + mean_right)
        return max(0.0, min(1.0, symmetry))
