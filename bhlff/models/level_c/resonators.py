"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Resonator analysis module for Level C.

This module implements comprehensive resonator analysis for the 7D phase field
theory, including resonator detection, frequency analysis, and resonance
characteristics.

Physical Meaning:
    Analyzes resonator structures in the 7D phase field, including:
    - Resonator detection and classification
    - Frequency response analysis
    - Resonance characteristics and quality factors
    - Resonator-field interactions

Mathematical Foundation:
    Implements resonator analysis using:
    - Frequency domain analysis
    - Resonance peak detection
    - Quality factor calculations
    - Impedance analysis

Example:
    >>> analyzer = ResonatorAnalyzer(bvp_core)
    >>> results = analyzer.analyze_resonators(envelope)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class ResonatorAnalyzer:
    """
    Resonator analyzer for Level C analysis.
    
    Physical Meaning:
        Analyzes resonator structures in the 7D phase field, including
        their frequency characteristics, resonance properties, and
        interactions with the field.
        
    Mathematical Foundation:
        Uses frequency domain analysis, resonance peak detection,
        and quality factor calculations to analyze resonator behavior.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize resonator analyzer.
        
        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def analyze_resonators(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive resonator analysis.
        
        Physical Meaning:
            Analyzes all aspects of resonators in the 7D phase field,
            including detection, frequency analysis, and resonance
            characteristics.
            
        Mathematical Foundation:
            Combines multiple resonator analysis methods:
            - Frequency domain analysis for resonance detection
            - Peak detection for resonance identification
            - Quality factor calculations for resonance characterization
            - Impedance analysis for resonator properties
            
        Args:
            envelope (np.ndarray): BVP envelope in 7D space-time.
            
        Returns:
            Dict[str, Any]: Comprehensive resonator analysis results.
        """
        self.logger.info("Starting comprehensive resonator analysis")
        
        # Perform different types of resonator analysis
        frequency_analysis = self._analyze_frequency_response(envelope)
        resonance_peaks = self._detect_resonance_peaks(envelope)
        quality_factors = self._calculate_quality_factors(envelope, resonance_peaks)
        impedance_analysis = self._analyze_impedance(envelope)
        
        # Combine results
        resonator_results = {
            "frequency_analysis": frequency_analysis,
            "resonance_peaks": resonance_peaks,
            "quality_factors": quality_factors,
            "impedance_analysis": impedance_analysis,
            "resonator_summary": self._create_resonator_summary(
                frequency_analysis, resonance_peaks, quality_factors, impedance_analysis
            )
        }
        
        self.logger.info("Resonator analysis completed")
        return resonator_results
    
    def _analyze_frequency_response(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency response of the field."""
        # Compute FFT for frequency analysis
        envelope_fft = np.fft.fftn(envelope)
        frequency_magnitude = np.abs(envelope_fft)
        
        # Analyze frequency spectrum
        frequency_spectrum = self._analyze_frequency_spectrum(frequency_magnitude)
        
        # Find dominant frequencies
        dominant_frequencies = self._find_dominant_frequencies(frequency_magnitude)
        
        # Analyze frequency distribution
        frequency_distribution = self._analyze_frequency_distribution(frequency_magnitude)
        
        return {
            "frequency_magnitude": frequency_magnitude,
            "frequency_spectrum": frequency_spectrum,
            "dominant_frequencies": dominant_frequencies,
            "frequency_distribution": frequency_distribution,
            "analysis_method": "fft"
        }
    
    def _detect_resonance_peaks(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Detect resonance peaks in the field."""
        # Compute frequency magnitude
        envelope_fft = np.fft.fftn(envelope)
        frequency_magnitude = np.abs(envelope_fft)
        
        # Find peaks in frequency domain
        peaks = self._find_frequency_peaks(frequency_magnitude)
        
        # Analyze peak properties
        resonance_peaks = []
        for peak in peaks:
            peak_analysis = self._analyze_peak_properties(peak, frequency_magnitude)
            resonance_peaks.append(peak_analysis)
        
        return resonance_peaks
    
    def _calculate_quality_factors(self, envelope: np.ndarray, resonance_peaks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality factors for resonance peaks."""
        quality_factors = {}
        
        for i, peak in enumerate(resonance_peaks):
            # Calculate quality factor for this peak
            q_factor = self._calculate_peak_quality_factor(peak, envelope)
            quality_factors[f"peak_{i}"] = {
                "quality_factor": q_factor,
                "peak_properties": peak
            }
        
        # Calculate overall quality metrics
        overall_quality = self._calculate_overall_quality_metrics(quality_factors)
        
        return {
            "individual_quality_factors": quality_factors,
            "overall_quality_metrics": overall_quality
        }
    
    def _analyze_impedance(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze impedance characteristics."""
        # Compute field derivatives for impedance calculation
        field_derivatives = self._compute_field_derivatives(envelope)
        
        # Calculate impedance components
        impedance_components = self._calculate_impedance_components(envelope, field_derivatives)
        
        # Analyze impedance characteristics
        impedance_characteristics = self._analyze_impedance_characteristics(impedance_components)
        
        return {
            "impedance_components": impedance_components,
            "impedance_characteristics": impedance_characteristics,
            "analysis_method": "derivative_based"
        }
    
    def _analyze_frequency_spectrum(self, frequency_magnitude: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency spectrum properties."""
        return {
            "total_power": float(np.sum(frequency_magnitude**2)),
            "mean_power": float(np.mean(frequency_magnitude**2)),
            "power_std": float(np.std(frequency_magnitude**2)),
            "max_power": float(np.max(frequency_magnitude**2)),
            "spectrum_width": float(np.std(frequency_magnitude))
        }
    
    def _find_dominant_frequencies(self, frequency_magnitude: np.ndarray) -> List[Dict[str, Any]]:
        """Find dominant frequencies in the spectrum."""
        # Find frequencies with high magnitude
        threshold = np.mean(frequency_magnitude) + 2 * np.std(frequency_magnitude)
        high_freq_mask = frequency_magnitude > threshold
        
        # Get coordinates of high-frequency points
        high_freq_coords = np.where(high_freq_mask)
        
        dominant_frequencies = []
        for i in range(len(high_freq_coords[0])):
            coords = tuple(high_freq_coords[j][i] for j in range(len(high_freq_coords)))
            dominant_frequencies.append({
                "frequency_coordinates": coords,
                "magnitude": float(frequency_magnitude[coords]),
                "relative_strength": float(frequency_magnitude[coords] / np.max(frequency_magnitude))
            })
        
        return dominant_frequencies
    
    def _analyze_frequency_distribution(self, frequency_magnitude: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution of frequencies."""
        # Compute frequency distribution statistics
        magnitude_flat = frequency_magnitude.flatten()
        
        return {
            "distribution_mean": float(np.mean(magnitude_flat)),
            "distribution_std": float(np.std(magnitude_flat)),
            "distribution_skewness": float(self._calculate_skewness(magnitude_flat)),
            "distribution_kurtosis": float(self._calculate_kurtosis(magnitude_flat)),
            "high_frequency_fraction": float(np.sum(magnitude_flat > np.mean(magnitude_flat)) / len(magnitude_flat))
        }
    
    def _find_frequency_peaks(self, frequency_magnitude: np.ndarray) -> List[Dict[str, Any]]:
        """Find peaks in frequency domain."""
        from scipy import ndimage
        
        # Find local maxima
        local_maxima = ndimage.maximum_filter(frequency_magnitude, size=3) == frequency_magnitude
        
        # Filter peaks by magnitude
        threshold = np.mean(frequency_magnitude) + np.std(frequency_magnitude)
        significant_peaks = local_maxima & (frequency_magnitude > threshold)
        
        # Extract peak coordinates and properties
        peak_coords = np.where(significant_peaks)
        peaks = []
        
        for i in range(len(peak_coords[0])):
            coords = tuple(peak_coords[j][i] for j in range(len(peak_coords)))
            peaks.append({
                "coordinates": coords,
                "magnitude": float(frequency_magnitude[coords]),
                "peak_index": i
            })
        
        return peaks
    
    def _analyze_peak_properties(self, peak: Dict[str, Any], frequency_magnitude: np.ndarray) -> Dict[str, Any]:
        """Analyze properties of a resonance peak."""
        coords = peak["coordinates"]
        magnitude = peak["magnitude"]
        
        # Calculate peak width (simplified)
        peak_width = self._estimate_peak_width(coords, frequency_magnitude)
        
        # Calculate peak sharpness
        peak_sharpness = self._calculate_peak_sharpness(coords, frequency_magnitude)
        
        return {
            "coordinates": coords,
            "magnitude": magnitude,
            "peak_width": peak_width,
            "peak_sharpness": peak_sharpness,
            "relative_strength": float(magnitude / np.max(frequency_magnitude))
        }
    
    def _calculate_peak_quality_factor(self, peak: Dict[str, Any], envelope: np.ndarray) -> float:
        """Calculate quality factor for a resonance peak."""
        # Simplified quality factor calculation
        magnitude = peak["magnitude"]
        width = peak.get("peak_width", 1.0)
        
        # Q = frequency / bandwidth (simplified)
        q_factor = magnitude / max(width, 1e-6)
        
        return float(q_factor)
    
    def _calculate_overall_quality_metrics(self, quality_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics."""
        if not quality_factors:
            return {
                "mean_quality_factor": 0.0,
                "max_quality_factor": 0.0,
                "quality_distribution": "none"
            }
        
        q_values = [qf["quality_factor"] for qf in quality_factors.values()]
        
        return {
            "mean_quality_factor": float(np.mean(q_values)),
            "max_quality_factor": float(np.max(q_values)),
            "quality_std": float(np.std(q_values)),
            "quality_distribution": "high" if np.mean(q_values) > 1.0 else "low"
        }
    
    def _compute_field_derivatives(self, envelope: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute field derivatives for impedance calculation."""
        derivatives = {}
        
        for dim in range(envelope.ndim):
            derivatives[f"dim_{dim}"] = np.gradient(envelope, axis=dim)
        
        return derivatives
    
    def _calculate_impedance_components(self, envelope: np.ndarray, derivatives: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate impedance components."""
        # Simplified impedance calculation
        field_magnitude = np.abs(envelope)
        derivative_magnitude = np.sqrt(sum(np.abs(deriv) for deriv in derivatives.values()))
        
        # Calculate impedance as ratio of field to derivative
        impedance = field_magnitude / (derivative_magnitude + 1e-15)
        
        return {
            "impedance": impedance,
            "field_magnitude": field_magnitude,
            "derivative_magnitude": derivative_magnitude,
            "mean_impedance": float(np.mean(impedance)),
            "impedance_std": float(np.std(impedance))
        }
    
    def _analyze_impedance_characteristics(self, impedance_components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impedance characteristics."""
        impedance = impedance_components["impedance"]
        
        return {
            "impedance_range": float(np.max(impedance) - np.min(impedance)),
            "impedance_variation": float(np.std(impedance) / np.mean(impedance)),
            "impedance_quality": "high" if np.std(impedance) < np.mean(impedance) else "low",
            "resonance_indicators": self._identify_resonance_indicators(impedance)
        }
    
    def _identify_resonance_indicators(self, impedance: np.ndarray) -> Dict[str, Any]:
        """Identify indicators of resonance behavior."""
        # Look for sharp changes in impedance
        impedance_gradient = np.gradient(impedance)
        sharp_changes = np.abs(impedance_gradient) > np.std(impedance_gradient)
        
        return {
            "sharp_change_count": int(np.sum(sharp_changes)),
            "resonance_likelihood": "high" if np.sum(sharp_changes) > 0 else "low",
            "impedance_smoothness": float(1.0 / (1.0 + np.std(impedance_gradient)))
        }
    
    def _estimate_peak_width(self, coords: Tuple, frequency_magnitude: np.ndarray) -> float:
        """Estimate width of a resonance peak."""
        # Simplified peak width estimation
        magnitude = frequency_magnitude[coords]
        threshold = magnitude * 0.5  # Half-maximum width
        
        # Count points above threshold in neighborhood
        neighborhood_size = 2
        width_count = 0
        
        for dim in range(len(coords)):
            for offset in range(-neighborhood_size, neighborhood_size + 1):
                new_coords = list(coords)
                new_coords[dim] = max(0, min(frequency_magnitude.shape[dim] - 1, coords[dim] + offset))
                new_coords = tuple(new_coords)
                
                if frequency_magnitude[new_coords] > threshold:
                    width_count += 1
        
        return float(width_count)
    
    def _calculate_peak_sharpness(self, coords: Tuple, frequency_magnitude: np.ndarray) -> float:
        """Calculate sharpness of a resonance peak."""
        # Calculate second derivative as measure of sharpness
        magnitude = frequency_magnitude[coords]
        
        # Simplified sharpness calculation
        sharpness = magnitude / max(np.mean(frequency_magnitude), 1e-15)
        
        return float(sharpness)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4))
    
    def _create_resonator_summary(self, frequency_analysis: Dict[str, Any],
                                 resonance_peaks: List[Dict[str, Any]],
                                 quality_factors: Dict[str, Any],
                                 impedance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of resonator analysis."""
        return {
            "total_resonators_detected": len(resonance_peaks),
            "resonance_quality": quality_factors["overall_quality_metrics"]["quality_distribution"],
            "frequency_analysis_complete": True,
            "impedance_analysis_complete": True,
            "analysis_methods": ["frequency_domain", "peak_detection", "quality_factors", "impedance"]
        }
