"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Beating analysis module for Level C.

This module implements comprehensive beating analysis for the 7D phase field
theory, including mode beating detection, interference analysis, and frequency
beating characteristics.

Physical Meaning:
    Analyzes mode beating in the 7D phase field, including:
    - Mode beating detection and classification
    - Interference pattern analysis
    - Beating frequency calculations
    - Mode coupling analysis

Mathematical Foundation:
    Implements beating analysis using:
    - Frequency domain analysis
    - Interference pattern detection
    - Beating frequency calculations
    - Mode coupling analysis

Example:
    >>> analyzer = BeatingAnalyzer(bvp_core)
    >>> results = analyzer.analyze_beating(envelope)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class BeatingAnalyzer:
    """
    Beating analyzer for Level C analysis.
    
    Physical Meaning:
        Analyzes mode beating in the 7D phase field, including
        interference patterns, beating frequencies, and mode
        coupling effects that emerge from field interactions.
        
    Mathematical Foundation:
        Uses frequency domain analysis, interference pattern detection,
        and beating frequency calculations to study mode interactions.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize beating analyzer.
        
        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def analyze_beating(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive beating analysis.
        
        Physical Meaning:
            Analyzes all aspects of mode beating in the 7D phase field,
            including detection, frequency analysis, interference patterns,
            and mode coupling effects.
            
        Mathematical Foundation:
            Combines multiple beating analysis methods:
            - Frequency domain analysis for mode detection
            - Interference pattern analysis for beating detection
            - Beating frequency calculations for mode interactions
            - Mode coupling analysis for interaction strength
            
        Args:
            envelope (np.ndarray): BVP envelope in 7D space-time.
            
        Returns:
            Dict[str, Any]: Comprehensive beating analysis results.
        """
        self.logger.info("Starting comprehensive beating analysis")
        
        # Perform different types of beating analysis
        mode_analysis = self._analyze_mode_beating(envelope)
        interference_analysis = self._analyze_interference_patterns(envelope)
        frequency_analysis = self._analyze_beating_frequencies(envelope)
        coupling_analysis = self._analyze_mode_coupling(envelope)
        
        # Combine results
        beating_results = {
            "mode_analysis": mode_analysis,
            "interference_analysis": interference_analysis,
            "frequency_analysis": frequency_analysis,
            "coupling_analysis": coupling_analysis,
            "beating_summary": self._create_beating_summary(
                mode_analysis, interference_analysis, 
                frequency_analysis, coupling_analysis
            )
        }
        
        self.logger.info("Beating analysis completed")
        return beating_results
    
    def _analyze_mode_beating(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze mode beating properties."""
        # Detect modes in the field
        modes = self._detect_modes(envelope)
        
        # Analyze mode interactions
        mode_interactions = self._analyze_mode_interactions(envelope, modes)
        
        # Calculate beating characteristics
        beating_characteristics = self._calculate_beating_characteristics(envelope, modes)
        
        return {
            "modes": modes,
            "mode_interactions": mode_interactions,
            "beating_characteristics": beating_characteristics,
            "analysis_method": "mode_detection"
        }
    
    def _analyze_interference_patterns(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze interference patterns."""
        # Detect interference patterns
        interference_patterns = self._detect_interference_patterns(envelope)
        
        # Analyze pattern properties
        pattern_properties = self._analyze_pattern_properties(envelope, interference_patterns)
        
        # Calculate interference strength
        interference_strength = self._calculate_interference_strength(envelope, interference_patterns)
        
        return {
            "interference_patterns": interference_patterns,
            "pattern_properties": pattern_properties,
            "interference_strength": interference_strength,
            "analysis_method": "interference_detection"
        }
    
    def _analyze_beating_frequencies(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze beating frequencies."""
        # Compute frequency spectrum
        frequency_spectrum = self._compute_frequency_spectrum(envelope)
        
        # Detect beating frequencies
        beating_frequencies = self._detect_beating_frequencies(envelope, frequency_spectrum)
        
        # Analyze frequency characteristics
        frequency_characteristics = self._analyze_frequency_characteristics(beating_frequencies)
        
        return {
            "frequency_spectrum": frequency_spectrum,
            "beating_frequencies": beating_frequencies,
            "frequency_characteristics": frequency_characteristics,
            "analysis_method": "frequency_analysis"
        }
    
    def _analyze_mode_coupling(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze mode coupling effects."""
        # Detect mode coupling
        mode_coupling = self._detect_mode_coupling(envelope)
        
        # Analyze coupling strength
        coupling_strength = self._analyze_coupling_strength(envelope, mode_coupling)
        
        # Calculate coupling efficiency
        coupling_efficiency = self._calculate_coupling_efficiency(envelope, mode_coupling)
        
        return {
            "mode_coupling": mode_coupling,
            "coupling_strength": coupling_strength,
            "coupling_efficiency": coupling_efficiency,
            "analysis_method": "coupling_analysis"
        }
    
    def _detect_modes(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Detect modes in the field."""
        modes = []
        
        # Use FFT to detect frequency modes
        envelope_fft = np.fft.fftn(envelope)
        frequency_magnitude = np.abs(envelope_fft)
        
        # Find dominant frequency modes
        threshold = np.mean(frequency_magnitude) + 2 * np.std(frequency_magnitude)
        mode_mask = frequency_magnitude > threshold
        
        # Extract mode properties
        mode_coords = np.where(mode_mask)
        for i in range(len(mode_coords[0])):
            coords = tuple(mode_coords[j][i] for j in range(len(mode_coords)))
            modes.append({
                "mode_id": i,
                "frequency_coordinates": coords,
                "amplitude": float(frequency_magnitude[coords]),
                "phase": float(np.angle(envelope_fft[coords])),
                "mode_strength": float(frequency_magnitude[coords] / np.max(frequency_magnitude))
            })
        
        return modes
    
    def _analyze_mode_interactions(self, envelope: np.ndarray, modes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interactions between modes."""
        if len(modes) < 2:
            return {
                "interaction_count": 0,
                "interaction_strength": 0.0,
                "interaction_type": "none"
            }
        
        # Calculate mode interaction matrix
        interaction_matrix = self._calculate_interaction_matrix(modes)
        
        # Analyze interaction patterns
        interaction_patterns = self._analyze_interaction_patterns(interaction_matrix)
        
        # Calculate overall interaction strength
        interaction_strength = self._calculate_interaction_strength(interaction_matrix)
        
        return {
            "interaction_matrix": interaction_matrix,
            "interaction_patterns": interaction_patterns,
            "interaction_strength": float(interaction_strength),
            "interaction_count": len(modes) * (len(modes) - 1) // 2,
            "interaction_type": "strong" if interaction_strength > 0.5 else "weak"
        }
    
    def _calculate_beating_characteristics(self, envelope: np.ndarray, modes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate beating characteristics."""
        if len(modes) < 2:
            return {
                "beating_frequency": 0.0,
                "beating_amplitude": 0.0,
                "beating_phase": 0.0,
                "beating_quality": "none"
            }
        
        # Calculate beating frequency between first two modes
        mode1_freq = self._frequency_from_coordinates(modes[0]["frequency_coordinates"])
        mode2_freq = self._frequency_from_coordinates(modes[1]["frequency_coordinates"])
        beating_frequency = abs(mode1_freq - mode2_freq)
        
        # Calculate beating amplitude
        beating_amplitude = min(modes[0]["amplitude"], modes[1]["amplitude"])
        
        # Calculate beating phase
        beating_phase = modes[0]["phase"] - modes[1]["phase"]
        
        return {
            "beating_frequency": float(beating_frequency),
            "beating_amplitude": float(beating_amplitude),
            "beating_phase": float(beating_phase),
            "beating_quality": "high" if beating_frequency > 0.1 else "low"
        }
    
    def _detect_interference_patterns(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Detect interference patterns in the field."""
        patterns = []
        
        # Analyze field amplitude for interference patterns
        amplitude = np.abs(envelope)
        
        # Detect periodic variations (interference fringes)
        periodic_variations = self._detect_periodic_variations(amplitude)
        patterns.extend(periodic_variations)
        
        # Detect standing wave patterns
        standing_waves = self._detect_standing_waves(amplitude)
        patterns.extend(standing_waves)
        
        # Detect beat patterns
        beat_patterns = self._detect_beat_patterns(amplitude)
        patterns.extend(beat_patterns)
        
        return patterns
    
    def _analyze_pattern_properties(self, envelope: np.ndarray, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze properties of interference patterns."""
        if not patterns:
            return {
                "pattern_count": 0,
                "pattern_complexity": 0.0,
                "pattern_regularity": "none"
            }
        
        # Calculate pattern statistics
        pattern_count = len(patterns)
        pattern_complexity = self._calculate_pattern_complexity(patterns)
        pattern_regularity = self._assess_pattern_regularity(patterns)
        
        return {
            "pattern_count": pattern_count,
            "pattern_complexity": float(pattern_complexity),
            "pattern_regularity": pattern_regularity,
            "pattern_quality": "high" if pattern_complexity > 0.5 else "low"
        }
    
    def _calculate_interference_strength(self, envelope: np.ndarray, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate interference strength."""
        if not patterns:
            return {
                "interference_strength": 0.0,
                "interference_visibility": 0.0,
                "interference_contrast": 0.0
            }
        
        # Calculate interference strength based on pattern amplitude
        pattern_amplitudes = [pattern.get("amplitude", 0.0) for pattern in patterns]
        interference_strength = np.mean(pattern_amplitudes) if pattern_amplitudes else 0.0
        
        # Calculate interference visibility
        interference_visibility = self._calculate_interference_visibility(envelope)
        
        # Calculate interference contrast
        interference_contrast = self._calculate_interference_contrast(envelope)
        
        return {
            "interference_strength": float(interference_strength),
            "interference_visibility": float(interference_visibility),
            "interference_contrast": float(interference_contrast),
            "interference_quality": "high" if interference_strength > 0.5 else "low"
        }
    
    def _compute_frequency_spectrum(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Compute frequency spectrum for beating analysis."""
        # Compute FFT
        envelope_fft = np.fft.fftn(envelope)
        frequency_magnitude = np.abs(envelope_fft)
        
        # Analyze spectrum properties
        spectrum_properties = {
            "total_power": float(np.sum(frequency_magnitude**2)),
            "mean_power": float(np.mean(frequency_magnitude**2)),
            "power_std": float(np.std(frequency_magnitude**2)),
            "max_power": float(np.max(frequency_magnitude**2)),
            "spectrum_width": float(np.std(frequency_magnitude))
        }
        
        return {
            "frequency_magnitude": frequency_magnitude,
            "spectrum_properties": spectrum_properties,
            "dominant_frequencies": self._find_dominant_frequencies(frequency_magnitude)
        }
    
    def _detect_beating_frequencies(self, envelope: np.ndarray, frequency_spectrum: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect beating frequencies."""
        beating_frequencies = []
        
        # Find frequency pairs that could produce beating
        dominant_frequencies = frequency_spectrum["dominant_frequencies"]
        
        for i in range(len(dominant_frequencies)):
            for j in range(i + 1, len(dominant_frequencies)):
                freq1 = dominant_frequencies[i]
                freq2 = dominant_frequencies[j]
                
                # Calculate beating frequency
                beating_freq = abs(freq1["frequency"] - freq2["frequency"])
                
                if beating_freq > 0.01:  # Minimum beating frequency threshold
                    beating_frequencies.append({
                        "beating_frequency": float(beating_freq),
                        "mode1_frequency": float(freq1["frequency"]),
                        "mode2_frequency": float(freq2["frequency"]),
                        "beating_amplitude": float(min(freq1["amplitude"], freq2["amplitude"])),
                        "beating_strength": "strong" if beating_freq > 0.1 else "weak"
                    })
        
        return beating_frequencies
    
    def _analyze_frequency_characteristics(self, beating_frequencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of beating frequencies."""
        if not beating_frequencies:
            return {
                "beating_count": 0,
                "mean_beating_frequency": 0.0,
                "beating_distribution": "none"
            }
        
        # Calculate beating statistics
        beating_freqs = [bf["beating_frequency"] for bf in beating_frequencies]
        beating_amplitudes = [bf["beating_amplitude"] for bf in beating_frequencies]
        
        return {
            "beating_count": len(beating_frequencies),
            "mean_beating_frequency": float(np.mean(beating_freqs)),
            "beating_frequency_std": float(np.std(beating_freqs)),
            "mean_beating_amplitude": float(np.mean(beating_amplitudes)),
            "beating_distribution": "wide" if np.std(beating_freqs) > 0.1 else "narrow",
            "beating_quality": "high" if np.mean(beating_amplitudes) > 0.5 else "low"
        }
    
    def _detect_mode_coupling(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Detect mode coupling effects."""
        # Analyze field correlations as coupling indicators
        field_correlations = self._analyze_field_correlations(envelope)
        
        # Detect coupling patterns
        coupling_patterns = self._detect_coupling_patterns(envelope)
        
        # Calculate coupling strength
        coupling_strength = self._calculate_coupling_strength_metric(envelope)
        
        return {
            "field_correlations": field_correlations,
            "coupling_patterns": coupling_patterns,
            "coupling_strength_metric": float(coupling_strength),
            "coupling_type": "strong" if coupling_strength > 0.5 else "weak"
        }
    
    def _analyze_coupling_strength(self, envelope: np.ndarray, mode_coupling: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coupling strength."""
        coupling_strength = mode_coupling["coupling_strength_metric"]
        
        return {
            "overall_coupling_strength": float(coupling_strength),
            "coupling_efficiency": float(coupling_strength),
            "coupling_stability": "high" if coupling_strength > 0.7 else "low",
            "coupling_quality": "strong" if coupling_strength > 0.5 else "weak"
        }
    
    def _calculate_coupling_efficiency(self, envelope: np.ndarray, mode_coupling: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate coupling efficiency."""
        coupling_strength = mode_coupling["coupling_strength_metric"]
        
        # Calculate efficiency based on coupling strength and field properties
        field_efficiency = np.mean(np.abs(envelope)) / np.max(np.abs(envelope))
        coupling_efficiency = coupling_strength * field_efficiency
        
        return {
            "coupling_efficiency": float(coupling_efficiency),
            "field_efficiency": float(field_efficiency),
            "efficiency_rating": "high" if coupling_efficiency > 0.5 else "low"
        }
    
    # Helper methods
    def _calculate_interaction_matrix(self, modes: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate interaction matrix between modes."""
        n_modes = len(modes)
        interaction_matrix = np.zeros((n_modes, n_modes))
        
        for i in range(n_modes):
            for j in range(n_modes):
                if i != j:
                    # Calculate interaction strength based on mode properties
                    interaction_strength = min(modes[i]["amplitude"], modes[j]["amplitude"]) / max(modes[i]["amplitude"], modes[j]["amplitude"])
                    interaction_matrix[i, j] = interaction_strength
        
        return interaction_matrix
    
    def _analyze_interaction_patterns(self, interaction_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze interaction patterns."""
        return {
            "interaction_symmetry": "symmetric" if np.allclose(interaction_matrix, interaction_matrix.T) else "asymmetric",
            "interaction_density": float(np.sum(interaction_matrix > 0) / interaction_matrix.size),
            "interaction_strength_distribution": "uniform" if np.std(interaction_matrix) < 0.1 else "varied"
        }
    
    def _calculate_interaction_strength(self, interaction_matrix: np.ndarray) -> float:
        """Calculate overall interaction strength."""
        return float(np.mean(interaction_matrix))
    
    def _frequency_from_coordinates(self, coords: Tuple) -> float:
        """Convert frequency coordinates to frequency value."""
        # Simplified frequency calculation
        return float(np.sqrt(sum(c**2 for c in coords)))
    
    def _detect_periodic_variations(self, amplitude: np.ndarray) -> List[Dict[str, Any]]:
        """Detect periodic variations in amplitude."""
        patterns = []
        
        # Use FFT to detect periodic variations
        amplitude_fft = np.fft.fftn(amplitude)
        freq_magnitude = np.abs(amplitude_fft)
        
        # Find significant frequency components
        threshold = np.mean(freq_magnitude) + np.std(freq_magnitude)
        significant_freqs = freq_magnitude > threshold
        
        if np.any(significant_freqs):
            patterns.append({
                "type": "periodic_variation",
                "frequency_count": int(np.sum(significant_freqs)),
                "amplitude": float(np.max(freq_magnitude)),
                "pattern_strength": "high" if np.sum(significant_freqs) > 1 else "low"
            })
        
        return patterns
    
    def _detect_standing_waves(self, amplitude: np.ndarray) -> List[Dict[str, Any]]:
        """Detect standing wave patterns."""
        patterns = []
        
        # Look for spatial patterns that could indicate standing waves
        spatial_variance = np.var(amplitude)
        if spatial_variance > np.mean(amplitude):
            patterns.append({
                "type": "standing_wave",
                "spatial_variance": float(spatial_variance),
                "pattern_strength": "high" if spatial_variance > 2 * np.mean(amplitude) else "low"
            })
        
        return patterns
    
    def _detect_beat_patterns(self, amplitude: np.ndarray) -> List[Dict[str, Any]]:
        """Detect beat patterns."""
        patterns = []
        
        # Look for amplitude modulation patterns
        amplitude_envelope = np.abs(amplitude)
        modulation_depth = (np.max(amplitude_envelope) - np.min(amplitude_envelope)) / np.mean(amplitude_envelope)
        
        if modulation_depth > 0.1:
            patterns.append({
                "type": "beat_pattern",
                "modulation_depth": float(modulation_depth),
                "pattern_strength": "high" if modulation_depth > 0.5 else "low"
            })
        
        return patterns
    
    def _calculate_pattern_complexity(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate pattern complexity."""
        if not patterns:
            return 0.0
        
        # Use number of patterns and their diversity as complexity measure
        pattern_types = set(pattern.get("type", "unknown") for pattern in patterns)
        complexity = len(pattern_types) * len(patterns) / 10.0  # Normalize
        return float(complexity)
    
    def _assess_pattern_regularity(self, patterns: List[Dict[str, Any]]) -> str:
        """Assess pattern regularity."""
        if not patterns:
            return "none"
        
        # Simple regularity assessment based on pattern count
        if len(patterns) == 1:
            return "regular"
        elif len(patterns) <= 3:
            return "semi_regular"
        else:
            return "irregular"
    
    def _calculate_interference_visibility(self, envelope: np.ndarray) -> float:
        """Calculate interference visibility."""
        amplitude = np.abs(envelope)
        max_amplitude = np.max(amplitude)
        min_amplitude = np.min(amplitude)
        
        if max_amplitude + min_amplitude == 0:
            return 0.0
        
        visibility = (max_amplitude - min_amplitude) / (max_amplitude + min_amplitude)
        return float(visibility)
    
    def _calculate_interference_contrast(self, envelope: np.ndarray) -> float:
        """Calculate interference contrast."""
        amplitude = np.abs(envelope)
        mean_amplitude = np.mean(amplitude)
        std_amplitude = np.std(amplitude)
        
        if mean_amplitude == 0:
            return 0.0
        
        contrast = std_amplitude / mean_amplitude
        return float(contrast)
    
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
                "frequency": self._frequency_from_coordinates(coords),
                "amplitude": float(frequency_magnitude[coords]),
                "coordinates": coords
            })
        
        return dominant_frequencies
    
    def _analyze_field_correlations(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze field correlations."""
        # Calculate spatial correlations
        if envelope.ndim >= 2:
            slice_1 = envelope[..., 0] if envelope.ndim > 2 else envelope
            slice_2 = envelope[..., 1] if envelope.ndim > 2 else envelope
            
            correlation = np.corrcoef(slice_1.flatten(), slice_2.flatten())[0, 1]
            if not np.isnan(correlation):
                return {
                    "spatial_correlation": float(correlation),
                    "correlation_strength": "strong" if abs(correlation) > 0.5 else "weak"
                }
        
        return {
            "spatial_correlation": 0.0,
            "correlation_strength": "none"
        }
    
    def _detect_coupling_patterns(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Detect coupling patterns."""
        patterns = []
        
        # Look for coupling indicators in field structure
        field_gradients = [np.gradient(envelope, axis=dim) for dim in range(envelope.ndim)]
        gradient_magnitude = np.sqrt(sum(grad**2 for grad in field_gradients))
        
        # Detect regions of high gradient (potential coupling regions)
        high_gradient_threshold = np.mean(gradient_magnitude) + 2 * np.std(gradient_magnitude)
        coupling_regions = gradient_magnitude > high_gradient_threshold
        
        if np.any(coupling_regions):
            patterns.append({
                "type": "gradient_coupling",
                "coupling_region_count": int(np.sum(coupling_regions)),
                "coupling_strength": float(np.mean(gradient_magnitude[coupling_regions])),
                "pattern_quality": "high" if np.sum(coupling_regions) > 1 else "low"
            })
        
        return patterns
    
    def _calculate_coupling_strength_metric(self, envelope: np.ndarray) -> float:
        """Calculate coupling strength metric."""
        # Use field variance as coupling strength indicator
        field_variance = np.var(envelope)
        field_mean = np.mean(np.abs(envelope))
        
        if field_mean == 0:
            return 0.0
        
        coupling_strength = field_variance / field_mean
        return float(coupling_strength)
    
    def _create_beating_summary(self, mode_analysis: Dict[str, Any],
                               interference_analysis: Dict[str, Any],
                               frequency_analysis: Dict[str, Any],
                               coupling_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of beating analysis."""
        return {
            "total_modes_detected": len(mode_analysis["modes"]),
            "beating_frequencies_detected": len(frequency_analysis["beating_frequencies"]),
            "interference_patterns_detected": interference_analysis["pattern_properties"]["pattern_count"],
            "mode_coupling_detected": coupling_analysis["coupling_strength"]["coupling_quality"] == "strong",
            "beating_quality": "high" if len(frequency_analysis["beating_frequencies"]) > 0 else "low",
            "analysis_complete": True,
            "analysis_methods": ["mode_detection", "interference_detection", "frequency_analysis", "coupling_analysis"]
        }
