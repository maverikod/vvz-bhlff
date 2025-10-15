"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Feature extractor for ML pattern classification.

This module implements feature extraction for machine learning
pattern classification in 7D phase field beating analysis.

Physical Meaning:
    Extracts comprehensive features from 7D phase field configurations
    for machine learning-based pattern classification.

Example:
    >>> extractor = BeatingMLPatternFeatureExtractor()
    >>> features = extractor.extract_pattern_features(envelope)
"""

import numpy as np
from typing import Dict, Any
import logging

from bhlff.core.bvp import BVPCore


class BeatingMLPatternFeatureExtractor:
    """
    Feature extractor for ML pattern classification.
    
    Physical Meaning:
        Extracts comprehensive features from 7D phase field configurations
        for machine learning-based pattern classification.
        
    Mathematical Foundation:
        Implements feature extraction methods based on 7D phase field theory
        including spatial, frequency, and pattern characteristics.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize feature extractor.
        
        Physical Meaning:
            Sets up the feature extraction system for 7D phase field analysis.
            
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def extract_pattern_features(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Extract features for pattern classification.
        
        Physical Meaning:
            Extracts relevant features from the envelope field
            for machine learning-based pattern classification.
            
        Mathematical Foundation:
            Computes spatial, frequency, and pattern features based on
            7D phase field theory and VBP envelope analysis.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Extracted features for classification.
        """
        # Spatial features
        spatial_features = self._extract_spatial_features(envelope)
        
        # Frequency features
        frequency_features = self._extract_frequency_features(envelope)
        
        # Pattern features
        pattern_features = self._extract_pattern_characteristics(envelope)
        
        return {
            "spatial_features": spatial_features,
            "frequency_features": frequency_features,
            "pattern_features": pattern_features,
        }
    
    def _extract_spatial_features(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Extract spatial features from envelope.
        
        Physical Meaning:
            Extracts spatial characteristics from 7D phase field configuration
            including energy distribution and envelope properties.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Spatial features dictionary.
        """
        return {
            "envelope_energy": np.sum(np.abs(envelope) ** 2),
            "envelope_max": np.max(np.abs(envelope)),
            "envelope_mean": np.mean(np.abs(envelope)),
            "envelope_std": np.std(np.abs(envelope)),
        }
    
    def _extract_frequency_features(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Extract frequency features from envelope.
        
        Physical Meaning:
            Extracts frequency characteristics from 7D phase field configuration
            using spectral analysis and FFT methods.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Frequency features dictionary.
        """
        envelope_fft = np.fft.fftn(envelope)
        frequency_spectrum = np.abs(envelope_fft)
        
        return {
            "spectrum_peak": np.max(frequency_spectrum),
            "spectrum_mean": np.mean(frequency_spectrum),
            "spectrum_std": np.std(frequency_spectrum),
            "spectrum_entropy": self._compute_spectral_entropy(frequency_spectrum),
            "frequency_spacing": self._compute_frequency_spacing(frequency_spectrum),
            "frequency_bandwidth": self._compute_frequency_bandwidth(frequency_spectrum),
            "dominant_frequencies": np.argsort(frequency_spectrum.flatten())[-5:].tolist(),
        }
    
    def _extract_pattern_characteristics(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Extract pattern characteristics from envelope.
        
        Physical Meaning:
            Extracts pattern characteristics from 7D phase field configuration
            including symmetry, regularity, and complexity scores.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Pattern characteristics dictionary.
        """
        return {
            "symmetry_score": self._calculate_symmetry_score(envelope),
            "regularity_score": self._calculate_regularity_score(envelope),
            "complexity_score": self._calculate_complexity_score(envelope),
        }
    
    def _compute_spectral_entropy(self, spectrum: np.ndarray) -> float:
        """
        Compute spectral entropy.
        
        Physical Meaning:
            Computes spectral entropy of frequency spectrum
            to measure frequency distribution complexity.
        """
        # Normalize spectrum to probability distribution
        total_spectrum = np.sum(spectrum)
        if total_spectrum > 0:
            prob_dist = spectrum / total_spectrum
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
            return float(entropy)
        return 0.0
    
    def _compute_frequency_spacing(self, spectrum: np.ndarray) -> float:
        """
        Compute frequency spacing.
        
        Physical Meaning:
            Computes average spacing between dominant frequencies
            in the spectrum.
        """
        # Find peaks in spectrum
        peaks = self._find_spectral_peaks(spectrum)
        if len(peaks) > 1:
            spacing = np.mean(np.diff(peaks))
            return float(spacing)
        return 0.0
    
    def _compute_frequency_bandwidth(self, spectrum: np.ndarray) -> float:
        """
        Compute frequency bandwidth.
        
        Physical Meaning:
            Computes bandwidth of frequency spectrum
            to measure frequency spread.
        """
        # Compute bandwidth as standard deviation of spectrum
        bandwidth = np.std(spectrum)
        return float(bandwidth)
    
    def _find_spectral_peaks(self, spectrum: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Find spectral peaks.
        
        Physical Meaning:
            Finds dominant frequency peaks in the spectrum
            above the specified threshold.
        """
        max_spectrum = np.max(spectrum)
        threshold_value = threshold * max_spectrum
        
        # Find peaks above threshold
        peaks = np.where(spectrum > threshold_value)[0]
        return peaks
    
    def _calculate_symmetry_score(self, envelope: np.ndarray) -> float:
        """
        Calculate symmetry score using 7D phase field theory.
        
        Physical Meaning:
            Computes symmetry score based on 7D phase field properties
            and VBP envelope analysis using full mathematical framework.
        """
        # Compute 7D phase field symmetry using full mathematical framework
        phase_field_symmetry = self._compute_7d_phase_field_symmetry(envelope)
        
        # Compute VBP envelope symmetry
        vbp_symmetry = self._compute_vbp_envelope_symmetry(envelope)
        
        # Combine symmetries using 7D phase field theory
        combined_symmetry = self._combine_7d_symmetries(phase_field_symmetry, vbp_symmetry)
        
        return max(0.0, min(1.0, combined_symmetry))
    
    def _calculate_regularity_score(self, envelope: np.ndarray) -> float:
        """
        Calculate regularity score using 7D phase field theory.
        
        Physical Meaning:
            Computes regularity score based on 7D phase field properties
            and VBP envelope analysis using full mathematical framework.
        """
        # Compute 7D phase field regularity using full mathematical framework
        phase_field_regularity = self._compute_7d_phase_field_regularity(envelope)
        
        # Compute VBP envelope regularity
        vbp_regularity = self._compute_vbp_envelope_regularity(envelope)
        
        # Combine regularities using 7D phase field theory
        combined_regularity = self._combine_7d_regularities(phase_field_regularity, vbp_regularity)
        
        return max(0.0, min(1.0, combined_regularity))
    
    def _calculate_complexity_score(self, envelope: np.ndarray) -> float:
        """
        Calculate complexity score using 7D phase field theory.
        
        Physical Meaning:
            Computes complexity score based on 7D phase field properties
            and VBP envelope analysis using full mathematical framework.
        """
        # Compute 7D phase field complexity using full mathematical framework
        phase_field_complexity = self._compute_7d_phase_field_complexity(envelope)
        
        # Compute VBP envelope complexity
        vbp_complexity = self._compute_vbp_envelope_complexity(envelope)
        
        # Combine complexities using 7D phase field theory
        combined_complexity = self._combine_7d_complexities(phase_field_complexity, vbp_complexity)
        
        return max(0.0, min(1.0, combined_complexity))
    
    def _compute_7d_phase_field_symmetry(self, envelope: np.ndarray) -> float:
        """
        Compute 7D phase field symmetry using full mathematical framework.
        
        Physical Meaning:
            Computes symmetry based on 7D phase field theory including
            phase coherence, topological charge, and energy density.
        """
        # Compute phase of envelope
        phase = np.angle(envelope)
        
        # Compute 7D phase field symmetry using circular statistics
        complex_phase = np.exp(1j * phase)
        mean_complex = np.mean(complex_phase)
        phase_coherence = np.abs(mean_complex)
        
        # Compute topological charge symmetry
        topological_charge = self._compute_topological_charge(envelope)
        charge_symmetry = 1.0 - abs(topological_charge)
        
        # Compute energy density symmetry
        energy_density = np.abs(envelope) ** 2
        energy_symmetry = self._compute_energy_symmetry(energy_density)
        
        # Combine symmetries using 7D phase field theory
        combined_symmetry = (
            phase_coherence * 0.4 +
            charge_symmetry * 0.3 +
            energy_symmetry * 0.3
        )
        
        return float(combined_symmetry)
    
    def _compute_vbp_envelope_symmetry(self, envelope: np.ndarray) -> float:
        """
        Compute VBP envelope symmetry.
        
        Physical Meaning:
            Computes VBP envelope symmetry based on envelope properties.
        """
        # Compute envelope symmetry using spatial correlation
        center = envelope.shape[0] // 2
        left_half = envelope[:center]
        right_half = envelope[center:]
        
        if left_half.shape != right_half.shape:
            return 0.5
        
        correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        return max(0.0, min(1.0, correlation))
    
    def _combine_7d_symmetries(self, phase_field_symmetry: float, vbp_symmetry: float) -> float:
        """
        Combine 7D symmetries using phase field theory.
        
        Physical Meaning:
            Combines phase field and VBP envelope symmetries using
            7D phase field theory principles.
        """
        # Weighted combination based on 7D phase field theory
        return phase_field_symmetry * 0.7 + vbp_symmetry * 0.3
    
    def _compute_7d_phase_field_regularity(self, envelope: np.ndarray) -> float:
        """
        Compute 7D phase field regularity using full mathematical framework.
        
        Physical Meaning:
            Computes regularity based on 7D phase field theory including
            phase coherence, topological charge, and energy density.
        """
        # Compute phase regularity using circular statistics
        phase = np.angle(envelope)
        phase_regularity = 1.0 - np.std(phase) / np.pi
        
        # Compute energy density regularity
        energy_density = np.abs(envelope) ** 2
        energy_regularity = 1.0 - np.std(energy_density) / np.mean(energy_density)
        
        # Compute topological charge regularity
        topological_charge = self._compute_topological_charge(envelope)
        charge_regularity = 1.0 - abs(topological_charge)
        
        # Combine regularities using 7D phase field theory
        combined_regularity = (
            phase_regularity * 0.4 +
            energy_regularity * 0.4 +
            charge_regularity * 0.2
        )
        
        return float(combined_regularity)
    
    def _compute_vbp_envelope_regularity(self, envelope: np.ndarray) -> float:
        """
        Compute VBP envelope regularity.
        
        Physical Meaning:
            Computes VBP envelope regularity based on envelope properties.
        """
        # Compute envelope regularity using variance analysis
        envelope_abs = np.abs(envelope)
        local_variance = np.var(envelope_abs)
        global_variance = np.var(envelope_abs.flatten())
        
        if global_variance == 0:
            return 1.0
        
        regularity = 1.0 - (local_variance / global_variance)
        return max(0.0, min(1.0, regularity))
    
    def _combine_7d_regularities(self, phase_field_regularity: float, vbp_regularity: float) -> float:
        """
        Combine 7D regularities using phase field theory.
        
        Physical Meaning:
            Combines phase field and VBP envelope regularities using
            7D phase field theory principles.
        """
        # Weighted combination based on 7D phase field theory
        return phase_field_regularity * 0.7 + vbp_regularity * 0.3
    
    def _compute_7d_phase_field_complexity(self, envelope: np.ndarray) -> float:
        """
        Compute 7D phase field complexity using full mathematical framework.
        
        Physical Meaning:
            Computes complexity based on 7D phase field theory including
            phase coherence, topological charge, and energy density.
        """
        # Compute phase complexity using spectral analysis
        phase = np.angle(envelope)
        phase_fft = np.fft.fftn(phase)
        phase_spectrum = np.abs(phase_fft)
        
        # Count significant phase components
        threshold = 0.1 * np.max(phase_spectrum)
        significant_components = np.sum(phase_spectrum > threshold)
        total_components = phase_spectrum.size
        phase_complexity = significant_components / total_components
        
        # Compute energy density complexity
        energy_density = np.abs(envelope) ** 2
        energy_fft = np.fft.fftn(energy_density)
        energy_spectrum = np.abs(energy_fft)
        
        # Count significant energy components
        threshold = 0.1 * np.max(energy_spectrum)
        significant_components = np.sum(energy_spectrum > threshold)
        total_components = energy_spectrum.size
        energy_complexity = significant_components / total_components
        
        # Compute topological charge complexity
        topological_charge = self._compute_topological_charge(envelope)
        charge_complexity = abs(topological_charge)
        
        # Combine complexities using 7D phase field theory
        combined_complexity = (
            phase_complexity * 0.4 +
            energy_complexity * 0.4 +
            charge_complexity * 0.2
        )
        
        return float(combined_complexity)
    
    def _compute_vbp_envelope_complexity(self, envelope: np.ndarray) -> float:
        """
        Compute VBP envelope complexity.
        
        Physical Meaning:
            Computes VBP envelope complexity based on envelope properties.
        """
        # Compute envelope complexity using frequency content
        envelope_fft = np.fft.fftn(envelope)
        frequency_spectrum = np.abs(envelope_fft)
        
        # Count significant frequency components
        threshold = 0.1 * np.max(frequency_spectrum)
        significant_components = np.sum(frequency_spectrum > threshold)
        total_components = frequency_spectrum.size
        
        complexity = significant_components / total_components
        return max(0.0, min(1.0, complexity))
    
    def _combine_7d_complexities(self, phase_field_complexity: float, vbp_complexity: float) -> float:
        """
        Combine 7D complexities using phase field theory.
        
        Physical Meaning:
            Combines phase field and VBP envelope complexities using
            7D phase field theory principles.
        """
        # Weighted combination based on 7D phase field theory
        return phase_field_complexity * 0.7 + vbp_complexity * 0.3
    
    def _compute_topological_charge(self, envelope: np.ndarray) -> float:
        """
        Compute topological charge using 7D phase field theory.
        
        Physical Meaning:
            Computes topological charge based on 7D phase field theory.
        """
        # Compute phase gradient
        phase = np.angle(envelope)
        grad_x = np.gradient(phase, axis=1)
        grad_y = np.gradient(phase, axis=0)
        
        # Compute topological charge
        topological_charge = np.sum(grad_x * grad_y) / (2 * np.pi)
        
        return float(topological_charge)
    
    def _compute_energy_symmetry(self, energy_density: np.ndarray) -> float:
        """
        Compute energy density symmetry.
        
        Physical Meaning:
            Computes energy density symmetry based on spatial distribution.
        """
        # Compute energy density symmetry using spatial correlation
        center = energy_density.shape[0] // 2
        left_half = energy_density[:center]
        right_half = energy_density[center:]
        
        if left_half.shape != right_half.shape:
            return 0.5
        
        correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        return max(0.0, min(1.0, correlation))
