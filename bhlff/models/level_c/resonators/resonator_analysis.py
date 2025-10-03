"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Resonator analysis functions for Level C.

This module provides additional analysis functions for resonator
analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class ResonatorAnalysis:
    """
    Additional analysis functions for resonator detection.
    
    Physical Meaning:
        Provides additional analysis functions for resonator detection,
        including correlation analysis, pattern detection, and
        statistical analysis of resonator behavior.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize resonator analysis.
        
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def analyze_resonator_correlations(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze resonator correlations.
        
        Physical Meaning:
            Analyzes correlations between different resonator modes
            and spatial/temporal patterns in the envelope field.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Correlation analysis results.
        """
        # Calculate spatial correlations
        spatial_correlations = self._calculate_spatial_correlations(envelope)
        
        # Calculate temporal correlations
        temporal_correlations = self._calculate_temporal_correlations(envelope)
        
        # Calculate phase correlations
        phase_correlations = self._calculate_phase_correlations(envelope)
        
        # Calculate cross-correlations
        cross_correlations = self._calculate_cross_correlations(envelope)
        
        return {
            'spatial_correlations': spatial_correlations,
            'temporal_correlations': temporal_correlations,
            'phase_correlations': phase_correlations,
            'cross_correlations': cross_correlations
        }
    
    def detect_resonator_patterns(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect resonator patterns in the envelope field.
        
        Physical Meaning:
            Detects specific resonator patterns that indicate
            structured resonator behavior in the 7D phase field.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            List[Dict[str, Any]]: List of detected resonator patterns.
        """
        patterns = []
        
        # Detect standing wave patterns
        standing_waves = self._detect_standing_wave_patterns(envelope)
        patterns.extend(standing_waves)
        
        # Detect traveling wave patterns
        traveling_waves = self._detect_traveling_wave_patterns(envelope)
        patterns.extend(traveling_waves)
        
        # Detect interference patterns
        interference_patterns = self._detect_interference_patterns(envelope)
        patterns.extend(interference_patterns)
        
        return patterns
    
    def calculate_resonator_statistics(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Calculate resonator statistics.
        
        Physical Meaning:
            Calculates statistical measures of resonator behavior,
            including amplitude distributions, frequency statistics,
            and resonator density.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Resonator statistics.
        """
        # Calculate amplitude statistics
        amplitude_stats = self._calculate_amplitude_statistics(envelope)
        
        # Calculate frequency statistics
        frequency_stats = self._calculate_frequency_statistics(envelope)
        
        # Calculate resonator density
        resonator_density = self._calculate_resonator_density(envelope)
        
        # Calculate resonator distribution
        resonator_distribution = self._calculate_resonator_distribution(envelope)
        
        return {
            'amplitude_stats': amplitude_stats,
            'frequency_stats': frequency_stats,
            'resonator_density': resonator_density,
            'resonator_distribution': resonator_distribution
        }
    
    def _calculate_spatial_correlations(self, envelope: np.ndarray) -> np.ndarray:
        """Calculate spatial correlations."""
        # Calculate correlations between spatial dimensions
        spatial_correlations = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    # Calculate correlation between dimensions i and j
                    slice_i = envelope.take(0, axis=i)
                    slice_j = envelope.take(0, axis=j)
                    correlation = np.corrcoef(slice_i.flatten(), slice_j.flatten())[0, 1]
                    spatial_correlations[i, j] = correlation if not np.isnan(correlation) else 0.0
        
        return spatial_correlations
    
    def _calculate_temporal_correlations(self, envelope: np.ndarray) -> np.ndarray:
        """Calculate temporal correlations."""
        # Calculate temporal autocorrelation
        temporal_slice = envelope.take(0, axis=-1)
        temporal_correlation = np.correlate(temporal_slice, temporal_slice, mode='full')
        temporal_correlation = temporal_correlation / np.max(temporal_correlation)
        
        return temporal_correlation
    
    def _calculate_phase_correlations(self, envelope: np.ndarray) -> np.ndarray:
        """Calculate phase correlations."""
        # Calculate correlations between phase dimensions
        phase_correlations = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    # Calculate correlation between phase dimensions i and j
                    slice_i = envelope.take(0, axis=i+3)
                    slice_j = envelope.take(0, axis=j+3)
                    correlation = np.corrcoef(slice_i.flatten(), slice_j.flatten())[0, 1]
                    phase_correlations[i, j] = correlation if not np.isnan(correlation) else 0.0
        
        return phase_correlations
    
    def _calculate_cross_correlations(self, envelope: np.ndarray) -> Dict[str, float]:
        """Calculate cross-correlations between different dimension types."""
        cross_correlations = {}
        
        # Spatial-temporal correlation
        spatial_slice = envelope.take(0, axis=0)  # First spatial dimension
        temporal_slice = envelope.take(0, axis=-1)  # Temporal dimension
        correlation = np.corrcoef(spatial_slice.flatten(), temporal_slice.flatten())[0, 1]
        cross_correlations['spatial_temporal'] = correlation if not np.isnan(correlation) else 0.0
        
        # Spatial-phase correlation
        phase_slice = envelope.take(0, axis=3)  # First phase dimension
        correlation = np.corrcoef(spatial_slice.flatten(), phase_slice.flatten())[0, 1]
        cross_correlations['spatial_phase'] = correlation if not np.isnan(correlation) else 0.0
        
        # Phase-temporal correlation
        correlation = np.corrcoef(phase_slice.flatten(), temporal_slice.flatten())[0, 1]
        cross_correlations['phase_temporal'] = correlation if not np.isnan(correlation) else 0.0
        
        return cross_correlations
    
    def _detect_standing_wave_patterns(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Detect standing wave patterns."""
        patterns = []
        
        # Analyze spatial dimensions for standing wave patterns
        for dim in range(3):
            spatial_slice = envelope.take(0, axis=dim)
            
            # Check for standing wave characteristics
            if self._has_standing_wave_characteristics(spatial_slice):
                patterns.append({
                    'type': 'standing_wave',
                    'dimension': dim,
                    'amplitude': np.max(spatial_slice),
                    'pattern': spatial_slice
                })
        
        return patterns
    
    def _detect_traveling_wave_patterns(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Detect traveling wave patterns."""
        patterns = []
        
        # Analyze temporal dimension for traveling wave patterns
        temporal_slice = envelope.take(0, axis=-1)
        
        # Check for traveling wave characteristics
        if self._has_traveling_wave_characteristics(temporal_slice):
            patterns.append({
                'type': 'traveling_wave',
                'amplitude': np.max(temporal_slice),
                'pattern': temporal_slice
            })
        
        return patterns
    
    def _detect_interference_patterns(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Detect interference patterns."""
        patterns = []
        
        # Analyze for interference patterns
        if self._has_interference_characteristics(envelope):
            patterns.append({
                'type': 'interference',
                'amplitude': np.max(envelope),
                'pattern': envelope
            })
        
        return patterns
    
    def _calculate_amplitude_statistics(self, envelope: np.ndarray) -> Dict[str, float]:
        """Calculate amplitude statistics."""
        return {
            'mean_amplitude': np.mean(envelope),
            'std_amplitude': np.std(envelope),
            'max_amplitude': np.max(envelope),
            'min_amplitude': np.min(envelope),
            'amplitude_range': np.max(envelope) - np.min(envelope)
        }
    
    def _calculate_frequency_statistics(self, envelope: np.ndarray) -> Dict[str, float]:
        """Calculate frequency statistics."""
        # Perform FFT analysis
        fft_result = np.fft.fftn(envelope)
        power_spectrum = np.abs(fft_result)**2
        
        return {
            'total_power': np.sum(power_spectrum),
            'max_power': np.max(power_spectrum),
            'mean_power': np.mean(power_spectrum),
            'std_power': np.std(power_spectrum)
        }
    
    def _calculate_resonator_density(self, envelope: np.ndarray) -> float:
        """Calculate resonator density."""
        # Calculate density based on local maxima
        local_maxima = self._find_local_maxima(envelope)
        total_points = envelope.size
        density = len(local_maxima) / total_points
        
        return density
    
    def _calculate_resonator_distribution(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Calculate resonator distribution."""
        # Calculate distribution across dimensions
        distribution = {}
        
        for dim in range(envelope.ndim):
            dim_slice = envelope.take(0, axis=dim)
            distribution[f'dimension_{dim}'] = {
                'mean': np.mean(dim_slice),
                'std': np.std(dim_slice),
                'max': np.max(dim_slice),
                'min': np.min(dim_slice)
            }
        
        return distribution
    
    def _has_standing_wave_characteristics(self, data: np.ndarray) -> bool:
        """Check for standing wave characteristics."""
        # Simplified standing wave detection
        return np.std(data) > 0.1 and np.max(data) > 0.5
    
    def _has_traveling_wave_characteristics(self, data: np.ndarray) -> bool:
        """Check for traveling wave characteristics."""
        # Simplified traveling wave detection
        return np.std(data) > 0.05 and len(data) > 10
    
    def _has_interference_characteristics(self, data: np.ndarray) -> bool:
        """Check for interference characteristics."""
        # Simplified interference detection
        return np.std(data) > 0.2 and np.max(data) > 0.3
    
    def _find_local_maxima(self, data: np.ndarray) -> List[Tuple[int, ...]]:
        """Find local maxima in the data."""
        maxima = []
        
        # Find local maxima in multi-dimensional array
        for idx in np.ndindex(data.shape):
            is_maximum = True
            for dim in range(len(idx)):
                if idx[dim] > 0 and data[idx] <= data[tuple(idx[i] - (1 if i == dim else 0) for i in range(len(idx)))]:
                    is_maximum = False
                    break
                if idx[dim] < data.shape[dim] - 1 and data[idx] <= data[tuple(idx[i] + (1 if i == dim else 0) for i in range(len(idx)))]:
                    is_maximum = False
                    break
            
            if is_maximum:
                maxima.append(idx)
        
        return maxima
