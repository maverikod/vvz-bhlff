"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Beating core analysis utilities for Level C.

This module implements core beating analysis functions for
analyzing mode beating in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class BeatingCoreAnalyzer:
    """
    Core beating analysis utilities for Level C analysis.
    
    Physical Meaning:
        Provides core beating analysis functions for analyzing
        mode beating in the 7D phase field, including interference
        patterns, beating frequencies, and mode coupling effects.
        
    Mathematical Foundation:
        Uses frequency domain analysis, interference pattern detection,
        and beating frequency calculations to study mode interactions.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize beating core analyzer.
        
        Physical Meaning:
            Sets up the analyzer with the BVP core for accessing
            field data and computational resources.
            
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis parameters
        self.beating_threshold = 1e-6
        self.frequency_tolerance = 1e-3
        self.interference_threshold = 0.1
    
    def analyze_beating(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mode beating in the envelope field.
        
        Physical Meaning:
            Analyzes mode beating phenomena in the 7D envelope field,
            identifying interference patterns, beating frequencies,
            and mode coupling effects.
            
        Mathematical Foundation:
            Uses frequency domain analysis to identify beating patterns:
            - FFT analysis for frequency identification
            - Interference pattern detection
            - Beating frequency calculations
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Analysis results including:
                - beating_frequencies: Detected beating frequencies
                - interference_patterns: Interference patterns
                - mode_coupling: Mode coupling analysis
                - beating_strength: Strength of beating effects
        """
        self.logger.info("Starting beating analysis")
        
        # Perform frequency analysis
        frequency_analysis = self._analyze_frequencies(envelope)
        
        # Detect interference patterns
        interference_patterns = self._detect_interference_patterns(envelope, frequency_analysis)
        
        # Calculate beating frequencies
        beating_frequencies = self._calculate_beating_frequencies(frequency_analysis)
        
        # Analyze mode coupling
        mode_coupling = self._analyze_mode_coupling(envelope, beating_frequencies)
        
        # Calculate beating strength
        beating_strength = self._calculate_beating_strength(envelope, beating_frequencies)
        
        results = {
            'beating_frequencies': beating_frequencies,
            'interference_patterns': interference_patterns,
            'mode_coupling': mode_coupling,
            'beating_strength': beating_strength,
            'frequency_analysis': frequency_analysis
        }
        
        self.logger.info(f"Beating analysis completed. Found {len(beating_frequencies)} beating frequencies")
        return results
    
    def _analyze_frequencies(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frequencies in the envelope field.
        
        Physical Meaning:
            Analyzes the frequency content of the envelope field
            to identify dominant frequencies and their characteristics.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Frequency analysis results.
        """
        # Calculate FFT
        fft_result = np.fft.fftn(envelope)
        
        # Calculate frequency magnitudes
        frequency_magnitudes = np.abs(fft_result)
        
        # Find dominant frequencies
        dominant_frequencies = self._find_dominant_frequencies(frequency_magnitudes)
        
        # Calculate frequency statistics
        frequency_stats = self._calculate_frequency_statistics(frequency_magnitudes)
        
        return {
            'fft_result': fft_result,
            'frequency_magnitudes': frequency_magnitudes,
            'dominant_frequencies': dominant_frequencies,
            'frequency_stats': frequency_stats
        }
    
    def _detect_interference_patterns(self, envelope: np.ndarray, frequency_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect interference patterns in the envelope field.
        
        Physical Meaning:
            Detects interference patterns that result from
            mode interactions and beating phenomena.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            frequency_analysis (Dict[str, Any]): Frequency analysis results.
            
        Returns:
            List[Dict[str, Any]]: List of detected interference patterns.
        """
        patterns = []
        
        # Analyze temporal interference
        temporal_interference = self._analyze_temporal_interference(envelope)
        if temporal_interference:
            patterns.append(temporal_interference)
        
        # Analyze spatial interference
        spatial_interference = self._analyze_spatial_interference(envelope)
        if spatial_interference:
            patterns.append(spatial_interference)
        
        # Analyze phase interference
        phase_interference = self._analyze_phase_interference(envelope)
        if phase_interference:
            patterns.append(phase_interference)
        
        return patterns
    
    def _calculate_beating_frequencies(self, frequency_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate beating frequencies from frequency analysis.
        
        Physical Meaning:
            Calculates beating frequencies as differences between
            dominant frequencies, representing mode interactions.
            
        Args:
            frequency_analysis (Dict[str, Any]): Frequency analysis results.
            
        Returns:
            List[Dict[str, Any]]: List of beating frequency analysis results.
        """
        dominant_frequencies = frequency_analysis['dominant_frequencies']
        beating_frequencies = []
        
        # Calculate beating frequencies between all pairs of dominant frequencies
        for i in range(len(dominant_frequencies)):
            for j in range(i+1, len(dominant_frequencies)):
                freq_i = dominant_frequencies[i]
                freq_j = dominant_frequencies[j]
                
                # Calculate beating frequency
                beating_freq = abs(freq_i['frequency'] - freq_j['frequency'])
                
                # Calculate beating amplitude
                beating_amplitude = min(freq_i['amplitude'], freq_j['amplitude'])
                
                beating_frequencies.append({
                    'frequency': beating_freq,
                    'amplitude': beating_amplitude,
                    'source_frequencies': [freq_i['frequency'], freq_j['frequency']],
                    'source_amplitudes': [freq_i['amplitude'], freq_j['amplitude']]
                })
        
        return beating_frequencies
    
    def _analyze_mode_coupling(self, envelope: np.ndarray, beating_frequencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze mode coupling effects.
        
        Physical Meaning:
            Analyzes mode coupling effects that result from
            beating between different frequency modes.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            beating_frequencies (List[Dict[str, Any]]): Beating frequency analysis results.
            
        Returns:
            Dict[str, Any]: Mode coupling analysis results.
        """
        # Calculate coupling strength
        coupling_strength = self._calculate_coupling_strength(envelope, beating_frequencies)
        
        # Analyze coupling patterns
        coupling_patterns = self._analyze_coupling_patterns(envelope, beating_frequencies)
        
        # Calculate coupling statistics
        coupling_stats = self._calculate_coupling_statistics(coupling_strength, coupling_patterns)
        
        return {
            'coupling_strength': coupling_strength,
            'coupling_patterns': coupling_patterns,
            'coupling_stats': coupling_stats
        }
    
    def _calculate_beating_strength(self, envelope: np.ndarray, beating_frequencies: List[Dict[str, Any]]) -> float:
        """
        Calculate the strength of beating effects.
        
        Physical Meaning:
            Calculates the overall strength of beating effects
            in the envelope field, providing a quantitative
            measure of mode interaction intensity.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            beating_frequencies (List[Dict[str, Any]]): Beating frequency analysis results.
            
        Returns:
            float: Beating strength value.
        """
        if not beating_frequencies:
            return 0.0
        
        # Calculate beating strength based on beating frequencies
        total_beating_amplitude = sum(bf['amplitude'] for bf in beating_frequencies)
        beating_strength = total_beating_amplitude / len(beating_frequencies)
        
        return beating_strength
    
    def _find_dominant_frequencies(self, frequency_magnitudes: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find dominant frequencies in the spectrum.
        
        Physical Meaning:
            Identifies the most significant frequencies in the
            spectrum based on their amplitudes.
            
        Args:
            frequency_magnitudes (np.ndarray): Frequency magnitude spectrum.
            
        Returns:
            List[Dict[str, Any]]: List of dominant frequency information.
        """
        # Find peaks in the frequency spectrum
        peaks = self._find_peaks(frequency_magnitudes)
        
        # Sort peaks by amplitude
        peaks.sort(key=lambda x: x['amplitude'], reverse=True)
        
        # Return top 10 peaks
        return peaks[:10]
    
    def _find_peaks(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find peaks in the data.
        
        Physical Meaning:
            Identifies local maxima in the data that represent
            significant frequency components.
            
        Args:
            data (np.ndarray): Input data array.
            
        Returns:
            List[Dict[str, Any]]: List of peak information.
        """
        peaks = []
        
        # Simple peak finding algorithm
        for i in range(1, len(data)-1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append({
                    'index': i,
                    'frequency': float(i),
                    'amplitude': float(data[i])
                })
        
        return peaks
    
    def _calculate_frequency_statistics(self, frequency_magnitudes: np.ndarray) -> Dict[str, float]:
        """
        Calculate frequency statistics.
        
        Physical Meaning:
            Calculates statistical measures of the frequency
            magnitudes for analysis.
            
        Args:
            frequency_magnitudes (np.ndarray): Frequency magnitude spectrum.
            
        Returns:
            Dict[str, float]: Frequency statistics.
        """
        return {
            'max_magnitude': float(np.max(frequency_magnitudes)),
            'mean_magnitude': float(np.mean(frequency_magnitudes)),
            'std_magnitude': float(np.std(frequency_magnitudes)),
            'total_power': float(np.sum(frequency_magnitudes**2))
        }
    
    def _analyze_temporal_interference(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze temporal interference patterns.
        
        Physical Meaning:
            Analyzes temporal interference patterns that may
            indicate beating phenomena.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Temporal interference analysis results.
        """
        # Calculate temporal interference
        temporal_interference = np.corrcoef(envelope.reshape(-1, envelope.shape[-1]))
        
        # Find interference patterns
        if np.max(temporal_interference) > self.interference_threshold:
            return {
                'type': 'temporal',
                'strength': float(np.max(temporal_interference)),
                'pattern': temporal_interference.tolist()
            }
        
        return None
    
    def _analyze_spatial_interference(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spatial interference patterns.
        
        Physical Meaning:
            Analyzes spatial interference patterns that may
            indicate beating phenomena.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Spatial interference analysis results.
        """
        # Calculate spatial interference
        spatial_interference = np.corrcoef(envelope.reshape(envelope.shape[0], -1))
        
        # Find interference patterns
        if np.max(spatial_interference) > self.interference_threshold:
            return {
                'type': 'spatial',
                'strength': float(np.max(spatial_interference)),
                'pattern': spatial_interference.tolist()
            }
        
        return None
    
    def _analyze_phase_interference(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze phase interference patterns.
        
        Physical Meaning:
            Analyzes phase interference patterns that may
            indicate beating phenomena.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Phase interference analysis results.
        """
        # Calculate phase interference
        phase_indices = [3, 4, 5]  # Phase dimensions
        phase_data = envelope.take(phase_indices, axis=0)
        phase_interference = np.corrcoef(phase_data.reshape(phase_data.shape[0], -1))
        
        # Find interference patterns
        if np.max(phase_interference) > self.interference_threshold:
            return {
                'type': 'phase',
                'strength': float(np.max(phase_interference)),
                'pattern': phase_interference.tolist()
            }
        
        return None
    
    def _calculate_coupling_strength(self, envelope: np.ndarray, beating_frequencies: List[Dict[str, Any]]) -> float:
        """
        Calculate mode coupling strength.
        
        Physical Meaning:
            Calculates the strength of mode coupling effects
            based on beating frequencies and field properties.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            beating_frequencies (List[Dict[str, Any]]): Beating frequency analysis results.
            
        Returns:
            float: Coupling strength value.
        """
        if not beating_frequencies:
            return 0.0
        
        # Calculate coupling strength based on beating frequencies
        total_coupling = sum(bf['amplitude'] for bf in beating_frequencies)
        coupling_strength = total_coupling / len(beating_frequencies)
        
        return coupling_strength
    
    def _analyze_coupling_patterns(self, envelope: np.ndarray, beating_frequencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze mode coupling patterns.
        
        Physical Meaning:
            Analyzes patterns in mode coupling that result
            from beating between different frequency modes.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            beating_frequencies (List[Dict[str, Any]]): Beating frequency analysis results.
            
        Returns:
            List[Dict[str, Any]]: List of coupling pattern analysis results.
        """
        patterns = []
        
        # Analyze coupling patterns for each beating frequency
        for beating_freq in beating_frequencies:
            freq = beating_freq['frequency']
            amplitude = beating_freq['amplitude']
            
            # Calculate coupling pattern for this beating frequency
            pattern = self._calculate_single_coupling_pattern(envelope, freq, amplitude)
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_coupling_statistics(self, coupling_strength: float, coupling_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate coupling statistics.
        
        Physical Meaning:
            Calculates statistical measures of mode coupling
            for analysis.
            
        Args:
            coupling_strength (float): Coupling strength value.
            coupling_patterns (List[Dict[str, Any]]): Coupling pattern analysis results.
            
        Returns:
            Dict[str, Any]: Coupling statistics.
        """
        return {
            'coupling_strength': coupling_strength,
            'num_coupling_patterns': len(coupling_patterns),
            'max_coupling_amplitude': max(pattern['amplitude'] for pattern in coupling_patterns) if coupling_patterns else 0.0,
            'mean_coupling_amplitude': np.mean([pattern['amplitude'] for pattern in coupling_patterns]) if coupling_patterns else 0.0
        }
    
    def _calculate_single_coupling_pattern(self, envelope: np.ndarray, frequency: float, amplitude: float) -> Dict[str, Any]:
        """
        Calculate coupling pattern for a single beating frequency.
        
        Physical Meaning:
            Calculates the coupling pattern that results from
            beating at a specific frequency.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            frequency (float): Beating frequency.
            amplitude (float): Beating amplitude.
            
        Returns:
            Dict[str, Any]: Coupling pattern information.
        """
        # Calculate coupling pattern
        coupling_pattern = amplitude * np.sin(2 * np.pi * frequency * np.arange(len(envelope.flatten())))
        
        # Calculate pattern statistics
        pattern_stats = {
            'max_amplitude': float(np.max(coupling_pattern)),
            'mean_amplitude': float(np.mean(coupling_pattern)),
            'std_amplitude': float(np.std(coupling_pattern)),
            'frequency': frequency,
            'amplitude': amplitude
        }
        
        return {
            'pattern': coupling_pattern.tolist(),
            'statistics': pattern_stats
        }

