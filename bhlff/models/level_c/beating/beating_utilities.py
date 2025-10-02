"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Beating analysis utilities for Level C.

This module implements utility functions and helper classes
for beating analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class BeatingUtilities:
    """
    Utility functions for beating analysis.
    
    Physical Meaning:
        Provides utility functions for beating analysis, including
        frequency analysis, pattern detection, and statistical
        analysis of mode interactions.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize beating utilities.
        
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def calculate_beating_spectrum(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Calculate beating spectrum from envelope field.
        
        Physical Meaning:
            Calculates the beating spectrum by analyzing frequency
            differences and interference patterns in the envelope field.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Beating spectrum analysis results.
        """
        # Perform FFT analysis
        fft_result = np.fft.fftn(envelope)
        power_spectrum = np.abs(fft_result)**2
        
        # Calculate beating spectrum
        beating_spectrum = self._calculate_frequency_differences(power_spectrum)
        
        # Find beating peaks
        beating_peaks = self._find_beating_peaks(beating_spectrum)
        
        # Calculate beating statistics
        beating_stats = self._calculate_beating_statistics(beating_spectrum)
        
        return {
            'beating_spectrum': beating_spectrum,
            'beating_peaks': beating_peaks,
            'beating_stats': beating_stats,
            'power_spectrum': power_spectrum
        }
    
    def detect_mode_interference(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect mode interference patterns.
        
        Physical Meaning:
            Detects interference patterns between different modes
            in the envelope field, identifying spatial and temporal
            interference effects.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            List[Dict[str, Any]]: List of detected interference patterns.
        """
        interference_patterns = []
        
        # Analyze spatial interference
        spatial_interference = self._analyze_spatial_interference(envelope)
        if spatial_interference:
            interference_patterns.append(spatial_interference)
        
        # Analyze temporal interference
        temporal_interference = self._analyze_temporal_interference(envelope)
        if temporal_interference:
            interference_patterns.append(temporal_interference)
        
        # Analyze phase interference
        phase_interference = self._analyze_phase_interference(envelope)
        if phase_interference:
            interference_patterns.append(phase_interference)
        
        return interference_patterns
    
    def calculate_beating_parameters(self, envelope: np.ndarray) -> Dict[str, float]:
        """
        Calculate beating parameters.
        
        Physical Meaning:
            Calculates key parameters characterizing beating effects,
            including beating frequency, amplitude, and phase.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, float]: Beating parameters.
        """
        # Calculate beating frequency
        beating_frequency = self._calculate_beating_frequency(envelope)
        
        # Calculate beating amplitude
        beating_amplitude = self._calculate_beating_amplitude(envelope)
        
        # Calculate beating phase
        beating_phase = self._calculate_beating_phase(envelope)
        
        # Calculate beating strength
        beating_strength = self._calculate_beating_strength(envelope)
        
        return {
            'beating_frequency': beating_frequency,
            'beating_amplitude': beating_amplitude,
            'beating_phase': beating_phase,
            'beating_strength': beating_strength
        }
    
    def analyze_mode_coupling(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mode coupling effects.
        
        Physical Meaning:
            Analyzes mode coupling effects that give rise to beating,
            including coupling strength, mechanisms, and interaction
            patterns between different modes.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Mode coupling analysis results.
        """
        # Calculate coupling strength
        coupling_strength = self._calculate_coupling_strength(envelope)
        
        # Identify coupling mechanisms
        coupling_mechanisms = self._identify_coupling_mechanisms(envelope)
        
        # Analyze coupling patterns
        coupling_patterns = self._analyze_coupling_patterns(envelope)
        
        # Calculate coupling statistics
        coupling_stats = self._calculate_coupling_statistics(envelope)
        
        return {
            'coupling_strength': coupling_strength,
            'coupling_mechanisms': coupling_mechanisms,
            'coupling_patterns': coupling_patterns,
            'coupling_stats': coupling_stats
        }
    
    def _calculate_frequency_differences(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Calculate frequency differences for beating analysis."""
        # Simplified frequency difference calculation
        freq_diffs = []
        for i in range(len(power_spectrum)):
            for j in range(i+1, len(power_spectrum)):
                diff = abs(i - j)
                freq_diffs.append(diff)
        
        return np.array(freq_diffs)
    
    def _find_beating_peaks(self, beating_spectrum: np.ndarray) -> List[Dict[str, Any]]:
        """Find peaks in beating spectrum."""
        peaks = []
        
        # Find local maxima
        for i in range(1, len(beating_spectrum) - 1):
            if beating_spectrum[i] > beating_spectrum[i-1] and beating_spectrum[i] > beating_spectrum[i+1]:
                peaks.append({
                    'frequency': float(i),
                    'amplitude': float(beating_spectrum[i]),
                    'index': i
                })
        
        return peaks
    
    def _calculate_beating_statistics(self, beating_spectrum: np.ndarray) -> Dict[str, float]:
        """Calculate beating statistics."""
        return {
            'total_power': float(np.sum(beating_spectrum)),
            'max_power': float(np.max(beating_spectrum)),
            'mean_power': float(np.mean(beating_spectrum)),
            'std_power': float(np.std(beating_spectrum)),
            'peak_count': len(self._find_beating_peaks(beating_spectrum))
        }
    
    def _analyze_spatial_interference(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial interference patterns."""
        # Calculate spatial correlation
        spatial_corr = np.corrcoef(envelope.reshape(envelope.shape[0], -1))
        
        # Find interference patterns
        if np.max(spatial_corr) > 0.1:
            return {
                'type': 'spatial',
                'strength': float(np.max(spatial_corr)),
                'pattern': spatial_corr.tolist()
            }
        
        return None
    
    def _analyze_temporal_interference(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal interference patterns."""
        # Calculate temporal correlation
        temporal_corr = np.corrcoef(envelope.reshape(-1, envelope.shape[-1]))
        
        # Find interference patterns
        if np.max(temporal_corr) > 0.1:
            return {
                'type': 'temporal',
                'strength': float(np.max(temporal_corr)),
                'pattern': temporal_corr.tolist()
            }
        
        return None
    
    def _analyze_phase_interference(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze phase interference patterns."""
        # Calculate phase correlation
        phase_indices = [3, 4, 5]  # Phase dimensions
        phase_data = envelope.take(phase_indices, axis=0)
        phase_corr = np.corrcoef(phase_data.reshape(phase_data.shape[0], -1))
        
        # Find interference patterns
        if np.max(phase_corr) > 0.1:
            return {
                'type': 'phase',
                'strength': float(np.max(phase_corr)),
                'pattern': phase_corr.tolist()
            }
        
        return None
    
    def _calculate_beating_frequency(self, envelope: np.ndarray) -> float:
        """Calculate beating frequency."""
        # Simplified beating frequency calculation
        fft_result = np.fft.fftn(envelope)
        power_spectrum = np.abs(fft_result)**2
        
        # Find dominant frequency
        dominant_freq = np.argmax(power_spectrum)
        return float(dominant_freq)
    
    def _calculate_beating_amplitude(self, envelope: np.ndarray) -> float:
        """Calculate beating amplitude."""
        # Simplified beating amplitude calculation
        return float(np.std(envelope))
    
    def _calculate_beating_phase(self, envelope: np.ndarray) -> float:
        """Calculate beating phase."""
        # Simplified beating phase calculation
        return float(np.angle(np.mean(envelope)))
    
    def _calculate_beating_strength(self, envelope: np.ndarray) -> float:
        """Calculate beating strength."""
        # Simplified beating strength calculation
        return float(np.max(np.abs(envelope)))
    
    def _calculate_coupling_strength(self, envelope: np.ndarray) -> float:
        """Calculate mode coupling strength."""
        # Simplified coupling strength calculation
        return float(np.var(envelope))
    
    def _identify_coupling_mechanisms(self, envelope: np.ndarray) -> List[str]:
        """Identify coupling mechanisms."""
        mechanisms = []
        
        # Check for nonlinear coupling
        if np.std(envelope) > 0.1:
            mechanisms.append('nonlinear')
        
        # Check for resonant coupling
        if np.max(np.abs(envelope)) > 0.5:
            mechanisms.append('resonant')
        
        # Check for parametric coupling
        if np.var(envelope) > 0.01:
            mechanisms.append('parametric')
        
        return mechanisms
    
    def _analyze_coupling_patterns(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Analyze coupling patterns."""
        return {
            'spatial_pattern': self._analyze_spatial_interference(envelope),
            'temporal_pattern': self._analyze_temporal_interference(envelope),
            'phase_pattern': self._analyze_phase_interference(envelope)
        }
    
    def _calculate_coupling_statistics(self, envelope: np.ndarray) -> Dict[str, float]:
        """Calculate coupling statistics."""
        return {
            'coupling_strength': float(np.var(envelope)),
            'coupling_range': float(np.max(envelope) - np.min(envelope)),
            'coupling_mean': float(np.mean(envelope)),
            'coupling_std': float(np.std(envelope))
        }


class BeatingVisualizer:
    """
    Visualization utilities for beating analysis.
    
    Physical Meaning:
        Provides visualization tools for beating analysis results,
        including plots of beating patterns, frequency spectra,
        and mode interactions.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize beating visualizer.
        
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def plot_beating_spectrum(self, beating_spectrum: Dict[str, Any], save_path: str = None) -> None:
        """
        Plot beating spectrum.
        
        Args:
            beating_spectrum (Dict[str, Any]): Beating spectrum data.
            save_path (str, optional): Path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(beating_spectrum['beating_spectrum'])
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            plt.title('Beating Spectrum')
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
    
    def plot_interference_patterns(self, interference_patterns: List[Dict[str, Any]], save_path: str = None) -> None:
        """
        Plot interference patterns.
        
        Args:
            interference_patterns (List[Dict[str, Any]]): Interference pattern data.
            save_path (str, optional): Path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots for different pattern types
            fig, axes = plt.subplots(len(interference_patterns), 1, figsize=(10, 6*len(interference_patterns)))
            
            if len(interference_patterns) == 1:
                axes = [axes]
            
            for i, pattern in enumerate(interference_patterns):
                axes[i].plot(pattern['pattern'])
                axes[i].set_title(f'{pattern["type"].title()} Interference Pattern')
                axes[i].set_xlabel('Position')
                axes[i].set_ylabel('Amplitude')
                axes[i].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
    
    def plot_mode_coupling(self, mode_coupling: Dict[str, Any], save_path: str = None) -> None:
        """
        Plot mode coupling analysis.
        
        Args:
            mode_coupling (Dict[str, Any]): Mode coupling data.
            save_path (str, optional): Path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(mode_coupling['coupling_mechanisms'])), 
                   [mode_coupling['coupling_strength']] * len(mode_coupling['coupling_mechanisms']))
            plt.xlabel('Coupling Mechanism')
            plt.ylabel('Coupling Strength')
            plt.title('Mode Coupling Analysis')
            plt.xticks(range(len(mode_coupling['coupling_mechanisms'])), 
                      mode_coupling['coupling_mechanisms'])
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
