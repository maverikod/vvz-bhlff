"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Resonator analysis utilities for Level C.

This module implements utility functions and helper classes
for resonator analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class ResonatorUtilities:
    """
    Utility functions for resonator analysis.
    
    Physical Meaning:
        Provides utility functions for resonator analysis, including
        frequency analysis, resonance detection, and statistical
        analysis of resonator behavior.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize resonator utilities.
        
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def calculate_resonance_spectrum(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Calculate resonance spectrum from envelope field.
        
        Physical Meaning:
            Calculates the resonance spectrum by analyzing frequency
            characteristics and resonance patterns in the envelope field.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Resonance spectrum analysis results.
        """
        # Perform FFT analysis
        fft_result = np.fft.fftn(envelope)
        power_spectrum = np.abs(fft_result)**2
        
        # Calculate resonance spectrum
        resonance_spectrum = self._calculate_resonance_characteristics(power_spectrum)
        
        # Find resonance peaks
        resonance_peaks = self._find_resonance_peaks(resonance_spectrum)
        
        # Calculate resonance statistics
        resonance_stats = self._calculate_resonance_statistics(resonance_spectrum)
        
        return {
            'resonance_spectrum': resonance_spectrum,
            'resonance_peaks': resonance_peaks,
            'resonance_stats': resonance_stats,
            'power_spectrum': power_spectrum
        }
    
    def detect_resonance_modes(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect resonance modes in the envelope field.
        
        Physical Meaning:
            Detects resonance modes that indicate resonator structures,
            including their frequencies, amplitudes, and characteristics.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            List[Dict[str, Any]]: List of detected resonance modes.
        """
        resonance_modes = []
        
        # Analyze spatial resonance modes
        spatial_modes = self._analyze_spatial_resonance_modes(envelope)
        resonance_modes.extend(spatial_modes)
        
        # Analyze temporal resonance modes
        temporal_modes = self._analyze_temporal_resonance_modes(envelope)
        resonance_modes.extend(temporal_modes)
        
        # Analyze phase resonance modes
        phase_modes = self._analyze_phase_resonance_modes(envelope)
        resonance_modes.extend(phase_modes)
        
        return resonance_modes
    
    def calculate_resonance_parameters(self, envelope: np.ndarray) -> Dict[str, float]:
        """
        Calculate resonance parameters.
        
        Physical Meaning:
            Calculates key parameters characterizing resonance effects,
            including resonance frequency, amplitude, and quality factor.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, float]: Resonance parameters.
        """
        # Calculate resonance frequency
        resonance_frequency = self._calculate_resonance_frequency(envelope)
        
        # Calculate resonance amplitude
        resonance_amplitude = self._calculate_resonance_amplitude(envelope)
        
        # Calculate quality factor
        quality_factor = self._calculate_quality_factor(envelope)
        
        # Calculate resonance strength
        resonance_strength = self._calculate_resonance_strength(envelope)
        
        return {
            'resonance_frequency': resonance_frequency,
            'resonance_amplitude': resonance_amplitude,
            'quality_factor': quality_factor,
            'resonance_strength': resonance_strength
        }
    
    def analyze_resonator_quality(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze resonator quality factors.
        
        Physical Meaning:
            Analyzes resonator quality factors that characterize
            the sharpness and selectivity of resonance peaks,
            indicating resonator efficiency and performance.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Resonator quality analysis results.
        """
        # Calculate quality factors
        quality_factors = self._calculate_quality_factors(envelope)
        
        # Analyze quality distribution
        quality_distribution = self._analyze_quality_distribution(quality_factors)
        
        # Calculate quality statistics
        quality_stats = self._calculate_quality_statistics(quality_factors)
        
        # Identify high-quality resonators
        high_quality_resonators = self._identify_high_quality_resonators(quality_factors)
        
        return {
            'quality_factors': quality_factors,
            'quality_distribution': quality_distribution,
            'quality_stats': quality_stats,
            'high_quality_resonators': high_quality_resonators
        }
    
    def _calculate_resonance_characteristics(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Calculate resonance characteristics from power spectrum."""
        # Simplified resonance characteristic calculation
        resonance_chars = []
        for i in range(len(power_spectrum)):
            char = power_spectrum[i] * (1.0 + np.sin(2 * np.pi * i / len(power_spectrum)))
            resonance_chars.append(char)
        
        return np.array(resonance_chars)
    
    def _find_resonance_peaks(self, resonance_spectrum: np.ndarray) -> List[Dict[str, Any]]:
        """Find peaks in resonance spectrum."""
        peaks = []
        
        # Find local maxima
        for i in range(1, len(resonance_spectrum) - 1):
            if resonance_spectrum[i] > resonance_spectrum[i-1] and resonance_spectrum[i] > resonance_spectrum[i+1]:
                peaks.append({
                    'frequency': float(i),
                    'amplitude': float(resonance_spectrum[i]),
                    'index': i
                })
        
        return peaks
    
    def _calculate_resonance_statistics(self, resonance_spectrum: np.ndarray) -> Dict[str, float]:
        """Calculate resonance statistics."""
        return {
            'total_power': float(np.sum(resonance_spectrum)),
            'max_power': float(np.max(resonance_spectrum)),
            'mean_power': float(np.mean(resonance_spectrum)),
            'std_power': float(np.std(resonance_spectrum)),
            'peak_count': len(self._find_resonance_peaks(resonance_spectrum))
        }
    
    def _analyze_spatial_resonance_modes(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze spatial resonance modes."""
        modes = []
        
        # Calculate spatial resonance characteristics
        spatial_resonance = np.corrcoef(envelope.reshape(envelope.shape[0], -1))
        
        # Find resonance modes
        if np.max(spatial_resonance) > 0.1:
            modes.append({
                'type': 'spatial',
                'strength': float(np.max(spatial_resonance)),
                'pattern': spatial_resonance.tolist()
            })
        
        return modes
    
    def _analyze_temporal_resonance_modes(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze temporal resonance modes."""
        modes = []
        
        # Calculate temporal resonance characteristics
        temporal_resonance = np.corrcoef(envelope.reshape(-1, envelope.shape[-1]))
        
        # Find resonance modes
        if np.max(temporal_resonance) > 0.1:
            modes.append({
                'type': 'temporal',
                'strength': float(np.max(temporal_resonance)),
                'pattern': temporal_resonance.tolist()
            })
        
        return modes
    
    def _analyze_phase_resonance_modes(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze phase resonance modes."""
        modes = []
        
        # Calculate phase resonance characteristics
        phase_indices = [3, 4, 5]  # Phase dimensions
        phase_data = envelope.take(phase_indices, axis=0)
        phase_resonance = np.corrcoef(phase_data.reshape(phase_data.shape[0], -1))
        
        # Find resonance modes
        if np.max(phase_resonance) > 0.1:
            modes.append({
                'type': 'phase',
                'strength': float(np.max(phase_resonance)),
                'pattern': phase_resonance.tolist()
            })
        
        return modes
    
    def _calculate_resonance_frequency(self, envelope: np.ndarray) -> float:
        """Calculate resonance frequency."""
        # Simplified resonance frequency calculation
        fft_result = np.fft.fftn(envelope)
        power_spectrum = np.abs(fft_result)**2
        
        # Find dominant frequency
        dominant_freq = np.argmax(power_spectrum)
        return float(dominant_freq)
    
    def _calculate_resonance_amplitude(self, envelope: np.ndarray) -> float:
        """Calculate resonance amplitude."""
        # Simplified resonance amplitude calculation
        return float(np.max(np.abs(envelope)))
    
    def _calculate_quality_factor(self, envelope: np.ndarray) -> float:
        """Calculate quality factor."""
        # Simplified quality factor calculation
        fft_result = np.fft.fftn(envelope)
        power_spectrum = np.abs(fft_result)**2
        
        # Calculate quality factor based on peak width
        peak_index = np.argmax(power_spectrum)
        peak_value = power_spectrum[peak_index]
        half_max = peak_value / 2.0
        
        # Find width at half maximum
        width = 0
        for i in range(len(power_spectrum)):
            if power_spectrum[i] >= half_max:
                width += 1
        
        if width > 0:
            quality_factor = peak_index / width
        else:
            quality_factor = 0.0
        
        return float(quality_factor)
    
    def _calculate_resonance_strength(self, envelope: np.ndarray) -> float:
        """Calculate resonance strength."""
        # Simplified resonance strength calculation
        return float(np.std(envelope))
    
    def _calculate_quality_factors(self, envelope: np.ndarray) -> List[float]:
        """Calculate quality factors for different modes."""
        quality_factors = []
        
        # Calculate quality factors for different frequency components
        fft_result = np.fft.fftn(envelope)
        power_spectrum = np.abs(fft_result)**2
        
        # Find peaks and calculate quality factors
        for i in range(1, len(power_spectrum) - 1):
            if power_spectrum[i] > power_spectrum[i-1] and power_spectrum[i] > power_spectrum[i+1]:
                # Calculate quality factor for this peak
                peak_value = power_spectrum[i]
                half_max = peak_value / 2.0
                
                # Find width at half maximum
                width = 0
                for j in range(len(power_spectrum)):
                    if power_spectrum[j] >= half_max:
                        width += 1
                
                if width > 0:
                    quality_factor = i / width
                else:
                    quality_factor = 0.0
                
                quality_factors.append(quality_factor)
        
        return quality_factors
    
    def _analyze_quality_distribution(self, quality_factors: List[float]) -> Dict[str, float]:
        """Analyze quality factor distribution."""
        if not quality_factors:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': float(np.mean(quality_factors)),
            'std': float(np.std(quality_factors)),
            'min': float(np.min(quality_factors)),
            'max': float(np.max(quality_factors))
        }
    
    def _calculate_quality_statistics(self, quality_factors: List[float]) -> Dict[str, Any]:
        """Calculate quality factor statistics."""
        if not quality_factors:
            return {'count': 0, 'high_quality_count': 0, 'high_quality_ratio': 0.0}
        
        high_quality_count = sum(1 for qf in quality_factors if qf > 1.0)
        high_quality_ratio = high_quality_count / len(quality_factors)
        
        return {
            'count': len(quality_factors),
            'high_quality_count': high_quality_count,
            'high_quality_ratio': float(high_quality_ratio)
        }
    
    def _identify_high_quality_resonators(self, quality_factors: List[float]) -> List[int]:
        """Identify high-quality resonators."""
        high_quality_indices = []
        
        for i, qf in enumerate(quality_factors):
            if qf > 1.0:  # High quality threshold
                high_quality_indices.append(i)
        
        return high_quality_indices


class ResonatorVisualizer:
    """
    Visualization utilities for resonator analysis.
    
    Physical Meaning:
        Provides visualization tools for resonator analysis results,
        including plots of resonance spectra, quality factors,
        and resonator characteristics.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize resonator visualizer.
        
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def plot_resonance_spectrum(self, resonance_spectrum: Dict[str, Any], save_path: str = None) -> None:
        """
        Plot resonance spectrum.
        
        Args:
            resonance_spectrum (Dict[str, Any]): Resonance spectrum data.
            save_path (str, optional): Path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(resonance_spectrum['resonance_spectrum'])
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            plt.title('Resonance Spectrum')
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
    
    def plot_quality_factors(self, quality_factors: List[float], save_path: str = None) -> None:
        """
        Plot quality factors.
        
        Args:
            quality_factors (List[float]): Quality factor data.
            save_path (str, optional): Path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(quality_factors)), quality_factors)
            plt.xlabel('Resonator Index')
            plt.ylabel('Quality Factor')
            plt.title('Resonator Quality Factors')
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
    
    def plot_resonance_modes(self, resonance_modes: List[Dict[str, Any]], save_path: str = None) -> None:
        """
        Plot resonance modes.
        
        Args:
            resonance_modes (List[Dict[str, Any]]): Resonance mode data.
            save_path (str, optional): Path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots for different mode types
            fig, axes = plt.subplots(len(resonance_modes), 1, figsize=(10, 6*len(resonance_modes)))
            
            if len(resonance_modes) == 1:
                axes = [axes]
            
            for i, mode in enumerate(resonance_modes):
                axes[i].plot(mode['pattern'])
                axes[i].set_title(f'{mode["type"].title()} Resonance Mode')
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
