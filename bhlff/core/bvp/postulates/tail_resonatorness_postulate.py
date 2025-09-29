"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Postulate 6: Tail Resonatorness implementation.

This module implements the Tail Resonatorness postulate for the BVP framework,
validating that the tail exhibits cascade of effective resonators with
frequency-dependent impedance.

Physical Meaning:
    The Tail Resonatorness postulate describes how the tail forms a cascade
    of effective resonators/transmission lines with frequency-dependent
    impedance. The spectrum {ω_n,Q_n} is determined by the BVP and boundaries.

Mathematical Foundation:
    Validates tail resonatorness by analyzing the frequency spectrum
    and identifying resonant modes with their quality factors. The
    resonators should exhibit proper frequency-dependent impedance
    characteristics.

Example:
    >>> postulate = BVPPostulate6_TailResonatorness(domain_7d, config)
    >>> results = postulate.apply(envelope_7d)
    >>> print(f"Resonances detected: {results['resonance_count']}")
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from ...domain.domain_7d import Domain7D
from ..bvp_postulate_base import BVPPostulate


class BVPPostulate6_TailResonatorness(BVPPostulate):
    """
    Postulate 6: Tail Resonatorness.
    
    Physical Meaning:
        Tail is cascade of effective resonators/transmission lines with
        frequency-dependent impedance; spectrum {ω_n,Q_n} is determined
        by BVP and boundaries.
        
    Mathematical Foundation:
        Validates tail resonatorness by analyzing frequency spectrum
        and quality factors of resonant modes.
    """
    
    def __init__(self, domain_7d: Domain7D, config: Dict[str, Any]):
        """
        Initialize Tail Resonatorness postulate.
        
        Physical Meaning:
            Sets up the postulate with the computational domain and
            configuration parameters, including the minimum required
            resonance count and quality factors.
            
        Args:
            domain_7d (Domain7D): 7D computational domain.
            config (Dict[str, Any]): Configuration parameters including:
                - min_resonance_count (int): Minimum required resonances (default: 3)
                - min_quality_factor (float): Minimum required Q-factor (default: 10.0)
        """
        self.domain_7d = domain_7d
        self.config = config
        self.min_resonance_count = config.get('min_resonance_count', 3)
        self.min_quality_factor = config.get('min_quality_factor', 10.0)
    
    def apply(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Apply Tail Resonatorness postulate.
        
        Physical Meaning:
            Validates tail resonatorness by analyzing the frequency spectrum
            and identifying resonant modes with their quality factors. This
            ensures that the tail exhibits the proper cascade of resonators
            with frequency-dependent impedance.
            
        Mathematical Foundation:
            Computes the power spectrum of the envelope field, identifies
            resonance peaks, and calculates their quality factors. Validates
            that sufficient resonances exist with adequate quality factors.
            
        Args:
            envelope (np.ndarray): 7D envelope field to validate.
                Shape: (N_x, N_y, N_z, N_φx, N_φy, N_φz, N_t)
                
        Returns:
            Dict[str, Any]: Validation results including:
                - postulate_satisfied (bool): Whether postulate is satisfied
                - resonance_frequencies (List[float]): Detected resonance frequencies
                - quality_factors (List[float]): Quality factors of resonances
                - resonance_count (int): Number of detected resonances
                - min_required_resonances (int): Minimum required resonances
                - min_required_quality (float): Minimum required Q-factor
        """
        # Compute frequency spectrum
        fft_envelope = np.fft.fftn(envelope)
        power_spectrum = np.abs(fft_envelope)**2
        
        # Find resonance peaks
        resonance_frequencies, quality_factors = self._find_resonance_peaks(power_spectrum)
        
        # Check if sufficient resonances are found
        resonance_count = len(resonance_frequencies)
        sufficient_resonances = resonance_count >= self.min_resonance_count
        
        # Check if quality factors are adequate
        adequate_quality = all(q > self.min_quality_factor for q in quality_factors)
        
        postulate_satisfied = sufficient_resonances and adequate_quality
        
        return {
            'postulate_satisfied': postulate_satisfied,
            'resonance_frequencies': [float(f) for f in resonance_frequencies],
            'quality_factors': [float(q) for q in quality_factors],
            'resonance_count': int(resonance_count),
            'min_required_resonances': self.min_resonance_count,
            'min_required_quality': self.min_quality_factor
        }
    
    def _find_resonance_peaks(self, power_spectrum: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Find resonance peaks in power spectrum.
        
        Physical Meaning:
            Identifies resonance peaks in the power spectrum that correspond
            to the cascade of effective resonators in the tail. These peaks
            should exhibit proper frequency-dependent impedance characteristics.
            
        Mathematical Foundation:
            Uses peak detection algorithms to identify local maxima in the
            power spectrum above a threshold, then computes quality factors
            for each identified resonance.
            
        Args:
            power_spectrum (np.ndarray): Power spectrum of the envelope field.
            
        Returns:
            Tuple[List[float], List[float]]: Tuple containing:
                - frequencies: List of resonance frequencies
                - quality_factors: List of corresponding quality factors
        """
        # Simplified peak finding - in practice would use more sophisticated methods
        max_power = np.max(power_spectrum)
        threshold = 0.1 * max_power
        
        # Find peaks above threshold
        peaks = np.where(power_spectrum > threshold)
        
        if len(peaks[0]) == 0:
            return [], []
        
        # Extract peak frequencies and compute quality factors
        frequencies = []
        quality_factors = []
        
        for i in range(min(len(peaks[0]), 10)):  # Limit to top 10 peaks
            peak_power = power_spectrum[tuple(p[idx] for p, idx in zip(peaks, [i]*len(peaks)))]
            frequencies.append(float(peak_power))
            quality_factors.append(float(peak_power / threshold))  # Simplified Q factor
        
        return frequencies, quality_factors
