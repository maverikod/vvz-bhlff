"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quench Detector implementation according to step 00 specification.

This module implements the detector for quench events in BVP,
monitoring local thresholds and detecting when BVP dissipatively
"dumps" energy into the medium.

Theoretical Background:
    Quenches represent threshold events in the BVP field where
    local thresholds (amplitude/detuning/gradient) are reached,
    causing the BVP to dissipatively "dump" energy into the medium.
    This results in a local regime transition with increased losses
    and Q-factor changes.

Example:
    >>> detector = QuenchDetector(domain_7d, config)
    >>> quenches = detector.detect_quenches(envelope)
    >>> print(f"Quenches detected: {quenches['quenches_detected']}")
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from ..domain.domain_7d import Domain7D


class QuenchDetector:
    """
    Detector for quench events in BVP.
    
    Physical Meaning:
        Monitors local thresholds (amplitude/detuning/gradient)
        and detects when BVP dissipatively "dumps" energy into
        the medium. Quenches represent threshold events where
        the BVP field undergoes a local regime transition.
        
    Mathematical Foundation:
        Applies three threshold criteria for quench detection:
        1. Amplitude threshold: |A| > |A_q|
        2. Detuning threshold: |ω - ω_0| > Δω_q  
        3. Gradient threshold: |∇A| > |∇A_q|
        where A_q, Δω_q, and ∇A_q are the quench thresholds.
    """
    
    def __init__(self, domain_7d: Domain7D, config: Dict[str, Any]):
        """
        Initialize quench detector.
        
        Physical Meaning:
            Sets up the quench detector with threshold parameters
            for detecting amplitude, detuning, and gradient quenches
            in the 7D BVP field.
            
        Args:
            domain_7d (Domain7D): 7D computational domain.
            config (Dict[str, Any]): Configuration parameters including:
                - amplitude_threshold (float): Amplitude quench threshold |A_q|
                - detuning_threshold (float): Detuning quench threshold Δω_q
                - gradient_threshold (float): Gradient quench threshold |∇A_q|
                - carrier_frequency (float): BVP carrier frequency ω₀
        """
        self.domain_7d = domain_7d
        self.config = config
        
        # Extract threshold parameters
        self.amplitude_threshold = config.get("amplitude_threshold", 0.8)
        self.detuning_threshold = config.get("detuning_threshold", 0.1)
        self.gradient_threshold = config.get("gradient_threshold", 0.5)
        self.carrier_frequency = config.get("carrier_frequency", 1.85e43)
        
        # Setup threshold validation
        self._validate_thresholds()
    
    def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quench events based on three thresholds.
        
        Physical Meaning:
            Applies three threshold criteria to detect quench events:
            - amplitude: |A| > |A_q| - detects high-amplitude quenches
            - detuning: |ω - ω_0| > Δω_q - detects frequency detuning quenches
            - gradient: |∇A| > |∇A_q| - detects high-gradient quenches
            
        Mathematical Foundation:
            For each point in 7D space-time, checks:
            1. |A(x,φ,t)| > |A_q|
            2. |ω_local - ω_0| > Δω_q
            3. |∇A(x,φ,t)| > |∇A_q|
            where ω_local is the local frequency derived from phase evolution.
            
        Args:
            envelope (np.ndarray): 7D envelope field with shape
                (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t)
                
        Returns:
            Dict[str, Any]: Quench detection results including:
                - quenches_detected (bool): Whether any quenches were found
                - quench_locations (List[Tuple]): 7D coordinates of quench events
                - quench_types (List[str]): Types of quenches detected
                - quench_strengths (List[float]): Strength of each quench
                - amplitude_quenches (List[Tuple]): Amplitude quench locations
                - detuning_quenches (List[Tuple]): Detuning quench locations
                - gradient_quenches (List[Tuple]): Gradient quench locations
        """
        # Detect different types of quenches
        amplitude_quenches = self._detect_amplitude_quenches(envelope)
        detuning_quenches = self._detect_detuning_quenches(envelope)
        gradient_quenches = self._detect_gradient_quenches(envelope)
        
        # Combine all quenches
        all_quenches = amplitude_quenches + detuning_quenches + gradient_quenches
        quench_locations = [q["location"] for q in all_quenches]
        quench_types = [q["type"] for q in all_quenches]
        quench_strengths = [q["strength"] for q in all_quenches]
        
        return {
            "quenches_detected": len(all_quenches) > 0,
            "quench_locations": quench_locations,
            "quench_types": quench_types,
            "quench_strengths": quench_strengths,
            "amplitude_quenches": [q["location"] for q in amplitude_quenches],
            "detuning_quenches": [q["location"] for q in detuning_quenches],
            "gradient_quenches": [q["location"] for q in gradient_quenches],
            "total_quenches": len(all_quenches)
        }
    
    def _detect_amplitude_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect amplitude quenches: |A| > |A_q|.
        
        Physical Meaning:
            Detects locations where the envelope amplitude exceeds
            the amplitude threshold, indicating potential quench events
            due to high field strength.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            List[Dict[str, Any]]: List of amplitude quench events.
        """
        quenches = []
        
        # Compute amplitude
        amplitude = np.abs(envelope)
        
        # Find locations exceeding threshold
        quench_mask = amplitude > self.amplitude_threshold
        
        if np.any(quench_mask):
            # Get coordinates of quench events
            quench_indices = np.where(quench_mask)
            
            for i in range(len(quench_indices[0])):
                location = tuple(idx[i] for idx in quench_indices)
                strength = amplitude[location]
                
                quenches.append({
                    "location": location,
                    "type": "amplitude",
                    "strength": float(strength),
                    "threshold": self.amplitude_threshold
                })
        
        return quenches
    
    def _detect_detuning_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect detuning quenches: |ω - ω_0| > Δω_q.
        
        Physical Meaning:
            Detects locations where the local frequency deviates
            significantly from the carrier frequency, indicating
            detuning quench events.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            List[Dict[str, Any]]: List of detuning quench events.
        """
        quenches = []
        
        # Compute local frequency from phase evolution
        if envelope.shape[-1] > 1:  # Need at least 2 time slices
            phase = np.angle(envelope)
            phase_diff = np.diff(phase, axis=-1)
            
            # Local frequency (avoid division by zero)
            dt = self.domain_7d.temporal_config.dt
            local_frequency = np.abs(phase_diff) / (dt + 1e-12)
            
            # Detuning from carrier frequency
            detuning = np.abs(local_frequency - self.carrier_frequency)
            
            # Find locations exceeding detuning threshold
            quench_mask = detuning > self.detuning_threshold
            
            if np.any(quench_mask):
                # Get coordinates of quench events
                quench_indices = np.where(quench_mask)
                
                for i in range(len(quench_indices[0])):
                    location = tuple(idx[i] for idx in quench_indices)
                    strength = detuning[location]
                    
                    quenches.append({
                        "location": location,
                        "type": "detuning",
                        "strength": float(strength),
                        "threshold": self.detuning_threshold
                    })
        
        return quenches
    
    def _detect_gradient_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect gradient quenches: |∇A| > |∇A_q|.
        
        Physical Meaning:
            Detects locations where the envelope gradient exceeds
            the gradient threshold, indicating potential quench events
            due to high spatial/phase gradients.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            List[Dict[str, Any]]: List of gradient quench events.
        """
        quenches = []
        
        # Compute gradients in all 7 dimensions
        differentials = self.domain_7d.get_differentials()
        dx = differentials["dx"]
        dy = differentials["dy"]
        dz = differentials["dz"]
        dphi1 = differentials["dphi_1"]
        dphi2 = differentials["dphi_2"]
        dphi3 = differentials["dphi_3"]
        
        # Spatial gradients
        grad_x = np.gradient(envelope, dx, axis=0)
        grad_y = np.gradient(envelope, dy, axis=1)
        grad_z = np.gradient(envelope, dz, axis=2)
        
        # Phase gradients
        grad_phi1 = np.gradient(envelope, dphi1, axis=3)
        grad_phi2 = np.gradient(envelope, dphi2, axis=4)
        grad_phi3 = np.gradient(envelope, dphi3, axis=5)
        
        # Total gradient magnitude
        grad_magnitude = np.sqrt(
            np.abs(grad_x)**2 + np.abs(grad_y)**2 + np.abs(grad_z)**2 +
            np.abs(grad_phi1)**2 + np.abs(grad_phi2)**2 + np.abs(grad_phi3)**2
        )
        
        # Find locations exceeding gradient threshold
        quench_mask = grad_magnitude > self.gradient_threshold
        
        if np.any(quench_mask):
            # Get coordinates of quench events
            quench_indices = np.where(quench_mask)
            
            for i in range(len(quench_indices[0])):
                location = tuple(idx[i] for idx in quench_indices)
                strength = grad_magnitude[location]
                
                quenches.append({
                    "location": location,
                    "type": "gradient",
                    "strength": float(strength),
                    "threshold": self.gradient_threshold
                })
        
        return quenches
    
    def _validate_thresholds(self) -> None:
        """
        Validate threshold parameters.
        
        Physical Meaning:
            Ensures that threshold parameters are physically reasonable
            and consistent with the BVP theory.
            
        Raises:
            ValueError: If thresholds are invalid.
        """
        if self.amplitude_threshold <= 0:
            raise ValueError("Amplitude threshold must be positive")
        
        if self.detuning_threshold <= 0:
            raise ValueError("Detuning threshold must be positive")
        
        if self.gradient_threshold <= 0:
            raise ValueError("Gradient threshold must be positive")
        
        if self.carrier_frequency <= 0:
            raise ValueError("Carrier frequency must be positive")
