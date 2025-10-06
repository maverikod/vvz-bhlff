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
from .quench_thresholds import QuenchThresholdComputer
from .quench_morphology import QuenchMorphology
from .quench_characteristics import QuenchCharacteristics


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

        # Initialize threshold computer, morphology processor, and characteristics computer
        self.threshold_computer = QuenchThresholdComputer(domain_7d)
        self.morphology = QuenchMorphology()
        self.characteristics = QuenchCharacteristics(domain_7d)

        # Compute physical thresholds from theoretical principles
        thresholds = self.threshold_computer.compute_all_thresholds()
        self.amplitude_threshold = thresholds["amplitude_threshold"]
        self.detuning_threshold = thresholds["detuning_threshold"]
        self.gradient_threshold = thresholds["gradient_threshold"]
        self.carrier_frequency = thresholds["carrier_frequency"]

        # Override with config values if provided (for testing/debugging)
        if "amplitude_threshold" in config:
            self.amplitude_threshold = config["amplitude_threshold"]
        if "detuning_threshold" in config:
            self.detuning_threshold = config["detuning_threshold"]
        if "gradient_threshold" in config:
            self.gradient_threshold = config["gradient_threshold"]
        if "carrier_frequency" in config:
            self.carrier_frequency = config["carrier_frequency"]

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
            "total_quenches": len(all_quenches),
        }

    def _detect_amplitude_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect amplitude quenches: |A| > |A_q| with advanced processing.

        Physical Meaning:
            Detects locations where the envelope amplitude exceeds
            the amplitude threshold, indicating potential quench events
            due to high field strength. Uses morphological operations
            to filter noise and find connected components.

        Mathematical Foundation:
            Applies morphological operations to filter noise:
            - Binary opening: removes small noise components
            - Binary closing: fills small gaps in quench regions
            - Connected component analysis: groups nearby quench events

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            List[Dict[str, Any]]: List of amplitude quench events with
                enhanced characteristics including size and center of mass.
        """
        quenches = []

        # Compute amplitude
        amplitude = np.abs(envelope)

        # Find locations exceeding threshold
        quench_mask = amplitude > self.amplitude_threshold

        if np.any(quench_mask):
            # Apply morphological operations to filter noise
            quench_mask = self.morphology.apply_morphological_operations(quench_mask)

            # Find connected components
            quench_components = self.morphology.find_connected_components(quench_mask)

            # Process each component
            for component_id, component_mask in quench_components.items():
                if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                    continue  # Skip small components

                # Compute component characteristics
                center = self.characteristics.compute_center_of_mass(component_mask)
                strength = self.characteristics.compute_quench_strength(
                    component_mask, amplitude
                )
                size = np.sum(component_mask)

                quenches.append(
                    {
                        "location": center,
                        "type": "amplitude",
                        "strength": float(strength),
                        "threshold": self.amplitude_threshold,
                        "size": int(size),
                        "component_id": component_id,
                    }
                )

        return quenches

    def _detect_detuning_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect detuning quenches: |ω - ω_0| > Δω_q with advanced processing.

        Physical Meaning:
            Detects locations where the local frequency deviates
            significantly from the carrier frequency, indicating
            detuning quench events. Uses advanced frequency analysis
            and morphological operations for robust detection.

        Mathematical Foundation:
            Computes local frequency using phase evolution:
            ω_local = |dφ/dt| / dt
            Detuning = |ω_local - ω_0|
            Applies same morphological operations as amplitude quenches.

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            List[Dict[str, Any]]: List of detuning quench events with
                enhanced characteristics.
        """
        quenches = []

        # Compute local frequency from phase evolution
        if envelope.shape[-1] > 1:  # Need at least 2 time slices
            local_frequency = self.characteristics.compute_local_frequency(envelope)

            # Detuning from carrier frequency
            detuning = np.abs(local_frequency - self.carrier_frequency)

            # Find locations exceeding detuning threshold
            quench_mask = detuning > self.detuning_threshold

            if np.any(quench_mask):
                # Apply morphological operations to filter noise
                quench_mask = self.morphology.apply_morphological_operations(
                    quench_mask
                )

                # Find connected components
                quench_components = self.morphology.find_connected_components(
                    quench_mask
                )

                # Process each component
                for component_id, component_mask in quench_components.items():
                    if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                        continue  # Skip small components

                    # Compute component characteristics
                    center = self.characteristics.compute_center_of_mass(component_mask)
                    strength = self.characteristics.compute_detuning_strength(
                        component_mask, detuning
                    )
                    size = np.sum(component_mask)

                    quenches.append(
                        {
                            "location": center,
                            "type": "detuning",
                            "strength": float(strength),
                            "threshold": self.detuning_threshold,
                            "size": int(size),
                            "component_id": component_id,
                        }
                    )

        return quenches

    def _detect_gradient_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect gradient quenches: |∇A| > |∇A_q| with advanced processing.

        Physical Meaning:
            Detects locations where the envelope gradient exceeds
            the gradient threshold, indicating potential quench events
            due to high spatial/phase gradients. Uses 7D gradient computation
            and morphological operations for robust detection.

        Mathematical Foundation:
            Computes 7D gradient: ∇A = (∂A/∂x, ∂A/∂y, ∂A/∂z, ∂A/∂φ₁, ∂A/∂φ₂, ∂A/∂φ₃, ∂A/∂t)
            Gradient magnitude: |∇A| = √(Σ|∂A/∂xᵢ|²)
            Applies same morphological operations as other quench types.

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            List[Dict[str, Any]]: List of gradient quench events with
                enhanced characteristics.
        """
        quenches = []

        # Compute 7D gradient
        gradient_magnitude = self.characteristics.compute_7d_gradient_magnitude(
            envelope
        )

        # Find locations exceeding gradient threshold
        quench_mask = gradient_magnitude > self.gradient_threshold

        if np.any(quench_mask):
            # Apply morphological operations to filter noise
            quench_mask = self.morphology.apply_morphological_operations(quench_mask)

            # Find connected components
            quench_components = self.morphology.find_connected_components(quench_mask)

            # Process each component
            for component_id, component_mask in quench_components.items():
                if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                    continue  # Skip small components

                # Compute component characteristics
                center = self.characteristics.compute_center_of_mass(component_mask)
                strength = self.characteristics.compute_gradient_strength(
                    component_mask, gradient_magnitude
                )
                size = np.sum(component_mask)

                quenches.append(
                    {
                        "location": center,
                        "type": "gradient",
                        "strength": float(strength),
                        "threshold": self.gradient_threshold,
                        "size": int(size),
                        "component_id": component_id,
                    }
                )

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
