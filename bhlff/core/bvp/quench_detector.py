"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quench detector for BVP events.

This module implements the detector for quench events in the Base
High-Frequency Field (BVP) when local thresholds are reached.

Physical Meaning:
    Monitors local thresholds (amplitude/detuning/gradient) and detects
    when BVP dissipatively "dumps" energy into the medium at local
    thresholds.

Mathematical Foundation:
    Applies three threshold criteria for quench detection:
    - amplitude: |A| > |A_q|
    - detuning: |ω - ω_0| > Δω_q
    - gradient: |∇A| > |∇A_q|
"""

import numpy as np
from typing import Dict, Any, List, Optional

from .bvp_constants import BVPConstants


class QuenchDetector:
    """
    Detector for quench events in BVP.

    Physical Meaning:
        Monitors local thresholds (amplitude/detuning/gradient)
        and detects when BVP dissipatively "dumps" energy.

    Mathematical Foundation:
        Applies three threshold criteria for quench detection:
        - amplitude: |A| > |A_q|
        - detuning: |ω - ω_0| > Δω_q
        - gradient: |∇A| > |∇A_q|

    Attributes:
        config (Dict[str, Any]): Quench detection configuration.
        amplitude_threshold (float): Amplitude threshold for quench detection.
        detuning_threshold (float): Detuning threshold for quench detection.
        gradient_threshold (float): Gradient threshold for quench detection.
    """

    def __init__(self, config: Dict[str, Any], constants: Optional[BVPConstants] = None) -> None:
        """
        Initialize quench detector with configuration.

        Physical Meaning:
            Sets up the detector with threshold values for identifying
            quench events when local conditions exceed critical values.

        Args:
            config (Dict[str, Any]): Quench detection configuration including:
                - amplitude_threshold: Threshold for amplitude quenches
                - detuning_threshold: Threshold for detuning quenches
                - gradient_threshold: Threshold for gradient quenches
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.config = config
        self.constants = constants or BVPConstants(config)
        self._setup_thresholds()

    def _setup_thresholds(self) -> None:
        """
        Setup quench detection thresholds.

        Physical Meaning:
            Initializes the threshold values used for detecting
            different types of quench events.
        """
        self.amplitude_threshold = self.constants.get_quench_threshold("amplitude_threshold")
        self.detuning_threshold = self.constants.get_quench_threshold("detuning_threshold")
        self.gradient_threshold = self.constants.get_quench_threshold("gradient_threshold")

    def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quench events based on three thresholds.

        Physical Meaning:
            Applies three threshold criteria to identify quench events:
            - amplitude: |A| > |A_q|
            - detuning: |ω - ω_0| > Δω_q
            - gradient: |∇A| > |∇A_q|

        Mathematical Foundation:
            Quench detection uses three criteria:
            1. Amplitude quench: |A(x)| > A_q
            2. Detuning quench: |ω - ω_0| > Δω_q
            3. Gradient quench: |∇A(x)| > |∇A_q|

        Args:
            envelope (np.ndarray): BVP envelope to analyze.
                Represents the field amplitude distribution.

        Returns:
            Dict[str, Any]: Quench detection results including:
                - quench_locations: Spatial locations of quenches
                - quench_types: Types of quenches detected
                - energy_dumped: Energy dumped at each quench
        """
        # Detect amplitude quenches
        amplitude_quenches = self._detect_amplitude_quenches(envelope)

        # Detect gradient quenches
        gradient_quenches = self._detect_gradient_quenches(envelope)

        # Combine results
        all_quenches = amplitude_quenches + gradient_quenches

        # Extract locations, types, and energy
        quench_locations = [q["location"] for q in all_quenches]
        quench_types = [q["type"] for q in all_quenches]
        energy_dumped = [q["energy"] for q in all_quenches]

        return {
            "quench_locations": quench_locations,
            "quench_types": quench_types,
            "energy_dumped": energy_dumped,
        }

    def _detect_amplitude_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect amplitude-based quench events.

        Physical Meaning:
            Identifies locations where the field amplitude exceeds
            the critical threshold, indicating potential quench events.

        Args:
            envelope (np.ndarray): Field envelope to analyze.

        Returns:
            List[Dict[str, Any]]: List of amplitude quench events.
        """
        quenches = []

        # Find locations where amplitude exceeds threshold
        amplitude = np.abs(envelope)
        quench_mask = amplitude > self.amplitude_threshold

        if np.any(quench_mask):
            # Find coordinates of quench events
            quench_indices = np.where(quench_mask)

            for i in range(len(quench_indices[0])):
                location = tuple(idx[i] for idx in quench_indices)
                energy = (
                    amplitude[location] ** 2
                )  # Energy proportional to amplitude squared

                quenches.append(
                    {
                        "location": location,
                        "type": "amplitude",
                        "energy": energy,
                    }
                )

        return quenches

    def _detect_gradient_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect gradient-based quench events.

        Physical Meaning:
            Identifies locations where the field gradient exceeds
            the critical threshold, indicating potential quench events.

        Args:
            envelope (np.ndarray): Field envelope to analyze.

        Returns:
            List[Dict[str, Any]]: List of gradient quench events.
        """
        quenches = []

        # Compute gradient magnitude
        if envelope.ndim == 1:
            gradient = np.gradient(envelope)
            gradient_magnitude = np.abs(gradient)
        elif envelope.ndim == 2:
            grad_x, grad_y = np.gradient(envelope)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        else:  # 3D
            grad_x, grad_y, grad_z = np.gradient(envelope)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        # Find locations where gradient exceeds threshold
        quench_mask = gradient_magnitude > self.gradient_threshold

        if np.any(quench_mask):
            # Find coordinates of quench events
            quench_indices = np.where(quench_mask)

            for i in range(len(quench_indices[0])):
                location = tuple(idx[i] for idx in quench_indices)
                energy = (
                    gradient_magnitude[location] ** 2
                )  # Energy proportional to gradient squared

                quenches.append(
                    {
                        "location": location,
                        "type": "gradient",
                        "energy": energy,
                    }
                )

        return quenches

    def get_thresholds(self) -> Dict[str, float]:
        """
        Get current threshold values.

        Physical Meaning:
            Returns the current threshold values used for quench detection.

        Returns:
            Dict[str, float]: Dictionary of threshold values.
        """
        return {
            "amplitude_threshold": self.amplitude_threshold,
            "detuning_threshold": self.detuning_threshold,
            "gradient_threshold": self.gradient_threshold,
        }

    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set new threshold values.

        Physical Meaning:
            Updates the threshold values used for quench detection.

        Args:
            thresholds (Dict[str, float]): New threshold values.
        """
        if "amplitude_threshold" in thresholds:
            self.amplitude_threshold = thresholds["amplitude_threshold"]
        if "detuning_threshold" in thresholds:
            self.detuning_threshold = thresholds["detuning_threshold"]
        if "gradient_threshold" in thresholds:
            self.gradient_threshold = thresholds["gradient_threshold"]

    def __repr__(self) -> str:
        """String representation of quench detector."""
        return (
            f"QuenchDetector(amplitude_threshold={self.amplitude_threshold}, "
            f"detuning_threshold={self.detuning_threshold}, "
            f"gradient_threshold={self.gradient_threshold})"
        )
