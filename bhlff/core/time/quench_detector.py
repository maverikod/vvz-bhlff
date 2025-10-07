"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quench detection system for 7D BVP framework.

This module implements quench detection for monitoring energy dumping
events during temporal integration of phase field equations.

Physical Meaning:
    Quench detection monitors the phase field for sudden energy dumping
    events that may indicate phase transitions or topological changes
    in the 7D phase field configuration.

Mathematical Foundation:
    Detects quench events based on energy thresholds and rate of change
    in the phase field configuration.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

from ..domain import Domain


class QuenchDetector:
    """
    Quench detection system for 7D BVP framework.

    Physical Meaning:
        Monitors the phase field for sudden energy dumping events that
        may indicate phase transitions, topological changes, or other
        critical events in the 7D phase field dynamics.

    Mathematical Foundation:
        Detects quench events based on:
        - Energy threshold: |E(t) - E(t-1)| > threshold
        - Rate of change: |∂E/∂t| > rate_threshold
        - Field magnitude: |a(x,t)| > magnitude_threshold

    Attributes:
        domain (Domain): Computational domain.
        energy_threshold (float): Energy change threshold for quench detection.
        rate_threshold (float): Rate of change threshold.
        magnitude_threshold (float): Field magnitude threshold.
        quench_history (List[Dict]): History of detected quench events.
        _previous_energy (Optional[float]): Previous energy value.
        _initialized (bool): Initialization status.
    """

    def __init__(
        self,
        domain: Domain,
        energy_threshold: float = 1e-3,
        rate_threshold: float = 1e-2,
        magnitude_threshold: float = 10.0,
    ) -> None:
        """
        Initialize quench detector.

        Physical Meaning:
            Sets up the quench detection system with specified thresholds
            for monitoring energy dumping events in the phase field.

        Args:
            domain (Domain): Computational domain.
            energy_threshold (float): Energy change threshold for quench detection.
            rate_threshold (float): Rate of change threshold.
            magnitude_threshold (float): Field magnitude threshold.
        """
        # Validate thresholds
        if energy_threshold <= 0:
            raise ValueError("Energy threshold must be positive")
        if rate_threshold <= 0:
            raise ValueError("Rate threshold must be positive")
        if magnitude_threshold <= 0:
            raise ValueError("Magnitude threshold must be positive")

        self.domain = domain
        self.energy_threshold = energy_threshold
        self.rate_threshold = rate_threshold
        self.magnitude_threshold = magnitude_threshold

        # Quench history
        self.quench_history = []
        self._previous_energy = None
        self._last_time = None
        self._initialized = True

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Quench detector initialized with thresholds: "
            f"energy={energy_threshold}, rate={rate_threshold}, "
            f"magnitude={magnitude_threshold}"
        )

    def detect_quench(self, field: np.ndarray, time: float) -> bool:
        """
        Detect quench events in the field.

        Physical Meaning:
            Analyzes the current field configuration for signs of energy
            dumping or sudden changes that may indicate quench events.

        Mathematical Foundation:
            Checks multiple criteria:
            1. Energy change: |E(t) - E(t-1)| > energy_threshold
            2. Rate of change: |∂E/∂t| > rate_threshold
            3. Field magnitude: max|a(x,t)| > magnitude_threshold

        Args:
            field (np.ndarray): Current field configuration.
            time (float): Current time.

        Returns:
            bool: True if quench detected, False otherwise.
        """
        if not self._initialized:
            raise RuntimeError("Quench detector not initialized")

        # Validate field shape
        if field.shape != self.domain.shape:
            raise ValueError(f"Field shape {field.shape} must match domain shape {self.domain.shape}")

        # Validate field type
        if not np.iscomplexobj(field):
            raise ValueError("Field must be complex")

        # Validate time
        if time < 0:
            raise ValueError("Time must be non-negative")

        quench_detected = False
        quench_reasons = []

        # Calculate current energy
        current_energy = self._calculate_energy(field)

        # Check energy change threshold
        if self._previous_energy is not None:
            energy_change = abs(current_energy - self._previous_energy)
            if energy_change > self.energy_threshold:
                quench_detected = True
                quench_reasons.append(f"energy_change={energy_change:.2e}")

        # Check rate of change threshold
        if self._previous_energy is not None and self._last_time is not None:
            dt = time - self._last_time
            if dt > 0:  # Avoid division by zero
                rate_of_change = abs(current_energy - self._previous_energy) / dt
                if rate_of_change > self.rate_threshold:
                    quench_detected = True
                    quench_reasons.append(f"rate_change={rate_of_change:.2e}")

        # Check field magnitude threshold
        max_magnitude = np.max(np.abs(field))
        if max_magnitude > self.magnitude_threshold:
            quench_detected = True
            quench_reasons.append(f"magnitude={max_magnitude:.2e}")

        # Record quench event
        if quench_detected:
            quench_event = {
                "time": time,
                "energy": current_energy,
                "magnitude": max_magnitude,
                "reasons": quench_reasons,
                "field_stats": {
                    "max_magnitude": max_magnitude,
                    "mean_magnitude": np.mean(np.abs(field)),
                    "std_magnitude": np.std(np.abs(field)),
                },
            }
            self.quench_history.append(quench_event)
            self.logger.warning(
                f"Quench detected at t={time:.3f}: {', '.join(quench_reasons)}"
            )

        # Update previous values
        self._previous_energy = current_energy
        self._last_time = time

        return quench_detected

    def _calculate_energy(self, field: np.ndarray) -> float:
        """
        Calculate energy of the field configuration.

        Physical Meaning:
            Computes the total energy of the phase field configuration,
            which is used for quench detection.

        Mathematical Foundation:
            Energy = ∫ |a(x)|² dx
            Approximated as: Energy ≈ Σ |a(x)|² Δx

        Args:
            field (np.ndarray): Field configuration.

        Returns:
            float: Total energy of the field.
        """
        # Calculate energy density
        energy_density = np.abs(field) ** 2

        # Integrate over domain
        # For 7D: Δx = (dx^3) * (dphi^3) * dt
        dx = self.domain.L_spatial / self.domain.N_spatial
        dphi = (2 * np.pi) / self.domain.N_phase
        dt = self.domain.T / self.domain.N_t
        volume_element = (dx**3) * (dphi**3) * dt

        total_energy = np.sum(energy_density) * volume_element
        return float(total_energy)

    def get_quench_history(self) -> List[Dict]:
        """
        Get history of detected quench events.

        Returns:
            List[Dict]: List of quench events with details.
        """
        return self.quench_history.copy()

    def clear_history(self) -> None:
        """Clear quench event history."""
        self.quench_history.clear()
        self._previous_energy = None
        self.logger.info("Quench history cleared")

    def set_thresholds(
        self,
        energy_threshold: Optional[float] = None,
        rate_threshold: Optional[float] = None,
        magnitude_threshold: Optional[float] = None,
    ) -> None:
        """
        Update detection thresholds.

        Physical Meaning:
            Adjusts the sensitivity of quench detection by modifying
            the thresholds for energy change, rate of change, and
            field magnitude.

        Args:
            energy_threshold (Optional[float]): New energy change threshold.
            rate_threshold (Optional[float]): New rate of change threshold.
            magnitude_threshold (Optional[float]): New magnitude threshold.
        """
        if energy_threshold is not None:
            if energy_threshold <= 0:
                raise ValueError(
                    f"Energy threshold must be positive, got {energy_threshold}"
                )
            self.energy_threshold = energy_threshold

        if rate_threshold is not None:
            if rate_threshold <= 0:
                raise ValueError(
                    f"Rate threshold must be positive, got {rate_threshold}"
                )
            self.rate_threshold = rate_threshold

        if magnitude_threshold is not None:
            if magnitude_threshold <= 0:
                raise ValueError(
                    f"Magnitude threshold must be positive, got {magnitude_threshold}"
                )
            self.magnitude_threshold = magnitude_threshold

        self.logger.info(
            f"Thresholds updated: energy={self.energy_threshold}, "
            f"rate={self.rate_threshold}, magnitude={self.magnitude_threshold}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get quench detection statistics.

        Returns:
            Dict[str, Any]: Statistics about detected quench events.
        """
        if not self.quench_history:
            return {"total_quenches": 0, "average_energy": 0.0, "quench_rate": 0.0}

        total_quenches = len(self.quench_history)
        energies = [event["energy"] for event in self.quench_history]
        average_energy = np.mean(energies)
        
        # Calculate average magnitude
        magnitudes = [event["field_stats"]["max_magnitude"] for event in self.quench_history]
        average_magnitude = np.mean(magnitudes)

        # Calculate quench rate (quenches per unit time)
        if len(self.quench_history) > 1:
            time_span = self.quench_history[-1]["time"] - self.quench_history[0]["time"]
            quench_rate = total_quenches / time_span if time_span > 0 else 0.0
        else:
            time_span = 0.0
            quench_rate = 0.0

        return {
            "total_quenches": total_quenches,
            "average_energy": average_energy,
            "average_magnitude": average_magnitude,
            "quench_rate": quench_rate,
            "time_span": time_span,
            "energy_range": (min(energies), max(energies)),
        }

    def __repr__(self) -> str:
        """String representation of quench detector."""
        return (
            f"QuenchDetector("
            f"domain={self.domain.shape}, "
            f"energy_threshold={self.energy_threshold}, "
            f"rate_threshold={self.rate_threshold}, "
            f"magnitude_threshold={self.magnitude_threshold})"
        )
