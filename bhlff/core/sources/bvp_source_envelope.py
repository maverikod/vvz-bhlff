"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP source envelope and carrier generation implementation.

This module provides envelope and carrier generation functionality for
BVP-modulated sources in the 7D phase field theory.

Physical Meaning:
    BVP source envelope and carrier generation creates the envelope modulation
    and high-frequency carrier components for BVP-modulated sources.

Mathematical Foundation:
    Implements envelope and carrier generation:
    - Envelope: A(x) = A₀ * f(x)
    - Carrier: exp(iω₀t) with frequency ω₀

Example:
    >>> envelope_generator = BVPSourceEnvelope(domain, config)
    >>> envelope = envelope_generator.generate_envelope()
    >>> carrier = envelope_generator.generate_carrier()
"""

import numpy as np
from typing import Dict, Any

from ..domain import Domain


class BVPSourceEnvelope:
    """
    BVP source envelope and carrier generator.

    Physical Meaning:
        Generates envelope modulation and high-frequency carrier components
        for BVP-modulated sources in phase field evolution.

    Mathematical Foundation:
        Implements envelope and carrier generation:
        - Envelope: A(x) = A₀ * f(x)
        - Carrier: exp(iω₀t) with frequency ω₀

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Envelope generator configuration.
        carrier_frequency (float): High-frequency carrier frequency.
        envelope_amplitude (float): Envelope amplitude.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize BVP source envelope generator.

        Physical Meaning:
            Sets up the BVP source envelope generator with domain and
            configuration for generating envelope and carrier components.

        Args:
            domain (Domain): Computational domain for envelope generation.
            config (Dict[str, Any]): Envelope generator configuration.
        """
        self.domain = domain
        self.config = config
        self.carrier_frequency = config.get("carrier_frequency", 1.85e43)
        self.envelope_amplitude = config.get("envelope_amplitude", 1.0)

    def generate_envelope(self) -> np.ndarray:
        """
        Generate envelope modulation.

        Physical Meaning:
            Creates the envelope modulation A(x) that modulates the
            base source in the BVP framework.

        Mathematical Foundation:
            Envelope: A(x) = A₀ * f(x)
            where A₀ is the envelope amplitude and f(x) is the spatial
            distribution function.

        Returns:
            np.ndarray: Envelope modulation field.
        """
        # Get envelope parameters
        envelope_type = self.config.get("envelope_type", "constant")

        # Create coordinate arrays
        x = np.linspace(0, 1, self.domain.N)
        y = np.linspace(0, 1, self.domain.N)
        z = np.linspace(0, 1, self.domain.N)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Generate envelope based on type
        if envelope_type == "constant":
            envelope = self.envelope_amplitude * np.ones_like(X)

        elif envelope_type == "gaussian":
            # Gaussian envelope
            center = self.config.get("envelope_center", [0.5, 0.5, 0.5])
            width = self.config.get("envelope_width", 0.2)

            dx = X - center[0]
            dy = Y - center[1]
            dz = Z - center[2]
            r_squared = dx**2 + dy**2 + dz**2

            envelope = self.envelope_amplitude * np.exp(-r_squared / (2 * width**2))

        elif envelope_type == "sine":
            # Sine wave envelope
            kx = self.config.get("envelope_kx", 2 * np.pi)
            ky = self.config.get("envelope_ky", 2 * np.pi)
            kz = self.config.get("envelope_kz", 2 * np.pi)

            envelope = self.envelope_amplitude * (
                np.sin(kx * X) * np.sin(ky * Y) * np.sin(kz * Z)
            )

        elif envelope_type == "exponential":
            # Exponential envelope
            decay_rate = self.config.get("envelope_decay_rate", 1.0)
            center = self.config.get("envelope_center", [0.5, 0.5, 0.5])

            dx = X - center[0]
            dy = Y - center[1]
            dz = Z - center[2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)

            envelope = self.envelope_amplitude * np.exp(-decay_rate * r)

        else:
            # Default to constant envelope
            envelope = self.envelope_amplitude * np.ones_like(X)

        return envelope

    def generate_carrier(self, time: float = 0.0) -> np.ndarray:
        """
        Generate high-frequency carrier.

        Physical Meaning:
            Creates the high-frequency carrier component exp(iω₀t) that
            modulates the source in the BVP framework.

        Mathematical Foundation:
            Carrier: exp(iω₀t)
            where ω₀ is the carrier frequency and t is time.

        Args:
            time (float): Time for carrier generation.

        Returns:
            np.ndarray: Carrier component (complex).
        """
        # Generate carrier phase
        carrier_phase = self.carrier_frequency * time

        # Create carrier as complex exponential
        carrier = np.exp(1j * carrier_phase)

        # Broadcast to domain shape
        carrier_field = carrier * np.ones(
            (self.domain.N, self.domain.N, self.domain.N), dtype=complex
        )

        return carrier_field

    def generate_modulated_source(
        self, base_source: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """
        Generate BVP-modulated source.

        Physical Meaning:
            Creates the complete BVP-modulated source by combining the
            base source, envelope, and carrier components.

        Mathematical Foundation:
            BVP-modulated source: s(x) = s₀(x) * A(x) * exp(iω₀t)
            where s₀(x) is the base source, A(x) is the envelope, and
            exp(iω₀t) is the carrier.

        Args:
            base_source (np.ndarray): Base source field.
            time (float): Time for carrier generation.

        Returns:
            np.ndarray: BVP-modulated source field.
        """
        # Generate envelope and carrier
        envelope = self.generate_envelope()
        carrier = self.generate_carrier(time)

        # Combine components
        modulated_source = base_source * envelope * carrier

        return modulated_source

    def get_envelope_info(self) -> Dict[str, Any]:
        """
        Get envelope information.

        Physical Meaning:
            Returns information about the envelope generation including
            parameters and configuration.

        Returns:
            Dict[str, Any]: Envelope information.
        """
        return {
            "envelope_amplitude": self.envelope_amplitude,
            "envelope_type": self.config.get("envelope_type", "constant"),
            "carrier_frequency": self.carrier_frequency,
            "supported_envelope_types": ["constant", "gaussian", "sine", "exponential"],
        }

    def get_carrier_info(self) -> Dict[str, Any]:
        """
        Get carrier information.

        Physical Meaning:
            Returns information about the carrier generation including
            frequency and mathematical description.

        Returns:
            Dict[str, Any]: Carrier information.
        """
        return {
            "carrier_frequency": self.carrier_frequency,
            "carrier_formula": "exp(iω₀t)",
            "frequency_units": "rad/s",
            "wavelength": (
                2 * np.pi / self.carrier_frequency if self.carrier_frequency > 0 else 0
            ),
        }
