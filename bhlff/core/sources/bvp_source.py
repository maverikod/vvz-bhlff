"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP-modulated source implementation.

This module implements BVP-modulated sources for the 7D phase field theory,
representing sources that are modulated by the Base High-Frequency Field.

Physical Meaning:
    BVP-modulated sources represent external excitations that are modulated
    by the high-frequency carrier field, creating envelope modulations
    in the source term.

Mathematical Foundation:
    BVP-modulated sources have the form:
    s(x) = s₀(x) * A(x) * exp(iω₀t)
    where s₀(x) is the base source, A(x) is the envelope, and ω₀ is the
    carrier frequency.

Example:
    >>> source = BVPSource(domain, config)
    >>> source_field = source.generate()
"""

import numpy as np
from typing import Dict, Any

from ..domain import Domain
from .source import Source


class BVPSource(Source):
    """
    BVP-modulated source for 7D phase field theory.

    Physical Meaning:
        Implements sources that are modulated by the Base High-Frequency
        Field, creating envelope modulations in the source term that
        drive phase field evolution.

    Mathematical Foundation:
        BVP-modulated sources have the form:
        s(x) = s₀(x) * A(x) * exp(iω₀t)
        where s₀(x) is the base source, A(x) is the envelope, and ω₀ is
        the carrier frequency.

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): BVP source configuration.
        carrier_frequency (float): High-frequency carrier frequency.
        envelope_amplitude (float): Envelope amplitude.
        base_source_type (str): Type of base source.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize BVP-modulated source.

        Physical Meaning:
            Sets up the BVP-modulated source with carrier frequency and
            envelope parameters for generating modulated source terms.

        Args:
            domain (Domain): Computational domain for the source.
            config (Dict[str, Any]): BVP source configuration including:
                - carrier_frequency: High-frequency carrier frequency
                - envelope_amplitude: Envelope amplitude
                - base_source_type: Type of base source
        """
        super().__init__(domain, config)
        self._setup_bvp_parameters()

    def _setup_bvp_parameters(self) -> None:
        """
        Setup BVP source parameters.

        Physical Meaning:
            Initializes the BVP source parameters from configuration
            including carrier frequency and envelope properties.
        """
        self.carrier_frequency = self.config.get("carrier_frequency", 1.85e43)
        self.envelope_amplitude = self.config.get("envelope_amplitude", 1.0)
        self.base_source_type = self.config.get("base_source_type", "gaussian")

    def generate(self) -> np.ndarray:
        """
        Generate BVP-modulated source field.

        Physical Meaning:
            Generates the BVP-modulated source field s(x) that represents
            external excitations modulated by the high-frequency carrier.

        Mathematical Foundation:
            Creates the BVP-modulated source:
            s(x) = s₀(x) * A(x) * exp(iω₀t)
            where s₀(x) is the base source, A(x) is the envelope, and
            ω₀ is the carrier frequency.

        Returns:
            np.ndarray: BVP-modulated source field s(x).
        """
        # Generate base source
        base_source = self._generate_base_source()

        # Generate envelope
        envelope = self._generate_envelope()

        # Generate carrier modulation
        carrier = self._generate_carrier()

        # Combine to create BVP-modulated source
        bvp_source = base_source * envelope * carrier

        return bvp_source

    def _generate_base_source(self) -> np.ndarray:
        """
        Generate base source term.

        Physical Meaning:
            Generates the base source term s₀(x) that represents the
            fundamental spatial distribution of the source.

        Mathematical Foundation:
            Creates the base source s₀(x) based on the specified type
            (gaussian, point, distributed, etc.).

        Returns:
            np.ndarray: Base source term s₀(x).
        """
        if self.base_source_type == "gaussian":
            return self._generate_gaussian_source()
        elif self.base_source_type == "point":
            return self._generate_point_source()
        elif self.base_source_type == "distributed":
            return self._generate_distributed_source()
        else:
            raise ValueError(f"Unsupported base source type: {self.base_source_type}")

    def _generate_gaussian_source(self) -> np.ndarray:
        """
        Generate Gaussian base source.

        Physical Meaning:
            Creates a Gaussian-distributed base source with specified
            width and amplitude.

        Mathematical Foundation:
            Gaussian source: s₀(x) = A * exp(-|x-x₀|²/(2σ²))
            where A is amplitude, x₀ is center, and σ is width.

        Returns:
            np.ndarray: Gaussian base source.
        """
        amplitude = self.config.get("amplitude", 1.0)
        width = self.config.get("width", 0.1)
        center = self.config.get("center", 0.0)

        # Create coordinate arrays
        if self.domain.dimensions == 1:
            x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            r = np.abs(x - center)
        elif self.domain.dimensions == 2:
            x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            X, Y = np.meshgrid(x, y, indexing="ij")
            r = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
        else:  # 3D
            x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            z = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            r = np.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)

        # Gaussian source
        gaussian_source = amplitude * np.exp(-(r**2) / (2 * width**2))

        return gaussian_source

    def _generate_point_source(self) -> np.ndarray:
        """
        Generate point source.

        Physical Meaning:
            Creates a point source at a specified location with
            specified amplitude.

        Mathematical Foundation:
            Point source: s₀(x) = A * δ(x - x₀)
            where A is amplitude and δ is the Dirac delta function.

        Returns:
            np.ndarray: Point source.
        """
        amplitude = self.config.get("amplitude", 1.0)
        center = self.config.get("center", 0.0)

        # Create point source
        point_source = np.zeros(self.domain.shape)

        if self.domain.dimensions == 1:
            center_idx = int(
                (center + self.domain.L / 2) * self.domain.N / self.domain.L
            )
            if 0 <= center_idx < self.domain.N:
                point_source[center_idx] = amplitude
        elif self.domain.dimensions == 2:
            center_idx = int(
                (center + self.domain.L / 2) * self.domain.N / self.domain.L
            )
            if 0 <= center_idx < self.domain.N:
                point_source[center_idx, center_idx] = amplitude
        else:  # 3D
            center_idx = int(
                (center + self.domain.L / 2) * self.domain.N / self.domain.L
            )
            if 0 <= center_idx < self.domain.N:
                point_source[center_idx, center_idx, center_idx] = amplitude

        return point_source

    def _generate_distributed_source(self) -> np.ndarray:
        """
        Generate distributed source.

        Physical Meaning:
            Creates a distributed source with random spatial distribution
            and specified statistical properties.

        Mathematical Foundation:
            Distributed source: s₀(x) = A * f(x) where f(x) is a random
            field with specified statistical properties.

        Returns:
            np.ndarray: Distributed source.
        """
        amplitude = self.config.get("amplitude", 1.0)
        # correlation_length = self.config.get("correlation_length", 0.1)
        seed = self.config.get("seed", 42)

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Generate random field
        random_field = np.random.randn(*self.domain.shape)

        # Apply correlation length (simplified)
        # In practice, would use proper random field generation
        distributed_source = amplitude * random_field

        return distributed_source

    def _generate_envelope(self) -> np.ndarray:
        """
        Generate envelope modulation.

        Physical Meaning:
            Generates the envelope A(x) that modulates the base source,
            representing the spatial variation of the modulation.

        Mathematical Foundation:
            Envelope: A(x) = A₀ * f(x) where A₀ is amplitude and f(x)
            is the spatial envelope function.

        Returns:
            np.ndarray: Envelope modulation A(x).
        """
        envelope_type = self.config.get("envelope_type", "constant")
        envelope_width = self.config.get("envelope_width", 0.2)

        if envelope_type == "constant":
            envelope = np.full(self.domain.shape, self.envelope_amplitude)
        elif envelope_type == "gaussian":
            # Create coordinate arrays
            if self.domain.dimensions == 1:
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                r = np.abs(x)
            elif self.domain.dimensions == 2:
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                X, Y = np.meshgrid(x, y, indexing="ij")
                r = np.sqrt(X**2 + Y**2)
            else:  # 3D
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                z = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
                r = np.sqrt(X**2 + Y**2 + Z**2)

            envelope = self.envelope_amplitude * np.exp(
                -(r**2) / (2 * envelope_width**2)
            )
        else:
            envelope = np.full(self.domain.shape, self.envelope_amplitude)

        return envelope

    def _generate_carrier(self) -> np.ndarray:
        """
        Generate carrier modulation.

        Physical Meaning:
            Generates the high-frequency carrier modulation exp(iω₀t)
            that represents the temporal oscillation of the source.

        Mathematical Foundation:
            Carrier: exp(iω₀t) where ω₀ is the carrier frequency and
            t is time (here represented as spatial phase).

        Returns:
            np.ndarray: Carrier modulation.
        """
        # For simplicity, use spatial phase instead of temporal
        # In practice, would include proper temporal evolution
        phase = self.config.get("phase", 0.0)

        # Create carrier with spatial phase variation
        if self.domain.dimensions == 1:
            x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            carrier = np.exp(1j * (self.carrier_frequency * x + phase))
        elif self.domain.dimensions == 2:
            x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            X, Y = np.meshgrid(x, y, indexing="ij")
            carrier = np.exp(1j * (self.carrier_frequency * (X + Y) + phase))
        else:  # 3D
            x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            z = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            carrier = np.exp(1j * (self.carrier_frequency * (X + Y + Z) + phase))

        return carrier

    def get_source_type(self) -> str:
        """
        Get the source type.

        Physical Meaning:
            Returns the type of source being used.

        Returns:
            str: Source type ("bvp_modulated").
        """
        return "bvp_modulated"

    def get_carrier_frequency(self) -> float:
        """
        Get the carrier frequency.

        Physical Meaning:
            Returns the high-frequency carrier frequency ω₀.

        Returns:
            float: Carrier frequency.
        """
        return float(self.carrier_frequency)

    def get_envelope_amplitude(self) -> float:
        """
        Get the envelope amplitude.

        Physical Meaning:
            Returns the amplitude of the envelope modulation.

        Returns:
            float: Envelope amplitude.
        """
        return float(self.envelope_amplitude)

    def __repr__(self) -> str:
        """String representation of the BVP source."""
        return (
            f"BVPSource(domain={self.domain}, "
            f"carrier_freq={self.carrier_frequency}, "
            f"envelope_amp={self.envelope_amplitude})"
        )
