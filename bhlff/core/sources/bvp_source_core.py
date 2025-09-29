"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP-modulated source core implementation.

This module implements the core BVP-modulated source for the 7D phase field theory,
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
from .bvp_source_generators import BVPSourceGenerators
from .bvp_source_envelope import BVPSourceEnvelope


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
        source_generators (BVPSourceGenerators): Source generators.
        envelope_generator (BVPSourceEnvelope): Envelope generator.
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
        
        # Initialize component generators
        self.source_generators = BVPSourceGenerators(domain, config)
        self.envelope_generator = BVPSourceEnvelope(domain, config)

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
            np.ndarray: BVP-modulated source field.
        """
        # Generate base source
        base_source = self.source_generators.generate_base_source(self.base_source_type)
        
        # Generate BVP-modulated source
        time = self.config.get("time", 0.0)
        modulated_source = self.envelope_generator.generate_modulated_source(base_source, time)
        
        return modulated_source

    def generate_base_source(self) -> np.ndarray:
        """
        Generate base source without modulation.

        Physical Meaning:
            Generates the base source s₀(x) without BVP modulation,
            representing the unmodulated external excitation.

        Returns:
            np.ndarray: Base source field.
        """
        return self.source_generators.generate_base_source(self.base_source_type)

    def generate_envelope(self) -> np.ndarray:
        """
        Generate envelope modulation.

        Physical Meaning:
            Generates the envelope modulation A(x) that modulates the
            base source in the BVP framework.

        Returns:
            np.ndarray: Envelope modulation field.
        """
        return self.envelope_generator.generate_envelope()

    def generate_carrier(self, time: float = 0.0) -> np.ndarray:
        """
        Generate high-frequency carrier.

        Physical Meaning:
            Generates the high-frequency carrier component exp(iω₀t) that
            modulates the source in the BVP framework.

        Args:
            time (float): Time for carrier generation.

        Returns:
            np.ndarray: Carrier component.
        """
        return self.envelope_generator.generate_carrier(time)

    def get_source_type(self) -> str:
        """
        Get the source type.

        Physical Meaning:
            Returns the type of BVP-modulated source being generated.

        Returns:
            str: Source type.
        """
        return f"BVP-modulated {self.base_source_type}"

    def get_carrier_frequency(self) -> float:
        """
        Get the carrier frequency.

        Physical Meaning:
            Returns the high-frequency carrier frequency used in
            BVP modulation.

        Returns:
            float: Carrier frequency.
        """
        return self.carrier_frequency

    def get_envelope_amplitude(self) -> float:
        """
        Get the envelope amplitude.

        Physical Meaning:
            Returns the envelope amplitude used in BVP modulation.

        Returns:
            float: Envelope amplitude.
        """
        return self.envelope_amplitude

    def get_base_source_type(self) -> str:
        """
        Get the base source type.

        Physical Meaning:
            Returns the type of base source being used.

        Returns:
            str: Base source type.
        """
        return self.base_source_type

    def get_supported_source_types(self) -> list:
        """
        Get supported source types.

        Physical Meaning:
            Returns the list of supported base source types for
            BVP modulation.

        Returns:
            list: Supported source types.
        """
        return self.source_generators.get_supported_source_types()

    def get_source_info(self) -> Dict[str, Any]:
        """
        Get source information.

        Physical Meaning:
            Returns comprehensive information about the BVP-modulated
            source including parameters and configuration.

        Returns:
            Dict[str, Any]: Source information.
        """
        return {
            "source_type": self.get_source_type(),
            "base_source_type": self.base_source_type,
            "carrier_frequency": self.carrier_frequency,
            "envelope_amplitude": self.envelope_amplitude,
            "envelope_info": self.envelope_generator.get_envelope_info(),
            "carrier_info": self.envelope_generator.get_carrier_info(),
            "supported_source_types": self.get_supported_source_types()
        }

    def __repr__(self) -> str:
        """String representation of the BVP-modulated source."""
        return (
            f"BVPSource(domain={self.domain}, "
            f"base_type={self.base_source_type}, "
            f"carrier_frequency={self.carrier_frequency}, "
            f"envelope_amplitude={self.envelope_amplitude})"
        )
