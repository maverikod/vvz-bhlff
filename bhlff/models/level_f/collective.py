"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Collective excitations implementation for Level F models.

This module provides a facade for collective excitations functionality
for Level F models in 7D phase field theory, ensuring proper functionality
of all collective excitation analysis components.

Theoretical Background:
    Collective excitations in multi-particle systems are described by
    linear response theory. The system response to external fields
    reveals collective modes and their dispersion relations.
    
    The response function is given by:
    R(ω) = χ(ω) F(ω)
    where χ(ω) is the susceptibility and F(ω) is the external field.

Example:
    >>> excitations = CollectiveExcitations(system, excitation_params)
    >>> response = excitations.excite_system(external_field)
    >>> analysis = excitations.analyze_response(response)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..base.abstract_model import AbstractModel
from .collective.excitation_analysis import ExcitationAnalyzer
from .collective.dispersion_analysis import DispersionAnalyzer


class CollectiveExcitations(AbstractModel):
    """
    Collective excitations in multi-particle systems.

    Physical Meaning:
        Studies the response of multi-particle systems to
        external fields, identifying collective modes and
        their dispersion relations.

    Mathematical Foundation:
        Implements linear response theory for collective
        excitations in the effective potential framework.

    Attributes:
        system (MultiParticleSystem): Multi-particle system
        excitation_params (Dict[str, Any]): Excitation parameters
        frequency_range (Tuple[float, float]): Frequency range for analysis
        amplitude (float): Excitation amplitude
        excitation_type (str): Type of excitation
    """

    def __init__(
        self, system: "MultiParticleSystem", excitation_params: Dict[str, Any]
    ):
        """
        Initialize collective excitations model.

        Physical Meaning:
            Sets up the model for studying collective excitations
            in the multi-particle system.

        Args:
            system (MultiParticleSystem): Multi-particle system
            excitation_params (Dict): Parameters including:
                - frequency_range: [ω_min, ω_max]
                - amplitude: A (excitation amplitude)
                - type: "harmonic", "impulse", "sweep"
        """
        super().__init__(system.domain)
        self.system = system
        self.excitation_params = excitation_params

        # Extract parameters
        self.frequency_range = excitation_params.get("frequency_range", [0.1, 10.0])
        self.amplitude = excitation_params.get("amplitude", 0.1)
        self.excitation_type = excitation_params.get("type", "harmonic")
        self.duration = excitation_params.get("duration", 100.0)

        # Initialize analysis components
        self.excitation_analyzer = ExcitationAnalyzer(system, excitation_params)
        self.dispersion_analyzer = DispersionAnalyzer(system)

        # Setup analysis parameters
        self._setup_analysis_parameters()

    def excite_system(self, external_field: np.ndarray) -> np.ndarray:
        """
        Excite the system with external field.

        Physical Meaning:
            Applies external field to the system and
            computes the response.

        Args:
            external_field (np.ndarray): External field F(x,t)

        Returns:
            np.ndarray: System response R(x,t)
        """
        return self.excitation_analyzer.excite_system(external_field)

    def analyze_response(self, response: np.ndarray) -> Dict[str, Any]:
        """
        Analyze system response to excitation.

        Physical Meaning:
            Extracts collective mode frequencies and
            amplitudes from the response.

        Args:
            response (np.ndarray): System response R(x,t)

        Returns:
            Dict containing:
                - frequencies: ω_n (collective frequencies)
                - amplitudes: A_n (mode amplitudes)
                - damping: γ_n (damping rates)
                - participation: p_n (particle participation)
        """
        return self.excitation_analyzer.analyze_response(response)

    def compute_dispersion_relations(self) -> Dict[str, Any]:
        """
        Compute dispersion relations for collective modes.

        Physical Meaning:
            Calculates ω(k) relations for collective
            excitations in the system.

        Returns:
            Dict containing:
                - wave_vectors: k (wave vector magnitudes)
                - frequencies: ω(k) (dispersion relation)
                - group_velocities: v_g = dω/dk
                - phase_velocities: v_φ = ω/k
        """
        return self.dispersion_analyzer.compute_dispersion_relations()

    def compute_susceptibility(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute susceptibility function χ(ω).

        Physical Meaning:
            Calculates the linear response susceptibility
            for collective excitations.

        Args:
            frequencies (np.ndarray): Frequency array

        Returns:
            np.ndarray: Susceptibility χ(ω)
        """
        return self.dispersion_analyzer.compute_susceptibility(frequencies)

    def _setup_analysis_parameters(self) -> None:
        """
        Setup analysis parameters for collective excitations.

        Physical Meaning:
            Initializes parameters needed for analysis
            of collective excitations.
        """
        self.dt = 0.01  # Time step
        self.k_max = 10.0  # Maximum wave vector
        self.n_k_points = 100  # Number of k points
        self.peak_threshold = 0.1  # Peak detection threshold
        self.damping_threshold = 0.01  # Damping analysis threshold

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data for this model.

        Physical Meaning:
            Performs comprehensive analysis of collective excitations,
            including response analysis and dispersion relations.

        Args:
            data (Any): Input data to analyze (external field)

        Returns:
            Dict: Analysis results including response and dispersion
        """
        # Create external field if not provided
        if data is None:
            external_field = np.random.randn(*self.domain.shape) * 0.1
        else:
            external_field = data

        # Excite system
        response = self.excite_system(external_field)

        # Analyze response
        response_analysis = self.analyze_response(response)

        # Compute dispersion relations
        dispersion = self.compute_dispersion_relations()

        return {
            "response": response,
            "response_analysis": response_analysis,
            "dispersion": dispersion,
        }