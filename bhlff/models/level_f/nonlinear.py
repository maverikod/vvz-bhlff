"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Nonlinear effects implementation for Level F models.

This module provides a facade for nonlinear effects functionality
for Level F models in 7D phase field theory, ensuring proper functionality
of all nonlinear analysis components.

Theoretical Background:
    Nonlinear effects in multi-particle systems arise from
    higher-order terms in the effective potential. These include
    cubic, quartic, and sine-Gordon type nonlinearities that
    lead to solitonic solutions and nonlinear collective modes.
    
    The nonlinear potential is given by:
    U_nonlinear = g * |ψ|^n + λ * sin(φ) + ...
    where g is the nonlinear strength and n is the order.

Example:
    >>> nonlinear = NonlinearEffects(system, nonlinear_params)
    >>> nonlinear.add_nonlinear_interactions(nonlinear_params)
    >>> modes = nonlinear.find_nonlinear_modes()
    >>> solitons = nonlinear.find_soliton_solutions()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..base.abstract_model import AbstractModel
from .basic_effects import BasicNonlinearEffects
from .soliton_analysis import SolitonAnalyzer
from .mode_analysis import NonlinearModeAnalyzer


class NonlinearEffects(AbstractModel):
    """
    Nonlinear effects in collective systems.

    Physical Meaning:
        Studies nonlinear interactions in multi-particle
        systems, including solitonic solutions and
        nonlinear modes.

    Mathematical Foundation:
        Implements nonlinear field equations with
        collective interaction terms.

    Attributes:
        system (MultiParticleSystem): Multi-particle system
        nonlinear_params (Dict[str, Any]): Nonlinear parameters
        nonlinear_strength (float): Nonlinear coupling strength
        nonlinear_order (int): Order of nonlinearity
        nonlinear_type (str): Type of nonlinearity
    """

    def __init__(self, system, nonlinear_params: Dict[str, Any]):
        """
        Initialize nonlinear effects.

        Physical Meaning:
            Sets up the nonlinear effects system with
            nonlinear parameters and analysis components.

        Args:
            system: Multi-particle system
            nonlinear_params (Dict[str, Any]): Nonlinear parameters
        """
        super().__init__()
        self.system = system
        self.nonlinear_params = nonlinear_params
        
        # Nonlinear parameters
        self.nonlinear_strength = nonlinear_params.get("strength", 1.0)
        self.nonlinear_order = nonlinear_params.get("order", 3)
        self.nonlinear_type = nonlinear_params.get("type", "cubic")
        
        # Initialize analysis components
        self.basic_effects = BasicNonlinearEffects(system, nonlinear_params)
        self.soliton_analyzer = SolitonAnalyzer(system, nonlinear_params)
        self.mode_analyzer = NonlinearModeAnalyzer(system, nonlinear_params)

    def add_nonlinear_interactions(self, nonlinear_params: Dict[str, Any]) -> None:
        """
        Add nonlinear interactions to the system.

        Physical Meaning:
            Adds nonlinear interaction terms to the system
            potential and equations of motion.

        Args:
            nonlinear_params (Dict): Nonlinear interaction parameters
        """
        # Update parameters
        self.nonlinear_strength = nonlinear_params.get(
            "strength", self.nonlinear_strength
        )
        self.nonlinear_order = nonlinear_params.get("order", self.nonlinear_order)
        self.nonlinear_type = nonlinear_params.get("type", self.nonlinear_type)

        # Add nonlinear terms to system
        self._add_nonlinear_potential()
        self._add_nonlinear_dynamics()

    def _add_nonlinear_potential(self) -> None:
        """
        Add nonlinear potential to system.

        Physical Meaning:
            Adds nonlinear potential terms to the system
            potential energy.
        """
        # Add nonlinear potential to system
        if hasattr(self.system, 'add_potential'):
            self.system.add_potential(self._nonlinear_potential)

    def _add_nonlinear_dynamics(self) -> None:
        """
        Add nonlinear dynamics to system.

        Physical Meaning:
            Adds nonlinear dynamics terms to the system
            equations of motion.
        """
        # Add nonlinear force to system
        if hasattr(self.system, 'add_force'):
            self.system.add_force(self._nonlinear_force)

    def _nonlinear_potential(self, psi: np.ndarray) -> np.ndarray:
        """
        Nonlinear potential function.

        Physical Meaning:
            Computes the nonlinear potential energy
            for the given field configuration.

        Args:
            psi (np.ndarray): Field configuration.

        Returns:
            np.ndarray: Nonlinear potential energy.
        """
        if self.nonlinear_type == "cubic":
            return self.nonlinear_strength * np.abs(psi) ** 3
        elif self.nonlinear_type == "quartic":
            return self.nonlinear_strength * np.abs(psi) ** 4
        elif self.nonlinear_type == "sine_gordon":
            return self.nonlinear_strength * (1 - np.cos(psi))
        else:
            raise ValueError(f"Unknown nonlinear type: {self.nonlinear_type}")

    def _nonlinear_force(self, psi: np.ndarray) -> np.ndarray:
        """
        Nonlinear force function.

        Physical Meaning:
            Computes the nonlinear force acting on the field
            due to nonlinear interactions.

        Args:
            psi (np.ndarray): Field configuration.

        Returns:
            np.ndarray: Nonlinear force.
        """
        if self.nonlinear_type == "cubic":
            return -3 * self.nonlinear_strength * np.abs(psi) * np.sign(psi)
        elif self.nonlinear_type == "quartic":
            return -4 * self.nonlinear_strength * np.abs(psi) ** 2 * np.sign(psi)
        elif self.nonlinear_type == "sine_gordon":
            return -self.nonlinear_strength * np.sin(psi)
        else:
            raise ValueError(f"Unknown nonlinear type: {self.nonlinear_type}")

    def find_nonlinear_modes(self) -> Dict[str, Any]:
        """
        Find nonlinear modes in the system.

        Physical Meaning:
            Identifies nonlinear collective modes that
            arise from nonlinear interactions.

        Returns:
            Dict containing:
                - frequencies: ω_n (nonlinear mode frequencies)
                - amplitudes: A_n (mode amplitudes)
                - stability: stability analysis
                - bifurcations: bifurcation points
        """
        # Use mode analyzer to find nonlinear modes
        return self.mode_analyzer.find_nonlinear_modes()

    def find_soliton_solutions(self) -> Dict[str, Any]:
        """
        Find soliton solutions.

        Physical Meaning:
            Finds soliton solutions in the nonlinear system
            using optimization methods.

        Returns:
            Dict[str, Any]: Soliton solutions including:
                - profiles: soliton profiles
                - energies: soliton energies
                - stability: stability analysis
                - interactions: soliton interactions
        """
        # Use soliton analyzer to find soliton solutions
        return self.soliton_analyzer.find_soliton_solutions()

    def analyze_nonlinear_strength(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Analyze nonlinear strength.

        Physical Meaning:
            Analyzes the strength of nonlinear effects
            in the field configuration.

        Args:
            field (np.ndarray): Field configuration.

        Returns:
            Dict[str, Any]: Nonlinear strength analysis.
        """
        # Use basic effects analyzer
        return self.basic_effects.analyze_nonlinear_strength(field)

    def compute_nonlinear_energy(self, field: np.ndarray) -> float:
        """
        Compute nonlinear energy.

        Physical Meaning:
            Computes the nonlinear energy contribution
            to the total system energy.

        Args:
            field (np.ndarray): Field configuration.

        Returns:
            float: Nonlinear energy.
        """
        # Use basic effects analyzer
        return self.basic_effects.compute_nonlinear_energy(field)

    def compute_nonlinear_force(self, field: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear force.

        Physical Meaning:
            Computes the nonlinear force acting on the field
            due to nonlinear interactions.

        Args:
            field (np.ndarray): Field configuration.

        Returns:
            np.ndarray: Nonlinear force.
        """
        # Use basic effects analyzer
        return self.basic_effects.compute_nonlinear_force(field)

    def analyze_nonlinear_stability(self) -> Dict[str, Any]:
        """
        Analyze nonlinear stability.

        Physical Meaning:
            Analyzes the stability of nonlinear modes
            in the system.

        Returns:
            Dict[str, Any]: Stability analysis.
        """
        # Use mode analyzer
        return self.mode_analyzer._analyze_nonlinear_stability()

    def find_bifurcation_points(self) -> List[Dict[str, Any]]:
        """
        Find bifurcation points.

        Physical Meaning:
            Identifies bifurcation points in the nonlinear
            system where qualitative changes occur.

        Returns:
            List[Dict[str, Any]]: Bifurcation points.
        """
        # Use mode analyzer
        return self.mode_analyzer._find_bifurcation_points()

    def compute_nonlinear_corrections(self, linear_modes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute nonlinear corrections to linear modes.

        Physical Meaning:
            Computes nonlinear corrections to linear
            collective modes.

        Args:
            linear_modes (Dict[str, Any]): Linear mode results.

        Returns:
            Dict[str, Any]: Nonlinear corrections.
        """
        # Use mode analyzer
        return self.mode_analyzer._compute_nonlinear_corrections(linear_modes)

    def analyze_soliton_stability(self, soliton_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze soliton stability.

        Physical Meaning:
            Analyzes the stability of soliton solutions.

        Args:
            soliton_profiles (List[Dict[str, Any]]): Soliton profiles.

        Returns:
            Dict[str, Any]: Stability analysis.
        """
        # Use soliton analyzer
        return self.soliton_analyzer._analyze_soliton_stability(soliton_profiles)

    def analyze_soliton_interactions(self, soliton_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze soliton interactions.

        Physical Meaning:
            Analyzes interactions between soliton solutions.

        Args:
            soliton_profiles (List[Dict[str, Any]]): Soliton profiles.

        Returns:
            Dict[str, Any]: Interaction analysis.
        """
        # Use soliton analyzer
        return self.soliton_analyzer._analyze_soliton_interactions(soliton_profiles)

    def compute_soliton_energies(self, soliton_profiles: List[Dict[str, Any]]) -> List[float]:
        """
        Compute soliton energies.

        Physical Meaning:
            Computes the energies of soliton solutions.

        Args:
            soliton_profiles (List[Dict[str, Any]]): Soliton profiles.

        Returns:
            List[float]: Soliton energies.
        """
        # Use soliton analyzer
        return self.soliton_analyzer._compute_soliton_energies(soliton_profiles)

    def validate_nonlinear_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate nonlinear analysis results.

        Physical Meaning:
            Validates the nonlinear analysis results to ensure
            they meet quality and consistency criteria.

        Args:
            results (Dict[str, Any]): Analysis results to validate.

        Returns:
            Dict[str, Any]: Validation results.
        """
        # Validate basic effects
        basic_validation = self._validate_basic_effects(results.get("basic_effects", {}))

        # Validate soliton analysis
        soliton_validation = self._validate_soliton_analysis(results.get("soliton_analysis", {}))

        # Validate mode analysis
        mode_validation = self._validate_mode_analysis(results.get("mode_analysis", {}))

        # Calculate overall validation
        overall_validation = self._calculate_overall_validation(
            basic_validation, soliton_validation, mode_validation
        )

        return {
            "basic_validation": basic_validation,
            "soliton_validation": soliton_validation,
            "mode_validation": mode_validation,
            "overall_validation": overall_validation,
            "validation_complete": True,
        }

    def _validate_basic_effects(self, basic_effects: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate basic effects results.

        Physical Meaning:
            Validates the basic effects analysis results.

        Args:
            basic_effects (Dict[str, Any]): Basic effects results.

        Returns:
            Dict[str, Any]: Basic effects validation.
        """
        # Check if basic effects are present
        is_present = len(basic_effects) > 0

        # Check quality metrics
        quality_metrics = basic_effects.get("quality_metrics", {})
        quality_score = quality_metrics.get("overall_quality", 0.0)

        return {
            "is_present": is_present,
            "quality_score": quality_score,
            "validation_passed": is_present and quality_score > 0.7,
        }

    def _validate_soliton_analysis(self, soliton_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate soliton analysis results.

        Physical Meaning:
            Validates the soliton analysis results.

        Args:
            soliton_analysis (Dict[str, Any]): Soliton analysis results.

        Returns:
            Dict[str, Any]: Soliton analysis validation.
        """
        # Check if soliton analysis is present
        is_present = len(soliton_analysis) > 0

        # Check soliton profiles
        profiles = soliton_analysis.get("profiles", [])
        num_profiles = len(profiles)

        # Check stability
        stability = soliton_analysis.get("stability", {})
        stability_score = stability.get("overall_stability", 0.0)

        return {
            "is_present": is_present,
            "num_profiles": num_profiles,
            "stability_score": stability_score,
            "validation_passed": is_present and num_profiles > 0 and stability_score > 0.5,
        }

    def _validate_mode_analysis(self, mode_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mode analysis results.

        Physical Meaning:
            Validates the mode analysis results.

        Args:
            mode_analysis (Dict[str, Any]): Mode analysis results.

        Returns:
            Dict[str, Any]: Mode analysis validation.
        """
        # Check if mode analysis is present
        is_present = len(mode_analysis) > 0

        # Check frequencies
        frequencies = mode_analysis.get("nonlinear_frequencies", [])
        num_frequencies = len(frequencies)

        # Check stability
        stability = mode_analysis.get("stability", {})
        stability_score = stability.get("overall_stability", 0.0)

        return {
            "is_present": is_present,
            "num_frequencies": num_frequencies,
            "stability_score": stability_score,
            "validation_passed": is_present and num_frequencies > 0 and stability_score > 0.5,
        }

    def _calculate_overall_validation(
        self, basic_validation: Dict[str, Any], soliton_validation: Dict[str, Any], mode_validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall validation.

        Physical Meaning:
            Calculates the overall validation of all analysis components.

        Args:
            basic_validation (Dict[str, Any]): Basic effects validation.
            soliton_validation (Dict[str, Any]): Soliton analysis validation.
            mode_validation (Dict[str, Any]): Mode analysis validation.

        Returns:
            Dict[str, Any]: Overall validation results.
        """
        # Calculate overall quality
        overall_quality = np.mean([
            basic_validation["quality_score"],
            soliton_validation["stability_score"],
            mode_validation["stability_score"],
        ])

        # Calculate overall validation status
        overall_passed = all([
            basic_validation["validation_passed"],
            soliton_validation["validation_passed"],
            mode_validation["validation_passed"],
        ])

        return {
            "overall_quality": overall_quality,
            "overall_passed": overall_passed,
            "validation_summary": {
                "basic_validation": basic_validation["validation_passed"],
                "soliton_validation": soliton_validation["validation_passed"],
                "mode_validation": mode_validation["validation_passed"],
            },
        }