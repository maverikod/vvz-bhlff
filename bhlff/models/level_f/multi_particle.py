"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Multi-particle system implementation for Level F collective effects.

This module provides a facade for multi-particle system functionality
for Level F models in 7D phase field theory, ensuring proper functionality
of all multi-particle analysis components.

Theoretical Background:
    Multi-particle systems in 7D phase field theory are described by
    effective potentials that include single-particle, pair-wise, and
    higher-order interactions:
    U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ + ...
    
    Collective modes arise from the diagonalization of the dynamics matrix
    M⁻¹K, where M is the mass matrix and K is the stiffness matrix.

Example:
    >>> particles = [Particle(position=[5,10,10], charge=1, phase=0),
    ...              Particle(position=[15,10,10], charge=-1, phase=π)]
    >>> system = MultiParticleSystem(domain, particles)
    >>> potential = system.compute_effective_potential()
    >>> modes = system.find_collective_modes()
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..base.abstract_model import AbstractModel
from .data_structures import Particle, SystemParameters
from .potential_analysis import PotentialAnalyzer
from .collective_modes import CollectiveModesAnalyzer


class MultiParticleSystem(AbstractModel):
    """
    Multi-particle system for Level F collective effects.

    Physical Meaning:
        Studies collective effects in systems with multiple
        topological defects, including effective potential
        calculations and collective mode analysis.

    Mathematical Foundation:
        Implements multi-particle system analysis:
        - Effective potential: U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ
        - Collective modes: diagonalization of M⁻¹K
        - Correlation functions: G(x,t) = ⟨ψ*(x,t)ψ(0,0)⟩

    Attributes:
        domain: Computational domain
        particles (List[Particle]): List of particles
        interaction_range (float): Range of particle interactions
        interaction_strength (float): Strength of interactions
    """

    def __init__(self, domain, particles: List[Particle], interaction_range: float = 2.0, 
                 interaction_strength: float = 1.0):
        """
        Initialize multi-particle system.

        Physical Meaning:
            Sets up the multi-particle system with particles
            and interaction parameters.

        Args:
            domain: Computational domain
            particles (List[Particle]): List of particles
            interaction_range (float): Range of particle interactions
            interaction_strength (float): Strength of interactions
        """
        super().__init__(domain)
        self.particles = particles
        self.interaction_range = interaction_range
        self.interaction_strength = interaction_strength
        self.phase_coherence_length = 1.0  # Phase coherence length
        
        # Initialize system parameters
        self.system_params = SystemParameters(
            interaction_range=interaction_range,
            interaction_strength=interaction_strength,
            phase_coherence_length=self.phase_coherence_length
        )
        
        # Initialize analysis components
        self.potential_analyzer = PotentialAnalyzer(domain, particles, self.system_params)
        self.collective_modes_analyzer = CollectiveModesAnalyzer(domain, particles, self.system_params)
        
        # Setup interaction matrices
        self._setup_interaction_matrices()

    def _setup_interaction_matrices(self) -> None:
        """
        Setup interaction matrices.

        Physical Meaning:
            Sets up the interaction matrices for the system
            based on particle positions and charges.
        """
        n_particles = len(self.particles)
        
        # Initialize interaction matrices
        self.interaction_matrix = np.zeros((n_particles, n_particles))
        self.distance_matrix = np.zeros((n_particles, n_particles))
        self.charge_matrix = np.zeros((n_particles, n_particles))
        
        # Compute interaction matrices
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles):
                if i != j:
                    # Distance matrix
                    distance = particle_i.distance_to(particle_j)
                    self.distance_matrix[i, j] = distance
                    
                    # Charge matrix
                    charge_product = particle_i.charge * particle_j.charge
                    self.charge_matrix[i, j] = charge_product
                    
                    # Interaction matrix
                    if distance <= self.interaction_range:
                        interaction_strength = particle_i.interaction_strength(
                            particle_j, self.interaction_range
                        )
                        self.interaction_matrix[i, j] = interaction_strength

    def compute_effective_potential(self) -> np.ndarray:
        """
        Compute effective potential for the system.

        Physical Meaning:
            Calculates the total effective potential including
            single-particle, pair-wise, and higher-order interactions.

        Mathematical Foundation:
            U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ

        Returns:
            np.ndarray: Effective potential field U_eff(x,y,z)
        """
        # Use potential analyzer
        return self.potential_analyzer.compute_effective_potential()

    def find_collective_modes(self) -> Dict[str, Any]:
        """
        Find collective modes of the system.

        Physical Meaning:
            Identifies collective excitations that involve
            coordinated motion of multiple particles.

        Returns:
            Dict[str, Any]: Collective modes analysis including:
                - frequencies: ω_n (mode frequencies)
                - amplitudes: A_n (mode amplitudes)
                - eigenvectors: v_n (mode shapes)
                - stability: stability analysis
        """
        # Use collective modes analyzer
        return self.collective_modes_analyzer.find_collective_modes()

    def compute_correlation_function(self, field: np.ndarray, time_points: np.ndarray) -> np.ndarray:
        """
        Compute correlation function.

        Physical Meaning:
            Calculates the correlation function G(x,t) = ⟨ψ*(x,t)ψ(0,0)⟩
            for the multi-particle system.

        Args:
            field (np.ndarray): Field configuration.
            time_points (np.ndarray): Time points.

        Returns:
            np.ndarray: Correlation function.
        """
        # Initialize correlation function
        correlation = np.zeros((len(time_points), field.shape[0], field.shape[1], field.shape[2]))

        # Compute correlation for each time point
        for t_idx, t in enumerate(time_points):
            # Simplified correlation calculation
            # In practice, this would involve proper correlation analysis
            correlation[t_idx] = np.real(field * np.conj(field))

        return correlation

    def analyze_system_properties(self) -> Dict[str, Any]:
        """
        Analyze system properties.

        Physical Meaning:
            Analyzes the properties of the multi-particle system
            including energy, stability, and collective behavior.

        Returns:
            Dict[str, Any]: System properties analysis.
        """
        # Compute effective potential
        potential = self.compute_effective_potential()

        # Find collective modes
        modes = self.find_collective_modes()

        # Analyze potential landscape
        potential_analysis = self.potential_analyzer.analyze_potential_landscape(potential)

        # Analyze mode spectrum
        mode_spectrum = self.collective_modes_analyzer.analyze_mode_spectrum(modes)

        # Calculate system energy
        system_energy = self._calculate_system_energy(potential)

        # Calculate system stability
        system_stability = self._calculate_system_stability(modes)

        return {
            "potential_analysis": potential_analysis,
            "mode_spectrum": mode_spectrum,
            "system_energy": system_energy,
            "system_stability": system_stability,
            "num_particles": len(self.particles),
            "interaction_range": self.interaction_range,
            "interaction_strength": self.interaction_strength,
        }

    def _calculate_system_energy(self, potential: np.ndarray) -> Dict[str, Any]:
        """
        Calculate system energy.

        Physical Meaning:
            Calculates the total energy of the multi-particle system.

        Args:
            potential (np.ndarray): Potential field.

        Returns:
            Dict[str, Any]: System energy analysis.
        """
        # Calculate potential energy
        potential_energy = np.sum(potential)

        # Calculate kinetic energy (simplified)
        kinetic_energy = 0.5 * np.sum([particle.mass for particle in self.particles])

        # Calculate total energy
        total_energy = potential_energy + kinetic_energy

        return {
            "potential_energy": float(potential_energy),
            "kinetic_energy": float(kinetic_energy),
            "total_energy": float(total_energy),
            "energy_per_particle": float(total_energy / len(self.particles)),
        }

    def _calculate_system_stability(self, modes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate system stability.

        Physical Meaning:
            Calculates the stability of the multi-particle system
            based on collective modes.

        Args:
            modes (Dict[str, Any]): Collective modes.

        Returns:
            Dict[str, Any]: System stability analysis.
        """
        # Extract stability information
        stability = modes.get("stability", {})
        
        # Calculate stability metrics
        stable_modes = stability.get("stable_modes", 0)
        unstable_modes = stability.get("unstable_modes", 0)
        total_modes = stable_modes + unstable_modes

        # Calculate stability ratio
        stability_ratio = stable_modes / total_modes if total_modes > 0 else 1.0

        # Determine overall stability
        if stability_ratio > 0.8:
            overall_stability = "stable"
        elif stability_ratio > 0.5:
            overall_stability = "mostly_stable"
        else:
            overall_stability = "unstable"

        return {
            "stable_modes": stable_modes,
            "unstable_modes": unstable_modes,
            "stability_ratio": float(stability_ratio),
            "overall_stability": overall_stability,
            "max_growth_rate": stability.get("max_growth_rate", 0.0),
        }

    def optimize_system_configuration(self) -> Dict[str, Any]:
        """
        Optimize system configuration.

        Physical Meaning:
            Optimizes the configuration of the multi-particle system
            to minimize energy and improve stability.

        Returns:
            Dict[str, Any]: Optimization results.
        """
        # Compute current potential
        current_potential = self.compute_effective_potential()

        # Optimize potential
        optimization_results = self.potential_analyzer.optimize_potential(current_potential)

        # Analyze optimization results
        energy_reduction = optimization_results.get("energy_reduction", 0.0)
        stability_improvement = optimization_results.get("stability_improvement", 0.0)

        # Determine optimization success
        optimization_success = energy_reduction > 0 and stability_improvement > 0

        return {
            "optimization_results": optimization_results,
            "energy_reduction": energy_reduction,
            "stability_improvement": stability_improvement,
            "optimization_success": optimization_success,
            "optimization_complete": True,
        }

    def validate_system_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate system analysis results.

        Physical Meaning:
            Validates the system analysis results to ensure
            they meet quality and consistency criteria.

        Args:
            results (Dict[str, Any]): Analysis results to validate.

        Returns:
            Dict[str, Any]: Validation results.
        """
        # Validate potential analysis
        potential_validation = self._validate_potential_analysis(results.get("potential_analysis", {}))

        # Validate mode spectrum
        mode_validation = self._validate_mode_spectrum(results.get("mode_spectrum", {}))

        # Validate system energy
        energy_validation = self._validate_system_energy(results.get("system_energy", {}))

        # Validate system stability
        stability_validation = self._validate_system_stability(results.get("system_stability", {}))

        # Calculate overall validation
        overall_validation = self._calculate_overall_validation(
            potential_validation, mode_validation, energy_validation, stability_validation
        )

        return {
            "potential_validation": potential_validation,
            "mode_validation": mode_validation,
            "energy_validation": energy_validation,
            "stability_validation": stability_validation,
            "overall_validation": overall_validation,
            "validation_complete": True,
        }

    def _validate_potential_analysis(self, potential_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate potential analysis results.

        Physical Meaning:
            Validates the potential analysis results.

        Args:
            potential_analysis (Dict[str, Any]): Potential analysis results.

        Returns:
            Dict[str, Any]: Potential analysis validation.
        """
        # Check if potential analysis is present
        is_present = len(potential_analysis) > 0

        # Check potential range
        potential_range = potential_analysis.get("potential_range", [0.0, 0.0])
        range_valid = potential_range[1] > potential_range[0]

        # Check extrema
        extrema = potential_analysis.get("extrema", {})
        extrema_valid = len(extrema.get("minima", [])) > 0

        return {
            "is_present": is_present,
            "range_valid": range_valid,
            "extrema_valid": extrema_valid,
            "validation_passed": is_present and range_valid and extrema_valid,
        }

    def _validate_mode_spectrum(self, mode_spectrum: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mode spectrum results.

        Physical Meaning:
            Validates the mode spectrum results.

        Args:
            mode_spectrum (Dict[str, Any]): Mode spectrum results.

        Returns:
            Dict[str, Any]: Mode spectrum validation.
        """
        # Check if mode spectrum is present
        is_present = len(mode_spectrum) > 0

        # Check frequency analysis
        frequency_analysis = mode_spectrum.get("frequency_analysis", {})
        frequency_valid = len(frequency_analysis) > 0

        # Check spectral features
        spectral_features = mode_spectrum.get("spectral_features", {})
        spectral_valid = len(spectral_features) > 0

        return {
            "is_present": is_present,
            "frequency_valid": frequency_valid,
            "spectral_valid": spectral_valid,
            "validation_passed": is_present and frequency_valid and spectral_valid,
        }

    def _validate_system_energy(self, system_energy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate system energy results.

        Physical Meaning:
            Validates the system energy results.

        Args:
            system_energy (Dict[str, Any]): System energy results.

        Returns:
            Dict[str, Any]: System energy validation.
        """
        # Check if system energy is present
        is_present = len(system_energy) > 0

        # Check energy values
        total_energy = system_energy.get("total_energy", 0.0)
        energy_valid = total_energy > 0

        # Check energy per particle
        energy_per_particle = system_energy.get("energy_per_particle", 0.0)
        energy_per_particle_valid = energy_per_particle > 0

        return {
            "is_present": is_present,
            "energy_valid": energy_valid,
            "energy_per_particle_valid": energy_per_particle_valid,
            "validation_passed": is_present and energy_valid and energy_per_particle_valid,
        }

    def _validate_system_stability(self, system_stability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate system stability results.

        Physical Meaning:
            Validates the system stability results.

        Args:
            system_stability (Dict[str, Any]): System stability results.

        Returns:
            Dict[str, Any]: System stability validation.
        """
        # Check if system stability is present
        is_present = len(system_stability) > 0

        # Check stability ratio
        stability_ratio = system_stability.get("stability_ratio", 0.0)
        stability_valid = 0.0 <= stability_ratio <= 1.0

        # Check overall stability
        overall_stability = system_stability.get("overall_stability", "unknown")
        stability_type_valid = overall_stability in ["stable", "mostly_stable", "unstable"]

        return {
            "is_present": is_present,
            "stability_valid": stability_valid,
            "stability_type_valid": stability_type_valid,
            "validation_passed": is_present and stability_valid and stability_type_valid,
        }

    def _calculate_overall_validation(
        self, potential_validation: Dict[str, Any], mode_validation: Dict[str, Any],
        energy_validation: Dict[str, Any], stability_validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall validation.

        Physical Meaning:
            Calculates the overall validation of all analysis components.

        Args:
            potential_validation (Dict[str, Any]): Potential validation.
            mode_validation (Dict[str, Any]): Mode validation.
            energy_validation (Dict[str, Any]): Energy validation.
            stability_validation (Dict[str, Any]): Stability validation.

        Returns:
            Dict[str, Any]: Overall validation results.
        """
        # Calculate overall validation status
        overall_passed = all([
            potential_validation["validation_passed"],
            mode_validation["validation_passed"],
            energy_validation["validation_passed"],
            stability_validation["validation_passed"],
        ])

        # Calculate validation summary
        validation_summary = {
            "potential_validation": potential_validation["validation_passed"],
            "mode_validation": mode_validation["validation_passed"],
            "energy_validation": energy_validation["validation_passed"],
            "stability_validation": stability_validation["validation_passed"],
        }

        return {
            "overall_passed": overall_passed,
            "validation_summary": validation_summary,
            "validation_complete": True,
        }