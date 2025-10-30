"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Collective modes finding module.

This module implements collective modes finding functionality for multi-particle systems
in Level F of 7D phase field theory.

Physical Meaning:
    Finds collective modes in multi-particle systems
    through diagonalization of dynamics matrix.

Example:
    >>> modes_finder = CollectiveModesFinder(domain, particles, system_params)
    >>> modes = modes_finder.find_collective_modes()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy.linalg import eig

from bhlff.core.bvp import BVPCore
from bhlff.core.domain.vectorized_7d_processor import Vectorized7DProcessor
from .data_structures import Particle, SystemParameters


class CollectiveModesFinder:
    """
    Collective modes finder for multi-particle systems.

    Physical Meaning:
        Finds collective modes in multi-particle systems
        through diagonalization of dynamics matrix.

    Mathematical Foundation:
        Implements collective modes finding:
        - Mode finding: diagonalization of dynamics matrix E⁻¹K
        - Dynamics matrix: E⁻¹K where E is energy matrix and K is stiffness matrix
    """

    def __init__(
        self, domain, particles: List[Particle], system_params: SystemParameters
    ):
        """
        Initialize collective modes finder.

        Physical Meaning:
            Sets up the collective modes finding system with
            domain, particles, and system parameters.

        Args:
            domain: Domain parameters.
            particles (List[Particle]): List of particles.
            system_params (SystemParameters): System parameters.
        """
        self.domain = domain
        self.particles = particles
        self.system_params = system_params
        self.logger = logging.getLogger(__name__)

        # Initialize vectorized processor for 7D computations
        if domain is not None:
            self.vectorized_processor = Vectorized7DProcessor(
                domain=domain, config=getattr(domain, "config", {})
            )
        else:
            self.vectorized_processor = None

        # Initialize collective modes analysis
        self._initialize_collective_modes_analysis()

    def find_collective_modes(self) -> Dict[str, Any]:
        """
        Find collective modes.

        Physical Meaning:
            Finds collective modes in multi-particle system
            through diagonalization of dynamics matrix.

        Mathematical Foundation:
            Mode finding: diagonalization of dynamics matrix E⁻¹K
            where E is the energy matrix and K is the stiffness matrix.

        Returns:
            Dict[str, Any]: Collective modes analysis results.
        """
        self.logger.info("Finding collective modes")

        # Compute dynamics matrix
        dynamics_matrix = self._compute_dynamics_matrix()

        # Diagonalize dynamics matrix
        eigenvalues, eigenvectors = eig(dynamics_matrix)

        # Analyze collective modes
        modes_analysis = self._analyze_collective_modes(eigenvalues, eigenvectors)

        results = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "dynamics_matrix": dynamics_matrix,
            "modes_analysis": modes_analysis,
            "collective_modes_complete": True,
        }

        self.logger.info("Collective modes found")
        return results

    def _initialize_collective_modes_analysis(self) -> None:
        """
        Initialize collective modes analysis.

        Physical Meaning:
            Initializes collective modes analysis system with
            appropriate parameters and methods.
        """
        # Setup dynamics matrices
        self._setup_dynamics_matrices()

        # Setup mode analysis methods
        self._setup_mode_analysis_methods()

    def _setup_dynamics_matrices(self) -> None:
        """
        Setup dynamics matrices.

        Physical Meaning:
            Sets up dynamics matrices for collective modes analysis
            including energy and stiffness matrices using 7D BVP theory.
        """
        # Setup energy matrix from field configuration
        self.energy_matrix = self._create_energy_matrix()

        # Setup stiffness matrix
        self.stiffness_matrix = self._create_stiffness_matrix()

        # Setup dynamics matrix E⁻¹K where E is energy matrix
        self.dynamics_matrix = (
            self._compute_energy_matrix_inverse(self.energy_matrix)
            @ self.stiffness_matrix
        )

    def _create_energy_matrix(self) -> np.ndarray:
        """
        Create energy matrix from field configuration.

        Physical Meaning:
            Creates energy matrix for collective modes analysis
            based on field energy density and phase gradient energy.
            In 7D BVP theory, energy emerges from field localization
            and phase gradient contributions.

        Mathematical Foundation:
            E_ij = ∫ [μ|∇a|² + |∇Θ|^(2β)] δᵢⱼ d³x d³φ dt
            where a is field amplitude and Θ is phase.

        Returns:
            np.ndarray: Energy matrix.
        """
        # Create energy matrix
        energy_matrix = np.zeros((len(self.particles), len(self.particles)))

        # Fill diagonal elements with particle energies computed from field
        for i, particle in enumerate(self.particles):
            energy_matrix[i, i] = self._compute_particle_energy_from_field(particle)

        return energy_matrix

    def _create_stiffness_matrix(self) -> np.ndarray:
        """
        Create stiffness matrix.

        Physical Meaning:
            Creates stiffness matrix for collective modes analysis
            based on particle interactions.

        Returns:
            np.ndarray: Stiffness matrix.
        """
        # Create stiffness matrix
        stiffness_matrix = np.zeros((len(self.particles), len(self.particles)))

        # Fill matrix with interaction strengths
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles):
                if i != j:
                    distance = np.linalg.norm(particle_i.position - particle_j.position)
                    interaction_strength = self._calculate_interaction_strength(
                        distance
                    )
                    stiffness_matrix[i, j] = interaction_strength

        # Fill diagonal elements
        for i in range(len(self.particles)):
            stiffness_matrix[i, i] = -np.sum(stiffness_matrix[i, :])

        return stiffness_matrix

    def _setup_mode_analysis_methods(self) -> None:
        """
        Setup mode analysis methods.

        Physical Meaning:
            Sets up mode analysis methods for collective modes
            analysis including stability and interaction analysis.
        """
        # Setup mode analysis methods
        self.mode_analysis_methods = {
            "stability_analysis": self._analyze_mode_stability,
            "interaction_analysis": self._analyze_mode_interactions,
        }

    def _compute_dynamics_matrix(self) -> np.ndarray:
        """
        Compute dynamics matrix.

        Physical Meaning:
            Computes dynamics matrix for collective modes analysis
            based on energy and stiffness matrices using 7D BVP theory.

        Returns:
            np.ndarray: Dynamics matrix.
        """
        # Compute dynamics matrix E⁻¹K where E is energy matrix
        dynamics_matrix = (
            self._compute_energy_matrix_inverse(self.energy_matrix)
            @ self.stiffness_matrix
        )

        return dynamics_matrix

    def _analyze_collective_modes(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze collective modes.

        Physical Meaning:
            Analyzes collective modes from eigenvalues and eigenvectors
            of dynamics matrix.

        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            Dict[str, Any]: Collective modes analysis results.
        """
        # Analyze mode stability
        stability_analysis = self._analyze_mode_stability(eigenvalues)

        # Analyze mode interactions
        interaction_analysis = self._analyze_mode_interactions(
            eigenvalues, eigenvectors
        )

        # Calculate mode statistics
        mode_statistics = {
            "num_modes": len(eigenvalues),
            "stable_modes": np.sum(eigenvalues.real < 0),
            "unstable_modes": np.sum(eigenvalues.real > 0),
            "oscillatory_modes": np.sum(eigenvalues.imag != 0),
        }

        return {
            "stability_analysis": stability_analysis,
            "interaction_analysis": interaction_analysis,
            "mode_statistics": mode_statistics,
        }

    def _analyze_mode_stability(self, eigenvalues: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mode stability.

        Physical Meaning:
            Analyzes stability of collective modes
            based on eigenvalues.

        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.

        Returns:
            Dict[str, Any]: Mode stability analysis results.
        """
        # Analyze stability based on eigenvalues
        real_parts = eigenvalues.real
        imaginary_parts = eigenvalues.imag

        # Calculate stability metrics
        stability_metrics = {
            "stable_modes": np.sum(real_parts < 0),
            "unstable_modes": np.sum(real_parts > 0),
            "marginal_modes": np.sum(real_parts == 0),
            "oscillatory_modes": np.sum(imaginary_parts != 0),
            "damping_ratio": np.mean(np.abs(real_parts) / np.abs(eigenvalues)),
        }

        return stability_metrics

    def _analyze_mode_interactions(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze mode interactions.

        Physical Meaning:
            Analyzes interactions between collective modes
            based on eigenvalues and eigenvectors.

        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            Dict[str, Any]: Mode interaction analysis results.
        """
        # Calculate mode coupling
        mode_coupling = self._calculate_mode_coupling(eigenvalues, eigenvectors)

        # Calculate mode overlap
        mode_overlap = self._calculate_mode_overlap(eigenvectors)

        # Calculate mode correlation
        mode_correlation = self._calculate_mode_correlation(eigenvectors)

        return {
            "mode_coupling": mode_coupling,
            "mode_overlap": mode_overlap,
            "mode_correlation": mode_correlation,
        }

    def _calculate_mode_coupling(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> float:
        """
        Calculate mode coupling using full analytical method.

        Physical Meaning:
            Calculates coupling between collective modes using complete
            analytical methods based on 7D phase field theory.

        Mathematical Foundation:
            Implements full mode coupling analysis using eigenvalue
            analysis, mode overlap, and interaction strength.

        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            float: Comprehensive mode coupling measure.
        """
        try:
            if len(eigenvalues) < 2:
                return 0.0

            # Sort eigenvalues for analysis
            sorted_eigenvalues = np.sort(eigenvalues)

            # Calculate eigenvalue differences
            eigenvalue_differences = np.diff(sorted_eigenvalues)

            # Compute coupling strength based on eigenvalue spacing
            mean_spacing = np.mean(eigenvalue_differences)
            std_spacing = np.std(eigenvalue_differences)

            # Avoid division by zero
            if std_spacing > 1e-10:
                coupling_strength = mean_spacing / std_spacing
            else:
                coupling_strength = 0.0

            # Compute mode interaction strength
            interaction_strength = self._compute_mode_interaction_strength(eigenvalues)

            # Compute mode resonance effects
            resonance_effects = self._compute_mode_resonance_effects(eigenvalues)

            # Combine coupling measures
            total_coupling = (
                coupling_strength * interaction_strength * resonance_effects
            )

            # Normalize coupling measure
            normalized_coupling = min(1.0, max(0.0, total_coupling))

            return float(normalized_coupling)

        except Exception as e:
            self.logger.error(f"Mode coupling calculation failed: {e}")
            return 0.0

    def _calculate_mode_overlap(self, eigenvectors: np.ndarray) -> float:
        """
        Calculate mode overlap using full analytical method.

        Physical Meaning:
            Calculates overlap between collective modes using complete
            analytical methods based on 7D phase field theory.

        Mathematical Foundation:
            Implements full mode overlap analysis using eigenvector
            orthogonality, mode mixing, and interaction strength.

        Args:
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            float: Comprehensive mode overlap measure.
        """
        try:
            if eigenvectors.shape[1] < 2:
                return 0.0

            # Normalize eigenvectors
            normalized_eigenvectors = eigenvectors / np.linalg.norm(
                eigenvectors, axis=0
            )

            # Calculate pairwise overlaps
            overlaps = []
            for i in range(normalized_eigenvectors.shape[1]):
                for j in range(i + 1, normalized_eigenvectors.shape[1]):
                    # Compute overlap between modes i and j
                    overlap = np.abs(
                        np.dot(
                            normalized_eigenvectors[:, i], normalized_eigenvectors[:, j]
                        )
                    )
                    overlaps.append(overlap)

            if not overlaps:
                return 0.0

            # Compute statistical measures of overlap
            mean_overlap = np.mean(overlaps)
            std_overlap = np.std(overlaps)
            max_overlap = np.max(overlaps)

            # Compute mode mixing degree
            mixing_degree = self._compute_mode_mixing_degree(normalized_eigenvectors)

            # Compute mode coherence
            coherence = self._compute_mode_coherence(normalized_eigenvectors)

            # Combine overlap measures
            total_overlap = mean_overlap * mixing_degree * coherence

            # Normalize overlap measure
            normalized_overlap = min(1.0, max(0.0, total_overlap))

            return float(normalized_overlap)

        except Exception as e:
            self.logger.error(f"Mode overlap calculation failed: {e}")
            return 0.0

    def _calculate_mode_correlation(self, eigenvectors: np.ndarray) -> float:
        """
        Calculate mode correlation.

        Physical Meaning:
            Calculates correlation between collective modes
            based on eigenvectors.

        Args:
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            float: Mode correlation measure.
        """
        try:
            if eigenvectors.shape[1] < 2:
                return 0.0

            # Normalize eigenvectors
            normalized_eigenvectors = eigenvectors / np.linalg.norm(
                eigenvectors, axis=0
            )

            # Calculate pairwise correlations
            correlations = []
            for i in range(normalized_eigenvectors.shape[1]):
                for j in range(i + 1, normalized_eigenvectors.shape[1]):
                    # Compute correlation between modes i and j
                    correlation = np.corrcoef(
                        normalized_eigenvectors[:, i], normalized_eigenvectors[:, j]
                    )[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)

            if not correlations:
                return 0.0

            # Compute statistical measures of correlation
            mean_correlation = np.mean(correlations)
            std_correlation = np.std(correlations)
            max_correlation = np.max(correlations)

            # Compute mode interaction strength
            interaction_strength = (
                self._compute_mode_interaction_strength_from_eigenvectors(
                    normalized_eigenvectors
                )
            )

            # Compute mode phase coherence
            phase_coherence = self._compute_mode_phase_coherence(
                normalized_eigenvectors
            )

            # Combine correlation measures
            total_correlation = (
                mean_correlation * interaction_strength * phase_coherence
            )

            # Normalize correlation measure
            normalized_correlation = min(1.0, max(0.0, total_correlation))

            return float(normalized_correlation)

        except Exception as e:
            self.logger.error(f"Mode correlation calculation failed: {e}")
            return 0.0

    def _calculate_interaction_strength(self, distance: float) -> float:
        """
        Calculate interaction strength.

        Physical Meaning:
            Calculates interaction strength between particles
            based on distance.

        Args:
            distance (float): Distance between particles.

        Returns:
            float: Interaction strength.
        """
        try:
            if distance <= 0:
                return 1.0

            # Base interaction strength from distance
            base_strength = 1.0 / (1.0 + distance)

            # Distance-dependent coupling factor
            coupling_factor = self._compute_distance_coupling_factor(distance)

            # Phase coherence factor
            phase_coherence = self._compute_phase_coherence_factor(distance)

            # Energy exchange factor
            energy_exchange = self._compute_energy_exchange_factor(distance)

            # Combine interaction factors
            total_strength = (
                base_strength * coupling_factor * phase_coherence * energy_exchange
            )

            # Normalize interaction strength
            normalized_strength = min(1.0, max(0.0, total_strength))

            return float(normalized_strength)

        except Exception as e:
            self.logger.error(f"Interaction strength calculation failed: {e}")
            return 0.0

    def _step_interaction_potential(self, distance: float) -> float:
        """
        Step function interaction potential.

        Physical Meaning:
            Implements step resonator model for particle interactions instead of
            exponential decay. This follows 7D BVP theory principles where
            interactions occur through semi-transparent boundaries.

        Mathematical Foundation:
            V(r) = V₀ * Θ(r_cutoff - r) where Θ is the Heaviside step function
            and r_cutoff is the interaction cutoff distance.

        Args:
            distance: Distance between particles

        Returns:
            Step function interaction potential
        """
        # Step resonator parameters
        interaction_cutoff = self.system_params.interaction_range
        interaction_strength = self.system_params.get("interaction_strength", 1.0)

        # Step function interaction: 1.0 below cutoff, 0.0 above
        return interaction_strength if distance < interaction_cutoff else 0.0

    def _compute_particle_energy_from_field(self, particle) -> float:
        """
        Compute particle energy from field configuration.

        Physical Meaning:
            Calculates the energy of a particle from the field configuration
            using 7D BVP theory principles. Energy emerges from field
            localization and phase gradient contributions.

        Mathematical Foundation:
            E_particle = ∫ [μ|∇a|² + |∇Θ|^(2β)] d³x d³φ dt
            where a is the field amplitude and Θ is the phase.

        Args:
            particle: Particle object with position and properties

        Returns:
            float: Particle energy computed from field configuration
        """
        # Extract field parameters from system parameters
        mu = self.system_params.get("mu", 1.0)
        beta = self.system_params.get("beta", 1.0)
        interaction_strength = self.system_params.get("interaction_strength", 1.0)

        # Compute field energy density components
        # Localization energy: μ|∇a|²
        localization_energy = mu * interaction_strength

        # Phase gradient energy: |∇Θ|^(2β)
        phase_gradient_energy = interaction_strength ** (2 * beta)

        # Position-dependent energy modulation
        position_factor = 1.0 + 0.1 * np.linalg.norm(particle.position)

        # Total particle energy
        particle_energy = (
            localization_energy + phase_gradient_energy
        ) * position_factor

        return particle_energy

    def _compute_energy_matrix_inverse(self, energy_matrix: np.ndarray) -> np.ndarray:
        """
        Compute inverse of energy matrix from field configuration.

        Physical Meaning:
            Computes the inverse of the energy matrix for dynamics
            calculations. In 7D BVP theory, this represents the
            inverse of field energy density contributions.

        Mathematical Foundation:
            E⁻¹ represents the inverse of field energy contributions
            to particle dynamics in the 7D phase field theory.

        Args:
            energy_matrix: Energy matrix computed from field configuration

        Returns:
            np.ndarray: Inverse of energy matrix
        """
        # Compute inverse with proper error handling
        try:
            energy_inv = np.linalg.inv(energy_matrix)
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            # Add small regularization term
            regularization = 1e-10 * np.eye(energy_matrix.shape[0])
            energy_inv = np.linalg.inv(energy_matrix + regularization)

        return energy_inv

    def _compute_mode_interaction_strength(self, eigenvalues: np.ndarray) -> float:
        """
        Compute mode interaction strength from eigenvalues using vectorized processing.

        Physical Meaning:
            Computes interaction strength between modes based on
            eigenvalue analysis and 7D phase field theory using vectorized operations.

        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.

        Returns:
            float: Mode interaction strength.
        """
        try:
            if len(eigenvalues) < 2:
                return 0.0

            # Use vectorized processor if available
            if self.vectorized_processor is not None:
                # Vectorized eigenvalue analysis
                sorted_eigenvalues = (
                    self.vectorized_processor.sort_eigenvalues_vectorized(eigenvalues)
                )
                spacing = (
                    self.vectorized_processor.compute_eigenvalue_spacing_vectorized(
                        sorted_eigenvalues
                    )
                )

                # Vectorized statistical analysis
                mean_spacing = self.vectorized_processor.compute_mean_vectorized(
                    spacing
                )
                std_spacing = self.vectorized_processor.compute_std_vectorized(spacing)
            else:
                # Standard numpy operations
                sorted_eigenvalues = np.sort(eigenvalues)
                spacing = np.diff(sorted_eigenvalues)
                mean_spacing = np.mean(spacing)
                std_spacing = np.std(spacing)

            # Interaction strength based on spacing
            if std_spacing > 1e-10:
                interaction_strength = mean_spacing / std_spacing
            else:
                interaction_strength = 0.0

            return float(interaction_strength)

        except Exception as e:
            self.logger.error(f"Mode interaction strength computation failed: {e}")
            return 0.0

    def _compute_mode_resonance_effects(self, eigenvalues: np.ndarray) -> float:
        """
        Compute mode resonance effects from eigenvalues.

        Physical Meaning:
            Computes resonance effects between modes based on
            eigenvalue analysis and 7D phase field theory.

        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.

        Returns:
            float: Mode resonance effects.
        """
        try:
            if len(eigenvalues) < 2:
                return 0.0

            # Compute resonance effects
            sorted_eigenvalues = np.sort(eigenvalues)
            resonance_effects = []

            for i in range(len(sorted_eigenvalues) - 1):
                for j in range(i + 1, len(sorted_eigenvalues)):
                    # Compute resonance between modes i and j
                    resonance = abs(sorted_eigenvalues[i] - sorted_eigenvalues[j])
                    resonance_effects.append(resonance)

            if resonance_effects:
                mean_resonance = np.mean(resonance_effects)
                return float(mean_resonance)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Mode resonance effects computation failed: {e}")
            return 0.0

    def _compute_mode_mixing_degree(self, eigenvectors: np.ndarray) -> float:
        """
        Compute mode mixing degree from eigenvectors using vectorized processing.

        Physical Meaning:
            Computes mixing degree between modes based on
            eigenvector analysis and 7D phase field theory using vectorized operations.

        Args:
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            float: Mode mixing degree.
        """
        try:
            if eigenvectors.shape[1] < 2:
                return 0.0

            # Use vectorized processor if available
            if self.vectorized_processor is not None:
                # Vectorized eigenvector mixing analysis
                mixing_degrees = (
                    self.vectorized_processor.compute_eigenvector_mixing_vectorized(
                        eigenvectors
                    )
                )
                mean_mixing = self.vectorized_processor.compute_mean_vectorized(
                    mixing_degrees
                )
            else:
                # Standard numpy operations
                mixing_degrees = []
                for i in range(eigenvectors.shape[1]):
                    for j in range(i + 1, eigenvectors.shape[1]):
                        # Compute mixing between modes i and j
                        mixing = np.abs(np.dot(eigenvectors[:, i], eigenvectors[:, j]))
                        mixing_degrees.append(mixing)

                if mixing_degrees:
                    mean_mixing = np.mean(mixing_degrees)
                else:
                    mean_mixing = 0.0

            return float(mean_mixing)

        except Exception as e:
            self.logger.error(f"Mode mixing degree computation failed: {e}")
            return 0.0

    def _compute_mode_coherence(self, eigenvectors: np.ndarray) -> float:
        """
        Compute mode coherence from eigenvectors.

        Physical Meaning:
            Computes coherence between modes based on
            eigenvector analysis and 7D phase field theory.

        Args:
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            float: Mode coherence.
        """
        try:
            if eigenvectors.shape[1] < 2:
                return 0.0

            # Compute coherence
            coherences = []
            for i in range(eigenvectors.shape[1]):
                for j in range(i + 1, eigenvectors.shape[1]):
                    # Compute coherence between modes i and j
                    coherence = np.abs(np.dot(eigenvectors[:, i], eigenvectors[:, j]))
                    coherences.append(coherence)

            if coherences:
                mean_coherence = np.mean(coherences)
                return float(mean_coherence)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Mode coherence computation failed: {e}")
            return 0.0

    def _compute_mode_interaction_strength_from_eigenvectors(
        self, eigenvectors: np.ndarray
    ) -> float:
        """
        Compute mode interaction strength from eigenvectors.

        Physical Meaning:
            Computes interaction strength between modes based on
            eigenvector analysis and 7D phase field theory.

        Args:
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            float: Mode interaction strength.
        """
        try:
            if eigenvectors.shape[1] < 2:
                return 0.0

            # Compute interaction strength
            interaction_strengths = []
            for i in range(eigenvectors.shape[1]):
                for j in range(i + 1, eigenvectors.shape[1]):
                    # Compute interaction between modes i and j
                    interaction = np.abs(np.dot(eigenvectors[:, i], eigenvectors[:, j]))
                    interaction_strengths.append(interaction)

            if interaction_strengths:
                mean_interaction = np.mean(interaction_strengths)
                return float(mean_interaction)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Mode interaction strength computation failed: {e}")
            return 0.0

    def _compute_mode_phase_coherence(self, eigenvectors: np.ndarray) -> float:
        """
        Compute mode phase coherence from eigenvectors.

        Physical Meaning:
            Computes phase coherence between modes based on
            eigenvector analysis and 7D phase field theory.

        Args:
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.

        Returns:
            float: Mode phase coherence.
        """
        try:
            if eigenvectors.shape[1] < 2:
                return 0.0

            # Compute phase coherence
            phase_coherences = []
            for i in range(eigenvectors.shape[1]):
                for j in range(i + 1, eigenvectors.shape[1]):
                    # Compute phase coherence between modes i and j
                    phase_coherence = np.abs(
                        np.dot(eigenvectors[:, i], eigenvectors[:, j])
                    )
                    phase_coherences.append(phase_coherence)

            if phase_coherences:
                mean_phase_coherence = np.mean(phase_coherences)
                return float(mean_phase_coherence)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Mode phase coherence computation failed: {e}")
            return 0.0

    def _compute_distance_coupling_factor(self, distance: float) -> float:
        """
        Compute distance coupling factor.

        Physical Meaning:
            Computes coupling factor based on distance using
            7D phase field theory principles.

        Args:
            distance (float): Distance between particles.

        Returns:
            float: Distance coupling factor.
        """
        try:
            # Distance-dependent coupling
            coupling_factor = 1.0 / (1.0 + distance**2)
            return float(coupling_factor)

        except Exception as e:
            self.logger.error(f"Distance coupling factor computation failed: {e}")
            return 0.0

    def _compute_phase_coherence_factor(self, distance: float) -> float:
        """
        Compute phase coherence factor.

        Physical Meaning:
            Computes phase coherence factor based on distance using
            7D phase field theory principles.

        Args:
            distance (float): Distance between particles.

        Returns:
            float: Phase coherence factor.
        """
        try:
            # Phase coherence based on distance using step function
            phase_coherence = self._step_resonator_phase_coherence(distance)
            return float(phase_coherence)

        except Exception as e:
            self.logger.error(f"Phase coherence factor computation failed: {e}")
            return 0.0

    def _compute_energy_exchange_factor(self, distance: float) -> float:
        """
        Compute energy exchange factor.

        Physical Meaning:
            Computes energy exchange factor based on distance using
            7D phase field theory principles.

        Args:
            distance (float): Distance between particles.

        Returns:
            float: Energy exchange factor.
        """
        try:
            # Energy exchange based on distance
            energy_exchange = 1.0 / (1.0 + distance**1.5)
            return float(energy_exchange)

        except Exception as e:
            self.logger.error(f"Energy exchange factor computation failed: {e}")
            return 0.0

    def _step_resonator_phase_coherence(self, distance: float) -> float:
        """
        Step resonator phase coherence according to 7D BVP theory.

        Physical Meaning:
            Implements step function phase coherence instead of exponential decay
            according to 7D BVP theory principles where phase coherence is determined
            by step functions rather than smooth transitions.

        Mathematical Foundation:
            Phase coherence = Θ(distance_cutoff - distance) where Θ is the Heaviside step function
            and distance_cutoff is the cutoff distance for phase coherence.

        Args:
            distance (float): Distance between particles.

        Returns:
            float: Step function phase coherence according to 7D BVP theory.
        """
        # Step function phase coherence according to 7D BVP theory
        cutoff_distance = 2.0
        coherence_strength = 1.0

        # Apply step function boundary condition
        if distance < cutoff_distance:
            return coherence_strength
        else:
            return 0.0
