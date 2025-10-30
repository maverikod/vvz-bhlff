"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Potential analysis computation module.

This module implements potential computation functionality for multi-particle systems
in Level F of 7D phase field theory.

Physical Meaning:
    Computes effective potentials for multi-particle systems
    including single-particle, pair-wise, and higher-order interactions.

Example:
    >>> potential_computer = PotentialComputationAnalyzer(domain, particles, system_params)
    >>> potential = potential_computer.compute_effective_potential()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from bhlff.core.bvp import BVPCore
from .data_structures import Particle, SystemParameters


class PotentialComputationAnalyzer:
    """
    Potential computation analyzer for multi-particle systems.

    Physical Meaning:
        Computes effective potentials for multi-particle systems
        including single-particle, pair-wise, and higher-order interactions.

    Mathematical Foundation:
        Implements potential computation:
        - Effective potential: U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ
        - Single-particle potential: Uᵢ = U₀(x - xᵢ)
        - Pair-wise potential: Uᵢⱼ = U₁(|xᵢ - xⱼ|)
    """

    def __init__(
        self, domain, particles: List[Particle], system_params: SystemParameters
    ):
        """
        Initialize potential computation analyzer.

        Physical Meaning:
            Sets up the potential computation system with
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

        # Initialize potential analysis
        self._initialize_potential_analysis()

    def compute_effective_potential(self) -> np.ndarray:
        """
        Compute effective potential.

        Physical Meaning:
            Computes effective potential for multi-particle system
            including all interaction terms.

        Mathematical Foundation:
            Effective potential: U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ

        Returns:
            np.ndarray: Effective potential field.
        """
        self.logger.info("Computing effective potential")

        # Initialize potential
        potential = np.zeros(self.domain.shape)

        # Add single-particle potentials
        for particle in self.particles:
            single_potential = self._compute_single_particle_potential(particle)
            potential += single_potential

        # Add pair-wise interactions
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles[i + 1 :], i + 1):
                pair_potential = self._compute_pair_interaction(particle_i, particle_j)
                potential += pair_potential

        # Add higher-order interactions
        higher_order_potential = self._compute_higher_order_interactions()
        potential += higher_order_potential

        self.logger.info("Effective potential computed")
        return potential

    def _initialize_potential_analysis(self) -> None:
        """
        Initialize potential analysis.

        Physical Meaning:
            Initializes potential analysis system with
            appropriate parameters and methods.
        """
        # Setup interaction matrices
        self._setup_interaction_matrices()

        # Setup potential functions
        self._setup_potential_functions()

    def _setup_interaction_matrices(self) -> None:
        """
        Setup interaction matrices.

        Physical Meaning:
            Sets up interaction matrices for multi-particle system
            to enable efficient potential calculations.
        """
        # Setup interaction matrices
        # In practice, this would involve proper matrix setup
        self.interaction_matrix = np.zeros((len(self.particles), len(self.particles)))

        # Calculate interaction strengths
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles):
                if i != j:
                    distance = np.linalg.norm(particle_i.position - particle_j.position)
                    interaction_strength = self._calculate_interaction_strength(
                        distance
                    )
                    self.interaction_matrix[i, j] = interaction_strength

    def _setup_potential_functions(self) -> None:
        """
        Setup potential functions.

        Physical Meaning:
            Sets up potential functions for multi-particle system
            to enable efficient potential calculations.
        """
        # Setup potential functions
        # In practice, this would involve proper function setup
        self.potential_functions = {
            "single_particle": self._create_single_particle_potential,
            "pair_interaction": self._create_pair_potential,
            "higher_order": self._create_higher_order_potential,
        }

    def _compute_single_particle_potential(self, particle: Particle) -> np.ndarray:
        """
        Compute single-particle potential.

        Physical Meaning:
            Computes potential contribution from single particle
            in the multi-particle system.

        Args:
            particle (Particle): Particle object.

        Returns:
            np.ndarray: Single-particle potential field.
        """
        # Create coordinate arrays
        x = np.linspace(0, self.domain.L, self.domain.N)
        y = np.linspace(0, self.domain.L, self.domain.N)
        z = np.linspace(0, self.domain.L, self.domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Calculate distance from particle
        distances = np.sqrt(
            (X - particle.position[0]) ** 2
            + (Y - particle.position[1]) ** 2
            + (Z - particle.position[2]) ** 2
        )

        # Create single-particle potential
        potential = self._create_single_particle_potential(distances, particle)

        return potential

    def _create_single_particle_potential(
        self, distances: np.ndarray, particle: Particle
    ) -> np.ndarray:
        """
        Create single-particle potential.

        Physical Meaning:
            Creates potential field from single particle
            based on distance calculations.

        Args:
            distances (np.ndarray): Distance field from particle.
            particle (Particle): Particle object.

        Returns:
            np.ndarray: Single-particle potential field.
        """
        # Create potential based on particle properties using step resonator model
        potential = particle.charge * self._step_interaction_potential(distances)

        return potential

    def _compute_pair_interaction(
        self, particle_i: Particle, particle_j: Particle
    ) -> np.ndarray:
        """
        Compute pair interaction potential.

        Physical Meaning:
            Computes potential contribution from pair interaction
            between two particles.

        Args:
            particle_i (Particle): First particle.
            particle_j (Particle): Second particle.

        Returns:
            np.ndarray: Pair interaction potential field.
        """
        # Create coordinate arrays
        x = np.linspace(0, self.domain.L, self.domain.N)
        y = np.linspace(0, self.domain.L, self.domain.N)
        z = np.linspace(0, self.domain.L, self.domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Calculate distances from both particles
        distances_i = np.sqrt(
            (X - particle_i.position[0]) ** 2
            + (Y - particle_i.position[1]) ** 2
            + (Z - particle_i.position[2]) ** 2
        )

        distances_j = np.sqrt(
            (X - particle_j.position[0]) ** 2
            + (Y - particle_j.position[1]) ** 2
            + (Z - particle_j.position[2]) ** 2
        )

        # Create pair potential
        potential = self._create_pair_potential(
            distances_i, distances_j, particle_i, particle_j
        )

        return potential

    def _create_pair_potential(
        self,
        distances_i: np.ndarray,
        distances_j: np.ndarray,
        particle_i: Particle,
        particle_j: Particle,
    ) -> np.ndarray:
        """
        Create pair potential.

        Physical Meaning:
            Creates potential field from pair interaction
            between two particles.

        Args:
            distances_i (np.ndarray): Distance field from particle i.
            distances_j (np.ndarray): Distance field from particle j.
            particle_i (Particle): First particle.
            particle_j (Particle): Second particle.

        Returns:
            np.ndarray: Pair potential field.
        """
        # Create pair potential based on particle interactions
        interaction_strength = particle_i.charge * particle_j.charge
        potential = interaction_strength * self._step_three_body_interaction_potential(
            distances_i, distances_j
        )

        return potential

    def _compute_higher_order_interactions(self) -> np.ndarray:
        """
        Compute higher-order interactions.

        Physical Meaning:
            Computes potential contribution from higher-order
            interactions in the multi-particle system.

        Returns:
            np.ndarray: Higher-order interaction potential field.
        """
        # Initialize potential
        potential = np.zeros(self.domain.shape)

        # Compute three-particle interactions
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles[i + 1 :], i + 1):
                for k, particle_k in enumerate(self.particles[j + 1 :], j + 1):
                    three_particle_potential = self._compute_three_particle_interaction(
                        particle_i, particle_j, particle_k
                    )
                    potential += three_particle_potential

        return potential

    def _compute_three_particle_interaction(
        self, particle_i: Particle, particle_j: Particle, particle_k: Particle
    ) -> np.ndarray:
        """
        Compute three-particle interaction potential.

        Physical Meaning:
            Computes potential contribution from three-particle
            interaction in the multi-particle system.

        Args:
            particle_i (Particle): First particle.
            particle_j (Particle): Second particle.
            particle_k (Particle): Third particle.

        Returns:
            np.ndarray: Three-particle interaction potential field.
        """
        # Create coordinate arrays
        x = np.linspace(0, self.domain.L, self.domain.N)
        y = np.linspace(0, self.domain.L, self.domain.N)
        z = np.linspace(0, self.domain.L, self.domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Calculate distances from all three particles
        distances_i = np.sqrt(
            (X - particle_i.position[0]) ** 2
            + (Y - particle_i.position[1]) ** 2
            + (Z - particle_i.position[2]) ** 2
        )

        distances_j = np.sqrt(
            (X - particle_j.position[0]) ** 2
            + (Y - particle_j.position[1]) ** 2
            + (Z - particle_j.position[2]) ** 2
        )

        distances_k = np.sqrt(
            (X - particle_k.position[0]) ** 2
            + (Y - particle_k.position[1]) ** 2
            + (Z - particle_k.position[2]) ** 2
        )

        # Create three-particle potential
        potential = self._create_higher_order_potential(
            distances_i, distances_j, distances_k, particle_i, particle_j, particle_k
        )

        return potential

    def _create_higher_order_potential(
        self,
        distances_i: np.ndarray,
        distances_j: np.ndarray,
        distances_k: np.ndarray,
        particle_i: Particle,
        particle_j: Particle,
        particle_k: Particle,
    ) -> np.ndarray:
        """
        Create higher-order potential.

        Physical Meaning:
            Creates potential field from higher-order interaction
            between three particles.

        Args:
            distances_i (np.ndarray): Distance field from particle i.
            distances_j (np.ndarray): Distance field from particle j.
            distances_k (np.ndarray): Distance field from particle k.
            particle_i (Particle): First particle.
            particle_j (Particle): Second particle.
            particle_k (Particle): Third particle.

        Returns:
            np.ndarray: Higher-order potential field.
        """
        # Create higher-order potential based on three-particle interaction
        interaction_strength = particle_i.charge * particle_j.charge * particle_k.charge
        potential = (
            interaction_strength
            * self._step_three_particle_interaction_potential(
                distances_i, distances_j, distances_k
            )
        )

        return potential

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
        # Simplified interaction strength calculation
        # In practice, this would involve proper interaction calculation
        if distance < self.system_params.interaction_range:
            return self._step_interaction_potential(distance)
        else:
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

    def _step_three_body_interaction_potential(
        self, distance_i: float, distance_j: float
    ) -> float:
        """
        Step function three-body interaction potential.

        Physical Meaning:
            Implements step resonator model for three-body interactions instead of
            exponential decay. This follows 7D BVP theory principles where
            multi-body interactions occur through semi-transparent boundaries.

        Mathematical Foundation:
            V(r₁,r₂) = V₀ * Θ(r_cutoff - r₁) * Θ(r_cutoff - r₂) where Θ is the Heaviside step function
            and r_cutoff is the interaction cutoff distance.

        Args:
            distance_i: Distance to first particle
            distance_j: Distance to second particle

        Returns:
            Step function three-body interaction potential
        """
        # Step resonator parameters
        interaction_cutoff = self.system_params.interaction_range
        interaction_strength = self.system_params.get("interaction_strength", 1.0)

        # Step function three-body interaction: 1.0 if both distances below cutoff, 0.0 otherwise
        if distance_i < interaction_cutoff and distance_j < interaction_cutoff:
            return interaction_strength
        else:
            return 0.0

    def _step_three_particle_interaction_potential(
        self, distances_i: np.ndarray, distances_j: np.ndarray, distances_k: np.ndarray
    ) -> np.ndarray:
        """
        Step function three-particle interaction potential.

        Physical Meaning:
            Implements step resonator model for three-particle interactions instead of
            exponential decay. This follows 7D BVP theory principles where
            multi-particle interactions occur through semi-transparent boundaries.

        Mathematical Foundation:
            V(r₁,r₂,r₃) = V₀ * Θ(r_cutoff - r₁) * Θ(r_cutoff - r₂) * Θ(r_cutoff - r₃)
            where Θ is the Heaviside step function and r_cutoff is the interaction cutoff distance.

        Args:
            distances_i: Distance field to first particle
            distances_j: Distance field to second particle
            distances_k: Distance field to third particle

        Returns:
            Step function three-particle interaction potential field
        """
        # Step resonator parameters
        interaction_cutoff = self.system_params.interaction_range

        # Step function three-particle interaction: 1.0 if all distances below cutoff, 0.0 otherwise
        step_condition = (
            (distances_i < interaction_cutoff)
            & (distances_j < interaction_cutoff)
            & (distances_k < interaction_cutoff)
        )

        return np.where(step_condition, 1.0, 0.0)
