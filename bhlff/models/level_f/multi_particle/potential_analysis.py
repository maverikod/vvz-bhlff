"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Potential analysis module for multi-particle systems.

This module implements potential analysis functionality
for Level F models in 7D phase field theory.

Physical Meaning:
    Implements potential analysis including effective potential
    computation, interaction analysis, and potential optimization.

Example:
    >>> analyzer = PotentialAnalyzer(domain, particles, system_params)
    >>> potential = analyzer.compute_effective_potential()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ...base.abstract_model import AbstractModel
from .data_structures import Particle, SystemParameters


class PotentialAnalyzer(AbstractModel):
    """
    Potential analysis for multi-particle systems.

    Physical Meaning:
        Analyzes the effective potential in multi-particle systems,
        including single-particle, pair-wise, and higher-order
        interactions.

    Mathematical Foundation:
        Implements potential analysis methods:
        - Effective potential: U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ
        - Single-particle potential: Uᵢ = U₀(x - xᵢ)
        - Pair-wise potential: Uᵢⱼ = U₁(|xᵢ - xⱼ|)
        - Higher-order potential: Uᵢⱼₖ = U₂(xᵢ, xⱼ, xₖ)
    """

    def __init__(self, domain, particles: List[Particle], system_params: SystemParameters):
        """
        Initialize potential analyzer.

        Physical Meaning:
            Sets up the potential analysis system with
            domain, particles, and system parameters.

        Args:
            domain: Computational domain
            particles (List[Particle]): List of particles
            system_params (SystemParameters): System parameters
        """
        super().__init__()
        self.domain = domain
        self.particles = particles
        self.system_params = system_params
        
        # Initialize potential analysis
        self._initialize_potential_analysis()

    def _initialize_potential_analysis(self) -> None:
        """
        Initialize potential analysis.

        Physical Meaning:
            Initializes the potential analysis system with
            interaction matrices and potential functions.
        """
        # Set up interaction matrices
        self._setup_interaction_matrices()
        
        # Set up potential functions
        self._setup_potential_functions()

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
                    if distance <= self.system_params.interaction_range:
                        interaction_strength = particle_i.interaction_strength(
                            particle_j, self.system_params.interaction_range
                        )
                        self.interaction_matrix[i, j] = interaction_strength

    def _setup_potential_functions(self) -> None:
        """
        Setup potential functions.

        Physical Meaning:
            Sets up the potential functions for different
            types of interactions.
        """
        # Single-particle potential function
        self.single_particle_potential = self._create_single_particle_potential
        
        # Pair-wise potential function
        self.pair_potential = self._create_pair_potential
        
        # Higher-order potential function
        self.higher_order_potential = self._create_higher_order_potential

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
        # Initialize potential field
        potential = np.zeros(self.domain.shape)

        # Single-particle contributions
        for particle in self.particles:
            potential += self._compute_single_particle_potential(particle)

        # Pair-wise interactions
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles[i + 1 :], i + 1):
                potential += self._compute_pair_interaction(particle_i, particle_j)

        # Higher-order interactions (if needed)
        if len(self.particles) > 2:
            potential += self._compute_higher_order_interactions()

        return potential

    def _compute_single_particle_potential(self, particle: Particle) -> np.ndarray:
        """
        Compute single-particle potential.

        Physical Meaning:
            Calculates the potential contribution from
            a single particle.

        Args:
            particle (Particle): Particle.

        Returns:
            np.ndarray: Single-particle potential.
        """
        # Create coordinate arrays
        x, y, z = np.meshgrid(
            np.arange(self.domain.N),
            np.arange(self.domain.N),
            np.arange(self.domain.N),
            indexing='ij'
        )

        # Calculate distances from particle
        distances = np.sqrt(
            (x - particle.x) ** 2 + (y - particle.y) ** 2 + (z - particle.z) ** 2
        )

        # Single-particle potential
        potential = self.single_particle_potential(distances, particle)

        return potential

    def _create_single_particle_potential(self, distances: np.ndarray, particle: Particle) -> np.ndarray:
        """
        Create single-particle potential.

        Physical Meaning:
            Creates the potential field for a single particle
            based on its position and charge.

        Args:
            distances (np.ndarray): Distances from particle.
            particle (Particle): Particle.

        Returns:
            np.ndarray: Single-particle potential.
        """
        # Gaussian potential centered at particle
        sigma = self.system_params.phase_coherence_length
        potential = particle.charge * np.exp(-distances ** 2 / (2 * sigma ** 2))

        return potential

    def _compute_pair_interaction(self, particle_i: Particle, particle_j: Particle) -> np.ndarray:
        """
        Compute pair interaction potential.

        Physical Meaning:
            Calculates the potential contribution from
            interaction between two particles.

        Args:
            particle_i (Particle): First particle.
            particle_j (Particle): Second particle.

        Returns:
            np.ndarray: Pair interaction potential.
        """
        # Calculate distance between particles
        distance = particle_i.distance_to(particle_j)

        # Check if particles are within interaction range
        if distance > self.system_params.interaction_range:
            return np.zeros(self.domain.shape)

        # Create coordinate arrays
        x, y, z = np.meshgrid(
            np.arange(self.domain.N),
            np.arange(self.domain.N),
            np.arange(self.domain.N),
            indexing='ij'
        )

        # Calculate distances from each particle
        distances_i = np.sqrt(
            (x - particle_i.x) ** 2 + (y - particle_i.y) ** 2 + (z - particle_i.z) ** 2
        )
        distances_j = np.sqrt(
            (x - particle_j.x) ** 2 + (y - particle_j.y) ** 2 + (z - particle_j.z) ** 2
        )

        # Pair interaction potential
        potential = self.pair_potential(distances_i, distances_j, particle_i, particle_j)

        return potential

    def _create_pair_potential(self, distances_i: np.ndarray, distances_j: np.ndarray, 
                              particle_i: Particle, particle_j: Particle) -> np.ndarray:
        """
        Create pair interaction potential.

        Physical Meaning:
            Creates the potential field for interaction
            between two particles.

        Args:
            distances_i (np.ndarray): Distances from particle i.
            distances_j (np.ndarray): Distances from particle j.
            particle_i (Particle): First particle.
            particle_j (Particle): Second particle.

        Returns:
            np.ndarray: Pair interaction potential.
        """
        # Interaction strength
        interaction_strength = particle_i.interaction_strength(
            particle_j, self.system_params.interaction_range
        )

        # Pair potential (simplified)
        potential = interaction_strength * np.exp(-(distances_i + distances_j) / self.system_params.phase_coherence_length)

        return potential

    def _compute_higher_order_interactions(self) -> np.ndarray:
        """
        Compute higher-order interactions.

        Physical Meaning:
            Calculates the potential contribution from
            higher-order interactions between multiple particles.

        Returns:
            np.ndarray: Higher-order interaction potential.
        """
        # Initialize potential
        potential = np.zeros(self.domain.shape)

        # Three-particle interactions
        if len(self.particles) >= 3:
            for i, particle_i in enumerate(self.particles):
                for j, particle_j in enumerate(self.particles[i + 1 :], i + 1):
                    for k, particle_k in enumerate(self.particles[j + 1 :], j + 1):
                        potential += self._compute_three_particle_interaction(
                            particle_i, particle_j, particle_k
                        )

        return potential

    def _compute_three_particle_interaction(self, particle_i: Particle, particle_j: Particle, 
                                          particle_k: Particle) -> np.ndarray:
        """
        Compute three-particle interaction.

        Physical Meaning:
            Calculates the potential contribution from
            interaction between three particles.

        Args:
            particle_i (Particle): First particle.
            particle_j (Particle): Second particle.
            particle_k (Particle): Third particle.

        Returns:
            np.ndarray: Three-particle interaction potential.
        """
        # Check if all particles are within interaction range
        distance_ij = particle_i.distance_to(particle_j)
        distance_ik = particle_i.distance_to(particle_k)
        distance_jk = particle_j.distance_to(particle_k)

        if (distance_ij > self.system_params.interaction_range or
            distance_ik > self.system_params.interaction_range or
            distance_jk > self.system_params.interaction_range):
            return np.zeros(self.domain.shape)

        # Create coordinate arrays
        x, y, z = np.meshgrid(
            np.arange(self.domain.N),
            np.arange(self.domain.N),
            np.arange(self.domain.N),
            indexing='ij'
        )

        # Calculate distances from each particle
        distances_i = np.sqrt(
            (x - particle_i.x) ** 2 + (y - particle_i.y) ** 2 + (z - particle_i.z) ** 2
        )
        distances_j = np.sqrt(
            (x - particle_j.x) ** 2 + (y - particle_j.y) ** 2 + (z - particle_j.z) ** 2
        )
        distances_k = np.sqrt(
            (x - particle_k.x) ** 2 + (y - particle_k.y) ** 2 + (z - particle_k.z) ** 2
        )

        # Three-particle potential
        potential = self.higher_order_potential(distances_i, distances_j, distances_k, 
                                               particle_i, particle_j, particle_k)

        return potential

    def _create_higher_order_potential(self, distances_i: np.ndarray, distances_j: np.ndarray, 
                                      distances_k: np.ndarray, particle_i: Particle, 
                                      particle_j: Particle, particle_k: Particle) -> np.ndarray:
        """
        Create higher-order interaction potential.

        Physical Meaning:
            Creates the potential field for higher-order
            interactions between multiple particles.

        Args:
            distances_i (np.ndarray): Distances from particle i.
            distances_j (np.ndarray): Distances from particle j.
            distances_k (np.ndarray): Distances from particle k.
            particle_i (Particle): First particle.
            particle_j (Particle): Second particle.
            particle_k (Particle): Third particle.

        Returns:
            np.ndarray: Higher-order interaction potential.
        """
        # Higher-order potential (simplified)
        # In practice, this would involve more sophisticated three-body interactions
        potential = 0.1 * np.exp(-(distances_i + distances_j + distances_k) / self.system_params.phase_coherence_length)

        return potential

    def analyze_potential_landscape(self, potential: np.ndarray) -> Dict[str, Any]:
        """
        Analyze potential landscape.

        Physical Meaning:
            Analyzes the potential landscape to identify
            minima, maxima, and saddle points.

        Args:
            potential (np.ndarray): Potential field.

        Returns:
            Dict[str, Any]: Potential landscape analysis.
        """
        # Find potential extrema
        extrema = self._find_potential_extrema(potential)

        # Analyze potential barriers
        barriers = self._analyze_potential_barriers(potential)

        # Analyze potential wells
        wells = self._analyze_potential_wells(potential)

        return {
            "extrema": extrema,
            "barriers": barriers,
            "wells": wells,
            "potential_range": [float(np.min(potential)), float(np.max(potential))],
            "potential_mean": float(np.mean(potential)),
            "potential_std": float(np.std(potential)),
        }

    def _find_potential_extrema(self, potential: np.ndarray) -> Dict[str, Any]:
        """
        Find potential extrema.

        Physical Meaning:
            Identifies minima, maxima, and saddle points
            in the potential landscape.

        Args:
            potential (np.ndarray): Potential field.

        Returns:
            Dict[str, Any]: Potential extrema analysis.
        """
        # Simplified extrema finding
        # In practice, this would involve proper extrema detection
        minima = []
        maxima = []
        saddle_points = []

        # Find global minimum and maximum
        min_idx = np.unravel_index(np.argmin(potential), potential.shape)
        max_idx = np.unravel_index(np.argmax(potential), potential.shape)

        minima.append({
            "position": min_idx,
            "value": float(potential[min_idx]),
            "type": "global_minimum",
        })

        maxima.append({
            "position": max_idx,
            "value": float(potential[max_idx]),
            "type": "global_maximum",
        })

        return {
            "minima": minima,
            "maxima": maxima,
            "saddle_points": saddle_points,
            "num_minima": len(minima),
            "num_maxima": len(maxima),
            "num_saddle_points": len(saddle_points),
        }

    def _analyze_potential_barriers(self, potential: np.ndarray) -> Dict[str, Any]:
        """
        Analyze potential barriers.

        Physical Meaning:
            Analyzes potential barriers in the landscape.

        Args:
            potential (np.ndarray): Potential field.

        Returns:
            Dict[str, Any]: Potential barrier analysis.
        """
        # Simplified barrier analysis
        # In practice, this would involve proper barrier detection
        barriers = []

        # Calculate barrier height
        barrier_height = np.max(potential) - np.min(potential)

        return {
            "barriers": barriers,
            "barrier_height": float(barrier_height),
            "num_barriers": len(barriers),
        }

    def _analyze_potential_wells(self, potential: np.ndarray) -> Dict[str, Any]:
        """
        Analyze potential wells.

        Physical Meaning:
            Analyzes potential wells in the landscape.

        Args:
            potential (np.ndarray): Potential field.

        Returns:
            Dict[str, Any]: Potential well analysis.
        """
        # Simplified well analysis
        # In practice, this would involve proper well detection
        wells = []

        # Calculate well depth
        well_depth = np.max(potential) - np.min(potential)

        return {
            "wells": wells,
            "well_depth": float(well_depth),
            "num_wells": len(wells),
        }

    def optimize_potential(self, potential: np.ndarray) -> Dict[str, Any]:
        """
        Optimize potential configuration.

        Physical Meaning:
            Optimizes the potential configuration to
            minimize energy and improve stability.

        Args:
            potential (np.ndarray): Potential field.

        Returns:
            Dict[str, Any]: Potential optimization results.
        """
        # Simplified potential optimization
        # In practice, this would involve proper optimization algorithms
        optimized_potential = potential.copy()

        # Apply smoothing
        from scipy.ndimage import gaussian_filter
        optimized_potential = gaussian_filter(optimized_potential, sigma=1.0)

        # Calculate optimization metrics
        energy_reduction = np.sum(potential) - np.sum(optimized_potential)
        stability_improvement = np.std(potential) - np.std(optimized_potential)

        return {
            "optimized_potential": optimized_potential,
            "energy_reduction": float(energy_reduction),
            "stability_improvement": float(stability_improvement),
            "optimization_success": energy_reduction > 0,
        }
