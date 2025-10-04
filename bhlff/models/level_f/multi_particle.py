"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Multi-particle system implementation for Level F collective effects.

This module implements the MultiParticleSystem class for studying collective
effects in systems with multiple topological defects. The system includes
effective potential calculations, collective mode analysis, and correlation
function computations.

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


@dataclass
class Particle:
    """
    Particle in multi-particle system.
    
    Physical Meaning:
        Represents a topological defect with position, charge, phase,
        and effective mass in the 7D phase field theory.
        
    Attributes:
        position (np.ndarray): 3D coordinates of the particle
        charge (int): Topological charge q ∈ ℤ
        phase (float): Initial phase φ ∈ [0, 2π)
        mass (float): Effective mass M_eff
    """
    position: np.ndarray
    charge: int
    phase: float
    mass: float = 1.0


class MultiParticleSystem(AbstractModel):
    """
    Multi-particle system for studying collective effects.
    
    Physical Meaning:
        Represents a system of multiple topological defects
        interacting through effective potentials, forming
        collective modes and phase transitions.
        
    Mathematical Foundation:
        Implements the effective potential hierarchy:
        U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ + ...
        where each term represents different orders of interaction.
        
    Attributes:
        domain (Domain): Computational domain
        particles (List[Particle]): List of particles in the system
        interaction_range (float): Range of particle interactions
        interaction_strength (float): Strength of interactions
    """
    
    def __init__(self, domain: 'Domain', particles: List[Particle],
                 interaction_range: float = 5.0, 
                 interaction_strength: float = 1.0):
        """
        Initialize multi-particle system.
        
        Physical Meaning:
            Sets up a system of multiple particles with specified
            interactions and computational domain.
            
        Args:
            domain (Domain): Computational domain
            particles (List[Particle]): List of particles with:
                - position: 3D coordinates
                - charge: topological charge q ∈ ℤ
                - phase: initial phase φ ∈ [0, 2π)
                - mass: effective mass M_eff
            interaction_range (float): Range of particle interactions
            interaction_strength (float): Strength of interactions
        """
        super().__init__(domain)
        self.particles = particles
        self.interaction_range = interaction_range
        self.interaction_strength = interaction_strength
        self._setup_interaction_matrices()
    
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
            for j, particle_j in enumerate(self.particles[i+1:], i+1):
                potential += self._compute_pair_interaction(particle_i, particle_j)
        
        # Higher-order interactions (if needed)
        if len(self.particles) > 2:
            potential += self._compute_higher_order_interactions()
        
        return potential
    
    def find_collective_modes(self) -> Dict[str, Any]:
        """
        Find collective modes of the system.
        
        Physical Meaning:
            Identifies collective excitations that involve
            coordinated motion of multiple particles.
            
        Returns:
            Dict containing:
                - frequencies: ω_n (collective mode frequencies)
                - amplitudes: A_n (mode amplitudes)
                - participation_ratios: p_n (particle participation)
        """
        # Compute dynamics matrix
        dynamics_matrix = self._compute_dynamics_matrix()
        
        # Diagonalize to find modes
        eigenvalues, eigenvectors = np.linalg.eigh(dynamics_matrix)
        
        # Extract mode properties
        frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
        amplitudes = np.linalg.norm(eigenvectors, axis=0)  # Convert to vector
        participation_ratios = self._compute_participation_ratios(eigenvectors)
        
        return {
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'participation_ratios': participation_ratios,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors
        }
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlation functions.
        
        Physical Meaning:
            Computes spatial and temporal correlations
            between particle positions and phases.
            
        Returns:
            Dict containing:
                - spatial_correlations: g(r) (pair correlation function)
                - temporal_correlations: C(t) (time correlation function)
                - phase_correlations: ⟨φᵢφⱼ⟩ (phase correlation matrix)
        """
        # Spatial correlations
        spatial_correlations = self._compute_spatial_correlations()
        
        # Phase correlations
        phase_correlations = self._compute_phase_correlations()
        
        # Temporal correlations (if time evolution is available)
        temporal_correlations = self._compute_temporal_correlations()
        
        return {
            'spatial_correlations': spatial_correlations,
            'temporal_correlations': temporal_correlations,
            'phase_correlations': phase_correlations
        }
    
    def check_stability(self) -> Dict[str, Any]:
        """
        Check stability of the multi-particle system.
        
        Physical Meaning:
            Analyzes the stability of the system by checking
            the eigenvalues of the dynamics matrix.
            
        Returns:
            Dict containing stability analysis results
        """
        # Compute dynamics matrix
        dynamics_matrix = self._compute_dynamics_matrix()
        
        # Check eigenvalues
        eigenvalues = np.linalg.eigvals(dynamics_matrix)
        
        # Stability analysis
        is_stable = np.all(np.real(eigenvalues) < 0)
        stability_margin = -np.max(np.real(eigenvalues))
        
        # Growth rates
        growth_rates = np.real(eigenvalues)
        
        return {
            'is_stable': is_stable,
            'stability_margin': stability_margin,
            'growth_rates': growth_rates,
            'eigenvalues': eigenvalues
        }
    
    def _setup_interaction_matrices(self) -> None:
        """
        Setup interaction matrices for efficient computation.
        
        Physical Meaning:
            Pre-computes interaction matrices to optimize
            collective mode calculations.
        """
        n_particles = len(self.particles)
        self._mass_matrix = np.zeros((n_particles, n_particles))
        self._stiffness_matrix = np.zeros((n_particles, n_particles))
        
        # Fill matrices
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles):
                if i == j:
                    # Diagonal elements
                    self._mass_matrix[i, j] = particle_i.mass
                    self._stiffness_matrix[i, j] = self._compute_self_stiffness(particle_i)
                else:
                    # Off-diagonal elements
                    self._mass_matrix[i, j] = self._compute_interaction_mass(particle_i, particle_j)
                    self._stiffness_matrix[i, j] = self._compute_interaction_stiffness(particle_i, particle_j)
    
    def _compute_single_particle_potential(self, particle: Particle) -> np.ndarray:
        """
        Compute single-particle potential contribution.
        
        Physical Meaning:
            Calculates the potential energy contribution
            from a single particle.
        """
        # Create 7D potential array directly
        potential = np.zeros(self.domain.shape)
        
        # Single-particle potential (Gaussian-like)
        sigma = 1.0  # Width parameter
        
        # Compute potential at particle position
        # For simplicity, use a localized potential
        # Make sure we get negative values for attractive potential
        potential_value = -abs(particle.charge) / (sigma * np.sqrt(2 * np.pi))
        
        # Place potential at particle position
        # This is a simplified approach for 7D space
        potential = np.full(self.domain.shape, potential_value)
        
        return potential
    
    def _compute_pair_interaction(self, particle_i: Particle, 
                                particle_j: Particle) -> np.ndarray:
        """
        Compute pair-wise interaction potential.
        
        Physical Meaning:
            Calculates the interaction potential between
            two particles.
        """
        # Distance between particles
        r_ij = np.linalg.norm(particle_i.position - particle_j.position)
        
        # Interaction strength
        if r_ij > self.interaction_range:
            return np.zeros(self.domain.shape)
        
        # Create 7D interaction array directly
        interaction = np.zeros(self.domain.shape)
        
        # Pair interaction in 7D space (simplified)
        interaction_value = (particle_i.charge * particle_j.charge * 
                           self.interaction_strength / (r_ij + 1e-10))
        
        # Place interaction throughout domain
        interaction = np.full(self.domain.shape, interaction_value)
        
        return interaction
    
    def _compute_higher_order_interactions(self) -> np.ndarray:
        """
        Compute higher-order (three-body, etc.) interactions.
        
        Physical Meaning:
            Calculates three-body and higher-order interaction
            contributions to the effective potential.
        """
        potential = np.zeros(self.domain.shape)
        
        # Three-body interactions
        if len(self.particles) >= 3:
            for i in range(len(self.particles)):
                for j in range(i+1, len(self.particles)):
                    for k in range(j+1, len(self.particles)):
                        potential += self._compute_three_body_interaction(
                            self.particles[i], self.particles[j], self.particles[k])
        
        return potential
    
    def _compute_three_body_interaction(self, particle_i: Particle,
                                      particle_j: Particle, 
                                      particle_k: Particle) -> np.ndarray:
        """
        Compute three-body interaction potential.
        
        Physical Meaning:
            Calculates the three-body interaction contribution
            to the effective potential.
        """
        # Three-body interaction strength
        strength = 0.1 * self.interaction_strength
        
        # Create 7D interaction array directly
        interaction = np.zeros(self.domain.shape)
        
        # Three-body interaction in 7D space (simplified)
        interaction_value = (strength * particle_i.charge * particle_j.charge * 
                           particle_k.charge / (1.0 + 1e-10))
        
        # Place interaction throughout domain
        interaction = np.full(self.domain.shape, interaction_value)
        
        return interaction
    
    def _compute_dynamics_matrix(self) -> np.ndarray:
        """
        Compute dynamics matrix M⁻¹K.
        
        Physical Meaning:
            Computes the dynamics matrix for collective modes
            from mass and stiffness matrices.
        """
        # Invert mass matrix
        mass_inv = np.linalg.inv(self._mass_matrix)
        
        # Dynamics matrix
        dynamics_matrix = mass_inv @ self._stiffness_matrix
        
        return dynamics_matrix
    
    def _compute_self_stiffness(self, particle: Particle) -> float:
        """
        Compute self-stiffness for a particle.
        
        Physical Meaning:
            Calculates the self-stiffness coefficient
            for a single particle.
        """
        # Self-stiffness based on particle properties
        stiffness = particle.mass * (2 * np.pi)**2  # Natural frequency squared
        
        return stiffness
    
    def _compute_interaction_mass(self, particle_i: Particle, 
                                 particle_j: Particle) -> float:
        """
        Compute interaction mass between particles.
        
        Physical Meaning:
            Calculates the effective mass contribution
            from particle interactions.
        """
        # Distance between particles
        r_ij = np.linalg.norm(particle_i.position - particle_j.position)
        
        # Interaction mass (decreases with distance)
        if r_ij > self.interaction_range:
            return 0.0
        
        interaction_mass = (self.interaction_strength * 
                           particle_i.charge * particle_j.charge / 
                           (r_ij + 1e-10))
        
        return interaction_mass
    
    def _compute_interaction_stiffness(self, particle_i: Particle,
                                     particle_j: Particle) -> float:
        """
        Compute interaction stiffness between particles.
        
        Physical Meaning:
            Calculates the interaction stiffness coefficient
            between two particles.
        """
        # Distance between particles
        r_ij = np.linalg.norm(particle_i.position - particle_j.position)
        
        # Interaction stiffness
        if r_ij > self.interaction_range:
            return 0.0
        
        stiffness = (self.interaction_strength * 
                   particle_i.charge * particle_j.charge / 
                   (r_ij**2 + 1e-10))
        
        return stiffness
    
    def _compute_participation_ratios(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Compute participation ratios for collective modes.
        
        Physical Meaning:
            Calculates how much each particle participates
            in each collective mode.
        """
        n_modes, n_particles = eigenvectors.shape
        participation_ratios = np.zeros((n_modes, n_particles))
        
        for mode_idx in range(n_modes):
            mode_vector = eigenvectors[mode_idx, :]
            
            # Normalize
            mode_vector = mode_vector / np.linalg.norm(mode_vector)
            
            # Participation ratios
            participation_ratios[mode_idx, :] = np.abs(mode_vector)**2
        
        return participation_ratios
    
    def _compute_spatial_correlations(self) -> Dict[str, Any]:
        """
        Compute spatial correlation functions.
        
        Physical Meaning:
            Calculates spatial correlations between
            particle positions.
        """
        n_particles = len(self.particles)
        positions = np.array([p.position for p in self.particles])
        
        # Pair correlation function
        distances = []
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        # Correlation length
        if distances:
            correlation_length = np.mean(distances)
        else:
            correlation_length = 0.0
        
        return {
            'distances': distances,
            'correlation_length': correlation_length,
            'mean_distance': np.mean(distances) if distances else 0.0
        }
    
    def _compute_phase_correlations(self) -> np.ndarray:
        """
        Compute phase correlation matrix.
        
        Physical Meaning:
            Calculates correlations between particle phases.
        """
        n_particles = len(self.particles)
        phases = np.array([p.phase for p in self.particles])
        
        # Phase correlation matrix
        phase_correlations = np.outer(phases, phases)
        
        return phase_correlations
    
    def _compute_temporal_correlations(self) -> Dict[str, Any]:
        """
        Compute temporal correlation functions.
        
        Physical Meaning:
            Calculates temporal correlations in the system
            (placeholder for time evolution).
        """
        # Placeholder for temporal correlations
        # This would require time evolution data
        return {
            'correlation_time': 0.0,
            'decay_rate': 0.0
        }
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data for this model.
        
        Physical Meaning:
            Performs comprehensive analysis of the multi-particle system,
            including effective potential, collective modes, and correlations.
            
        Args:
            data (Any): Input data to analyze (not used for this model)
            
        Returns:
            Dict: Analysis results including potential, modes, and correlations
        """
        # Compute effective potential
        potential = self.compute_effective_potential()
        
        # Find collective modes
        modes = self.find_collective_modes()
        
        # Analyze correlations
        correlations = self.analyze_correlations()
        
        # Check stability
        stability = self.check_stability()
        
        return {
            'effective_potential': potential,
            'collective_modes': modes,
            'correlations': correlations,
            'stability': stability
        }

