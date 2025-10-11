"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Multi-particle collective modes module.

This module implements collective modes analysis functionality for multi-particle systems
in Level F of 7D phase field theory.

Physical Meaning:
    Analyzes collective modes in multi-particle systems
    including mode spectrum and correlation functions.

Example:
    >>> modes_analyzer = MultiParticleModesAnalyzer(domain, particles)
    >>> modes = modes_analyzer.find_collective_modes()
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from bhlff.core.bvp import BVPCore
from .data_structures import Particle, SystemParameters


class MultiParticleModesAnalyzer:
    """
    Multi-particle collective modes analyzer for Level F.
    
    Physical Meaning:
        Analyzes collective modes in multi-particle systems
        including mode spectrum and correlation functions.
        
    Mathematical Foundation:
        Implements collective modes analysis:
        - Collective modes: diagonalization of M⁻¹K
        - Correlation functions: G(x,t) = ⟨ψ*(x,t)ψ(0,0)⟩
    """
    
    def __init__(self, domain, particles: List[Particle], interaction_range: float = 2.0, params: Dict[str, Any] = None):
        """
        Initialize multi-particle modes analyzer.
        
        Physical Meaning:
            Sets up the collective modes analysis system with
            appropriate parameters and methods.
            
        Args:
            domain: Domain parameters.
            particles (List[Particle]): List of particles.
            interaction_range (float): Interaction range parameter.
            params (Dict[str, Any]): Additional parameters for step resonator model.
        """
        self.domain = domain
        self.particles = particles
        self.interaction_range = interaction_range
        self.params = params or {}
        self.logger = logging.getLogger(__name__)
    
    def find_collective_modes(self) -> Dict[str, Any]:
        """
        Find collective modes.
        
        Physical Meaning:
            Finds collective modes in multi-particle system
            through diagonalization of dynamics matrix.
            
        Mathematical Foundation:
            Collective modes: diagonalization of M⁻¹K
            where M is the mass matrix and K is the stiffness matrix.
            
        Returns:
            Dict[str, Any]: Collective modes analysis results.
        """
        self.logger.info("Finding collective modes")
        
        # Create dynamics matrix
        dynamics_matrix = self._create_dynamics_matrix()
        
        # Diagonalize dynamics matrix
        eigenvalues, eigenvectors = np.linalg.eigh(dynamics_matrix)
        
        # Analyze mode spectrum
        mode_spectrum = self._analyze_mode_spectrum(eigenvalues, eigenvectors)
        
        # Calculate mode frequencies
        mode_frequencies = self._calculate_mode_frequencies(eigenvalues)
        
        # Calculate mode amplitudes
        mode_amplitudes = self._calculate_mode_amplitudes(eigenvectors)
        
        results = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "mode_spectrum": mode_spectrum,
            "mode_frequencies": mode_frequencies,
            "mode_amplitudes": mode_amplitudes,
            "collective_modes_complete": True,
        }
        
        self.logger.info("Collective modes found")
        return results
    
    def compute_correlation_function(self, field: np.ndarray, time_points: np.ndarray) -> np.ndarray:
        """
        Compute correlation function.
        
        Physical Meaning:
            Computes correlation function for multi-particle system
            to analyze collective behavior.
            
        Mathematical Foundation:
            Correlation function: G(x,t) = ⟨ψ*(x,t)ψ(0,0)⟩
            
        Args:
            field (np.ndarray): Field configuration.
            time_points (np.ndarray): Time points for correlation.
            
        Returns:
            np.ndarray: Correlation function.
        """
        self.logger.info("Computing correlation function")
        
        # Initialize correlation function
        correlation_function = np.zeros((len(time_points), field.shape[0], field.shape[1], field.shape[2]))
        
        # Compute correlation for each time point
        for i, t in enumerate(time_points):
            correlation_function[i] = self._compute_time_correlation(field, t)
        
        self.logger.info("Correlation function computed")
        return correlation_function
    
    def _create_dynamics_matrix(self) -> np.ndarray:
        """
        Create dynamics matrix.
        
        Physical Meaning:
            Creates dynamics matrix for collective modes analysis
            based on particle interactions.
            
        Returns:
            np.ndarray: Dynamics matrix.
        """
        # Create mass matrix
        mass_matrix = self._create_mass_matrix()
        
        # Create stiffness matrix
        stiffness_matrix = self._create_stiffness_matrix()
        
        # Compute dynamics matrix M⁻¹K
        dynamics_matrix = np.linalg.inv(mass_matrix) @ stiffness_matrix
        
        return dynamics_matrix
    
    def _create_mass_matrix(self) -> np.ndarray:
        """
        Create mass matrix.
        
        Physical Meaning:
            Creates mass matrix for collective modes analysis
            based on particle properties.
            
        Returns:
            np.ndarray: Mass matrix.
        """
        # Create mass matrix
        mass_matrix = np.zeros((len(self.particles), len(self.particles)))
        
        # Fill diagonal elements with particle masses
        for i, particle in enumerate(self.particles):
            mass_matrix[i, i] = particle.mass
        
        return mass_matrix
    
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
        for i, particle1 in enumerate(self.particles):
            for j, particle2 in enumerate(self.particles):
                if i != j:
                    distance = np.linalg.norm(particle1.position - particle2.position)
                    interaction_strength = self._calculate_interaction_strength(distance)
                    stiffness_matrix[i, j] = interaction_strength
        
        # Fill diagonal elements
        for i in range(len(self.particles)):
            stiffness_matrix[i, i] = -np.sum(stiffness_matrix[i, :])
        
        return stiffness_matrix
    
    def _analyze_mode_spectrum(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mode spectrum.
        
        Physical Meaning:
            Analyzes mode spectrum of collective modes
            in multi-particle system.
            
        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.
            
        Returns:
            Dict[str, Any]: Mode spectrum analysis results.
        """
        # Analyze eigenvalues
        positive_modes = np.sum(eigenvalues > 0)
        negative_modes = np.sum(eigenvalues < 0)
        zero_modes = np.sum(eigenvalues == 0)
        
        # Calculate mode statistics
        mode_statistics = {
            "total_modes": len(eigenvalues),
            "positive_modes": positive_modes,
            "negative_modes": negative_modes,
            "zero_modes": zero_modes,
            "mode_range": [np.min(eigenvalues), np.max(eigenvalues)],
        }
        
        return mode_statistics
    
    def _calculate_mode_frequencies(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Calculate mode frequencies.
        
        Physical Meaning:
            Calculates frequencies of collective modes
            from eigenvalues.
            
        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.
            
        Returns:
            np.ndarray: Mode frequencies.
        """
        # Calculate mode frequencies
        mode_frequencies = np.sqrt(np.abs(eigenvalues))
        
        return mode_frequencies
    
    def _calculate_mode_amplitudes(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Calculate mode amplitudes.
        
        Physical Meaning:
            Calculates amplitudes of collective modes
            from eigenvectors.
            
        Args:
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.
            
        Returns:
            np.ndarray: Mode amplitudes.
        """
        # Calculate mode amplitudes
        mode_amplitudes = np.linalg.norm(eigenvectors, axis=0)
        
        return mode_amplitudes
    
    def _compute_time_correlation(self, field: np.ndarray, time: float) -> np.ndarray:
        """
        Compute time correlation.
        
        Physical Meaning:
            Computes time correlation for field configuration
            at specific time.
            
        Args:
            field (np.ndarray): Field configuration.
            time (float): Time for correlation.
            
        Returns:
            np.ndarray: Time correlation field.
        """
        # Simplified time correlation calculation
        # In practice, this would involve proper correlation calculation
        correlation = np.fft.fftn(field) * np.conj(np.fft.fftn(field))
        correlation = np.fft.ifftn(correlation).real
        
        return correlation
    
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
        # Step resonator interaction strength calculation
        # Based on 7D BVP theory principles
        return self._step_interaction_potential(distance)
    
    def _step_interaction_potential(self, distance: float) -> float:
        """
        Step function interaction potential.
        
        Physical Meaning:
            Implements step resonator model for particle interactions instead of
            exponential decay. This follows 7D BVP theory principles where
            energy exchange occurs through semi-transparent boundaries.
            
        Mathematical Foundation:
            V(r) = V₀ * Θ(r_cutoff - r) where Θ is the Heaviside step function
            and r_cutoff is the cutoff distance for the interaction.
            
        Args:
            distance (float): Distance between particles
            
        Returns:
            float: Step function interaction potential
        """
        # Step resonator parameters
        interaction_strength = self.params.get("interaction_strength", 1.0)
        
        # Step function interaction: 1.0 below cutoff, 0.0 above
        return interaction_strength if distance < self.interaction_range else 0.0
