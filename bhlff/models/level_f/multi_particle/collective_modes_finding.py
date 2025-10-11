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
from .data_structures import Particle, SystemParameters


class CollectiveModesFinder:
    """
    Collective modes finder for multi-particle systems.
    
    Physical Meaning:
        Finds collective modes in multi-particle systems
        through diagonalization of dynamics matrix.
        
    Mathematical Foundation:
        Implements collective modes finding:
        - Mode finding: diagonalization of dynamics matrix M⁻¹K
        - Dynamics matrix: M⁻¹K where M is mass matrix and K is stiffness matrix
    """
    
    def __init__(self, domain, particles: List[Particle], system_params: SystemParameters):
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
        
        # Initialize collective modes analysis
        self._initialize_collective_modes_analysis()
    
    def find_collective_modes(self) -> Dict[str, Any]:
        """
        Find collective modes.
        
        Physical Meaning:
            Finds collective modes in multi-particle system
            through diagonalization of dynamics matrix.
            
        Mathematical Foundation:
            Mode finding: diagonalization of dynamics matrix M⁻¹K
            where M is the mass matrix and K is the stiffness matrix.
            
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
            including mass and stiffness matrices.
        """
        # Setup mass matrix
        self.mass_matrix = self._create_mass_matrix()
        
        # Setup stiffness matrix
        self.stiffness_matrix = self._create_stiffness_matrix()
        
        # Setup dynamics matrix
        self.dynamics_matrix = np.linalg.inv(self.mass_matrix) @ self.stiffness_matrix
    
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
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles):
                if i != j:
                    distance = np.linalg.norm(particle_i.position - particle_j.position)
                    interaction_strength = self._calculate_interaction_strength(distance)
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
            based on mass and stiffness matrices.
            
        Returns:
            np.ndarray: Dynamics matrix.
        """
        # Compute dynamics matrix M⁻¹K
        dynamics_matrix = np.linalg.inv(self.mass_matrix) @ self.stiffness_matrix
        
        return dynamics_matrix
    
    def _analyze_collective_modes(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Dict[str, Any]:
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
        interaction_analysis = self._analyze_mode_interactions(eigenvalues, eigenvectors)
        
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
    
    def _analyze_mode_interactions(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Dict[str, Any]:
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
    
    def _calculate_mode_coupling(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> float:
        """
        Calculate mode coupling.
        
        Physical Meaning:
            Calculates coupling between collective modes
            based on eigenvalues and eigenvectors.
            
        Args:
            eigenvalues (np.ndarray): Eigenvalues of dynamics matrix.
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.
            
        Returns:
            float: Mode coupling measure.
        """
        # Simplified mode coupling calculation
        # In practice, this would involve proper coupling analysis
        if len(eigenvalues) > 1:
            # Calculate coupling based on eigenvalue differences
            eigenvalue_differences = np.diff(np.sort(eigenvalues))
            coupling = np.mean(eigenvalue_differences) / np.std(eigenvalue_differences)
        else:
            coupling = 0.0
        
        return coupling
    
    def _calculate_mode_overlap(self, eigenvectors: np.ndarray) -> float:
        """
        Calculate mode overlap.
        
        Physical Meaning:
            Calculates overlap between collective modes
            based on eigenvectors.
            
        Args:
            eigenvectors (np.ndarray): Eigenvectors of dynamics matrix.
            
        Returns:
            float: Mode overlap measure.
        """
        # Simplified mode overlap calculation
        # In practice, this would involve proper overlap analysis
        if eigenvectors.shape[1] > 1:
            # Calculate overlap between different modes
            overlaps = []
            for i in range(eigenvectors.shape[1]):
                for j in range(i + 1, eigenvectors.shape[1]):
                    overlap = np.abs(np.dot(eigenvectors[:, i], eigenvectors[:, j]))
                    overlaps.append(overlap)
            
            mode_overlap = np.mean(overlaps) if overlaps else 0.0
        else:
            mode_overlap = 0.0
        
        return mode_overlap
    
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
        # Simplified mode correlation calculation
        # In practice, this would involve proper correlation analysis
        if eigenvectors.shape[1] > 1:
            # Calculate correlation between different modes
            correlations = []
            for i in range(eigenvectors.shape[1]):
                for j in range(i + 1, eigenvectors.shape[1]):
                    correlation = np.corrcoef(eigenvectors[:, i], eigenvectors[:, j])[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
            
            mode_correlation = np.mean(correlations) if correlations else 0.0
        else:
            mode_correlation = 0.0
        
        return mode_correlation
    
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
