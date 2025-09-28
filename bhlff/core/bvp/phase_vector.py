"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

U(1)³ phase vector structure for BVP.

This module implements the U(1)³ phase vector structure Θ_a (a=1..3)
as required by the 7D phase field theory, providing the fundamental
phase structure for the Base High-Frequency Field.

Physical Meaning:
    Implements the three-component phase vector Θ_a (a=1..3) that
    represents the fundamental phase structure of the BVP field.
    Each component corresponds to a different U(1) symmetry group,
    and together they form the U(1)³ structure required by the theory.

Mathematical Foundation:
    The phase vector Θ = (Θ₁, Θ₂, Θ₃) represents three independent
    U(1) phase degrees of freedom. The BVP field is constructed as:
    a(x) = |A(x)| * exp(i * Θ(x))
    where Θ(x) = Σ_a Θ_a(x) * e_a and e_a are the basis vectors.

Example:
    >>> phase_vector = PhaseVector(domain, config)
    >>> theta_components = phase_vector.get_phase_components()
    >>> electroweak_currents = phase_vector.compute_electroweak_currents(envelope)
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from abc import ABC, abstractmethod

from ..domain import Domain
from .bvp_constants import BVPConstants


class PhaseVector:
    """
    U(1)³ phase vector structure for BVP.
    
    Physical Meaning:
        Implements the three-component phase vector Θ_a (a=1..3)
        that represents the fundamental phase structure of the BVP field.
        Each component corresponds to a different U(1) symmetry group.
        
    Mathematical Foundation:
        The phase vector Θ = (Θ₁, Θ₂, Θ₃) represents three independent
        U(1) phase degrees of freedom with weak hierarchical coupling
        to SU(2)/core through invariant mixed terms.
        
    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Phase vector configuration.
        constants (BVPConstants): BVP constants instance.
        theta_components (List[np.ndarray]): Three phase components Θ_a.
        coupling_matrix (np.ndarray): SU(2) coupling matrix.
        electroweak_coefficients (Dict[str, float]): Electroweak coupling coefficients.
    """
    
    def __init__(self, domain: Domain, config: Dict[str, Any], constants: BVPConstants = None) -> None:
        """
        Initialize U(1)³ phase vector structure.
        
        Physical Meaning:
            Sets up the three-component phase vector Θ_a (a=1..3)
            with proper U(1)³ structure and weak SU(2) coupling.
            
        Args:
            domain (Domain): Computational domain.
            config (Dict[str, Any]): Phase vector configuration including:
                - phase_amplitudes: Amplitudes for each phase component
                - phase_frequencies: Frequencies for each phase component
                - su2_coupling_strength: Strength of SU(2) coupling
                - electroweak_coefficients: Electroweak coupling parameters
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.domain = domain
        self.config = config
        self.constants = constants or BVPConstants(config)
        self._setup_phase_components()
        self._setup_su2_coupling()
        self._setup_electroweak_coefficients()
    
    def _setup_phase_components(self) -> None:
        """
        Setup the three U(1) phase components Θ_a (a=1..3).
        
        Physical Meaning:
            Initializes the three independent U(1) phase components
            that form the U(1)³ structure of the BVP field.
        """
        self.theta_components = []
        
        # Get phase configuration
        phase_config = self.config.get("phase_components", {})
        
        for a in range(3):  # Three U(1) components
            # Initialize phase component Θ_a
            theta_a = np.zeros(self.domain.shape, dtype=complex)
            
            # Set amplitude and frequency for this component
            amplitude = phase_config.get(f"amplitude_{a+1}", 1.0)
            frequency = phase_config.get(f"frequency_{a+1}", 1.0)
            
            # Create spatial phase distribution
            if self.domain.dimensions == 1:
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                theta_a = amplitude * np.exp(1j * frequency * x)
            elif self.domain.dimensions == 2:
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                X, Y = np.meshgrid(x, y, indexing="ij")
                theta_a = amplitude * np.exp(1j * frequency * (X + Y))
            else:  # 3D
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                z = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
                theta_a = amplitude * np.exp(1j * frequency * (X + Y + Z))
            
            self.theta_components.append(theta_a)
    
    def _setup_su2_coupling(self) -> None:
        """
        Setup weak hierarchical coupling to SU(2)/core.
        
        Physical Meaning:
            Establishes the weak hierarchical coupling between
            the U(1)³ phase structure and SU(2)/core through
            invariant mixed terms.
        """
        su2_config = self.config.get("su2_coupling", {})
        coupling_strength = su2_config.get("coupling_strength", 0.1)
        
        # Create SU(2) coupling matrix (weak coupling)
        # This represents the invariant mixed terms between U(1)³ and SU(2)
        self.coupling_matrix = np.array([
            [1.0, coupling_strength, 0.0],
            [coupling_strength, 1.0, coupling_strength],
            [0.0, coupling_strength, 1.0]
        ], dtype=complex)
        
        # Add weak coupling terms
        self.su2_coupling_terms = {
            "theta_1_theta_2": coupling_strength * 0.1,
            "theta_2_theta_3": coupling_strength * 0.1,
            "theta_1_theta_3": coupling_strength * 0.05,  # Weaker coupling
        }
    
    def _setup_electroweak_coefficients(self) -> None:
        """
        Setup electroweak coupling coefficients.
        
        Physical Meaning:
            Initializes the coefficients for electroweak currents
            that are generated as functionals of the envelope.
        """
        electroweak_config = self.config.get("electroweak", {})
        
        self.electroweak_coefficients = {
            "em_coupling": electroweak_config.get("em_coupling", 1.0),
            "weak_coupling": electroweak_config.get("weak_coupling", 0.1),
            "mixing_angle": electroweak_config.get("mixing_angle", 0.23),  # Weinberg angle
            "gauge_coupling": electroweak_config.get("gauge_coupling", 0.65),
        }
    
    def get_phase_components(self) -> List[np.ndarray]:
        """
        Get the three U(1) phase components Θ_a (a=1..3).
        
        Physical Meaning:
            Returns the three independent U(1) phase components
            that form the U(1)³ structure.
            
        Returns:
            List[np.ndarray]: List of three phase components Θ_a.
        """
        return self.theta_components.copy()
    
    def get_total_phase(self) -> np.ndarray:
        """
        Get the total phase from U(1)³ structure.
        
        Physical Meaning:
            Computes the total phase by combining the three
            U(1) components with proper SU(2) coupling.
            
        Mathematical Foundation:
            Θ_total = Σ_a Θ_a + Σ_{a,b} g_{ab} Θ_a Θ_b
            where g_{ab} are the SU(2) coupling coefficients.
            
        Returns:
            np.ndarray: Total phase field.
        """
        # Start with sum of individual components
        total_phase = np.zeros_like(self.theta_components[0])
        
        for theta_a in self.theta_components:
            total_phase += theta_a
        
        # Add SU(2) coupling terms
        for i, theta_i in enumerate(self.theta_components):
            for j, theta_j in enumerate(self.theta_components):
                if i != j:
                    coupling_strength = self.coupling_matrix[i, j]
                    total_phase += coupling_strength * theta_i * theta_j
        
        return total_phase
    
    def compute_electroweak_currents(self, envelope: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute electroweak currents as functionals of the envelope.
        
        Physical Meaning:
            Computes electromagnetic and weak currents that are
            generated as functionals of the BVP envelope through
            the U(1)³ phase structure.
            
        Mathematical Foundation:
            J_EM = g_EM * |A|² * ∇Θ_EM
            J_weak = g_weak * |A|⁴ * ∇Θ_weak
            where Θ_EM and Θ_weak are combinations of Θ_a components.
            
        Args:
            envelope (np.ndarray): BVP envelope |A|.
            
        Returns:
            Dict[str, np.ndarray]: Electroweak currents including:
                - em_current: Electromagnetic current
                - weak_current: Weak interaction current
                - mixed_current: Mixed electroweak current
        """
        # Compute phase gradients
        phase_gradients = []
        for theta_a in self.theta_components:
            if self.domain.dimensions == 1:
                grad_theta = np.gradient(theta_a, self.domain.dx)
            elif self.domain.dimensions == 2:
                grad_x, grad_y = np.gradient(theta_a, self.domain.dx, self.domain.dx)
                grad_theta = np.sqrt(grad_x**2 + grad_y**2)
            else:  # 3D
                grad_x, grad_y, grad_z = np.gradient(theta_a, self.domain.dx, self.domain.dx, self.domain.dx)
                grad_theta = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            phase_gradients.append(grad_theta)
        
        # Electromagnetic current (primarily from Θ₁)
        em_gradient = phase_gradients[0]  # Primary EM component
        em_current = (
            self.electroweak_coefficients["em_coupling"] * 
            envelope**2 * 
            em_gradient
        )
        
        # Weak current (primarily from Θ₂ and Θ₃)
        weak_gradient = phase_gradients[1] + phase_gradients[2]  # Weak components
        weak_current = (
            self.electroweak_coefficients["weak_coupling"] * 
            envelope**4 * 
            weak_gradient
        )
        
        # Mixed electroweak current (Weinberg mixing)
        mixing_angle = self.electroweak_coefficients["mixing_angle"]
        mixed_current = (
            self.electroweak_coefficients["gauge_coupling"] * 
            envelope**3 * 
            (np.cos(mixing_angle) * em_gradient + np.sin(mixing_angle) * weak_gradient)
        )
        
        return {
            "em_current": em_current,
            "weak_current": weak_current,
            "mixed_current": mixed_current,
        }
    
    def compute_phase_coherence(self) -> np.ndarray:
        """
        Compute phase coherence measure.
        
        Physical Meaning:
            Computes a measure of phase coherence across the
            U(1)³ structure, indicating the degree of
            synchronization between the three phase components.
            
        Mathematical Foundation:
            Coherence = |Σ_a exp(iΘ_a)| / 3
            where the magnitude indicates coherence strength.
            
        Returns:
            np.ndarray: Phase coherence measure.
        """
        # Sum of complex exponentials
        coherence_sum = np.zeros_like(self.theta_components[0])
        
        for theta_a in self.theta_components:
            coherence_sum += np.exp(1j * np.angle(theta_a))
        
        # Normalize by number of components
        coherence = np.abs(coherence_sum) / 3.0
        
        return coherence
    
    def get_su2_coupling_strength(self) -> float:
        """
        Get the SU(2) coupling strength.
        
        Physical Meaning:
            Returns the strength of the weak hierarchical
            coupling to SU(2)/core.
            
        Returns:
            float: SU(2) coupling strength.
        """
        return np.abs(self.coupling_matrix[0, 1])  # Off-diagonal element
    
    def set_su2_coupling_strength(self, strength: float) -> None:
        """
        Set the SU(2) coupling strength.
        
        Physical Meaning:
            Updates the strength of the weak hierarchical
            coupling to SU(2)/core.
            
        Args:
            strength (float): New SU(2) coupling strength.
        """
        # Update coupling matrix
        self.coupling_matrix[0, 1] = strength
        self.coupling_matrix[1, 0] = strength
        self.coupling_matrix[1, 2] = strength
        self.coupling_matrix[2, 1] = strength
        
        # Update coupling terms
        self.su2_coupling_terms["theta_1_theta_2"] = strength * 0.1
        self.su2_coupling_terms["theta_2_theta_3"] = strength * 0.1
        self.su2_coupling_terms["theta_1_theta_3"] = strength * 0.05
    
    def __repr__(self) -> str:
        """String representation of phase vector."""
        coupling_strength = self.get_su2_coupling_strength()
        return (
            f"PhaseVector(domain={self.domain}, "
            f"su2_coupling={coupling_strength:.3f}, "
            f"em_coupling={self.electroweak_coefficients['em_coupling']:.3f})"
        )
