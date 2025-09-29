"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

U(1)³ phase vector structure for BVP.

This module implements the main PhaseVector class that coordinates
the three U(1) phase components and electroweak coupling for the
Base High-Frequency Field.

Physical Meaning:
    Implements the three-component phase vector Θ_a (a=1..3)
    that represents the fundamental phase structure of the BVP field.
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
from typing import Dict, Any, List

from ...domain import Domain
from ..bvp_constants import BVPConstants
from .phase_components import PhaseComponents
from .electroweak_coupling import ElectroweakCoupling


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
        _phase_components (PhaseComponents): Phase components manager.
        _electroweak_coupling (ElectroweakCoupling): Electroweak coupling.
        coupling_matrix (np.ndarray): SU(2) coupling matrix.
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
        
        # Initialize components
        self._phase_components = PhaseComponents(domain, config)
        self._electroweak_coupling = ElectroweakCoupling(config)
        
        # Setup SU(2) coupling
        self._setup_su2_coupling()
    
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
    
    def get_phase_components(self) -> List[np.ndarray]:
        """
        Get the three U(1) phase components Θ_a (a=1..3).
        
        Physical Meaning:
            Returns the three independent U(1) phase components
            that form the U(1)³ structure.
            
        Returns:
            List[np.ndarray]: List of three phase components Θ_a.
        """
        return self._phase_components.get_components()
    
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
        return self._phase_components.get_total_phase(self.coupling_matrix)
    
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
        phase_components = self._phase_components.get_components()
        return self._electroweak_coupling.compute_electroweak_currents(
            envelope, phase_components, self.domain
        )
    
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
        return self._phase_components.compute_phase_coherence()
    
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
    
    def update_phase_components(self, envelope: np.ndarray) -> None:
        """
        Update phase components from solved envelope.
        
        Physical Meaning:
            Updates the three U(1) phase components Θ_a (a=1..3)
            from the solved BVP envelope field.
            
        Mathematical Foundation:
            Extracts phase components from the envelope solution
            and updates the U(1)³ phase structure.
            
        Args:
            envelope (np.ndarray): Solved BVP envelope in 7D space-time.
        """
        self._phase_components.update_components(envelope)
    
    def get_electroweak_coefficients(self) -> Dict[str, float]:
        """
        Get electroweak coupling coefficients.
        
        Physical Meaning:
            Returns the current electroweak coupling coefficients
            used for current calculations.
            
        Returns:
            Dict[str, float]: Electroweak coupling coefficients.
        """
        return self._electroweak_coupling.get_electroweak_coefficients()
    
    def set_electroweak_coefficients(self, coefficients: Dict[str, float]) -> None:
        """
        Set electroweak coupling coefficients.
        
        Physical Meaning:
            Updates the electroweak coupling coefficients
            used for current calculations.
            
        Args:
            coefficients (Dict[str, float]): New coupling coefficients.
        """
        self._electroweak_coupling.set_electroweak_coefficients(coefficients)
    
    def __repr__(self) -> str:
        """String representation of phase vector."""
        coupling_strength = self.get_su2_coupling_strength()
        em_coupling = self.get_electroweak_coefficients()['em_coupling']
        return (
            f"PhaseVector(domain={self.domain}, "
            f"su2_coupling={coupling_strength:.3f}, "
            f"em_coupling={em_coupling:.3f})"
        )
