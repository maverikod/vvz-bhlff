"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

U(1)³ phase field implementation for 7D BVP theory.

This module implements the U(1)³ phase field structure according to the theory,
where the field a(x,φ,t) ∈ ℂ³ is a 3-component complex vector representing
the phase structure in 7D space-time.

Physical Meaning:
    Implements the U(1)³ phase field a(x,φ,t) ∈ ℂ³ where:
    - Each component represents a different phase degree of freedom
    - The field lives in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    - Each component has U(1) gauge symmetry
    - The field represents the fundamental phase structure of matter

Mathematical Foundation:
    The U(1)³ phase field has the structure:
    a(x,φ,t) = (a₁(x,φ,t), a₂(x,φ,t), a₃(x,φ,t))
    where each aᵢ(x,φ,t) ∈ ℂ is a complex scalar field with U(1) symmetry.

Example:
    >>> phase_field = U1PhaseField(domain, initial_amplitudes, initial_phases)
    >>> field_value = phase_field.get_field_at_point(x, phi, t)
    >>> phase_coherence = phase_field.compute_phase_coherence()
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..domain.domain_7d_bvp import Domain7DBVP


class U1PhaseField:
    """
    U(1)³ phase field implementation.
    
    Physical Meaning:
        Represents the U(1)³ phase field a(x,φ,t) ∈ ℂ³ in 7D space-time,
        where each component has U(1) gauge symmetry and represents
        a different phase degree of freedom.
        
    Mathematical Foundation:
        The field has the structure:
        a(x,φ,t) = (a₁(x,φ,t), a₂(x,φ,t), a₃(x,φ,t))
        where each aᵢ(x,φ,t) ∈ ℂ is a complex scalar field.
        
    Attributes:
        domain (Domain7DBVP): 7D BVP computational domain.
        field_components (List[np.ndarray]): List of 3 complex field components.
        amplitudes (List[np.ndarray]): Amplitude of each component |aᵢ|.
        phases (List[np.ndarray]): Phase of each component arg(aᵢ).
    """
    
    def __init__(self, domain: 'Domain7DBVP', 
                 initial_amplitudes: Optional[List[np.ndarray]] = None,
                 initial_phases: Optional[List[np.ndarray]] = None):
        """
        Initialize U(1)³ phase field.
        
        Physical Meaning:
            Creates a U(1)³ phase field with specified initial conditions
            for amplitudes and phases of each component.
            
        Args:
            domain (Domain7DBVP): 7D BVP computational domain.
            initial_amplitudes (Optional[List[np.ndarray]]): Initial amplitudes for each component.
            initial_phases (Optional[List[np.ndarray]]): Initial phases for each component.
        """
        self.domain = domain
        self.logger = logging.getLogger(__name__)
        
        # Initialize field components
        self.field_components = []
        self.amplitudes = []
        self.phases = []
        
        # Create initial field components
        for i in range(3):
            if initial_amplitudes is not None and i < len(initial_amplitudes):
                amplitude = initial_amplitudes[i]
            else:
                # Default: random amplitude
                amplitude = np.random.randn(*domain.shape).astype(np.complex128) * 0.1
            
            if initial_phases is not None and i < len(initial_phases):
                phase = initial_phases[i]
            else:
                # Default: random phase
                phase = np.random.uniform(0, 2*np.pi, domain.shape)
            
            # Create complex field component
            field_component = amplitude * np.exp(1j * phase)
            
            self.field_components.append(field_component)
            self.amplitudes.append(np.abs(field_component))
            self.phases.append(np.angle(field_component))
        
        self.logger.info(f"U1PhaseField initialized with {len(self.field_components)} components")
    
    def get_field_at_point(self, x: Tuple[int, int, int], 
                          phi: Tuple[int, int, int], 
                          t: int) -> np.ndarray:
        """
        Get field value at specific point.
        
        Physical Meaning:
            Returns the U(1)³ phase field value at a specific point
            in 7D space-time.
            
        Args:
            x (Tuple[int, int, int]): Spatial coordinates (x, y, z).
            phi (Tuple[int, int, int]): Phase coordinates (φ₁, φ₂, φ₃).
            t (int): Time coordinate.
            
        Returns:
            np.ndarray: 3-component complex field vector.
        """
        field_vector = np.zeros(3, dtype=np.complex128)
        
        for i, component in enumerate(self.field_components):
            field_vector[i] = component[x[0], x[1], x[2], phi[0], phi[1], phi[2], t]
        
        return field_vector
    
    def set_field_at_point(self, x: Tuple[int, int, int], 
                          phi: Tuple[int, int, int], 
                          t: int, 
                          field_vector: np.ndarray) -> None:
        """
        Set field value at specific point.
        
        Physical Meaning:
            Sets the U(1)³ phase field value at a specific point
            in 7D space-time.
            
        Args:
            x (Tuple[int, int, int]): Spatial coordinates (x, y, z).
            phi (Tuple[int, int, int]): Phase coordinates (φ₁, φ₂, φ₃).
            t (int): Time coordinate.
            field_vector (np.ndarray): 3-component complex field vector.
        """
        for i, component in enumerate(self.field_components):
            component[x[0], x[1], x[2], phi[0], phi[1], phi[2], t] = field_vector[i]
            # Update amplitude and phase
            self.amplitudes[i][x[0], x[1], x[2], phi[0], phi[1], phi[2], t] = np.abs(field_vector[i])
            self.phases[i][x[0], x[1], x[2], phi[0], phi[1], phi[2], t] = np.angle(field_vector[i])
    
    def compute_phase_coherence(self) -> np.ndarray:
        """
        Compute phase coherence for each component.
        
        Physical Meaning:
            Computes the phase coherence for each U(1) component,
            measuring the uniformity of phase distribution.
            
        Mathematical Foundation:
            Phase coherence = |⟨e^(iφ)⟩| where ⟨⟩ is spatial average
            and φ is the phase of each component.
            
        Returns:
            np.ndarray: Array of 3 coherence values, one for each component.
        """
        coherence = np.zeros(3)
        
        for i, phase in enumerate(self.phases):
            # Compute complex phase
            complex_phase = np.exp(1j * phase)
            
            # Compute spatial average
            spatial_average = np.mean(complex_phase)
            
            # Coherence is the magnitude of the average
            coherence[i] = np.abs(spatial_average)
        
        return coherence
    
    def compute_amplitude_distribution(self) -> Dict[str, np.ndarray]:
        """
        Compute amplitude distribution statistics.
        
        Physical Meaning:
            Computes statistical properties of the amplitude distribution
            for each component of the U(1)³ field.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with amplitude statistics.
        """
        stats = {
            'mean': np.zeros(3),
            'std': np.zeros(3),
            'min': np.zeros(3),
            'max': np.zeros(3)
        }
        
        for i, amplitude in enumerate(self.amplitudes):
            stats['mean'][i] = np.mean(amplitude)
            stats['std'][i] = np.std(amplitude)
            stats['min'][i] = np.min(amplitude)
            stats['max'][i] = np.max(amplitude)
        
        return stats
    
    def apply_gauge_transformation(self, gauge_function: List[np.ndarray]) -> None:
        """
        Apply U(1) gauge transformation to each component.
        
        Physical Meaning:
            Applies a U(1) gauge transformation to each component of the field:
            aᵢ(x,φ,t) → aᵢ(x,φ,t) * e^(iαᵢ(x,φ,t))
            where αᵢ(x,φ,t) is the gauge function for component i.
            
        Mathematical Foundation:
            U(1) gauge transformation: aᵢ → aᵢ * e^(iαᵢ)
            where αᵢ is the gauge function.
            
        Args:
            gauge_function (List[np.ndarray]): List of 3 gauge functions αᵢ(x,φ,t).
        """
        for i, (component, gauge) in enumerate(zip(self.field_components, gauge_function)):
            # Apply gauge transformation
            self.field_components[i] = component * np.exp(1j * gauge)
            
            # Update amplitude and phase
            self.amplitudes[i] = np.abs(self.field_components[i])
            self.phases[i] = np.angle(self.field_components[i])
        
        self.logger.info("U(1) gauge transformation applied to all components")
    
    def compute_field_norm(self) -> np.ndarray:
        """
        Compute field norm at each point.
        
        Physical Meaning:
            Computes the norm of the U(1)³ field at each point in space-time:
            |a(x,φ,t)| = √(|a₁|² + |a₂|² + |a₃|²)
            
        Returns:
            np.ndarray: Field norm at each point.
        """
        norm_squared = np.zeros(self.domain.shape)
        
        for amplitude in self.amplitudes:
            norm_squared += amplitude**2
        
        return np.sqrt(norm_squared)
    
    def get_field_component(self, index: int) -> np.ndarray:
        """
        Get specific field component.
        
        Args:
            index (int): Component index (0, 1, or 2).
            
        Returns:
            np.ndarray: Field component.
        """
        if 0 <= index < 3:
            return self.field_components[index]
        else:
            raise ValueError(f"Component index must be 0, 1, or 2, got {index}")
    
    def set_field_component(self, index: int, component: np.ndarray) -> None:
        """
        Set specific field component.
        
        Args:
            index (int): Component index (0, 1, or 2).
            component (np.ndarray): New field component.
        """
        if 0 <= index < 3:
            self.field_components[index] = component
            self.amplitudes[index] = np.abs(component)
            self.phases[index] = np.angle(component)
        else:
            raise ValueError(f"Component index must be 0, 1, or 2, got {index}")
    
    def __repr__(self) -> str:
        """String representation of U(1)³ phase field."""
        coherence = self.compute_phase_coherence()
        return (f"U1PhaseField("
                f"domain={self.domain.shape}, "
                f"coherence={coherence}, "
                f"components={len(self.field_components)})")
