"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

U(1)³ Phase Structure Postulate implementation for BVP framework.

This module implements Postulate 4 of the BVP framework, which states that
BVP has U(1)³ phase structure with phase vector Θ_a (a=1..3) and
phase coherence is maintained across the field.

Theoretical Background:
    The U(1)³ phase structure represents three independent phase degrees
    of freedom in the BVP field. Phase coherence ensures that phase
    relationships are maintained across spatial and temporal scales.

Example:
    >>> postulate = U1PhaseStructurePostulate(domain, constants)
    >>> results = postulate.apply(envelope)
"""

import numpy as np
from typing import Dict, Any, List
from ..domain.domain import Domain
from .bvp_constants import BVPConstants
from .bvp_postulate_base import BVPPostulate


class U1PhaseStructurePostulate(BVPPostulate):
    """
    Postulate 4: U(1)³ Phase Structure.
    
    Physical Meaning:
        BVP has U(1)³ phase structure with phase vector Θ_a (a=1..3)
        and phase coherence is maintained across the field.
    """
    
    def __init__(self, domain: Domain, constants: BVPConstants):
        """
        Initialize U(1)³ phase structure postulate.
        
        Physical Meaning:
            Sets up the postulate with domain and constants for
            analyzing U(1)³ phase structure properties.
            
        Args:
            domain (Domain): Computational domain for analysis.
            constants (BVPConstants): BVP physical constants.
        """
        self.domain = domain
        self.constants = constants
        self.phase_coherence_threshold = constants.get_quench_parameter("phase_coherence_threshold", 0.8)
        self.phase_variance_threshold = constants.get_quench_parameter("phase_variance_threshold", 0.1)
    
    def apply(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Apply U(1)³ phase structure postulate.
        
        Physical Meaning:
            Verifies that BVP field exhibits U(1)³ phase structure
            with proper phase coherence and phase vector properties.
            
        Mathematical Foundation:
            Analyzes phase components Θ_a (a=1..3) and their
            coherence properties across the field.
            
        Args:
            envelope (np.ndarray): BVP envelope to analyze.
            
        Returns:
            Dict[str, Any]: Results including phase structure analysis,
                phase coherence, and U(1)³ validation.
        """
        # Analyze phase structure
        phase_structure = self._analyze_phase_structure(envelope)
        
        # Analyze phase coherence
        phase_coherence = self._analyze_phase_coherence(envelope)
        
        # Check U(1)³ properties
        u1_properties = self._check_u1_properties(phase_structure, phase_coherence)
        
        # Validate U(1)³ phase structure
        satisfies_postulate = self._validate_u1_phase_structure(u1_properties)
        
        return {
            "phase_structure": phase_structure,
            "phase_coherence": phase_coherence,
            "u1_properties": u1_properties,
            "satisfies_postulate": satisfies_postulate,
            "postulate_satisfied": satisfies_postulate
        }
    
    def _analyze_phase_structure(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze U(1)³ phase structure of the field.
        
        Physical Meaning:
            Extracts and analyzes the three phase components Θ_a (a=1..3)
            from the complex envelope field.
            
        Mathematical Foundation:
            Envelope A = |A|e^(iΘ) where Θ = Σ_a Θ_a represents
            the total phase with three independent components.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            Dict[str, Any]: Phase structure analysis.
        """
        # Extract phase from complex envelope
        total_phase = np.angle(envelope)
        
        # Decompose into three U(1) components
        phase_components = self._decompose_phase_components(total_phase)
        
        # Analyze phase statistics
        phase_stats = self._compute_phase_statistics(phase_components)
        
        return {
            "total_phase": total_phase,
            "phase_components": phase_components,
            "phase_statistics": phase_stats
        }
    
    def _decompose_phase_components(self, total_phase: np.ndarray) -> List[np.ndarray]:
        """
        Decompose total phase into three U(1) components.
        
        Physical Meaning:
            Separates total phase into three independent U(1)
            phase components using spatial frequency analysis.
            
        Args:
            total_phase (np.ndarray): Total phase field.
            
        Returns:
            List[np.ndarray]: Three phase components.
        """
        # Use spatial FFT to decompose phase
        phase_fft = np.fft.fftn(total_phase)
        
        # Create three frequency bands
        shape = total_phase.shape
        phase_components = []
        
        for i in range(3):
            # Create frequency mask for each component
            freq_mask = self._create_frequency_mask(shape, i)
            
            # Extract component in frequency space
            component_fft = phase_fft * freq_mask
            
            # Transform back to real space
            component = np.fft.ifftn(component_fft).real
            phase_components.append(component)
        
        return phase_components
    
    def _create_frequency_mask(self, shape: tuple, component_idx: int) -> np.ndarray:
        """
        Create frequency mask for phase component extraction.
        
        Physical Meaning:
            Creates frequency domain mask to separate different
            phase components based on spatial frequencies.
            
        Args:
            shape (tuple): Field shape.
            component_idx (int): Component index (0, 1, 2).
            
        Returns:
            np.ndarray: Frequency mask.
        """
        # Create frequency axes
        freq_axes = []
        for i, size in enumerate(shape[:3]):  # Spatial dimensions only
            freq_axis = np.fft.fftfreq(size, self.domain.dx)
            freq_axes.append(freq_axis)
        
        # Create frequency grid
        freq_grid = np.meshgrid(*freq_axes, indexing='ij')
        freq_magnitude = np.sqrt(sum(f**2 for f in freq_grid))
        
        # Create frequency bands
        max_freq = np.max(freq_magnitude)
        band_width = max_freq / 3
        
        # Create mask for this component
        freq_mask = np.zeros_like(freq_magnitude)
        lower_bound = component_idx * band_width
        upper_bound = (component_idx + 1) * band_width
        
        freq_mask[(freq_magnitude >= lower_bound) & (freq_magnitude < upper_bound)] = 1.0
        
        return freq_mask
    
    def _compute_phase_statistics(self, phase_components: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute statistics for phase components.
        
        Physical Meaning:
            Calculates statistical properties of each phase
            component to characterize U(1)³ structure.
            
        Args:
            phase_components (List[np.ndarray]): Three phase components.
            
        Returns:
            Dict[str, Any]: Phase statistics.
        """
        phase_stats = {}
        
        for i, component in enumerate(phase_components):
            # Compute component statistics
            mean_phase = np.mean(component)
            std_phase = np.std(component)
            phase_variance = np.var(component)
            
            # Check phase wrapping
            phase_range = np.max(component) - np.min(component)
            is_wrapped = phase_range > np.pi
            
            phase_stats[f"component_{i}"] = {
                "mean_phase": mean_phase,
                "std_phase": std_phase,
                "phase_variance": phase_variance,
                "phase_range": phase_range,
                "is_wrapped": is_wrapped
            }
        
        return phase_stats
    
    def _analyze_phase_coherence(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze phase coherence across the field.
        
        Physical Meaning:
            Computes phase coherence measures to verify that
            phase relationships are maintained across spatial scales.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            Dict[str, Any]: Phase coherence analysis.
        """
        # Extract phase
        phase = np.angle(envelope)
        
        # Compute phase coherence measures
        local_coherence = self._compute_local_phase_coherence(phase)
        global_coherence = self._compute_global_phase_coherence(phase)
        
        # Compute coherence statistics
        mean_local_coherence = np.mean(local_coherence)
        std_local_coherence = np.std(local_coherence)
        
        return {
            "local_coherence": local_coherence,
            "global_coherence": global_coherence,
            "mean_local_coherence": mean_local_coherence,
            "std_local_coherence": std_local_coherence
        }
    
    def _compute_local_phase_coherence(self, phase: np.ndarray) -> np.ndarray:
        """
        Compute local phase coherence.
        
        Physical Meaning:
            Calculates phase coherence in local neighborhoods
            to measure phase consistency.
            
        Args:
            phase (np.ndarray): Phase field.
            
        Returns:
            np.ndarray: Local coherence field.
        """
        # Compute phase gradients
        phase_gradients = []
        for axis in range(3):  # Spatial dimensions only
            gradient = np.gradient(phase, self.domain.dx, axis=axis)
            phase_gradients.append(gradient)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sum(g**2 for g in phase_gradients))
        
        # Local coherence is inverse of gradient magnitude
        local_coherence = 1.0 / (gradient_magnitude + 1e-12)
        
        # Normalize
        local_coherence = local_coherence / np.max(local_coherence)
        
        return local_coherence
    
    def _compute_global_phase_coherence(self, phase: np.ndarray) -> float:
        """
        Compute global phase coherence.
        
        Physical Meaning:
            Calculates overall phase coherence across the
            entire field domain.
            
        Args:
            phase (np.ndarray): Phase field.
            
        Returns:
            float: Global coherence measure.
        """
        # Compute phase variance
        phase_variance = np.var(phase)
        
        # Global coherence is inverse of variance
        global_coherence = 1.0 / (phase_variance + 1e-12)
        
        # Normalize
        global_coherence = min(global_coherence, 1.0)
        
        return global_coherence
    
    def _check_u1_properties(self, phase_structure: Dict[str, Any], 
                           phase_coherence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check U(1)³ properties of the field.
        
        Physical Meaning:
            Verifies that field exhibits proper U(1)³ phase
            structure with adequate coherence.
            
        Args:
            phase_structure (Dict[str, Any]): Phase structure analysis.
            phase_coherence (Dict[str, Any]): Phase coherence analysis.
            
        Returns:
            Dict[str, Any]: U(1)³ properties.
        """
        # Check phase component independence
        phase_stats = phase_structure["phase_statistics"]
        independent_components = True
        
        for i in range(3):
            component_stats = phase_stats[f"component_{i}"]
            phase_variance = component_stats["phase_variance"]
            
            # Check if component has sufficient variance
            if phase_variance < self.phase_variance_threshold:
                independent_components = False
                break
        
        # Check phase coherence
        mean_local_coherence = phase_coherence["mean_local_coherence"]
        global_coherence = phase_coherence["global_coherence"]
        
        adequate_coherence = (mean_local_coherence > self.phase_coherence_threshold and
                            global_coherence > self.phase_coherence_threshold)
        
        # Overall U(1)³ properties
        has_u1_structure = independent_components and adequate_coherence
        
        return {
            "independent_components": independent_components,
            "adequate_coherence": adequate_coherence,
            "has_u1_structure": has_u1_structure,
            "structure_quality": (mean_local_coherence + global_coherence) / 2
        }
    
    def _validate_u1_phase_structure(self, u1_properties: Dict[str, Any]) -> bool:
        """
        Validate U(1)³ phase structure postulate.
        
        Physical Meaning:
            Checks that field exhibits proper U(1)³ phase
            structure for BVP framework validity.
            
        Args:
            u1_properties (Dict[str, Any]): U(1)³ properties.
            
        Returns:
            bool: True if U(1)³ phase structure is satisfied.
        """
        return u1_properties["has_u1_structure"]
