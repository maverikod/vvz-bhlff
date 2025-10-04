"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Topological charge analyzer for BVP framework.

This module implements comprehensive topological charge analysis for the
7D BVP field, including winding number computation, defect identification,
and topological characterization according to the theoretical framework.

Physical Meaning:
    Analyzes topological charge in the BVP field, identifying
    topological defects and their properties according to the
    theoretical framework.

Mathematical Foundation:
    Implements topological charge analysis with proper winding
    number computation and defect characterization for 7D phase field theory.

Example:
    >>> analyzer = TopologicalChargeAnalyzer(domain, config)
    >>> results = analyzer.compute_topological_charge(field)
    >>> print(f"Total charge: {results['topological_charge']}")
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.ndimage import label, center_of_mass

# CUDA optimization imports
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = np

from ..domain import Domain
from .bvp_constants import BVPConstants
from .memory_decorator import memory_protected_class_method
from .topological_defect_analyzer import TopologicalDefectAnalyzer


class TopologicalChargeAnalyzer:
    """
    Analyzer for topological charge in BVP field.
    
    Physical Meaning:
        Computes the topological charge of the BVP field,
        identifying topological defects and their properties
        according to the theoretical framework.
    
    Mathematical Foundation:
        Implements topological charge analysis with proper winding
        number computation and defect characterization for 7D phase field theory.
    """
    
    def __init__(self, domain: Domain, config: Dict[str, Any], constants: BVPConstants = None):
        """
        Initialize topological charge analyzer.
        
        Physical Meaning:
            Sets up the topological charge analyzer with the computational domain
            and configuration parameters for analyzing topological defects
            in the BVP field.
        
        Args:
            domain (Domain): Computational domain for analysis.
            config (Dict[str, Any]): Analysis configuration including:
                - charge_threshold: Threshold for significant charge
                - defect_size: Minimum size for defect detection
                - winding_precision: Precision for winding number computation
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.domain = domain
        self.config = config
        self.constants = constants or BVPConstants(config)
        self._setup_analysis_parameters()
    
    def _setup_analysis_parameters(self) -> None:
        """
        Setup analysis parameters.
        
        Physical Meaning:
            Initializes parameters for topological charge analysis based on
            the domain properties and configuration.
        """
        # Topological analysis parameters
        self.charge_threshold = self.config.get("charge_threshold", 0.1)
        self.winding_precision = self.config.get("winding_precision", 1e-6)
        
        # Analysis precision
        self.min_charge = self.config.get("min_charge", 0.01)
        self.max_charge = self.config.get("max_charge", 10.0)
        self.stability_threshold = self.config.get("stability_threshold", 0.8)
        
        # Initialize defect analyzer
        self.defect_analyzer = TopologicalDefectAnalyzer(domain, config, constants)
    
    @memory_protected_class_method(memory_threshold=0.8, shape_param='field', dtype_param='field')
    def compute_topological_charge(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Compute topological charge of the field.
        
        Physical Meaning:
            Computes the topological charge using the winding
            number formula for the 7D BVP field.
        
        Mathematical Foundation:
            Q = (1/2π) ∮ ∇φ · dl
            where φ is the phase field and the integral is over
            a closed loop in the field.
        
        Args:
            field (np.ndarray): BVP field for analysis.
        
        Returns:
            Dict[str, Any]: Analysis results including:
                - topological_charge: Total topological charge
                - charge_locations: List of charge locations
                - charge_stability: Stability measure of charges
                - defect_analysis: Detailed defect analysis
        """
        # Convert to complex field for phase analysis
        if np.iscomplexobj(field):
            complex_field = field
        else:
            complex_field = field.astype(complex)
        
        # Compute phase field
        phase = np.angle(complex_field)
        
        # Find topological defects
        defects = self.defect_analyzer.find_topological_defects(phase)
        
        # Compute topological charge for each defect
        charges = []
        charge_locations = []
        
        for defect in defects:
            charge = self._compute_defect_charge(phase, defect)
            if abs(charge) > self.min_charge:
                charges.append(charge)
                charge_locations.append(defect)
        
        # Compute total topological charge
        total_charge = sum(charges)
        
        # Compute charge stability
        charge_stability = self._compute_charge_stability(charges, charge_locations)
        
        # Analyze defects
        defect_analysis = self._analyze_defects(phase, charge_locations, charges)
        
        return {
            'topological_charge': float(total_charge),
            'charge_locations': charge_locations,
            'charge_stability': float(charge_stability),
            'defect_analysis': defect_analysis,
            'individual_charges': charges
        }
    
    
    def _compute_defect_charge(self, phase: np.ndarray, defect_location: Tuple[int, ...]) -> float:
        """
        Compute topological charge around a defect with CUDA optimization.
        
        Physical Meaning:
            Computes the winding number around a topological defect
            using the circulation of phase gradients with CUDA acceleration.
        
        Mathematical Foundation:
            Q = (1/2π) ∮ ∇φ · dl
            where the integral is over a small loop around the defect.
        
        Args:
            phase (np.ndarray): Phase field.
            defect_location (Tuple[int, ...]): Location of the defect.
        
        Returns:
            float: Topological charge of the defect.
        """
        # Use CUDA if available
        if CUDA_AVAILABLE:
            return self._compute_defect_charge_cuda(phase, defect_location)
        else:
            return self._compute_defect_charge_cpu(phase, defect_location)
    
    def _compute_defect_charge_cuda(self, phase: np.ndarray, defect_location: Tuple[int, ...]) -> float:
        """
        Compute topological charge using CUDA acceleration.
        
        Physical Meaning:
            CUDA-accelerated computation of topological charge
            using vectorized operations on GPU.
        """
        try:
            # Move data to GPU
            phase_gpu = cp.asarray(phase)
            
            # Create a small loop around the defect
            loop_radius = 2
            
            # Extract neighborhood around defect
            slices = []
            for i, coord in enumerate(defect_location):
                start = max(0, coord - loop_radius)
                end = min(phase.shape[i], coord + loop_radius + 1)
                slices.append(slice(start, end))
            
            neighborhood_gpu = phase_gpu[tuple(slices)]
            
            # Compute circulation around the loop using CUDA
            circulation = 0.0
            
            # For 2D case (most common for topological defects)
            if neighborhood_gpu.ndim >= 2:
                # Extract boundary of the neighborhood using CUDA
                boundary_phase = cp.concatenate([
                    neighborhood_gpu[0, :],      # Top edge
                    neighborhood_gpu[-1, :],     # Bottom edge
                    neighborhood_gpu[:, 0],      # Left edge
                    neighborhood_gpu[:, -1]      # Right edge
                ])
                
                # Compute phase differences around the boundary using CUDA
                phase_diffs = cp.diff(boundary_phase)
                
                # Handle phase wrapping using CUDA
                phase_diffs = cp.unwrap(phase_diffs)
                
                # Total circulation using CUDA reduction
                circulation = cp.sum(phase_diffs).get()  # Move back to CPU
            
            # Convert to topological charge
            charge = circulation / (2 * np.pi)
            
            return float(charge)
            
        except Exception:
            # Fallback to CPU if CUDA fails
            return self._compute_defect_charge_cpu(phase, defect_location)
    
    def _compute_defect_charge_cpu(self, phase: np.ndarray, defect_location: Tuple[int, ...]) -> float:
        """
        Compute topological charge using CPU with vectorized operations.
        
        Physical Meaning:
            CPU-optimized computation of topological charge
            using vectorized NumPy operations.
        """
        # Create a small loop around the defect
        loop_radius = 2
        
        # Extract neighborhood around defect
        slices = []
        for i, coord in enumerate(defect_location):
            start = max(0, coord - loop_radius)
            end = min(phase.shape[i], coord + loop_radius + 1)
            slices.append(slice(start, end))
        
        neighborhood = phase[tuple(slices)]
        
        # Compute circulation around the loop
        circulation = 0.0
        
        # For 2D case (most common for topological defects)
        if neighborhood.ndim >= 2:
            try:
                # Extract boundary of the neighborhood using vectorized operations
                boundary_phase = np.concatenate([
                    neighborhood[0, :],      # Top edge
                    neighborhood[-1, :],     # Bottom edge
                    neighborhood[:, 0],      # Left edge
                    neighborhood[:, -1]      # Right edge
                ])
                
                # Compute phase differences around the boundary using vectorized operations
                phase_diffs = np.diff(boundary_phase)
                
                # Handle phase wrapping using vectorized operations
                phase_diffs = np.unwrap(phase_diffs)
                
                # Total circulation using vectorized reduction
                circulation = np.sum(phase_diffs)
                
            except (IndexError, ValueError):
                circulation = 0.0
        
        # Convert to topological charge
        charge = circulation / (2 * np.pi)
        
        return float(charge)
    
    def _compute_charge_stability(self, charges: List[float], locations: List[Tuple[int, ...]]) -> float:
        """
        Compute stability of topological charges.
        
        Physical Meaning:
            Computes a measure of how stable the topological charges are
            based on their magnitudes and spatial distribution.
        
        Args:
            charges (List[float]): List of individual charges.
            locations (List[Tuple[int, ...]]): List of charge locations.
        
        Returns:
            float: Charge stability measure (0.0 to 1.0).
        """
        if not charges:
            return 0.0
        
        # Stability based on charge magnitudes
        charge_magnitudes = [abs(charge) for charge in charges]
        magnitude_stability = min(1.0, np.mean(charge_magnitudes))
        
        # Stability based on spatial distribution
        if len(locations) > 1:
            # Compute distances between charges
            distances = []
            for i in range(len(locations)):
                for j in range(i + 1, len(locations)):
                    dist = np.sqrt(sum((a - b)**2 for a, b in zip(locations[i], locations[j])))
                    distances.append(dist)
            
            # Stability based on charge separation
            if distances:
                avg_distance = np.mean(distances)
                spatial_stability = min(1.0, avg_distance / 10.0)  # Normalize by domain size
            else:
                spatial_stability = 0.0
        else:
            spatial_stability = 1.0
        
        # Combined stability
        stability = (magnitude_stability + spatial_stability) / 2.0
        
        return stability
    
    def _analyze_defects(self, phase: np.ndarray, charge_locations: List[Tuple[int, ...]], 
                        charges: List[float]) -> Dict[str, Any]:
        """
        Analyze topological defects in detail.
        
        Physical Meaning:
            Performs detailed analysis of topological defects including
            their types, strengths, and interactions.
        
        Args:
            phase (np.ndarray): Phase field.
            charge_locations (List[Tuple[int, ...]]): Charge locations.
            charges (List[float]): Individual charges.
        
        Returns:
            Dict[str, Any]: Detailed defect analysis.
        """
        if not charge_locations:
            return {
                'defect_count': 0,
                'defect_types': [],
                'defect_strengths': [],
                'defect_interactions': []
            }
        
        # Analyze defect types
        defect_types = []
        for charge in charges:
            if charge > 0:
                defect_types.append('positive')
            elif charge < 0:
                defect_types.append('negative')
            else:
                defect_types.append('neutral')
        
        # Analyze defect strengths
        defect_strengths = [abs(charge) for charge in charges]
        
        # Analyze defect interactions
        defect_interactions = []
        if len(charge_locations) > 1:
            for i in range(len(charge_locations)):
                for j in range(i + 1, len(charge_locations)):
                    # Compute interaction strength
                    dist = np.sqrt(sum((a - b)**2 for a, b in zip(charge_locations[i], charge_locations[j])))
                    interaction = charges[i] * charges[j] / (dist + 1e-6)  # Avoid division by zero
                    defect_interactions.append(interaction)
        
        return {
            'defect_count': len(charge_locations),
            'defect_types': defect_types,
            'defect_strengths': defect_strengths,
            'defect_interactions': defect_interactions,
            'total_positive_charge': sum(charge for charge in charges if charge > 0),
            'total_negative_charge': sum(charge for charge in charges if charge < 0)
        }
    
    def analyze_phase_structure(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Analyze phase structure of the field.
        
        Physical Meaning:
            Analyzes the phase structure of the BVP field to understand
            the topological characteristics and phase coherence.
        
        Args:
            field (np.ndarray): BVP field for analysis.
        
        Returns:
            Dict[str, Any]: Phase structure analysis.
        """
        # Convert to complex field for phase analysis
        if np.iscomplexobj(field):
            complex_field = field
        else:
            complex_field = field.astype(complex)
        
        # Compute phase field
        phase = np.angle(complex_field)
        amplitude = np.abs(complex_field)
        
        # Compute phase gradients
        gradients = []
        for i in range(phase.ndim):
            grad = np.gradient(phase, axis=i)
            gradients.append(grad)
        
        # Compute phase gradient magnitude
        grad_magnitude = np.sqrt(sum(grad**2 for grad in gradients))
        
        # Analyze phase coherence
        phase_coherence = np.mean(np.cos(phase))
        phase_variance = np.var(phase)
        
        # Analyze phase gradient statistics
        grad_mean = np.mean(grad_magnitude)
        grad_std = np.std(grad_magnitude)
        grad_max = np.max(grad_magnitude)
        
        # Find regions of high phase gradient (potential defects)
        high_grad_threshold = grad_mean + 2 * grad_std
        high_grad_regions = grad_magnitude > high_grad_threshold
        high_grad_fraction = np.sum(high_grad_regions) / high_grad_regions.size
        
        return {
            'phase_coherence': float(phase_coherence),
            'phase_variance': float(phase_variance),
            'gradient_mean': float(grad_mean),
            'gradient_std': float(grad_std),
            'gradient_max': float(grad_max),
            'high_gradient_fraction': float(high_grad_fraction)
        }
    
    def get_analysis_parameters(self) -> Dict[str, Any]:
        """
        Get current analysis parameters.
        
        Physical Meaning:
            Returns the current parameters used for topological charge analysis.
        
        Returns:
            Dict[str, Any]: Analysis parameters.
        """
        return {
            'charge_threshold': self.charge_threshold,
            'defect_size': self.defect_size,
            'winding_precision': self.winding_precision,
            'min_charge': self.min_charge,
            'max_charge': self.max_charge,
            'stability_threshold': self.stability_threshold
        }
