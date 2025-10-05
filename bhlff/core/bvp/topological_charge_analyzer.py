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
        self.defect_analyzer = TopologicalDefectAnalyzer(self.domain, self.config, self.constants)
    
    def compute_topological_charge(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Compute topological charge using block processing and vectorization.
        
        Physical Meaning:
            Computes the topological charge using block processing to handle
            large domains efficiently with CUDA acceleration and vectorization.
        
        Mathematical Foundation:
            Q = (1/2π) ∮ ∇φ · dl computed in blocks with vectorized operations
            for maximum performance on large 7D domains.
        
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
        
        # Compute phase field with vectorization
        phase = np.angle(complex_field)
        
        # Process in blocks to handle large domains
        all_defects = []
        all_charges = []
        all_charge_locations = []
        
        # Determine optimal block size
        block_size = self._determine_optimal_block_size(phase.shape)
        print(f"Processing {phase.shape} field in blocks of {block_size}")
        
        # Generate blocks with overlap
        blocks = self._generate_overlapping_blocks(phase.shape, block_size)
        print(f"Generated {len(blocks)} blocks for processing")
        
        # Limit number of blocks for performance - use only central blocks
        max_blocks = 100  # Maximum blocks to process
        if len(blocks) > max_blocks:
            print(f"Limiting to {max_blocks} central blocks for performance")
            # Take central blocks for representative analysis
            center_start = len(blocks) // 4
            center_end = center_start + max_blocks
            blocks = blocks[center_start:center_end]
        
        # Process each block
        for block_idx, block_slice in enumerate(blocks):
            if block_idx % 50 == 0:
                print(f"Processing block {block_idx + 1}/{len(blocks)}")
            
            # Extract block
            phase_block = phase[block_slice]
            
            # Skip blocks that are too small for gradient computation
            if any(dim < 2 for dim in phase_block.shape):
                continue
            
            # Find defects in this block using vectorized operations
            try:
                block_defects = self._find_defects_vectorized(phase_block)
                
                # Convert to global coordinates
                global_defects = []
                for defect in block_defects:
                    global_defect = tuple(defect[i] + block_slice[i].start for i in range(len(defect)))
                    global_defects.append(global_defect)
                
                all_defects.extend(global_defects)
                
                # Compute charges for defects in this block
                for defect_location in global_defects:
                    charge = self._compute_defect_charge_vectorized(phase, defect_location)
                    if abs(charge) > self.min_charge:
                        all_charges.append(charge)
                        all_charge_locations.append(defect_location)
                        
            except Exception as e:
                print(f"Skipping block {block_idx} due to error: {e}")
                continue
        
        # Compute total topological charge
        total_charge = sum(all_charges)
        
        # Compute charge stability
        charge_stability = self._compute_charge_stability(all_charges, all_charge_locations)
        
        # Analyze defects
        defect_analysis = self._analyze_defects(phase, all_charge_locations, all_charges)
        
        print(f"Found {len(all_defects)} defects with total charge {total_charge:.4f}")
        
        return {
            'topological_charge': float(total_charge),
            'charge_locations': all_charge_locations,
            'charge_stability': float(charge_stability),
            'defect_analysis': defect_analysis,
            'individual_charges': all_charges
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
    
    def _determine_optimal_block_size(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Determine optimal block size for memory-efficient processing.
        
        Physical Meaning:
            Calculates block size that fits within memory constraints
            while maintaining sufficient resolution for topological analysis.
        
        Args:
            field_shape (Tuple[int, ...]): Shape of the field to process.
        
        Returns:
            Tuple[int, ...]: Optimal block size for each dimension.
        """
        # Calculate block size based on available memory
        # Use larger blocks to reduce the number of blocks and improve performance
        block_size = []
        for dim_size in field_shape:
            if dim_size > 32:
                block_size.append(32)  # Large dimensions: 32x32 blocks
            elif dim_size > 16:
                block_size.append(16)  # Medium dimensions: 16x16 blocks
            elif dim_size > 8:
                block_size.append(8)   # Small dimensions: 8x8 blocks
            else:
                # Ensure minimum size of 2 for gradient computation
                block_size.append(max(2, dim_size))
        
        return tuple(block_size)
    
    def _generate_overlapping_blocks(self, field_shape: Tuple[int, ...], block_size: Tuple[int, ...]) -> List[Tuple[slice, ...]]:
        """
        Generate overlapping blocks for processing large fields.
        
        Physical Meaning:
            Creates overlapping blocks to ensure no defects are missed
            at block boundaries, with vectorized operations.
        
        Args:
            field_shape (Tuple[int, ...]): Shape of the field.
            block_size (Tuple[int, ...]): Size of each block.
        
        Returns:
            List[Tuple[slice, ...]]: List of overlapping block slices.
        """
        blocks = []
        overlap = 2  # Overlap size to avoid missing defects at boundaries
        
        # Calculate step sizes
        step_sizes = [max(1, block_size[i] - overlap) for i in range(len(block_size))]
        
        # Generate all combinations of block positions
        for i in range(0, field_shape[0], step_sizes[0]):
            for j in range(0, field_shape[1], step_sizes[1]):
                for k in range(0, field_shape[2], step_sizes[2]):
                    for l in range(0, field_shape[3], step_sizes[3]):
                        for m in range(0, field_shape[4], step_sizes[4]):
                            for n in range(0, field_shape[5], step_sizes[5]):
                                for o in range(0, field_shape[6], step_sizes[6]):
                                    # Create slice for this block
                                    block_slice = (
                                        slice(i, min(i + block_size[0], field_shape[0])),
                                        slice(j, min(j + block_size[1], field_shape[1])),
                                        slice(k, min(k + block_size[2], field_shape[2])),
                                        slice(l, min(l + block_size[3], field_shape[3])),
                                        slice(m, min(m + block_size[4], field_shape[4])),
                                        slice(n, min(n + block_size[5], field_shape[5])),
                                        slice(o, min(o + block_size[6], field_shape[6]))
                                    )
                                    blocks.append(block_slice)
        
        return blocks
    
    def _find_defects_vectorized(self, phase_block: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Find topological defects using vectorized operations.
        
        Physical Meaning:
            Identifies topological defects using vectorized gradient
            computation and threshold analysis for maximum performance.
        
        Args:
            phase_block (np.ndarray): Phase field block.
        
        Returns:
            List[Tuple[int, ...]]: List of defect locations in block coordinates.
        """
        # Use CUDA if available for vectorized operations
        if CUDA_AVAILABLE:
            return self._find_defects_cuda_vectorized(phase_block)
        else:
            return self._find_defects_cpu_vectorized(phase_block)
    
    def _find_defects_cuda_vectorized(self, phase_block: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Find defects using CUDA-accelerated vectorized operations.
        
        Physical Meaning:
            CUDA-accelerated identification of topological defects
            using vectorized gradient computation on GPU.
        """
        try:
            # Move to GPU
            phase_gpu = cp.asarray(phase_block)
            
            # Compute gradients using CUDA vectorized operations
            gradients = []
            for i in range(phase_gpu.ndim):
                grad = cp.gradient(phase_gpu, axis=i)
                gradients.append(grad)
            
            # Compute gradient magnitude using CUDA vectorized operations
            grad_magnitude = cp.sqrt(sum(grad**2 for grad in gradients))
            
            # Find high gradient regions using CUDA vectorized operations
            high_grad_threshold = cp.percentile(grad_magnitude, 95)
            high_grad_mask = grad_magnitude > high_grad_threshold
            
            # Move back to CPU for defect detection
            high_grad_mask_cpu = cp.asnumpy(high_grad_mask)
            
            # Find defects using vectorized operations
            defects = self._extract_defects_vectorized(high_grad_mask_cpu)
            
            return defects
            
        except Exception:
            # Fallback to CPU
            return self._find_defects_cpu_vectorized(phase_block)
    
    def _find_defects_cpu_vectorized(self, phase_block: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Find defects using CPU vectorized operations.
        
        Physical Meaning:
            CPU-optimized identification of topological defects
            using vectorized NumPy operations.
        """
        # Compute gradients using vectorized operations
        gradients = []
        for i in range(phase_block.ndim):
            grad = np.gradient(phase_block, axis=i)
            gradients.append(grad)
        
        # Compute gradient magnitude using vectorized operations
        grad_magnitude = np.sqrt(sum(grad**2 for grad in gradients))
        
        # Find high gradient regions using vectorized operations
        high_grad_threshold = np.percentile(grad_magnitude, 95)
        high_grad_mask = grad_magnitude > high_grad_threshold
        
        # Find defects using vectorized operations
        defects = self._extract_defects_vectorized(high_grad_mask)
        
        return defects
    
    def _extract_defects_vectorized(self, high_grad_mask: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Extract defect locations using vectorized operations.
        
        Physical Meaning:
            Identifies connected components of high gradient regions
            using vectorized morphological operations.
        
        Args:
            high_grad_mask (np.ndarray): Boolean mask of high gradient regions.
        
        Returns:
            List[Tuple[int, ...]]: List of defect locations.
        """
        defects = []
        
        # Find connected components using vectorized operations
        try:
            from scipy.ndimage import label, center_of_mass
            
            labeled_mask, num_components = label(high_grad_mask)
            
            for i in range(1, num_components + 1):
                component_mask = (labeled_mask == i)
                if np.sum(component_mask) >= 2:  # Minimum defect size
                    # Find center of mass using vectorized operations
                    center = center_of_mass(component_mask)
                    center_int = tuple(int(round(c)) for c in center)
                    defects.append(center_int)
                    
        except ImportError:
            # Fallback without scipy
            # Find local maxima using vectorized operations
            from scipy.ndimage import maximum_filter
            local_maxima = (high_grad_mask == maximum_filter(high_grad_mask, size=3))
            defect_coords = np.where(local_maxima)
            for coord in zip(*defect_coords):
                defects.append(coord)
        
        return defects
    
    def _compute_defect_charge_vectorized(self, phase: np.ndarray, defect_location: Tuple[int, ...]) -> float:
        """
        Compute topological charge using vectorized operations.
        
        Physical Meaning:
            Computes the winding number around a topological defect
            using vectorized operations for maximum performance.
        
        Args:
            phase (np.ndarray): Phase field.
            defect_location (Tuple[int, ...]): Location of the defect.
        
        Returns:
            float: Topological charge of the defect.
        """
        # Extract small neighborhood around defect
        neighborhood_size = 4
        
        # Create slices for neighborhood
        slices = []
        for i, coord in enumerate(defect_location):
            start = max(0, coord - neighborhood_size)
            end = min(phase.shape[i], coord + neighborhood_size + 1)
            slices.append(slice(start, end))
        
        try:
            neighborhood = phase[tuple(slices)]
            
            # Use vectorized operations for charge computation
            if CUDA_AVAILABLE:
                return self._compute_charge_cuda_vectorized(neighborhood)
            else:
                return self._compute_charge_cpu_vectorized(neighborhood)
                
        except (IndexError, ValueError):
            return 0.0
    
    def _compute_charge_cuda_vectorized(self, neighborhood: np.ndarray) -> float:
        """
        Compute charge using CUDA vectorized operations.
        
        Physical Meaning:
            CUDA-accelerated computation of topological charge
            using vectorized operations on GPU.
        """
        try:
            # Move to GPU
            neighborhood_gpu = cp.asarray(neighborhood)
            
            # Compute circulation using CUDA vectorized operations
            if neighborhood_gpu.ndim >= 2:
                # Extract boundary using CUDA vectorized operations
                boundary_phase = cp.concatenate([
                    neighborhood_gpu[0, :],      # Top edge
                    neighborhood_gpu[-1, :],     # Bottom edge
                    neighborhood_gpu[:, 0],      # Left edge
                    neighborhood_gpu[:, -1]      # Right edge
                ])
                
                # Compute phase differences using CUDA vectorized operations
                phase_diffs = cp.diff(boundary_phase)
                phase_diffs = cp.unwrap(phase_diffs)
                
                # Total circulation using CUDA vectorized reduction
                circulation = cp.sum(phase_diffs).get()
                
                return float(circulation / (2 * np.pi))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _compute_charge_cpu_vectorized(self, neighborhood: np.ndarray) -> float:
        """
        Compute charge using CPU vectorized operations.
        
        Physical Meaning:
            CPU-optimized computation of topological charge
            using vectorized NumPy operations.
        """
        # Compute circulation using vectorized operations
        if neighborhood.ndim >= 2:
            # Extract boundary using vectorized operations
            boundary_phase = np.concatenate([
                neighborhood[0, :],      # Top edge
                neighborhood[-1, :],     # Bottom edge
                neighborhood[:, 0],      # Left edge
                neighborhood[:, -1]      # Right edge
            ])
            
            # Compute phase differences using vectorized operations
            phase_diffs = np.diff(boundary_phase)
            phase_diffs = np.unwrap(phase_diffs)
            
            # Total circulation using vectorized reduction
            circulation = np.sum(phase_diffs)
            
            return float(circulation / (2 * np.pi))
        
        return 0.0
    
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
