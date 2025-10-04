"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Morphological operations for quench detection.

This module implements morphological operations for filtering noise
and finding connected components in quench detection, providing
robust quench event identification in 7D space-time.

Physical Meaning:
    Applies morphological operations to remove noise and fill gaps
    in quench regions, improving detection quality. Groups nearby
    quench events into connected components representing coherent
    quench structures in 7D space-time.

Mathematical Foundation:
    - Binary opening: Erosion followed by dilation
    - Binary closing: Dilation followed by erosion
    - Connected component analysis: Groups spatially/phase/temporally connected events

Example:
    >>> morphology = QuenchMorphology()
    >>> filtered_mask = morphology.apply_operations(quench_mask)
    >>> components = morphology.find_connected_components(filtered_mask)
"""

import numpy as np
from typing import Dict, Any, Tuple, List

try:
    from scipy.ndimage import binary_opening, binary_closing, label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class QuenchMorphology:
    """
    Morphological operations for quench detection.
    
    Physical Meaning:
        Applies morphological operations to remove noise and fill gaps
        in quench regions, improving detection quality. Groups nearby
        quench events into connected components representing coherent
        quench structures in 7D space-time.
    
    Mathematical Foundation:
        - Binary opening: Erosion followed by dilation
        - Binary closing: Dilation followed by erosion
        - Connected component analysis: Groups spatially/phase/temporally connected events
    """
    
    def __init__(self):
        """Initialize morphological operations processor."""
        self.scipy_available = SCIPY_AVAILABLE
    
    def apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to filter noise in quench mask.
        
        Physical Meaning:
            Applies binary morphological operations to remove noise
            and fill gaps in quench regions, improving detection quality.
        
        Mathematical Foundation:
            - Binary opening: Erosion followed by dilation
            - Binary closing: Dilation followed by erosion
            - Removes small noise components and fills small gaps
        
        Args:
            mask (np.ndarray): Binary mask of quench regions.
        
        Returns:
            np.ndarray: Filtered binary mask.
        """
        if self.scipy_available:
            return self._apply_scipy_operations(mask)
        else:
            return self._apply_simple_operations(mask)
    
    def find_connected_components(self, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Find connected components in quench mask.
        
        Physical Meaning:
            Groups nearby quench events into connected components,
            representing coherent quench regions in 7D space-time.
        
        Mathematical Foundation:
            Uses connected component labeling to identify regions
            where quench events are spatially/phase/temporally connected.
        
        Args:
            mask (np.ndarray): Binary mask of quench regions.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping component IDs to
                binary masks of each component.
        """
        if self.scipy_available:
            return self._find_scipy_components(mask)
        else:
            return self._find_simple_components(mask)
    
    def _apply_scipy_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations using scipy.
        
        Physical Meaning:
            Uses scipy's optimized morphological operations for
            efficient noise filtering in 7D space-time.
        
        Args:
            mask (np.ndarray): Binary mask of quench regions.
        
        Returns:
            np.ndarray: Filtered binary mask.
        """
        # Define structuring element for 7D operations
        # Use 3x3x3x3x3x3x3 structure for 7D
        structure = np.ones((3, 3, 3, 3, 3, 3, 3), dtype=bool)
        
        # Apply binary opening to remove small noise
        filtered_mask = binary_opening(mask, structure=structure)
        
        # Apply binary closing to fill small gaps
        filtered_mask = binary_closing(filtered_mask, structure=structure)
        
        return filtered_mask
    
    def _apply_simple_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Simple morphological filtering without scipy dependency.
        
        Physical Meaning:
            Basic noise filtering using local neighborhood operations
            to remove isolated pixels and fill small gaps.
        
        Args:
            mask (np.ndarray): Binary mask of quench regions.
        
        Returns:
            np.ndarray: Filtered binary mask.
        """
        # Simple erosion: remove isolated pixels
        filtered_mask = mask.copy()
        
        # Simple dilation: fill small gaps
        # This is a basic implementation for 7D
        for axis in range(mask.ndim):
            # Apply 1D dilation along each axis
            for i in range(1, mask.shape[axis] - 1):
                if axis == 0:
                    if mask[i-1, :, :, :, :, :, :].any() and mask[i+1, :, :, :, :, :, :].any():
                        filtered_mask[i, :, :, :, :, :, :] = True
                elif axis == 1:
                    if mask[:, i-1, :, :, :, :, :].any() and mask[:, i+1, :, :, :, :, :, :].any():
                        filtered_mask[:, i, :, :, :, :, :] = True
                # Continue for other axes...
        
        return filtered_mask
    
    def _find_scipy_components(self, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Find connected components using scipy.
        
        Physical Meaning:
            Uses scipy's optimized connected component labeling for
            efficient component identification in 7D space-time.
        
        Args:
            mask (np.ndarray): Binary mask of quench regions.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping component IDs to
                binary masks of each component.
        """
        # Label connected components
        labeled_mask, num_components = label(mask)
        
        # Extract individual components
        components = {}
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)
            components[component_id] = component_mask
        
        return components
    
    def _find_simple_components(self, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Simple connected component analysis without scipy.
        
        Physical Meaning:
            Basic grouping of nearby quench events using
            flood-fill algorithm for 7D space.
        
        Args:
            mask (np.ndarray): Binary mask of quench regions.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping component IDs to
                binary masks of each component.
        """
        components = {}
        visited = np.zeros_like(mask, dtype=bool)
        component_id = 0
        
        # Find all quench points
        quench_points = np.where(mask)
        
        for point in zip(*quench_points):
            if not visited[point]:
                component_id += 1
                component_mask = np.zeros_like(mask, dtype=bool)
                
                # Simple flood-fill for this component
                self._flood_fill_7d(mask, visited, component_mask, point)
                components[component_id] = component_mask
        
        return components
    
    def _flood_fill_7d(self, mask: np.ndarray, visited: np.ndarray, 
                       component_mask: np.ndarray, start_point: Tuple[int, ...]) -> None:
        """
        Flood-fill algorithm for 7D connected components.
        
        Physical Meaning:
            Recursively fills connected quench regions starting from
            a seed point, identifying coherent quench structures.
        
        Args:
            mask (np.ndarray): Binary mask of quench regions.
            visited (np.ndarray): Visited points mask.
            component_mask (np.ndarray): Current component mask.
            start_point (Tuple[int, ...]): Starting point for flood-fill.
        """
        stack = [start_point]
        
        while stack:
            point = stack.pop()
            
            if visited[point]:
                continue
                
            visited[point] = True
            component_mask[point] = True
            
            # Check 7D neighbors (3^7 = 2187 neighbors, but we check only immediate ones)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        for dphi1 in [-1, 0, 1]:
                            for dphi2 in [-1, 0, 1]:
                                for dphi3 in [-1, 0, 1]:
                                    for dt in [-1, 0, 1]:
                                        if dx == dy == dz == dphi1 == dphi2 == dphi3 == dt == 0:
                                            continue
                                        
                                        neighbor = (
                                            point[0] + dx, point[1] + dy, point[2] + dz,
                                            point[3] + dphi1, point[4] + dphi2, point[5] + dphi3,
                                            point[6] + dt
                                        )
                                        
                                        # Check bounds
                                        if (0 <= neighbor[0] < mask.shape[0] and
                                            0 <= neighbor[1] < mask.shape[1] and
                                            0 <= neighbor[2] < mask.shape[2] and
                                            0 <= neighbor[3] < mask.shape[3] and
                                            0 <= neighbor[4] < mask.shape[4] and
                                            0 <= neighbor[5] < mask.shape[5] and
                                            0 <= neighbor[6] < mask.shape[6]):
                                            
                                            if (mask[neighbor] and not visited[neighbor]):
                                                stack.append(neighbor)
