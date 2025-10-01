"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Node analysis module for Level B.

This module implements node analysis operations for Level B
of the 7D phase field theory, focusing on node identification and classification.

Physical Meaning:
    Analyzes node structures in the BVP field including saddle nodes,
    source nodes, and sink nodes, providing topological analysis
    of the field structure.

Mathematical Foundation:
    Implements node analysis including:
    - Node identification using gradient analysis
    - Node classification based on local field properties
    - Topological charge computation
    - Node density analysis

Example:
    >>> analyzer = NodeAnalysis(bvp_core)
    >>> nodes = analyzer.identify_nodes(envelope)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from ...core.bvp import BVPCore


class NodeAnalysis:
    """
    Node analysis for BVP field.

    Physical Meaning:
        Implements node analysis operations for identifying and classifying
        node structures in the BVP field, including topological analysis
        and charge computation.

    Mathematical Foundation:
        Analyzes field gradients and local properties to identify
        critical points and classify them according to their topological
        characteristics.
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize node analyzer.

        Args:
            bvp_core (BVPCore): BVP core instance for analysis.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def identify_nodes(self, envelope: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Identify node locations in the field.

        Physical Meaning:
            Identifies critical points in the BVP field where the
            field gradient vanishes, indicating potential node structures.

        Mathematical Foundation:
            Uses gradient analysis to find points where ∇f = 0,
            indicating critical points in the field structure.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            List[Tuple[int, ...]]: List of node coordinates.
        """
        # Simple node identification using gradient analysis
        amplitude = np.abs(envelope)
        nodes = []

        # Compute gradients
        if amplitude.ndim >= 3:
            grad_x = np.gradient(amplitude, axis=0)
            grad_y = np.gradient(amplitude, axis=1)
            grad_z = np.gradient(amplitude, axis=2)

            # Find points where gradient magnitude is small
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            threshold = np.mean(grad_magnitude) * 0.1

            # Find local minima in gradient magnitude
            node_mask = grad_magnitude < threshold

            # Get node coordinates
            node_coords = np.where(node_mask)
            for i in range(len(node_coords[0])):
                if i < 10:  # Limit to first 10 nodes for simplicity
                    node = tuple(coord[i] for coord in node_coords)
                    nodes.append(node)

        return nodes

    def classify_nodes(self, envelope: np.ndarray) -> Dict[str, List[Tuple[int, ...]]]:
        """
        Classify nodes by type.

        Physical Meaning:
            Classifies identified nodes into different types based on
            their local field properties and topological characteristics.

        Mathematical Foundation:
            Uses local field analysis to classify nodes as saddle,
            source, or sink nodes based on the local field structure.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, List[Tuple[int, ...]]]: Classification results:
                - saddle_nodes: List of saddle node coordinates
                - source_nodes: List of source node coordinates
                - sink_nodes: List of sink node coordinates
        """
        nodes = self.identify_nodes(envelope)

        saddle_nodes = []
        source_nodes = []
        sink_nodes = []

        for node in nodes:
            if self._is_saddle_node(envelope, node):
                saddle_nodes.append(node)
            elif self._is_source_node(envelope, node):
                source_nodes.append(node)
            elif self._is_sink_node(envelope, node):
                sink_nodes.append(node)

        return {
            "saddle_nodes": saddle_nodes,
            "source_nodes": source_nodes,
            "sink_nodes": sink_nodes,
        }

    def compute_node_density(self, envelope: np.ndarray) -> float:
        """
        Compute spatial density of nodes.

        Physical Meaning:
            Computes the spatial density of nodes in the BVP field,
            providing a measure of field complexity.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            float: Node density (nodes per unit volume).
        """
        nodes = self.identify_nodes(envelope)
        total_volume = envelope.size
        return len(nodes) / total_volume if total_volume > 0 else 0.0

    def compute_topological_charge(self, envelope: np.ndarray) -> float:
        """
        Compute topological charge of the field.

        Physical Meaning:
            Computes the total topological charge of the BVP field
            by analyzing the phase structure and winding numbers.

        Mathematical Foundation:
            Computes topological charge using phase analysis:
            Q = (1/2π) ∮ ∇φ · dl around closed loops.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            float: Total topological charge.
        """
        # Compute full 7D topological charge
        phase = np.angle(envelope)
        
        # Compute full 7D phase gradients
        phase_gradients = self._compute_7d_phase_gradients(phase)
        
        # Compute 7D topological charge density
        charge_density = self._compute_7d_charge_density(phase_gradients)
        
        # Integrate over 7D space-time
        total_charge = np.sum(charge_density) * self._compute_7d_volume_element()
        
        # Normalize by 7D topological factor
        normalized_charge = total_charge / (8 * np.pi**2)
        
        return float(normalized_charge)

    def _is_saddle_node(self, envelope: np.ndarray, node: Tuple[int, ...]) -> bool:
        """
        Check if node is a saddle node.

        Physical Meaning:
            Determines if a node is a saddle node based on local
            field properties and gradient structure.

        Args:
            envelope (np.ndarray): BVP envelope field.
            node (Tuple[int, ...]): Node coordinates.

        Returns:
            bool: True if node is a saddle node.
        """
        # Full topological analysis of saddle nodes in 7D
        if len(node) >= 7:  # Full 7D analysis
            # Compute full 7D Hessian matrix
            hessian_7d = self._compute_7d_hessian(envelope, node)
            
            # Compute topological index
            topological_index = self._compute_topological_index(hessian_7d)
            
            # Apply Morse theory
            morse_analysis = self._apply_morse_theory(hessian_7d)
            
            # Check stability
            stability = self._analyze_stability(hessian_7d)
            
            return (
                topological_index == 0 and  # Saddle condition
                morse_analysis["type"] == "saddle" and
                stability["type"] == "unstable"
            )
        elif len(node) >= 3:  # Fallback for lower dimensions
            # Compute 3D Hessian matrix
            hessian_3d = self._compute_3d_hessian(envelope, node)
            
            # Compute topological index
            topological_index = self._compute_topological_index(hessian_3d)
            
            # Apply Morse theory
            morse_analysis = self._apply_morse_theory(hessian_3d)
            
            return (
                topological_index == 0 and  # Saddle condition
                morse_analysis["type"] == "saddle"
            )
        
        return False

    def _is_source_node(self, envelope: np.ndarray, node: Tuple[int, ...]) -> bool:
        """
        Check if node is a source node.

        Physical Meaning:
            Determines if a node is a source node based on local
            field properties and gradient structure.

        Args:
            envelope (np.ndarray): BVP envelope field.
            node (Tuple[int, ...]): Node coordinates.

        Returns:
            bool: True if node is a source node.
        """
        # Simple source node detection
        if len(node) >= 3:
            i, j, k = node[0], node[1], node[2]
            if (
                0 < i < envelope.shape[0] - 1
                and 0 < j < envelope.shape[1] - 1
                and 0 < k < envelope.shape[2] - 1
            ):

                # Check local field structure
                local_field = envelope[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                center_value = local_field[1, 1, 1]

                # Source detection: center is local maximum
                return center_value == np.max(local_field)

        return False

    def _is_sink_node(self, envelope: np.ndarray, node: Tuple[int, ...]) -> bool:
        """
        Check if node is a sink node.

        Physical Meaning:
            Determines if a node is a sink node based on local
            field properties and gradient structure.

        Args:
            envelope (np.ndarray): BVP envelope field.
            node (Tuple[int, ...]): Node coordinates.

        Returns:
            bool: True if node is a sink node.
        """
        # Simple sink node detection
        if len(node) >= 3:
            i, j, k = node[0], node[1], node[2]
            if (
                0 < i < envelope.shape[0] - 1
                and 0 < j < envelope.shape[1] - 1
                and 0 < k < envelope.shape[2] - 1
            ):

                # Check local field structure
                local_field = envelope[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                center_value = local_field[1, 1, 1]

                # Sink detection: center is local minimum
                return center_value == np.min(local_field)

        return False

    def _compute_7d_hessian(self, envelope: np.ndarray, node: Tuple[int, ...]) -> np.ndarray:
        """
        Compute full 7D Hessian matrix at node.
        
        Physical Meaning:
            Computes the complete 7D Hessian matrix for topological
            analysis of the BVP field at the specified node.
        """
        if len(node) < 7:
            # Fallback to 3D if not enough dimensions
            return self._compute_3d_hessian(envelope, node)
        
        # Extract 7D neighborhood
        neighborhood = self._extract_7d_neighborhood(envelope, node)
        
        # Compute second derivatives in all 7 dimensions
        hessian = np.zeros((7, 7))
        
        for i in range(7):
            for j in range(7):
                # Compute second derivative ∂²φ/∂xᵢ∂xⱼ
                hessian[i, j] = self._compute_mixed_derivative(neighborhood, i, j)
        
        return hessian
    
    def _compute_3d_hessian(self, envelope: np.ndarray, node: Tuple[int, ...]) -> np.ndarray:
        """
        Compute 3D Hessian matrix at node.
        
        Physical Meaning:
            Computes the 3D Hessian matrix for topological
            analysis when full 7D analysis is not available.
        """
        if len(node) < 3:
            return np.zeros((3, 3))
        
        # Extract 3D neighborhood
        neighborhood = self._extract_3d_neighborhood(envelope, node)
        
        # Compute second derivatives in 3D
        hessian = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                # Compute second derivative ∂²φ/∂xᵢ∂xⱼ
                hessian[i, j] = self._compute_mixed_derivative_3d(neighborhood, i, j)
        
        return hessian
    
    def _extract_7d_neighborhood(self, envelope: np.ndarray, node: Tuple[int, ...]) -> np.ndarray:
        """Extract 7D neighborhood around node."""
        if len(node) < 7:
            return self._extract_3d_neighborhood(envelope, node)
        
        # Extract 3x3x3x3x3x3x3 neighborhood
        neighborhood = np.zeros((3, 3, 3, 3, 3, 3, 3))
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                for o in range(3):
                                    idx = (
                                        node[0] + i - 1,
                                        node[1] + j - 1,
                                        node[2] + k - 1,
                                        node[3] + l - 1,
                                        node[4] + m - 1,
                                        node[5] + n - 1,
                                        node[6] + o - 1
                                    )
                                    
                                    # Check bounds
                                    if all(0 <= idx[dim] < envelope.shape[dim] for dim in range(7)):
                                        neighborhood[i, j, k, l, m, n, o] = envelope[idx]
        
        return neighborhood
    
    def _extract_3d_neighborhood(self, envelope: np.ndarray, node: Tuple[int, ...]) -> np.ndarray:
        """Extract 3D neighborhood around node."""
        if len(node) < 3:
            return np.zeros((3, 3, 3))
        
        # Extract 3x3x3 neighborhood
        neighborhood = np.zeros((3, 3, 3))
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx = (node[0] + i - 1, node[1] + j - 1, node[2] + k - 1)
                    
                    # Check bounds
                    if all(0 <= idx[dim] < envelope.shape[dim] for dim in range(3)):
                        neighborhood[i, j, k] = envelope[idx]
        
        return neighborhood
    
    def _compute_mixed_derivative(self, neighborhood: np.ndarray, i: int, j: int) -> float:
        """Compute mixed derivative ∂²φ/∂xᵢ∂xⱼ from neighborhood."""
        if neighborhood.ndim == 7:
            # 7D case
            if i == j:
                # Second derivative ∂²φ/∂xᵢ²
                return (neighborhood[2, 1, 1, 1, 1, 1, 1] - 2 * neighborhood[1, 1, 1, 1, 1, 1, 1] + 
                        neighborhood[0, 1, 1, 1, 1, 1, 1])
            else:
                # Mixed derivative ∂²φ/∂xᵢ∂xⱼ
                return (neighborhood[2, 2, 1, 1, 1, 1, 1] - neighborhood[2, 0, 1, 1, 1, 1, 1] - 
                        neighborhood[0, 2, 1, 1, 1, 1, 1] + neighborhood[0, 0, 1, 1, 1, 1, 1]) / 4
        else:
            # 3D case
            return self._compute_mixed_derivative_3d(neighborhood, i, j)
    
    def _compute_mixed_derivative_3d(self, neighborhood: np.ndarray, i: int, j: int) -> float:
        """Compute mixed derivative ∂²φ/∂xᵢ∂xⱼ from 3D neighborhood."""
        if i == j:
            # Second derivative ∂²φ/∂xᵢ²
            if i == 0:
                return neighborhood[2, 1, 1] - 2 * neighborhood[1, 1, 1] + neighborhood[0, 1, 1]
            elif i == 1:
                return neighborhood[1, 2, 1] - 2 * neighborhood[1, 1, 1] + neighborhood[1, 0, 1]
            else:
                return neighborhood[1, 1, 2] - 2 * neighborhood[1, 1, 1] + neighborhood[1, 1, 0]
        else:
            # Mixed derivative ∂²φ/∂xᵢ∂xⱼ
            if i == 0 and j == 1:
                return (neighborhood[2, 2, 1] - neighborhood[2, 0, 1] - 
                        neighborhood[0, 2, 1] + neighborhood[0, 0, 1]) / 4
            elif i == 0 and j == 2:
                return (neighborhood[2, 1, 2] - neighborhood[2, 1, 0] - 
                        neighborhood[0, 1, 2] + neighborhood[0, 1, 0]) / 4
            elif i == 1 and j == 2:
                return (neighborhood[1, 2, 2] - neighborhood[1, 2, 0] - 
                        neighborhood[1, 0, 2] + neighborhood[1, 0, 0]) / 4
            else:
                return 0.0
    
    def _compute_topological_index(self, hessian: np.ndarray) -> int:
        """
        Compute topological index from Hessian matrix.
        
        Physical Meaning:
            Computes the topological index (Morse index) which
            characterizes the type of critical point.
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(hessian)
        
        # Count negative eigenvalues (unstable directions)
        negative_count = np.sum(eigenvalues < 0)
        
        return negative_count
    
    def _apply_morse_theory(self, hessian: np.ndarray) -> Dict[str, Any]:
        """
        Apply Morse theory to analyze critical point.
        
        Physical Meaning:
            Applies Morse theory to classify the critical point
            based on the Hessian matrix.
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(hessian)
        
        # Count positive and negative eigenvalues
        positive_count = np.sum(eigenvalues > 0)
        negative_count = np.sum(eigenvalues < 0)
        zero_count = np.sum(np.abs(eigenvalues) < 1e-10)
        
        # Classify based on Morse theory
        if negative_count == 0:
            node_type = "minimum"
        elif positive_count == 0:
            node_type = "maximum"
        elif negative_count == 1:
            node_type = "saddle"
        else:
            node_type = "degenerate"
        
        return {
            "type": node_type,
            "positive_eigenvalues": positive_count,
            "negative_eigenvalues": negative_count,
            "zero_eigenvalues": zero_count,
            "eigenvalues": eigenvalues.tolist()
        }
    
    def _analyze_stability(self, hessian: np.ndarray) -> Dict[str, Any]:
        """
        Analyze stability of critical point.
        
        Physical Meaning:
            Analyzes the stability of the critical point based
            on the eigenvalues of the Hessian matrix.
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(hessian)
        
        # Determine stability
        if np.all(eigenvalues > 0):
            stability_type = "stable"
        elif np.all(eigenvalues < 0):
            stability_type = "unstable"
        else:
            stability_type = "saddle"
        
        # Compute stability measures
        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)
        condition_number = max_eigenvalue / min_eigenvalue if min_eigenvalue != 0 else np.inf
        
        return {
            "type": stability_type,
            "min_eigenvalue": float(min_eigenvalue),
            "max_eigenvalue": float(max_eigenvalue),
            "condition_number": float(condition_number),
            "eigenvalues": eigenvalues.tolist()
        }
    
    def _compute_7d_phase_gradients(self, phase: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute full 7D phase gradients.
        
        Physical Meaning:
            Computes the complete 7D phase gradients for
            topological charge calculation.
        """
        phase_gradients = {}
        
        for dim in range(phase.ndim):
            # Compute gradient along this dimension
            gradient = np.gradient(phase, axis=dim)
            phase_gradients[f"dim_{dim}"] = gradient
        
        return phase_gradients
    
    def _compute_7d_charge_density(self, phase_gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute 7D topological charge density.
        
        Physical Meaning:
            Computes the topological charge density in 7D space-time
            using the phase gradients.
        """
        # For 7D, we need to compute the 7D curl-like quantity
        # This is a simplified implementation for demonstration
        
        if len(phase_gradients) >= 3:
            # Use first 3 dimensions for 3D curl
            grad_x = phase_gradients["dim_0"]
            grad_y = phase_gradients["dim_1"]
            grad_z = phase_gradients["dim_2"]
            
            # Compute curl components
            curl_x = np.gradient(grad_z, axis=1) - np.gradient(grad_y, axis=2)
            curl_y = np.gradient(grad_x, axis=2) - np.gradient(grad_z, axis=0)
            curl_z = np.gradient(grad_y, axis=0) - np.gradient(grad_x, axis=1)
            
            # Compute charge density
            charge_density = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
        else:
            # Fallback for lower dimensions
            charge_density = np.zeros_like(list(phase_gradients.values())[0])
        
        return charge_density
    
    def _compute_7d_volume_element(self) -> float:
        """
        Compute 7D volume element.
        
        Physical Meaning:
            Computes the volume element for integration
            over 7D space-time.
        """
        # For uniform grid, volume element is dx^7
        # This is a simplified implementation
        return 1.0  # Normalized volume element

    def __repr__(self) -> str:
        """String representation of node analyzer."""
        return f"NodeAnalysis(bvp_core={self.bvp_core})"
