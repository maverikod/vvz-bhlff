"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Zone analysis module for Level B.

This module implements zone analysis operations for Level B
of the 7D phase field theory, focusing on zone identification and classification.

Physical Meaning:
    Analyzes zone separation in the BVP field including core, transition,
    and tail regions, providing spatial analysis of field structure.

Mathematical Foundation:
    Implements zone analysis including:
    - Zone boundary identification
    - Zone classification based on field properties
    - Zone property analysis
    - Transition region identification

Example:
    >>> analyzer = ZoneAnalysis(bvp_core)
    >>> zones = analyzer.identify_zone_boundaries(envelope)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from ...core.bvp import BVPCore


class ZoneAnalysis:
    """
    Zone analysis for BVP field.

    Physical Meaning:
        Implements zone analysis operations for identifying and analyzing
        different zones in the BVP field including core, transition,
        and tail regions.

    Mathematical Foundation:
        Analyzes spatial field properties to identify zones with
        different characteristics and transition regions between them.
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize zone analyzer.

        Args:
            bvp_core (BVPCore): BVP core instance for analysis.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def identify_zone_boundaries(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify boundaries between different zones.

        Physical Meaning:
            Identifies boundaries between different zones in the BVP field
            based on field properties and spatial gradients.

        Mathematical Foundation:
            Uses gradient analysis and field property thresholds
            to identify transition regions between zones.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            List[Dict[str, Any]]: List of zone boundaries with properties:
                - boundary_type: Type of boundary (core-transition, transition-tail)
                - boundary_location: Location of the boundary
                - boundary_strength: Strength of the boundary
        """
        boundaries = []

        # Analyze field amplitude
        amplitude = np.abs(envelope)

        # Level set analysis for boundary detection
        level_sets = self._compute_level_sets(amplitude)
        
        # Phase field method for boundary evolution
        phase_field_boundaries = self._compute_phase_field_boundaries(amplitude)
        
        # Topological analysis of boundaries
        topological_boundaries = self._analyze_boundary_topology(amplitude)
        
        # Energy landscape analysis
        energy_landscape = self._compute_energy_landscape(amplitude)
        
        # Combine all boundary analyses
        boundaries = {
            "level_set_boundaries": level_sets,
            "phase_field_boundaries": phase_field_boundaries,
            "topological_boundaries": topological_boundaries,
            "energy_landscape": energy_landscape
        }
        
        return boundaries

    def classify_zones(self, envelope: np.ndarray) -> Dict[str, List[Tuple[int, ...]]]:
        """
        Classify spatial zones in the field.

        Physical Meaning:
            Classifies different spatial zones in the BVP field
            based on field properties and local characteristics.

        Mathematical Foundation:
            Uses field amplitude and gradient analysis to classify
            regions into core, transition, and tail zones.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, List[Tuple[int, ...]]]: Zone classification:
                - core_zones: List of core zone coordinates
                - transition_zones: List of transition zone coordinates
                - tail_zones: List of tail zone coordinates
        """
        amplitude = np.abs(envelope)

        # Define zone thresholds
        max_amplitude = np.max(amplitude)
        mean_amplitude = np.mean(amplitude)

        core_threshold = 0.8 * max_amplitude
        tail_threshold = 0.2 * mean_amplitude

        # Classify zones
        core_mask = amplitude > core_threshold
        tail_mask = amplitude < tail_threshold
        transition_mask = ~(core_mask | tail_mask)

        # Get zone coordinates
        core_zones = list(zip(*np.where(core_mask)))
        transition_zones = list(zip(*np.where(transition_mask)))
        tail_zones = list(zip(*np.where(tail_mask)))

        return {
            "core_zones": core_zones,
            "transition_zones": transition_zones,
            "tail_zones": tail_zones,
        }

    def analyze_zone_properties(
        self, envelope: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze properties of different zones.

        Physical Meaning:
            Analyzes properties of different zones in the BVP field
            including amplitude, gradient, and coherence properties.

        Mathematical Foundation:
            Computes statistical properties for each zone including
            mean, variance, and characteristic scales.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, Dict[str, float]]: Zone properties:
                - core_properties: Properties of core zones
                - transition_properties: Properties of transition zones
                - tail_properties: Properties of tail zones
        """
        amplitude = np.abs(envelope)

        # Define zone thresholds
        max_amplitude = np.max(amplitude)
        mean_amplitude = np.mean(amplitude)

        core_threshold = 0.8 * max_amplitude
        tail_threshold = 0.2 * mean_amplitude

        # Analyze core zones
        core_mask = amplitude > core_threshold
        core_properties = {
            "mean_amplitude": (
                np.mean(amplitude[core_mask]) if np.any(core_mask) else 0.0
            ),
            "variance": np.var(amplitude[core_mask]) if np.any(core_mask) else 0.0,
            "volume_fraction": np.sum(core_mask) / amplitude.size,
        }

        # Analyze transition zones
        transition_mask = ~(core_mask | (amplitude < tail_threshold))
        transition_properties = {
            "mean_amplitude": (
                np.mean(amplitude[transition_mask]) if np.any(transition_mask) else 0.0
            ),
            "variance": (
                np.var(amplitude[transition_mask]) if np.any(transition_mask) else 0.0
            ),
            "volume_fraction": np.sum(transition_mask) / amplitude.size,
        }

        # Analyze tail zones
        tail_mask = amplitude < tail_threshold
        tail_properties = {
            "mean_amplitude": (
                np.mean(amplitude[tail_mask]) if np.any(tail_mask) else 0.0
            ),
            "variance": np.var(amplitude[tail_mask]) if np.any(tail_mask) else 0.0,
            "volume_fraction": np.sum(tail_mask) / amplitude.size,
        }

        return {
            "core_properties": core_properties,
            "transition_properties": transition_properties,
            "tail_properties": tail_properties,
        }

    def identify_transition_regions(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify transition regions between zones.

        Physical Meaning:
            Identifies transition regions between different zones
            in the BVP field, focusing on regions with intermediate
            field properties.

        Mathematical Foundation:
            Uses gradient analysis and field property analysis
            to identify transition regions with intermediate
            characteristics.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            List[Dict[str, Any]]: List of transition regions:
                - region_type: Type of transition region
                - region_location: Location of the region
                - transition_strength: Strength of the transition
        """
        transition_regions = []

        # Analyze field gradients
        amplitude = np.abs(envelope)

        if amplitude.ndim >= 3:
            # Compute gradients
            grad_x = np.gradient(amplitude, axis=0)
            grad_y = np.gradient(amplitude, axis=1)
            grad_z = np.gradient(amplitude, axis=2)

            # Compute gradient magnitude
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

            # Find high-gradient regions (potential transitions)
            threshold = np.mean(grad_magnitude) + np.std(grad_magnitude)
            transition_mask = grad_magnitude > threshold

            # Get transition region coordinates
            transition_coords = np.where(transition_mask)

            if len(transition_coords[0]) > 0:
                # Create transition region
                transition_region = {
                    "region_type": "gradient_transition",
                    "region_location": (
                        transition_coords[0][0],
                        transition_coords[1][0],
                        transition_coords[2][0],
                    ),
                    "transition_strength": np.mean(grad_magnitude[transition_mask]),
                }
                transition_regions.append(transition_region)

        return transition_regions

    def _compute_level_sets(self, amplitude: np.ndarray) -> Dict[str, Any]:
        """
        Compute level sets for boundary detection.
        
        Physical Meaning:
            Computes level sets of the field amplitude to identify
            boundaries between different zones using level set methods.
        """
        # Define level set thresholds
        max_amp = np.max(amplitude)
        min_amp = np.min(amplitude)
        
        # Create level sets at different thresholds
        level_sets = {}
        thresholds = np.linspace(min_amp, max_amp, 10)
        
        for i, threshold in enumerate(thresholds):
            # Create level set
            level_set = amplitude >= threshold
            
            # Compute level set properties
            level_set_properties = {
                "threshold": float(threshold),
                "volume_fraction": float(np.sum(level_set) / level_set.size),
                "boundary_length": self._compute_boundary_length(level_set),
                "connectivity": self._compute_connectivity(level_set)
            }
            
            level_sets[f"level_{i}"] = level_set_properties
        
        return level_sets
    
    def _compute_phase_field_boundaries(self, amplitude: np.ndarray) -> Dict[str, Any]:
        """
        Compute boundaries using phase field method.
        
        Physical Meaning:
            Uses phase field methods to identify and evolve
            boundaries between different zones.
        """
        # Compute phase field order parameter
        max_amp = np.max(amplitude)
        phase_field = amplitude / max_amp
        
        # Compute phase field gradients
        gradients = self._compute_phase_field_gradients(phase_field)
        
        # Identify boundary regions (high gradient regions)
        gradient_magnitude = np.sqrt(sum(grad**2 for grad in gradients.values()))
        boundary_threshold = np.percentile(gradient_magnitude, 90)
        boundary_mask = gradient_magnitude > boundary_threshold
        
        # Compute boundary properties
        boundary_properties = {
            "boundary_mask": boundary_mask,
            "gradient_magnitude": gradient_magnitude,
            "boundary_threshold": float(boundary_threshold),
            "boundary_density": float(np.sum(boundary_mask) / boundary_mask.size),
            "gradients": gradients
        }
        
        return boundary_properties
    
    def _analyze_boundary_topology(self, amplitude: np.ndarray) -> Dict[str, Any]:
        """
        Analyze topology of boundaries.
        
        Physical Meaning:
            Analyzes the topological properties of boundaries
            between different zones in the BVP field.
        """
        # Compute field gradients
        gradients = self._compute_field_gradients(amplitude)
        
        # Compute curvature of level sets
        curvature = self._compute_curvature(amplitude, gradients)
        
        # Identify critical points
        critical_points = self._identify_critical_points(gradients)
        
        # Compute topological invariants
        topological_invariants = self._compute_topological_invariants(amplitude, gradients)
        
        return {
            "curvature": curvature,
            "critical_points": critical_points,
            "topological_invariants": topological_invariants,
            "gradients": gradients
        }
    
    def _compute_energy_landscape(self, amplitude: np.ndarray) -> Dict[str, Any]:
        """
        Compute energy landscape for boundary analysis.
        
        Physical Meaning:
            Computes the energy landscape of the BVP field
            to identify energy barriers and transition regions.
        """
        # Compute local energy density
        energy_density = self._compute_energy_density(amplitude)
        
        # Compute energy gradients
        energy_gradients = self._compute_energy_gradients(energy_density)
        
        # Identify energy barriers
        energy_barriers = self._identify_energy_barriers(energy_density, energy_gradients)
        
        # Compute transition regions
        transition_regions = self._identify_transition_regions(energy_density)
        
        return {
            "energy_density": energy_density,
            "energy_gradients": energy_gradients,
            "energy_barriers": energy_barriers,
            "transition_regions": transition_regions
        }
    
    def _compute_boundary_length(self, level_set: np.ndarray) -> float:
        """Compute boundary length of level set."""
        # Simple boundary length estimation
        # Count edges between different regions
        boundary_length = 0.0
        
        if level_set.ndim >= 2:
            # 2D or higher case
            for axis in range(level_set.ndim):
                # Compute differences along this axis
                diff = np.diff(level_set.astype(int), axis=axis)
                boundary_length += np.sum(np.abs(diff))
        else:
            # 1D case
            diff = np.diff(level_set.astype(int))
            boundary_length = np.sum(np.abs(diff))
        
        return float(boundary_length)
    
    def _compute_connectivity(self, level_set: np.ndarray) -> Dict[str, Any]:
        """Compute connectivity properties of level set."""
        # Count connected components
        try:
            from scipy import ndimage
            labeled, num_components = ndimage.label(level_set)
            
            # Compute component sizes
            component_sizes = []
            for i in range(1, num_components + 1):
                size = np.sum(labeled == i)
                component_sizes.append(size)
            
            return {
                "num_components": num_components,
                "component_sizes": component_sizes,
                "largest_component": max(component_sizes) if component_sizes else 0,
                "connectivity_ratio": max(component_sizes) / np.sum(level_set) if np.sum(level_set) > 0 else 0
            }
        except ImportError:
            # Fallback if scipy not available
            return {
                "num_components": 1,
                "component_sizes": [np.sum(level_set)],
                "largest_component": np.sum(level_set),
                "connectivity_ratio": 1.0
            }
    
    def _compute_phase_field_gradients(self, phase_field: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients of phase field."""
        gradients = {}
        
        for dim in range(phase_field.ndim):
            gradient = np.gradient(phase_field, axis=dim)
            gradients[f"dim_{dim}"] = gradient
        
        return gradients
    
    def _compute_field_gradients(self, field: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients of field."""
        gradients = {}
        
        for dim in range(field.ndim):
            gradient = np.gradient(field, axis=dim)
            gradients[f"dim_{dim}"] = gradient
        
        return gradients
    
    def _compute_curvature(self, field: np.ndarray, gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute curvature of level sets."""
        if field.ndim >= 2:
            # Compute second derivatives
            second_derivatives = {}
            for dim in range(field.ndim):
                second_derivatives[f"dim_{dim}"] = np.gradient(gradients[f"dim_{dim}"], axis=dim)
            
            # Compute mean curvature (simplified)
            if field.ndim == 2:
                # 2D case: κ = (f_xx * f_y² - 2*f_xy*f_x*f_y + f_yy*f_x²) / (f_x² + f_y²)^(3/2)
                f_x = gradients["dim_0"]
                f_y = gradients["dim_1"]
                f_xx = second_derivatives["dim_0"]
                f_yy = second_derivatives["dim_1"]
                
                # Approximate f_xy
                f_xy = np.gradient(f_x, axis=1)
                
                denominator = (f_x**2 + f_y**2)**(3/2)
                denominator = np.where(denominator < 1e-10, 1e-10, denominator)
                
                curvature = (f_xx * f_y**2 - 2 * f_xy * f_x * f_y + f_yy * f_x**2) / denominator
            else:
                # Higher dimensions: simplified curvature
                curvature = np.zeros_like(field)
                for dim in range(field.ndim):
                    curvature += second_derivatives[f"dim_{dim}"]
                curvature /= field.ndim
        else:
            # 1D case
            curvature = np.gradient(gradients["dim_0"], axis=0)
        
        return curvature
    
    def _identify_critical_points(self, gradients: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Identify critical points where gradients are zero."""
        critical_points = []
        
        # Find points where all gradients are close to zero
        gradient_magnitude = np.sqrt(sum(grad**2 for grad in gradients.values()))
        critical_mask = gradient_magnitude < np.percentile(gradient_magnitude, 5)
        
        # Find critical point coordinates
        critical_coords = np.where(critical_mask)
        
        for i in range(len(critical_coords[0])):
            coords = tuple(critical_coords[dim][i] for dim in range(len(critical_coords)))
            critical_points.append({
                "coordinates": coords,
                "gradient_magnitude": float(gradient_magnitude[coords]),
                "type": "critical_point"
            })
        
        return critical_points
    
    def _compute_topological_invariants(self, field: np.ndarray, gradients: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute topological invariants of the field."""
        # Compute basic topological measures
        gradient_magnitude = np.sqrt(sum(grad**2 for grad in gradients.values()))
        
        # Compute topological measures
        max_gradient = np.max(gradient_magnitude)
        mean_gradient = np.mean(gradient_magnitude)
        gradient_variance = np.var(gradient_magnitude)
        
        # Compute field topology
        field_max = np.max(field)
        field_min = np.min(field)
        field_range = field_max - field_min
        
        return {
            "max_gradient": float(max_gradient),
            "mean_gradient": float(mean_gradient),
            "gradient_variance": float(gradient_variance),
            "field_range": float(field_range),
            "topological_complexity": float(gradient_variance / mean_gradient) if mean_gradient > 0 else 0.0
        }
    
    def _compute_energy_density(self, amplitude: np.ndarray) -> np.ndarray:
        """Compute local energy density."""
        # Simple energy density: proportional to amplitude squared
        energy_density = amplitude**2
        
        # Add gradient energy contribution
        gradients = self._compute_field_gradients(amplitude)
        gradient_energy = sum(grad**2 for grad in gradients.values())
        energy_density += 0.1 * gradient_energy  # Small coefficient for gradient energy
        
        return energy_density
    
    def _compute_energy_gradients(self, energy_density: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients of energy density."""
        return self._compute_field_gradients(energy_density)
    
    def _identify_energy_barriers(self, energy_density: np.ndarray, energy_gradients: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Identify energy barriers in the landscape."""
        # Find local maxima in energy density
        gradient_magnitude = np.sqrt(sum(grad**2 for grad in energy_gradients.values()))
        
        # Energy barriers are regions with high energy and low gradient magnitude
        energy_threshold = np.percentile(energy_density, 80)
        gradient_threshold = np.percentile(gradient_magnitude, 20)
        
        barrier_mask = (energy_density > energy_threshold) & (gradient_magnitude < gradient_threshold)
        
        return {
            "barrier_mask": barrier_mask,
            "energy_threshold": float(energy_threshold),
            "gradient_threshold": float(gradient_threshold),
            "barrier_density": float(np.sum(barrier_mask) / barrier_mask.size)
        }
    
    def _identify_transition_regions(self, energy_density: np.ndarray) -> Dict[str, Any]:
        """Identify transition regions in energy landscape."""
        # Transition regions are regions with intermediate energy
        energy_min = np.min(energy_density)
        energy_max = np.max(energy_density)
        energy_range = energy_max - energy_min
        
        # Define transition region as middle 40% of energy range
        lower_threshold = energy_min + 0.3 * energy_range
        upper_threshold = energy_min + 0.7 * energy_range
        
        transition_mask = (energy_density >= lower_threshold) & (energy_density <= upper_threshold)
        
        return {
            "transition_mask": transition_mask,
            "lower_threshold": float(lower_threshold),
            "upper_threshold": float(upper_threshold),
            "transition_density": float(np.sum(transition_mask) / transition_mask.size)
        }

    def __repr__(self) -> str:
        """String representation of zone analyzer."""
        return f"ZoneAnalysis(bvp_core={self.bvp_core})"
