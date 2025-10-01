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
from .boundary_detection import BoundaryDetection
from .zone_properties import ZoneProperties


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
        
        # Initialize specialized analyzers
        self._boundary_detector = BoundaryDetection(bvp_core)
        self._zone_properties_analyzer = ZoneProperties(bvp_core)

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
        return self._boundary_detector.identify_zone_boundaries(envelope)

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
        return self._zone_properties_analyzer.analyze_zone_properties(envelope)

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

    def __repr__(self) -> str:
        """String representation of zone analyzer."""
        return f"ZoneAnalysis(bvp_core={self.bvp_core})"

