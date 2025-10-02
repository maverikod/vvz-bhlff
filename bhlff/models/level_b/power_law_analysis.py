"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level B power law analysis facade for BVP framework.

This module provides the main facade interface for Level B power law analysis,
coordinating all analysis operations while maintaining modular architecture.

Physical Meaning:
    Level B power law analysis examines the fundamental properties
    of the BVP field including power law tails, node analysis,
    topological charge, and zone separation.

Mathematical Foundation:
    Implements comprehensive analysis of BVP field properties
    including statistical analysis, topological analysis,
    and spatial pattern recognition.

Example:
    >>> analyzer = LevelBPowerLawAnalyzer(bvp_core)
    >>> results = analyzer.analyze_power_laws(envelope)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore
from ...core.domain import Domain
from ...core.bvp.unified_power_law_analyzer import UnifiedPowerLawAnalyzer
from .node_analysis import NodeAnalysis
from .zone_analysis import ZoneAnalysis


class LevelBPowerLawAnalyzer:
    """
    Level B power law analyzer for BVP framework.

    Physical Meaning:
        Analyzes fundamental properties of the BVP field including
        power law tails, node structures, topological charge,
        and zone separation patterns.

    Mathematical Foundation:
        Implements comprehensive analysis of BVP field properties
        including statistical analysis, topological analysis,
        and spatial pattern recognition.
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize Level B power law analyzer.

        Args:
            bvp_core (BVPCore): BVP core instance for analysis.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

        # Initialize modular components
        self._power_law_core = UnifiedPowerLawAnalyzer(bvp_core)
        self._node_analysis = NodeAnalysis(bvp_core)
        self._zone_analysis = ZoneAnalysis(bvp_core)

    def analyze_power_laws(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze power law properties of BVP field.

        Physical Meaning:
            Examines the power law characteristics of the BVP field
            distribution, identifying scaling behavior and critical
            exponents.

        Returns:
            Dict[str, Any]: Power law analysis results including:
                - power_law_exponents: Critical exponents
                - scaling_regions: Regions with power law behavior
                - correlation_functions: Spatial correlation analysis
                - critical_behavior: Critical point analysis
        """
        self.logger.info("Starting power law analysis")

        results = self._power_law_core.analyze_power_laws(envelope)

        self.logger.info("Power law analysis completed")
        return results

    def analyze_nodes(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze node structures in BVP field.

        Physical Meaning:
            Identifies and classifies node structures in the BVP field,
            including saddle nodes, source nodes, and sink nodes.

        Returns:
            Dict[str, Any]: Node analysis results including:
                - node_locations: Coordinates of identified nodes
                - node_types: Classification of node types
                - node_density: Spatial density of nodes
                - topological_charge: Total topological charge
        """
        self.logger.info("Starting node analysis")

        results = {
            "node_locations": self._node_analysis.identify_nodes(envelope),
            "node_types": self._node_analysis.classify_nodes(envelope),
            "node_density": self._node_analysis.compute_node_density(envelope),
            "topological_charge": self._node_analysis.compute_topological_charge(
                envelope
            ),
        }

        self.logger.info("Node analysis completed")
        return results

    def analyze_zones(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze zone separation in BVP field.

        Physical Meaning:
            Identifies and analyzes different zones in the BVP field
            including core, transition, and tail regions.

        Returns:
            Dict[str, Any]: Zone analysis results including:
                - zone_boundaries: Boundaries between different zones
                - zone_classification: Classification of spatial zones
                - zone_properties: Properties of each zone
                - transition_regions: Transition regions between zones
        """
        self.logger.info("Starting zone analysis")

        results = {
            "zone_boundaries": self._zone_analysis.identify_zone_boundaries(envelope),
            "zone_classification": self._zone_analysis.classify_zones(envelope),
            "zone_properties": self._zone_analysis.analyze_zone_properties(envelope),
            "transition_regions": self._zone_analysis.identify_transition_regions(
                envelope
            ),
        }

        self.logger.info("Zone analysis completed")
        return results

    def analyze_all(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of all Level B properties.

        Physical Meaning:
            Performs complete analysis of all fundamental properties
            of the BVP field including power laws, nodes, and zones.

        Returns:
            Dict[str, Any]: Complete analysis results from all components.
        """
        self.logger.info("Starting comprehensive Level B analysis")

        results = {
            "power_laws": self.analyze_power_laws(envelope),
            "nodes": self.analyze_nodes(envelope),
            "zones": self.analyze_zones(envelope),
            "analysis_status": "completed",
        }

        self.logger.info("Comprehensive Level B analysis completed")
        return results

    def analyze_power_law_tails(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze power law tails of BVP field.

        Physical Meaning:
            Analyzes the power law decay behavior in the tails of the BVP field,
            identifying scaling exponents and decay characteristics.

        Returns:
            Dict[str, Any]: Power law tail analysis results.
        """
        return self._power_law_core.analyze_power_law_tails(envelope)

    def compute_radial_profile(self, envelope: np.ndarray, n_bins: int = 50) -> Dict[str, Any]:
        """
        Compute radial profile of BVP field.

        Physical Meaning:
            Computes the radial distribution of the BVP field amplitude,
            providing insight into the spatial structure and decay behavior.

        Args:
            envelope (np.ndarray): BVP field envelope.
            n_bins (int): Number of radial bins for analysis.

        Returns:
            Dict[str, Any]: Radial profile analysis results.
        """
        return self._power_law_core.compute_radial_profile(envelope, n_bins)

    def __repr__(self) -> str:
        """String representation of analyzer."""
        return f"LevelBPowerLawAnalyzer(bvp_core={self.bvp_core})"
