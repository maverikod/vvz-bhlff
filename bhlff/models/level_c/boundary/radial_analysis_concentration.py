"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Radial analysis concentration module.

This module implements field concentration analysis functionality for radial analysis
in Level C test C1 of 7D phase field theory.

Physical Meaning:
    Analyzes field concentration patterns for boundary effects,
    including near-boundary and far-boundary concentration analysis.

Example:
    >>> concentration_analyzer = RadialConcentrationAnalyzer(bvp_core)
    >>> results = concentration_analyzer.analyze_field_concentration(domain, boundary, field)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

from bhlff.core.bvp import BVPCore
from .data_structures import BoundaryGeometry, RadialProfile


class RadialConcentrationAnalyzer:
    """
    Radial concentration analyzer for boundary effects.

    Physical Meaning:
        Analyzes field concentration patterns for boundary effects,
        including near-boundary and far-boundary concentration analysis.

    Mathematical Foundation:
        Implements concentration analysis:
        - Field concentration analysis
        - Near-boundary concentration analysis
        - Far-boundary concentration analysis
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize radial concentration analyzer.

        Physical Meaning:
            Sets up the concentration analysis system with
            appropriate parameters and methods.

        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def analyze_field_concentration(
        self, domain: Dict[str, Any], boundary: BoundaryGeometry, field: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze field concentration.

        Physical Meaning:
            Analyzes field concentration patterns for boundary effects
            including near-boundary and far-boundary concentration.

        Mathematical Foundation:
            Analyzes concentration patterns through:
            - Near-boundary concentration analysis
            - Far-boundary concentration analysis
            - Overall concentration pattern analysis

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            field (np.ndarray): Field data.

        Returns:
            Dict[str, Any]: Field concentration analysis results.
        """
        self.logger.info("Starting field concentration analysis")

        # Analyze near-boundary concentration
        near_boundary_concentration = self._analyze_near_boundary_concentration(
            domain, boundary, field
        )

        # Analyze far-boundary concentration
        far_boundary_concentration = self._analyze_far_boundary_concentration(
            domain, boundary, field
        )

        # Analyze overall concentration pattern
        overall_concentration = self._analyze_overall_concentration_pattern(
            domain, boundary, field
        )

        results = {
            "near_boundary_concentration": near_boundary_concentration,
            "far_boundary_concentration": far_boundary_concentration,
            "overall_concentration": overall_concentration,
            "concentration_analysis_complete": True,
        }

        self.logger.info("Field concentration analysis completed")
        return results

    def _analyze_near_boundary_concentration(
        self, domain: Dict[str, Any], boundary: BoundaryGeometry, field: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze near-boundary concentration.

        Physical Meaning:
            Analyzes field concentration near boundaries
            for boundary effects analysis.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            field (np.ndarray): Field data.

        Returns:
            Dict[str, Any]: Near-boundary concentration analysis results.
        """
        # Extract domain parameters
        N = domain["N"]
        L = domain["L"]

        # Define near-boundary region
        boundary_thickness = L / (4 * N)  # Boundary thickness

        # Create near-boundary mask
        near_boundary_mask = self._create_near_boundary_mask(domain, boundary_thickness)

        # Analyze concentration in near-boundary region
        near_boundary_field = field[near_boundary_mask]
        concentration_metrics = {
            "mean_concentration": np.mean(np.abs(near_boundary_field)),
            "max_concentration": np.max(np.abs(near_boundary_field)),
            "concentration_variance": np.var(np.abs(near_boundary_field)),
            "concentration_ratio": np.mean(np.abs(near_boundary_field))
            / np.mean(np.abs(field)),
        }

        return concentration_metrics

    def _analyze_far_boundary_concentration(
        self, domain: Dict[str, Any], boundary: BoundaryGeometry, field: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze far-boundary concentration.

        Physical Meaning:
            Analyzes field concentration far from boundaries
            for boundary effects analysis.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            field (np.ndarray): Field data.

        Returns:
            Dict[str, Any]: Far-boundary concentration analysis results.
        """
        # Extract domain parameters
        N = domain["N"]
        L = domain["L"]

        # Define far-boundary region
        boundary_thickness = L / (4 * N)  # Boundary thickness

        # Create far-boundary mask
        far_boundary_mask = self._create_far_boundary_mask(domain, boundary_thickness)

        # Analyze concentration in far-boundary region
        far_boundary_field = field[far_boundary_mask]
        concentration_metrics = {
            "mean_concentration": np.mean(np.abs(far_boundary_field)),
            "max_concentration": np.max(np.abs(far_boundary_field)),
            "concentration_variance": np.var(np.abs(far_boundary_field)),
            "concentration_ratio": np.mean(np.abs(far_boundary_field))
            / np.mean(np.abs(field)),
        }

        return concentration_metrics

    def _analyze_overall_concentration_pattern(
        self, domain: Dict[str, Any], boundary: BoundaryGeometry, field: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze overall concentration pattern.

        Physical Meaning:
            Analyzes overall field concentration pattern
            for boundary effects analysis.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            field (np.ndarray): Field data.

        Returns:
            Dict[str, Any]: Overall concentration pattern analysis results.
        """
        # Analyze overall concentration pattern
        overall_metrics = {
            "total_concentration": np.sum(np.abs(field)),
            "mean_concentration": np.mean(np.abs(field)),
            "max_concentration": np.max(np.abs(field)),
            "concentration_variance": np.var(np.abs(field)),
            "concentration_skewness": self._calculate_skewness(np.abs(field)),
            "concentration_kurtosis": self._calculate_kurtosis(np.abs(field)),
        }

        return overall_metrics

    def _create_near_boundary_mask(
        self, domain: Dict[str, Any], boundary_thickness: float
    ) -> np.ndarray:
        """
        Create near-boundary mask.

        Physical Meaning:
            Creates mask for near-boundary region
            for concentration analysis.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary_thickness (float): Boundary thickness.

        Returns:
            np.ndarray: Near-boundary mask.
        """
        # Extract domain parameters
        N = domain["N"]
        L = domain["L"]

        # Create coordinate arrays
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create near-boundary mask
        near_boundary_mask = (
            (X <= boundary_thickness)
            | (X >= L - boundary_thickness)
            | (Y <= boundary_thickness)
            | (Y >= L - boundary_thickness)
            | (Z <= boundary_thickness)
            | (Z >= L - boundary_thickness)
        )

        return near_boundary_mask

    def _create_far_boundary_mask(
        self, domain: Dict[str, Any], boundary_thickness: float
    ) -> np.ndarray:
        """
        Create far-boundary mask.

        Physical Meaning:
            Creates mask for far-boundary region
            for concentration analysis.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary_thickness (float): Boundary thickness.

        Returns:
            np.ndarray: Far-boundary mask.
        """
        # Extract domain parameters
        N = domain["N"]
        L = domain["L"]

        # Create coordinate arrays
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create far-boundary mask
        far_boundary_mask = (
            (X > boundary_thickness)
            & (X < L - boundary_thickness)
            & (Y > boundary_thickness)
            & (Y < L - boundary_thickness)
            & (Z > boundary_thickness)
            & (Z < L - boundary_thickness)
        )

        return far_boundary_mask

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """
        Calculate skewness.

        Physical Meaning:
            Calculates skewness of field concentration
            for pattern analysis.

        Args:
            data (np.ndarray): Data for skewness calculation.

        Returns:
            float: Skewness value.
        """
        # Calculate skewness
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            skewness = np.mean(((data - mean) / std) ** 3)
        else:
            skewness = 0.0

        return skewness

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """
        Calculate kurtosis.

        Physical Meaning:
            Calculates kurtosis of field concentration
            for pattern analysis.

        Args:
            data (np.ndarray): Data for kurtosis calculation.

        Returns:
            float: Kurtosis value.
        """
        # Calculate kurtosis
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            kurtosis = np.mean(((data - mean) / std) ** 4)
        else:
            kurtosis = 0.0

        return kurtosis
