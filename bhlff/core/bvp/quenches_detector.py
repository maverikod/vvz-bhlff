"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quenches detector implementation for BVP framework.

This module implements the detection functionality for quench events in the BVP field,
identifying localized regions where field amplitude drops significantly.

Physical Meaning:
    Detects quench events as localized regions where field amplitude
    drops below critical thresholds, creating energy dissipation zones.

Mathematical Foundation:
    Uses statistical analysis and morphological operations to identify
    and filter quench regions based on amplitude thresholds and size criteria.

Example:
    >>> detector = QuenchesDetector(domain, constants)
    >>> quench_data = detector.detect_quenches(envelope)
"""

import numpy as np
from typing import Dict, Any, List
from scipy import ndimage

from ..domain.domain import Domain
from .bvp_constants import BVPConstants


class QuenchesDetector:
    """
    Detector for quench events in BVP field.

    Physical Meaning:
        Identifies localized regions where field amplitude drops
        below critical thresholds, indicating energy dissipation.
    """

    def __init__(self, domain: Domain, constants: BVPConstants):
        """
        Initialize quenches detector.

        Args:
            domain (Domain): Computational domain for analysis.
            constants (BVPConstants): BVP physical constants.
        """
        self.domain = domain
        self.constants = constants
        self.quench_threshold = constants.get_quench_parameter("quench_threshold", 0.1)
        self.min_quench_size = constants.get_quench_parameter("min_quench_size", 5)

    def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quench events in the field.

        Physical Meaning:
            Identifies localized regions where field amplitude
            drops below critical thresholds.

        Args:
            envelope (np.ndarray): BVP envelope.

        Returns:
            Dict[str, Any]: Quench detection results.
        """
        amplitude = np.abs(envelope)

        # Compute amplitude statistics
        mean_amplitude = np.mean(amplitude)
        std_amplitude = np.std(amplitude)

        # Define quench threshold
        quench_threshold_value = mean_amplitude - self.quench_threshold * std_amplitude

        # Find quench regions
        quench_mask = amplitude < quench_threshold_value

        # Filter small quenches
        quench_mask = self._filter_small_quenches(quench_mask)

        # Find quench locations
        quench_locations = self._find_quench_locations(quench_mask)

        # Compute quench statistics
        num_quenches = len(quench_locations)
        quench_volume = np.sum(quench_mask)
        quench_fraction = quench_volume / amplitude.size

        return {
            "quench_mask": quench_mask,
            "quench_locations": quench_locations,
            "num_quenches": num_quenches,
            "quench_volume": quench_volume,
            "quench_fraction": quench_fraction,
            "quench_threshold_value": quench_threshold_value,
        }

    def _filter_small_quenches(self, quench_mask: np.ndarray) -> np.ndarray:
        """
        Filter out small quench regions.

        Physical Meaning:
            Removes quench regions that are too small to be
            physically significant.

        Args:
            quench_mask (np.ndarray): Binary quench mask.

        Returns:
            np.ndarray: Filtered quench mask.
        """
        # Use morphological operations to filter small regions
        # Remove small connected components
        filtered_mask = ndimage.binary_opening(
            quench_mask, structure=ndimage.generate_binary_structure(3, 1)
        )

        # Remove very small regions
        labeled_mask, num_features = ndimage.label(filtered_mask)
        for i in range(1, num_features + 1):
            region_size = np.sum(labeled_mask == i)
            if region_size < self.min_quench_size:
                filtered_mask[labeled_mask == i] = False

        return filtered_mask

    def _find_quench_locations(self, quench_mask: np.ndarray) -> List[tuple]:
        """
        Find center locations of quench regions.

        Physical Meaning:
            Identifies center coordinates of each quench region
            for further analysis.

        Args:
            quench_mask (np.ndarray): Binary quench mask.

        Returns:
            List[tuple]: List of quench center coordinates.
        """
        # Label connected components
        labeled_mask, num_features = ndimage.label(quench_mask)

        # Find centers of mass for each quench
        quench_locations = []
        for i in range(1, num_features + 1):
            region_mask = labeled_mask == i
            center = ndimage.center_of_mass(region_mask)
            quench_locations.append(tuple(center))

        return quench_locations
