"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Boundary analysis for phase mapping.

This module implements boundary analysis functionality
for identifying transition boundaries between different
system behavior regimes.

Theoretical Background:
    Boundary analysis identifies transition points between
    different regimes in parameter space, revealing the
    structure of the phase diagram.

Example:
    >>> analyzer = BoundaryAnalyzer()
    >>> boundaries = analyzer.analyze_regime_boundaries(parameter_grid, classifications)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class BoundaryAnalyzer:
    """
    Boundary analysis for regime transitions.

    Physical Meaning:
        Analyzes boundaries between different system behavior
        regimes in parameter space, identifying transition
        points and regime characteristics.
    """

    def __init__(self):
        """
        Initialize boundary analyzer.

        Physical Meaning:
            Sets up the analyzer for studying regime boundaries
            in parameter space.
        """
        pass

    def analyze_regime_boundaries(
        self, parameter_grid: Dict[str, np.ndarray], classifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze boundaries between regimes."""
        boundaries = {}

        # Extract regime information
        regime_data = []
        for point_id, classification in classifications.items():
            params = classification["parameters"]
            regime = classification.get("primary_regime", "unknown")
            regime_data.append(
                {
                    "eta": params["eta"],
                    "chi_double_prime": params["chi_double_prime"],
                    "beta": params["beta"],
                    "regime": regime,
                }
            )

        # Analyze boundaries between different regimes
        regime_pairs = [("PL", "R"), ("PL", "FRZ"), ("R", "FRZ"), ("FRZ", "LEAK")]

        for regime1, regime2 in regime_pairs:
            boundary = self._find_regime_boundary(regime_data, regime1, regime2)
            boundaries[f"{regime1}_{regime2}"] = boundary

        return boundaries

    def _find_regime_boundary(
        self, regime_data: List[Dict[str, Any]], regime1: str, regime2: str
    ) -> Dict[str, Any]:
        """Find boundary between two regimes."""
        # Filter data for the two regimes
        regime1_data = [d for d in regime_data if d["regime"] == regime1]
        regime2_data = [d for d in regime_data if d["regime"] == regime2]

        if not regime1_data or not regime2_data:
            return {"boundary": None, "separation": 0.0}

        # Compute separation between regimes
        separation = self._compute_regime_separation(regime1_data, regime2_data)

        # Find boundary points
        boundary_points = self._find_boundary_points(regime1_data, regime2_data)

        return {
            "separation": separation,
            "boundary_points": boundary_points,
            "regime1_count": len(regime1_data),
            "regime2_count": len(regime2_data),
        }

    def _compute_regime_separation(
        self, regime1_data: List[Dict[str, Any]], regime2_data: List[Dict[str, Any]]
    ) -> float:
        """Compute separation between two regimes."""
        # Extract parameter values
        regime1_params = np.array(
            [[d["eta"], d["chi_double_prime"], d["beta"]] for d in regime1_data]
        )
        regime2_params = np.array(
            [[d["eta"], d["chi_double_prime"], d["beta"]] for d in regime2_data]
        )

        # Compute mean parameter values
        mean1 = np.mean(regime1_params, axis=0)
        mean2 = np.mean(regime2_params, axis=0)

        # Compute separation distance
        separation = np.linalg.norm(mean1 - mean2)

        return separation

    def _find_boundary_points(
        self, regime1_data: List[Dict[str, Any]], regime2_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find boundary points between regimes."""
        boundary_points = []

        # Find points that are close to the boundary
        for d1 in regime1_data:
            for d2 in regime2_data:
                # Compute distance between points
                params1 = np.array([d1["eta"], d1["chi_double_prime"], d1["beta"]])
                params2 = np.array([d2["eta"], d2["chi_double_prime"], d2["beta"]])
                distance = np.linalg.norm(params1 - params2)

                # If distance is small, this is a boundary point
                if distance < 0.1:  # Threshold for boundary proximity
                    boundary_points.append(
                        {"regime1_point": d1, "regime2_point": d2, "distance": distance}
                    )

        return boundary_points
