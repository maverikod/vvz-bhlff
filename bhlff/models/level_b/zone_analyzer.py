"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Zone analysis for Level B fundamental properties.

This module implements zone separation analysis for the 7D phase field theory,
quantitatively separating the field into core, transition, and tail regions.

Theoretical Background:
    The phase field exhibits three characteristic zones: core (high density,
    nonlinear), transition (balance between core and tail), and tail
    (linear wave region). Each zone plays a specific role in particle formation.

Example:
    >>> analyzer = LevelBZoneAnalyzer()
    >>> result = analyzer.separate_zones(field, center, thresholds)
"""

import numpy as np
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path


class LevelBZoneAnalyzer:
    """
    Zone analysis for Level B fundamental properties.

    Physical Meaning:
        Separates the phase field into three characteristic zones:
        core (high density, nonlinear), transition (balance), and
        tail (linear wave region). Each zone plays a specific role
        in particle formation and stability.

    Mathematical Foundation:
        Zone separation is based on local indicators N (norm gradient),
        S (second derivative), and C (coherence). Thresholds determine
        the boundaries between zones.
    """

    def __init__(self):
        """Initialize zone analyzer."""
        pass

    def separate_zones(
        self, field: np.ndarray, center: List[float], thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Separate field into zones (core/transition/tail).

        Physical Meaning:
            Quantitatively separates the phase field into three
            characteristic zones based on local indicators, allowing
            analysis of the structure and role of each zone in
            particle formation.

        Mathematical Foundation:
            - Core: N > N_core, S > S_core (high density, nonlinear)
            - Tail: N < N_tail, S < S_tail (linear wave region)
            - Transition: intermediate values (balance zone)

        Args:
            field (np.ndarray): Phase field solution
            center (List[float]): Center of the defect [x, y, z]
            thresholds (Dict[str, float]): Threshold values for zone separation

        Returns:
            Dict[str, Any]: Zone separation results with masks, radii, and statistics
        """
        # 1. Compute zone indicators
        indicators = self._compute_zone_indicators(field)
        N = indicators["N"]
        S = indicators["S"]
        C = indicators["C"]

        # 2. Normalize indicators
        N_norm = N / np.max(N) if np.max(N) > 0 else N
        S_norm = S / np.max(S) if np.max(S) > 0 else S

        # 3. Separate zones by thresholds
        core_mask = (N_norm > thresholds["N_core"]) & (S_norm > thresholds["S_core"])
        tail_mask = (N_norm < thresholds["N_tail"]) & (S_norm < thresholds["S_tail"])
        transition_mask = ~(core_mask | tail_mask)

        # 4. Compute zone radii
        r_core = self._compute_zone_radius(core_mask, center)
        r_tail = self._compute_zone_radius(tail_mask, center)
        r_transition = self._compute_zone_radius(transition_mask, center)

        # 5. Compute zone statistics
        zone_stats = self._compute_zone_statistics(
            field, core_mask, transition_mask, tail_mask
        )

        # 6. Quality assessment
        quality_metrics = self._assess_zone_separation_quality(
            core_mask, tail_mask, transition_mask, zone_stats
        )

        return {
            "core_mask": core_mask,
            "transition_mask": transition_mask,
            "tail_mask": tail_mask,
            "r_core": r_core,
            "r_tail": r_tail,
            "r_transition": r_transition,
            "indicators": indicators,
            "zone_stats": zone_stats,
            "quality_metrics": quality_metrics,
            "thresholds": thresholds,
        }

    def _compute_zone_indicators(self, field: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute zone indicators N, S, C.

        Physical Meaning:
            Computes local indicators that characterize the properties
            of the phase field and allow quantitative zone separation.

        Mathematical Foundation:
            - N: norm of gradient (density indicator)
            - S: second derivative (curvature indicator)
            - C: coherence indicator (local "stiffness")
        """
        # 1. Indicator N: norm of gradient
        N = self._compute_norm_gradient(field)

        # 2. Indicator S: second derivative
        S = self._compute_second_derivative(field)

        # 3. Indicator C: coherence
        C = self._compute_coherence(field)

        return {"N": N, "S": S, "C": C}

    def _compute_norm_gradient(self, field: np.ndarray) -> np.ndarray:
        """Compute norm of field gradient."""
        # Compute gradients in all spatial dimensions
        grad_x = np.gradient(field, axis=0)
        grad_y = np.gradient(field, axis=1)
        grad_z = np.gradient(field, axis=2)

        # Compute norm
        N = np.sqrt(np.abs(grad_x) ** 2 + np.abs(grad_y) ** 2 + np.abs(grad_z) ** 2)

        return N

    def _compute_second_derivative(self, field: np.ndarray) -> np.ndarray:
        """Compute second derivative indicator."""
        # Compute Laplacian
        laplacian = self._compute_laplacian(field)

        # Second derivative indicator
        S = np.abs(laplacian)

        return S

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute Laplacian of the field."""
        # Second derivatives in each dimension
        d2dx2 = np.gradient(np.gradient(field, axis=0), axis=0)
        d2dy2 = np.gradient(np.gradient(field, axis=1), axis=1)
        d2dz2 = np.gradient(np.gradient(field, axis=2), axis=2)

        # Laplacian
        laplacian = d2dx2 + d2dy2 + d2dz2

        return laplacian

    def _compute_coherence(self, field: np.ndarray) -> np.ndarray:
        """Compute coherence indicator."""
        # Field amplitude
        amplitude = np.abs(field)

        # Gradient of amplitude
        grad_amp_x = np.gradient(amplitude, axis=0)
        grad_amp_y = np.gradient(amplitude, axis=1)
        grad_amp_z = np.gradient(amplitude, axis=2)

        # Coherence indicator (local "stiffness")
        C = np.sqrt(grad_amp_x**2 + grad_amp_y**2 + grad_amp_z**2)

        return C

    def _compute_zone_radius(self, mask: np.ndarray, center: List[float]) -> float:
        """Compute effective radius of a zone."""
        if not np.any(mask):
            return 0.0

        # Find all points in the zone
        zone_points = np.where(mask)

        # Compute distances to center
        distances = []
        for i, j, k in zip(zone_points[0], zone_points[1], zone_points[2]):
            dist = np.sqrt(
                (i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2
            )
            distances.append(dist)

        # Effective radius (mean distance)
        return np.mean(distances) if distances else 0.0

    def _compute_zone_statistics(
        self,
        field: np.ndarray,
        core_mask: np.ndarray,
        transition_mask: np.ndarray,
        tail_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute statistics for each zone."""
        stats = {}

        for zone_name, mask in [
            ("core", core_mask),
            ("transition", transition_mask),
            ("tail", tail_mask),
        ]:
            if np.any(mask):
                zone_field = field[mask]
                stats[zone_name] = {
                    "volume_fraction": np.sum(mask) / mask.size,
                    "mean_amplitude": np.mean(np.abs(zone_field)),
                    "max_amplitude": np.max(np.abs(zone_field)),
                    "std_amplitude": np.std(np.abs(zone_field)),
                    "mean_phase": np.mean(np.angle(zone_field)),
                    "phase_std": np.std(np.angle(zone_field)),
                }
            else:
                stats[zone_name] = {
                    "volume_fraction": 0.0,
                    "mean_amplitude": 0.0,
                    "max_amplitude": 0.0,
                    "std_amplitude": 0.0,
                    "mean_phase": 0.0,
                    "phase_std": 0.0,
                }

        return stats

    def _assess_zone_separation_quality(
        self,
        core_mask: np.ndarray,
        tail_mask: np.ndarray,
        transition_mask: np.ndarray,
        zone_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess quality of zone separation."""
        # Check for reasonable zone sizes
        core_fraction = zone_stats["core"]["volume_fraction"]
        tail_fraction = zone_stats["tail"]["volume_fraction"]
        transition_fraction = zone_stats["transition"]["volume_fraction"]

        # Check for proper ordering of amplitudes
        core_amplitude = zone_stats["core"]["mean_amplitude"]
        tail_amplitude = zone_stats["tail"]["mean_amplitude"]

        # Quality metrics
        quality = {
            "core_fraction": core_fraction,
            "tail_fraction": tail_fraction,
            "transition_fraction": transition_fraction,
            "amplitude_ordering": core_amplitude > tail_amplitude,
            "zone_balance": abs(core_fraction - tail_fraction)
            < 0.5,  # Not too imbalanced
            "total_coverage": core_fraction + tail_fraction + transition_fraction > 0.8,
        }

        # Overall quality score
        quality["overall_score"] = (
            sum(
                [
                    quality["amplitude_ordering"],
                    quality["zone_balance"],
                    quality["total_coverage"],
                ]
            )
            / 3
        )

        return quality

    def visualize_zone_analysis(
        self, analysis_result: Dict[str, Any], output_path: str = "zone_analysis.png"
    ) -> None:
        """
        Visualize zone analysis results.

        Physical Meaning:
            Creates visualization of the zone analysis showing
            zone maps, indicators, and separation quality.

        Args:
            analysis_result (Dict[str, Any]): Results from separate_zones
            output_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Zone map (2D slice)
        ax1 = axes[0, 0]
        field_slice = analysis_result["core_mask"][
            :, :, analysis_result["core_mask"].shape[2] // 2
        ]
        im1 = ax1.imshow(field_slice, cmap="viridis", origin="lower")
        ax1.set_title("Core Zone Map")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.colorbar(im1, ax=ax1)

        # Plot 2: Tail zone map
        ax2 = axes[0, 1]
        tail_slice = analysis_result["tail_mask"][
            :, :, analysis_result["tail_mask"].shape[2] // 2
        ]
        im2 = ax2.imshow(tail_slice, cmap="plasma", origin="lower")
        ax2.set_title("Tail Zone Map")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        plt.colorbar(im2, ax=ax2)

        # Plot 3: N indicator
        ax3 = axes[1, 0]
        N_slice = analysis_result["indicators"]["N"][
            :, :, analysis_result["indicators"]["N"].shape[2] // 2
        ]
        im3 = ax3.imshow(N_slice, cmap="hot", origin="lower")
        ax3.set_title("N Indicator (Norm Gradient)")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        plt.colorbar(im3, ax=ax3)

        # Plot 4: S indicator
        ax4 = axes[1, 1]
        S_slice = analysis_result["indicators"]["S"][
            :, :, analysis_result["indicators"]["S"].shape[2] // 2
        ]
        im4 = ax4.imshow(S_slice, cmap="cool", origin="lower")
        ax4.set_title("S Indicator (Second Derivative)")
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        plt.colorbar(im4, ax=ax4)

        # Add text with results
        zone_stats = analysis_result["zone_stats"]
        quality = analysis_result["quality_metrics"]

        textstr = f'Core Radius: {analysis_result["r_core"]:.2f}\n'
        textstr += f'Tail Radius: {analysis_result["r_tail"]:.2f}\n'
        textstr += f'Core Fraction: {zone_stats["core"]["volume_fraction"]:.3f}\n'
        textstr += f'Tail Fraction: {zone_stats["tail"]["volume_fraction"]:.3f}\n'
        textstr += f'Quality Score: {quality["overall_score"]:.3f}'

        ax4.text(
            0.05,
            0.95,
            textstr,
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def run_zone_analysis_variations(
        self,
        field: np.ndarray,
        center: List[float],
        threshold_ranges: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Run zone analysis for different threshold values.

        Physical Meaning:
            Analyzes zone separation sensitivity to threshold parameters,
            helping to determine optimal separation criteria.

        Args:
            field (np.ndarray): Phase field solution
            center (List[float]): Center of the defect
            threshold_ranges (Dict[str, List[float]]): Ranges of threshold values

        Returns:
            Dict[str, Any]: Results for all threshold combinations
        """
        results = {}

        for N_core in threshold_ranges.get("N_core", [3.0]):
            for S_core in threshold_ranges.get("S_core", [1.0]):
                for N_tail in threshold_ranges.get("N_tail", [0.3]):
                    for S_tail in threshold_ranges.get("S_tail", [0.3]):
                        thresholds = {
                            "N_core": N_core,
                            "S_core": S_core,
                            "N_tail": N_tail,
                            "S_tail": S_tail,
                        }

                        try:
                            result = self.separate_zones(field, center, thresholds)
                            key = f"Nc{N_core}_Sc{S_core}_Nt{N_tail}_St{S_tail}"
                            results[key] = result
                        except Exception as e:
                            key = f"Nc{N_core}_Sc{S_core}_Nt{N_tail}_St{S_tail}"
                            results[key] = {"error": str(e), "passed": False}

        return results
