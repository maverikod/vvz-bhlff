"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power law tail analyzer for Level B BVP interface.

This module implements analysis of power law tails in homogeneous medium
for the Level B BVP interface, providing fundamental field properties
analysis including power law decay characteristics.

Physical Meaning:
    Analyzes the power law decay of BVP envelope amplitude in the tail
    region, which characterizes the field's long-range behavior in
    homogeneous medium according to the 7D phase field theory.

Mathematical Foundation:
    Computes power law decay A(r) ∝ r^(2β-3) in the tail region,
    where β is the fractional order and r is the radial distance
    from the field center.

Example:
    >>> analyzer = PowerLawAnalyzer()
    >>> tail_data = analyzer.analyze_power_law_tails(envelope)
"""

import numpy as np
from typing import Dict, Any


class PowerLawAnalyzer:
    """
    Power law tail analyzer for Level B BVP interface.

    Physical Meaning:
        Analyzes the power law decay of BVP envelope amplitude in the
        tail region, which characterizes the field's long-range behavior
        in homogeneous medium according to the 7D phase field theory.

    Mathematical Foundation:
        Computes power law decay A(r) ∝ r^(2β-3) in the tail region,
        where β is the fractional order and r is the radial distance
        from the field center.
    """

    def analyze_power_law_tails(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze power law tails in homogeneous medium.

        Physical Meaning:
            Computes the power law decay of BVP envelope amplitude
            in the tail region, which characterizes the field's
            long-range behavior in homogeneous medium.

        Mathematical Foundation:
            Fits power law A(r) ∝ r^α in the tail region using
            linear regression on log-log scale: log(A) = α log(r) + C

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - tail_slope: Power law exponent α
                - r_squared: R-squared value of the fit
                - power_law_range: Range of radial distances used
        """
        amplitude = np.abs(envelope)

        # Compute radial profile from center
        center = np.array(amplitude.shape) // 2
        x, y, z = np.meshgrid(
            np.arange(amplitude.shape[0]) - center[0],
            np.arange(amplitude.shape[1]) - center[1],
            np.arange(amplitude.shape[2]) - center[2],
            indexing="ij",
        )
        r = np.sqrt(x**2 + y**2 + z**2)

        # Find tail region (outer 50% of domain)
        r_max = np.max(r)
        tail_mask = r > 0.5 * r_max

        if np.sum(tail_mask) < 10:  # Need sufficient points
            return {
                "tail_slope": -2.0,
                "r_squared": 0.0,
                "power_law_range": [0.5 * r_max, r_max],
            }

        # Extract tail data
        r_tail = r[tail_mask]
        amp_tail = amplitude[tail_mask]

        # Remove zeros and take log
        valid_mask = amp_tail > 1e-12
        if np.sum(valid_mask) < 5:
            return {
                "tail_slope": -2.0,
                "r_squared": 0.0,
                "power_law_range": [0.5 * r_max, r_max],
            }

        log_r = np.log(r_tail[valid_mask])
        log_amp = np.log(amp_tail[valid_mask])

        # Linear fit: log(amp) = slope * log(r) + intercept
        try:
            slope, intercept = np.polyfit(log_r, log_amp, 1)

            # Compute R-squared
            amp_pred = np.exp(slope * log_r + intercept)
            ss_res = np.sum((amp_tail[valid_mask] - amp_pred) ** 2)
            ss_tot = np.sum((amp_tail[valid_mask] - np.mean(amp_tail[valid_mask])) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            return {
                "tail_slope": float(slope),
                "r_squared": float(r_squared),
                "power_law_range": [float(0.5 * r_max), float(r_max)],
            }
        except (np.linalg.LinAlgError, ValueError):
            return {
                "tail_slope": -2.0,
                "r_squared": 0.0,
                "power_law_range": [0.5 * r_max, r_max],
            }

    def compute_radial_profile(
        self, envelope: np.ndarray, n_bins: int = 50
    ) -> Dict[str, Any]:
        """
        Compute radial profile of envelope amplitude.

        Physical Meaning:
            Computes the radial average of envelope amplitude for
            analysis of field structure and power law behavior.

        Args:
            envelope (np.ndarray): BVP envelope field.
            n_bins (int): Number of radial bins.

        Returns:
            Dict[str, Any]: Dictionary containing radial profile data.
        """
        amplitude = np.abs(envelope)

        # Compute radial profile from center
        center = np.array(amplitude.shape) // 2
        x, y, z = np.meshgrid(
            np.arange(amplitude.shape[0]) - center[0],
            np.arange(amplitude.shape[1]) - center[1],
            np.arange(amplitude.shape[2]) - center[2],
            indexing="ij",
        )
        r = np.sqrt(x**2 + y**2 + z**2)

        # Find maximum radius
        r_max = np.max(r)

        # Compute radial average
        r_bins = np.linspace(0, r_max, n_bins)
        radial_profile = []
        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if np.sum(mask) > 0:
                radial_profile.append(np.mean(amplitude[mask]))
            else:
                radial_profile.append(0.0)

        radial_profile = np.array(radial_profile)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2

        return {
            "radial_distances": r_centers,
            "radial_amplitudes": radial_profile,
            "r_max": r_max,
        }
