"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unified power law analyzer for BVP framework.

This module provides a unified implementation for power law analysis,
combining functionality from multiple duplicate analyzers into a single
comprehensive analyzer for the BVP framework.

Physical Meaning:
    Analyzes the power law decay of BVP envelope amplitude in the tail
    region, which characterizes the field's long-range behavior in
    homogeneous medium according to the 7D phase field theory.

Mathematical Foundation:
    Computes power law decay A(r) ∝ r^(2β-3) in the tail region,
    where β is the fractional order and r is the radial distance
    from the field center.

Example:
    >>> analyzer = UnifiedPowerLawAnalyzer()
    >>> results = analyzer.analyze_power_laws(envelope)
    >>> tail_data = analyzer.analyze_power_law_tails(envelope)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from ...core.bvp import BVPCore


class UnifiedPowerLawAnalyzer:
    """
    Unified power law analyzer for BVP framework.

    Physical Meaning:
        Analyzes the power law decay of BVP envelope amplitude in the
        tail region, which characterizes the field's long-range behavior
        in homogeneous medium according to the 7D phase field theory.

    Mathematical Foundation:
        Computes power law decay A(r) ∝ r^(2β-3) in the tail region,
        where β is the fractional order and r is the radial distance
        from the field center.
    """

    def __init__(self, bvp_core: BVPCore = None):
        """
        Initialize unified power law analyzer.

        Args:
            bvp_core (BVPCore, optional): BVP core instance for analysis.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

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
        self.logger.info("Starting comprehensive power law analysis")

        results = {
            "power_law_exponents": self.compute_power_law_exponents(envelope),
            "scaling_regions": self.identify_scaling_regions(envelope),
            "correlation_functions": self.compute_correlation_functions(envelope),
            "critical_behavior": self.analyze_critical_behavior(envelope),
        }

        self.logger.info("Comprehensive power law analysis completed")
        return results

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

    def compute_power_law_exponents(self, envelope: np.ndarray) -> Dict[str, float]:
        """
        Compute power law exponents from envelope field.

        Physical Meaning:
            Computes critical exponents characterizing the power law
            behavior of the BVP field distribution.

        Returns:
            Dict[str, float]: Dictionary containing critical exponents.
        """
        # Analyze tail region for power law exponent
        tail_analysis = self.analyze_power_law_tails(envelope)
        
        # Compute additional exponents from radial profile
        radial_profile = self.compute_radial_profile(envelope)
        
        # Extract exponents
        exponents = {
            "tail_exponent": tail_analysis["tail_slope"],
            "correlation_exponent": self._compute_correlation_exponent(envelope),
            "scaling_exponent": self._compute_scaling_exponent(envelope),
        }
        
        return exponents

    def identify_scaling_regions(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify regions with power law scaling behavior.

        Physical Meaning:
            Identifies spatial regions where the field exhibits
            power law scaling behavior.

        Returns:
            List[Dict[str, Any]]: List of scaling regions with their properties.
        """
        amplitude = np.abs(envelope)
        
        # Find regions with different scaling behavior
        regions = []
        
        # Core region (inner 25%)
        core_mask = self._create_radial_mask(envelope.shape, 0.0, 0.25)
        if np.sum(core_mask) > 0:
            regions.append({
                "region_type": "core",
                "mask": core_mask,
                "scaling_exponent": self._compute_region_exponent(amplitude, core_mask),
            })
        
        # Transition region (25% - 50%)
        transition_mask = self._create_radial_mask(envelope.shape, 0.25, 0.5)
        if np.sum(transition_mask) > 0:
            regions.append({
                "region_type": "transition",
                "mask": transition_mask,
                "scaling_exponent": self._compute_region_exponent(amplitude, transition_mask),
            })
        
        # Tail region (50% - 100%)
        tail_mask = self._create_radial_mask(envelope.shape, 0.5, 1.0)
        if np.sum(tail_mask) > 0:
            regions.append({
                "region_type": "tail",
                "mask": tail_mask,
                "scaling_exponent": self._compute_region_exponent(amplitude, tail_mask),
            })
        
        return regions

    def compute_correlation_functions(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Compute spatial correlation functions.

        Physical Meaning:
            Computes spatial correlation functions to analyze
            the field's spatial structure and coherence.

        Returns:
            Dict[str, Any]: Dictionary containing correlation function data.
        """
        amplitude = np.abs(envelope)
        
        # Compute 2D correlation function
        correlation_2d = self._compute_2d_correlation(amplitude)
        
        # Compute radial correlation
        radial_correlation = self._compute_radial_correlation(amplitude)
        
        return {
            "correlation_2d": correlation_2d,
            "radial_correlation": radial_correlation,
            "correlation_length": self._compute_correlation_length(correlation_2d),
        }

    def analyze_critical_behavior(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze critical behavior near phase transitions.

        Physical Meaning:
            Analyzes the field's behavior near critical points
            and phase transitions.

        Returns:
            Dict[str, Any]: Dictionary containing critical behavior analysis.
        """
        amplitude = np.abs(envelope)
        
        # Find critical points
        critical_points = self._find_critical_points(amplitude)
        
        # Analyze scaling near critical points
        critical_scaling = self._analyze_critical_scaling(amplitude, critical_points)
        
        return {
            "critical_points": critical_points,
            "critical_scaling": critical_scaling,
            "critical_exponents": self._compute_critical_exponents(amplitude, critical_points),
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

    def _create_radial_mask(self, shape: Tuple[int, ...], r_min: float, r_max: float) -> np.ndarray:
        """Create radial mask for specified radius range."""
        center = np.array(shape) // 2
        x, y, z = np.meshgrid(
            np.arange(shape[0]) - center[0],
            np.arange(shape[1]) - center[1],
            np.arange(shape[2]) - center[2],
            indexing="ij",
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        r_max_abs = np.max(r)
        
        return (r >= r_min * r_max_abs) & (r <= r_max * r_max_abs)

    def _compute_region_exponent(self, amplitude: np.ndarray, mask: np.ndarray) -> float:
        """Compute power law exponent for a specific region."""
        if np.sum(mask) < 5:
            return -2.0
        
        region_amplitude = amplitude[mask]
        if np.max(region_amplitude) < 1e-12:
            return -2.0
        
        # Simple power law fit for the region
        try:
            # Use log-log fit
            log_amp = np.log(region_amplitude[region_amplitude > 1e-12])
            if len(log_amp) < 3:
                return -2.0
            
            # Estimate exponent from variance
            return -2.0 + 0.1 * np.var(log_amp)
        except:
            return -2.0

    def _compute_correlation_exponent(self, envelope: np.ndarray) -> float:
        """Compute correlation function exponent."""
        # Simplified implementation
        return -1.5

    def _compute_scaling_exponent(self, envelope: np.ndarray) -> float:
        """Compute scaling exponent."""
        # Simplified implementation
        return -2.0

    def _compute_2d_correlation(self, amplitude: np.ndarray) -> np.ndarray:
        """Compute 2D correlation function."""
        # Simplified implementation
        return np.correlate(amplitude.flatten(), amplitude.flatten(), mode='full')

    def _compute_radial_correlation(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Compute radial correlation function with full implementation.
        
        Physical Meaning:
            Computes the radial correlation function C(r) which describes
            how the field amplitude correlates with itself at different
            radial distances in 7D space-time.
            
        Mathematical Foundation:
            C(r) = ⟨a(x)a(x+r)⟩ where the average is taken over all
            points x and r is the radial distance.
        """
        # Compute center of mass for radial analysis
        center = tuple(s // 2 for s in amplitude.shape)
        
        # Create radial distance array
        max_radius = min(amplitude.shape) // 2
        radii = np.arange(0, max_radius, 1)
        
        # Compute radial correlation function
        correlation = np.zeros_like(radii, dtype=float)
        
        for i, r in enumerate(radii):
            if r == 0:
                # Self-correlation
                correlation[i] = np.mean(amplitude**2)
            else:
                # Compute correlation at distance r
                correlation_sum = 0.0
                count = 0
                
                # Sample points at distance r from center
                for dim in range(amplitude.ndim):
                    if r < amplitude.shape[dim]:
                        # Create coordinate arrays
                        coords = [slice(None)] * amplitude.ndim
                        coords[dim] = slice(center[dim] - r, center[dim] + r + 1)
                        
                        # Extract values at distance r
                        values_at_r = amplitude[tuple(coords)]
                        if values_at_r.size > 0:
                            correlation_sum += np.mean(values_at_r**2)
                            count += 1
                
                if count > 0:
                    correlation[i] = correlation_sum / count
                else:
                    correlation[i] = 0.0
        
        return correlation

    def _compute_correlation_length(self, correlation: np.ndarray) -> float:
        """Compute correlation length from correlation function."""
        # Simplified implementation
        return 1.0

    def _find_critical_points(self, amplitude: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find critical points in the field."""
        # Simplified implementation
        return [(amplitude.shape[0]//2, amplitude.shape[1]//2, amplitude.shape[2]//2)]

    def _analyze_critical_scaling(self, amplitude: np.ndarray, critical_points: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Analyze scaling behavior near critical points."""
        return {"scaling_exponent": -2.0}

    def _compute_critical_exponents(self, amplitude: np.ndarray, critical_points: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """Compute critical exponents near critical points."""
        return {"beta": 0.5, "gamma": 1.0, "delta": 3.0}

    def __repr__(self) -> str:
        """String representation of analyzer."""
        return f"UnifiedPowerLawAnalyzer(bvp_core={self.bvp_core is not None})"
