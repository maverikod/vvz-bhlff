"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power law core analysis module for Level B.

This module implements core power law analysis operations for Level B
of the 7D phase field theory, focusing on power law behavior and scaling.

Physical Meaning:
    Analyzes power law characteristics of the BVP field distribution,
    identifying scaling behavior, critical exponents, and correlation
    functions in the 7D space-time.

Mathematical Foundation:
    Implements power law analysis including:
    - Power law exponent computation
    - Scaling region identification
    - Correlation function analysis
    - Critical behavior analysis

Example:
    >>> core = PowerLawCore(bvp_core)
    >>> exponents = core.compute_power_law_exponents(envelope)
"""

import numpy as np
from typing import Dict, Any, List
import logging

from ...core.bvp import BVPCore


class PowerLawCore:
    """
    Core power law analysis for BVP field.

    Physical Meaning:
        Implements core power law analysis operations for identifying
        scaling behavior and critical exponents in BVP field distributions.

    Mathematical Foundation:
        Analyzes power law behavior using statistical methods including
        log-log regression, correlation analysis, and scaling region
        identification.
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize power law core analyzer.

        Args:
            bvp_core (BVPCore): BVP core instance for analysis.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def compute_power_law_exponents(self, envelope: np.ndarray) -> Dict[str, float]:
        """
        Compute power law exponents from field distribution.

        Physical Meaning:
            Computes power law exponents by analyzing the amplitude
            distribution of the BVP field, identifying scaling behavior
            in the field structure.

        Mathematical Foundation:
            Uses log-log regression to fit power law distributions:
            P(x) ~ x^(-α) where α is the power law exponent.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, float]: Power law exponents including:
                - amplitude_exponent: Power law exponent for amplitude distribution
        """
        # Analyze amplitude distribution
        amplitudes = np.abs(envelope)
        amplitudes = amplitudes[amplitudes > 0]  # Remove zeros

        if len(amplitudes) == 0:
            return {"amplitude_exponent": 0.0}

        # Simple power law fit (log-log regression)
        sorted_amplitudes = np.sort(amplitudes)[::-1]  # Descending order
        ranks = np.arange(1, len(sorted_amplitudes) + 1)

        # Fit power law: P(x) ~ x^(-α)
        log_ranks = np.log(ranks)
        log_amplitudes = np.log(sorted_amplitudes)

        # Linear regression in log space
        if len(log_ranks) > 1:
            slope = np.polyfit(log_ranks, log_amplitudes, 1)[0]
            exponent = -slope
        else:
            exponent = 0.0

        return {"amplitude_exponent": exponent}

    def identify_scaling_regions(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify regions with power law scaling behavior.

        Physical Meaning:
            Identifies spatial regions where the BVP field exhibits
            power law scaling behavior, indicating critical regions
            in the field structure.

        Mathematical Foundation:
            Analyzes different spatial regions to identify consistent
            scaling behavior using power law fitting methods.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            List[Dict[str, Any]]: List of scaling regions with properties:
                - center: Center coordinates of the region
                - radius: Radius of the region
                - scaling_type: Type of scaling behavior
                - exponent: Power law exponent for the region
        """
        # Simple implementation: identify regions with consistent scaling
        regions = []

        # Analyze different spatial regions
        domain = self.bvp_core.domain
        if hasattr(domain, "shape"):
            shape = domain.shape
            if len(shape) >= 3:
                # Analyze center region
                center = tuple(s // 2 for s in shape)
                region = {
                    "center": center,
                    "radius": min(shape) // 4,
                    "scaling_type": "central",
                    "exponent": self.compute_power_law_exponents(envelope)[
                        "amplitude_exponent"
                    ],
                }
                regions.append(region)

        return regions

    def compute_correlation_functions(
        self, envelope: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute spatial correlation functions.

        Physical Meaning:
            Computes spatial correlation functions to analyze the
            spatial structure and coherence of the BVP field.

        Mathematical Foundation:
            Computes correlation functions C(r) = ⟨f(x)f(x+r)⟩
            to analyze spatial correlations in the field.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, np.ndarray]: Correlation functions including:
                - spatial_correlation: Spatial correlation function
                - correlation_length: Characteristic correlation length
        """
        # Simple correlation analysis
        amplitude = np.abs(envelope)

        # Compute spatial correlation (simplified)
        if amplitude.ndim >= 3:
            # Compute correlation along one dimension
            correlation = np.correlate(
                amplitude.flatten(), amplitude.flatten(), mode="full"
            )
            correlation = correlation[correlation.size // 2 :]
            correlation = correlation / correlation[0]  # Normalize
        else:
            correlation = np.array([1.0])

        return {
            "spatial_correlation": correlation,
            "correlation_length": len(correlation) // 2,
        }

    def analyze_critical_behavior(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze critical behavior in the field.

        Physical Meaning:
            Analyzes critical behavior and phase transitions in the
            BVP field, identifying critical points and scaling behavior.

        Mathematical Foundation:
            Analyzes field properties near critical points including
            scaling exponents and critical behavior indicators.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, Any]: Critical behavior analysis including:
                - critical_exponents: Critical scaling exponents
                - critical_regions: Regions with critical behavior
                - scaling_dimension: Effective scaling dimension
        """
        # Simple critical behavior analysis
        amplitude = np.abs(envelope)

        # Compute basic critical indicators
        max_amplitude = np.max(amplitude)
        mean_amplitude = np.mean(amplitude)

        # Simple critical exponent estimation
        if max_amplitude > 0:
            critical_exponent = np.log(mean_amplitude) / np.log(max_amplitude)
        else:
            critical_exponent = 0.0

        return {
            "critical_exponents": {"amplitude": critical_exponent},
            "critical_regions": [],
            "scaling_dimension": envelope.ndim,
        }

    def __repr__(self) -> str:
        """String representation of power law core."""
        return f"PowerLawCore(bvp_core={self.bvp_core})"
