"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Beating comparison module.

This module implements comparison functionality
for analyzing mode beating in the 7D phase field according to the
theoretical framework.

Physical Meaning:
    Implements comparison of beating analysis results
    to identify differences, similarities, and consistency.

Example:
    >>> comparator = BeatingComparator(bvp_core)
    >>> results = comparator.compare_analyses(results1, results2)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class BeatingComparator:
    """
    Beating comparison for Level C.

    Physical Meaning:
        Compares beating analysis results to identify
        differences, similarities, and consistency.

    Mathematical Foundation:
        Implements comparison methods for beating analysis:
        - Statistical comparison of results
        - Pattern similarity analysis
        - Consistency validation
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize beating comparator.

        Physical Meaning:
            Sets up the comparison system with
            comparison parameters and methods.

        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

        # Comparison parameters
        self.similarity_threshold = 0.8
        self.difference_threshold = 0.2
        self.consistency_threshold = 0.9

    def compare_analyses(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two beating analysis results.

        Physical Meaning:
            Compares two sets of beating analysis results to
            identify differences, similarities, and consistency.

        Args:
            results1 (Dict[str, Any]): First analysis results.
            results2 (Dict[str, Any]): Second analysis results.

        Returns:
            Dict[str, Any]: Comparison results.
        """
        self.logger.info("Starting beating analysis comparison")

        # Compare basic analysis
        basic_comparison = self._compare_basic_analysis(results1, results2)

        # Compare interference patterns
        interference_comparison = self._compare_interference_patterns(results1, results2)

        # Compare mode coupling
        coupling_comparison = self._compare_mode_coupling(results1, results2)

        # Compare phase coherence
        phase_comparison = self._compare_phase_coherence(results1, results2)

        # Compare beating frequencies
        frequency_comparison = self._compare_beating_frequencies(results1, results2)

        # Calculate overall comparison
        overall_comparison = self._calculate_overall_comparison(
            basic_comparison, interference_comparison, coupling_comparison, phase_comparison, frequency_comparison
        )

        # Combine all comparison results
        comparison_results = {
            "basic_comparison": basic_comparison,
            "interference_comparison": interference_comparison,
            "coupling_comparison": coupling_comparison,
            "phase_comparison": phase_comparison,
            "frequency_comparison": frequency_comparison,
            "overall_comparison": overall_comparison,
            "comparison_complete": True,
        }

        self.logger.info("Beating analysis comparison completed")
        return comparison_results

    def _compare_basic_analysis(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare basic analysis results.

        Physical Meaning:
            Compares basic analysis results between
            two analysis runs.

        Args:
            results1 (Dict[str, Any]): First analysis results.
            results2 (Dict[str, Any]): Second analysis results.

        Returns:
            Dict[str, Any]: Basic analysis comparison.
        """
        # Extract basic analysis results
        basic1 = results1.get("basic_analysis", {})
        basic2 = results2.get("basic_analysis", {})

        # Compare metrics
        comparison_metrics = self._compare_metrics(basic1, basic2)

        # Calculate similarity
        similarity = self._calculate_similarity(comparison_metrics)

        # Calculate differences
        differences = self._calculate_differences(comparison_metrics)

        return {
            "comparison_metrics": comparison_metrics,
            "similarity": similarity,
            "differences": differences,
            "are_similar": similarity > self.similarity_threshold,
        }

    def _compare_metrics(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare metrics between two analyses.

        Physical Meaning:
            Compares specific metrics between two
            analysis results.

        Args:
            metrics1 (Dict[str, Any]): First analysis metrics.
            metrics2 (Dict[str, Any]): Second analysis metrics.

        Returns:
            Dict[str, Any]: Metrics comparison.
        """
        # Compare amplitude metrics
        amplitude_comparison = self._compare_amplitude_metrics(metrics1, metrics2)

        # Compare energy metrics
        energy_comparison = self._compare_energy_metrics(metrics1, metrics2)

        # Compare variance metrics
        variance_comparison = self._compare_variance_metrics(metrics1, metrics2)

        return {
            "amplitude_comparison": amplitude_comparison,
            "energy_comparison": energy_comparison,
            "variance_comparison": variance_comparison,
        }

    def _compare_amplitude_metrics(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare amplitude metrics.

        Physical Meaning:
            Compares amplitude-related metrics between
            two analysis results.

        Args:
            metrics1 (Dict[str, Any]): First analysis metrics.
            metrics2 (Dict[str, Any]): Second analysis metrics.

        Returns:
            Dict[str, Any]: Amplitude metrics comparison.
        """
        # Extract amplitude metrics
        mean_amp1 = metrics1.get("mean_amplitude", 0.0)
        mean_amp2 = metrics2.get("mean_amplitude", 0.0)
        max_amp1 = metrics1.get("max_amplitude", 0.0)
        max_amp2 = metrics2.get("max_amplitude", 0.0)
        min_amp1 = metrics1.get("min_amplitude", 0.0)
        min_amp2 = metrics2.get("min_amplitude", 0.0)

        # Calculate differences
        mean_diff = abs(mean_amp1 - mean_amp2)
        max_diff = abs(max_amp1 - max_amp2)
        min_diff = abs(min_amp1 - min_amp2)

        # Calculate relative differences
        mean_rel_diff = mean_diff / (mean_amp1 + mean_amp2 + 1e-12)
        max_rel_diff = max_diff / (max_amp1 + max_amp2 + 1e-12)
        min_rel_diff = min_diff / (min_amp1 + min_amp2 + 1e-12)

        return {
            "mean_difference": mean_diff,
            "max_difference": max_diff,
            "min_difference": min_diff,
            "mean_relative_difference": mean_rel_diff,
            "max_relative_difference": max_rel_diff,
            "min_relative_difference": min_rel_diff,
        }

    def _compare_energy_metrics(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare energy metrics.

        Physical Meaning:
            Compares energy-related metrics between
            two analysis results.

        Args:
            metrics1 (Dict[str, Any]): First analysis metrics.
            metrics2 (Dict[str, Any]): Second analysis metrics.

        Returns:
            Dict[str, Any]: Energy metrics comparison.
        """
        # Extract energy metrics
        energy1 = metrics1.get("field_energy", 0.0)
        energy2 = metrics2.get("field_energy", 0.0)

        # Calculate differences
        energy_diff = abs(energy1 - energy2)
        energy_rel_diff = energy_diff / (energy1 + energy2 + 1e-12)

        return {
            "energy_difference": energy_diff,
            "energy_relative_difference": energy_rel_diff,
        }

    def _compare_variance_metrics(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare variance metrics.

        Physical Meaning:
            Compares variance-related metrics between
            two analysis results.

        Args:
            metrics1 (Dict[str, Any]): First analysis metrics.
            metrics2 (Dict[str, Any]): Second analysis metrics.

        Returns:
            Dict[str, Any]: Variance metrics comparison.
        """
        # Extract variance metrics
        variance1 = metrics1.get("spatial_variance", 0.0)
        variance2 = metrics2.get("spatial_variance", 0.0)

        # Calculate differences
        variance_diff = abs(variance1 - variance2)
        variance_rel_diff = variance_diff / (variance1 + variance2 + 1e-12)

        return {
            "variance_difference": variance_diff,
            "variance_relative_difference": variance_rel_diff,
        }

    def _calculate_similarity(self, comparison_metrics: Dict[str, Any]) -> float:
        """
        Calculate similarity between analyses.

        Physical Meaning:
            Calculates the overall similarity between
            two analysis results.

        Args:
            comparison_metrics (Dict[str, Any]): Comparison metrics.

        Returns:
            float: Similarity score.
        """
        # Calculate similarity based on relative differences
        amplitude_sim = 1.0 - comparison_metrics["amplitude_comparison"]["mean_relative_difference"]
        energy_sim = 1.0 - comparison_metrics["energy_comparison"]["energy_relative_difference"]
        variance_sim = 1.0 - comparison_metrics["variance_comparison"]["variance_relative_difference"]

        # Calculate overall similarity
        overall_similarity = np.mean([amplitude_sim, energy_sim, variance_sim])

        return float(overall_similarity)

    def _calculate_differences(self, comparison_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate differences between analyses.

        Physical Meaning:
            Calculates the differences between
            two analysis results.

        Args:
            comparison_metrics (Dict[str, Any]): Comparison metrics.

        Returns:
            Dict[str, Any]: Differences analysis.
        """
        # Calculate overall differences
        overall_difference = np.mean([
            comparison_metrics["amplitude_comparison"]["mean_relative_difference"],
            comparison_metrics["energy_comparison"]["energy_relative_difference"],
            comparison_metrics["variance_comparison"]["variance_relative_difference"],
        ])

        # Determine difference level
        if overall_difference < 0.1:
            difference_level = "minimal"
        elif overall_difference < 0.3:
            difference_level = "moderate"
        else:
            difference_level = "significant"

        return {
            "overall_difference": overall_difference,
            "difference_level": difference_level,
            "are_different": overall_difference > self.difference_threshold,
        }

    def _compare_interference_patterns(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare interference patterns.

        Physical Meaning:
            Compares interference patterns between
            two analysis results.

        Args:
            results1 (Dict[str, Any]): First analysis results.
            results2 (Dict[str, Any]): Second analysis results.

        Returns:
            Dict[str, Any]: Interference patterns comparison.
        """
        # Extract interference results
        interference1 = results1.get("interference_patterns", {})
        interference2 = results2.get("interference_patterns", {})

        # Compare interference strength
        strength1 = interference1.get("interference_strength", 0.0)
        strength2 = interference2.get("interference_strength", 0.0)
        strength_diff = abs(strength1 - strength2)

        # Compare interference regions
        regions1 = interference1.get("interference_regions", [])
        regions2 = interference2.get("interference_regions", [])
        regions_similarity = self._compare_interference_regions(regions1, regions2)

        # Compare interference coherence
        coherence1 = interference1.get("interference_coherence", {})
        coherence2 = interference2.get("interference_coherence", {})
        coherence_similarity = self._compare_interference_coherence(coherence1, coherence2)

        return {
            "strength_difference": strength_diff,
            "regions_similarity": regions_similarity,
            "coherence_similarity": coherence_similarity,
            "overall_similarity": np.mean([regions_similarity, coherence_similarity]),
        }

    def _compare_interference_regions(self, regions1: List[Dict[str, Any]], regions2: List[Dict[str, Any]]) -> float:
        """
        Compare interference regions.

        Physical Meaning:
            Compares interference regions between
            two analysis results.

        Args:
            regions1 (List[Dict[str, Any]]): First analysis regions.
            regions2 (List[Dict[str, Any]]): Second analysis regions.

        Returns:
            float: Regions similarity.
        """
        # Calculate regions similarity
        if len(regions1) == 0 and len(regions2) == 0:
            return 1.0
        elif len(regions1) == 0 or len(regions2) == 0:
            return 0.0
        else:
            # Simplified similarity calculation
            # In practice, this would involve proper region comparison
            return 0.8  # Placeholder value

    def _compare_interference_coherence(self, coherence1: Dict[str, Any], coherence2: Dict[str, Any]) -> float:
        """
        Compare interference coherence.

        Physical Meaning:
            Compares interference coherence between
            two analysis results.

        Args:
            coherence1 (Dict[str, Any]): First analysis coherence.
            coherence2 (Dict[str, Any]): Second analysis coherence.

        Returns:
            float: Coherence similarity.
        """
        # Calculate coherence similarity
        overall_coherence1 = coherence1.get("overall_coherence", 0.0)
        overall_coherence2 = coherence2.get("overall_coherence", 0.0)
        coherence_diff = abs(overall_coherence1 - overall_coherence2)

        # Calculate similarity
        similarity = 1.0 - coherence_diff

        return float(similarity)

    def _compare_mode_coupling(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare mode coupling.

        Physical Meaning:
            Compares mode coupling between
            two analysis results.

        Args:
            results1 (Dict[str, Any]): First analysis results.
            results2 (Dict[str, Any]): Second analysis results.

        Returns:
            Dict[str, Any]: Mode coupling comparison.
        """
        # Extract coupling results
        coupling1 = results1.get("mode_coupling", {})
        coupling2 = results2.get("mode_coupling", {})

        # Compare coupling strength
        strength1 = coupling1.get("coupling_strength", 0.0)
        strength2 = coupling2.get("coupling_strength", 0.0)
        strength_diff = abs(strength1 - strength2)

        # Compare coupling efficiency
        efficiency1 = coupling1.get("coupling_efficiency", 0.0)
        efficiency2 = coupling2.get("coupling_efficiency", 0.0)
        efficiency_diff = abs(efficiency1 - efficiency2)

        # Calculate overall similarity
        overall_similarity = 1.0 - np.mean([strength_diff, efficiency_diff])

        return {
            "strength_difference": strength_diff,
            "efficiency_difference": efficiency_diff,
            "overall_similarity": overall_similarity,
        }

    def _compare_phase_coherence(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare phase coherence.

        Physical Meaning:
            Compares phase coherence between
            two analysis results.

        Args:
            results1 (Dict[str, Any]): First analysis results.
            results2 (Dict[str, Any]): Second analysis results.

        Returns:
            Dict[str, Any]: Phase coherence comparison.
        """
        # Extract phase results
        phase1 = results1.get("phase_coherence", {})
        phase2 = results2.get("phase_coherence", {})

        # Compare phase coherence
        coherence1 = phase1.get("phase_coherence", 0.0)
        coherence2 = phase2.get("phase_coherence", 0.0)
        coherence_diff = abs(coherence1 - coherence2)

        # Compare phase stability
        stability1 = phase1.get("phase_stability", 0.0)
        stability2 = phase2.get("phase_stability", 0.0)
        stability_diff = abs(stability1 - stability2)

        # Calculate overall similarity
        overall_similarity = 1.0 - np.mean([coherence_diff, stability_diff])

        return {
            "coherence_difference": coherence_diff,
            "stability_difference": stability_diff,
            "overall_similarity": overall_similarity,
        }

    def _compare_beating_frequencies(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare beating frequencies.

        Physical Meaning:
            Compares beating frequencies between
            two analysis results.

        Args:
            results1 (Dict[str, Any]): First analysis results.
            results2 (Dict[str, Any]): Second analysis results.

        Returns:
            Dict[str, Any]: Beating frequencies comparison.
        """
        # Extract frequency results
        frequency1 = results1.get("beating_frequencies", {})
        frequency2 = results2.get("beating_frequencies", {})

        # Compare frequencies
        frequencies1 = frequency1.get("beating_frequencies", [])
        frequencies2 = frequency2.get("beating_frequencies", [])
        frequency_similarity = self._compare_frequency_lists(frequencies1, frequencies2)

        # Compare beating strength
        strength1 = frequency1.get("beating_strength", 0.0)
        strength2 = frequency2.get("beating_strength", 0.0)
        strength_diff = abs(strength1 - strength2)

        # Calculate overall similarity
        overall_similarity = np.mean([frequency_similarity, 1.0 - strength_diff])

        return {
            "frequency_similarity": frequency_similarity,
            "strength_difference": strength_diff,
            "overall_similarity": overall_similarity,
        }

    def _compare_frequency_lists(self, frequencies1: List[float], frequencies2: List[float]) -> float:
        """
        Compare frequency lists.

        Physical Meaning:
            Compares lists of frequencies between
            two analysis results.

        Args:
            frequencies1 (List[float]): First analysis frequencies.
            frequencies2 (List[float]): Second analysis frequencies.

        Returns:
            float: Frequency similarity.
        """
        # Calculate frequency similarity
        if len(frequencies1) == 0 and len(frequencies2) == 0:
            return 1.0
        elif len(frequencies1) == 0 or len(frequencies2) == 0:
            return 0.0
        else:
            # Simplified frequency comparison
            # In practice, this would involve proper frequency matching
            return 0.7  # Placeholder value

    def _calculate_overall_comparison(
        self, basic_comparison: Dict[str, Any], interference_comparison: Dict[str, Any],
        coupling_comparison: Dict[str, Any], phase_comparison: Dict[str, Any],
        frequency_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall comparison.

        Physical Meaning:
            Calculates the overall comparison between
            two analysis results.

        Args:
            basic_comparison (Dict[str, Any]): Basic analysis comparison.
            interference_comparison (Dict[str, Any]): Interference comparison.
            coupling_comparison (Dict[str, Any]): Coupling comparison.
            phase_comparison (Dict[str, Any]): Phase comparison.
            frequency_comparison (Dict[str, Any]): Frequency comparison.

        Returns:
            Dict[str, Any]: Overall comparison.
        """
        # Calculate overall similarity
        overall_similarity = np.mean([
            basic_comparison["similarity"],
            interference_comparison["overall_similarity"],
            coupling_comparison["overall_similarity"],
            phase_comparison["overall_similarity"],
            frequency_comparison["overall_similarity"],
        ])

        # Calculate overall consistency
        overall_consistency = self._calculate_consistency(
            basic_comparison, interference_comparison, coupling_comparison,
            phase_comparison, frequency_comparison
        )

        # Determine comparison result
        if overall_similarity > self.similarity_threshold:
            comparison_result = "highly_similar"
        elif overall_similarity > 0.5:
            comparison_result = "moderately_similar"
        else:
            comparison_result = "different"

        return {
            "overall_similarity": overall_similarity,
            "overall_consistency": overall_consistency,
            "comparison_result": comparison_result,
            "are_consistent": overall_consistency > self.consistency_threshold,
        }

    def _calculate_consistency(
        self, basic_comparison: Dict[str, Any], interference_comparison: Dict[str, Any],
        coupling_comparison: Dict[str, Any], phase_comparison: Dict[str, Any],
        frequency_comparison: Dict[str, Any]
    ) -> float:
        """
        Calculate consistency.

        Physical Meaning:
            Calculates the consistency between
            different aspects of the analysis.

        Args:
            basic_comparison (Dict[str, Any]): Basic analysis comparison.
            interference_comparison (Dict[str, Any]): Interference comparison.
            coupling_comparison (Dict[str, Any]): Coupling comparison.
            phase_comparison (Dict[str, Any]): Phase comparison.
            frequency_comparison (Dict[str, Any]): Frequency comparison.

        Returns:
            float: Consistency score.
        """
        # Calculate consistency based on similarity across all aspects
        similarities = [
            basic_comparison["similarity"],
            interference_comparison["overall_similarity"],
            coupling_comparison["overall_similarity"],
            phase_comparison["overall_similarity"],
            frequency_comparison["overall_similarity"],
        ]

        # Calculate consistency as inverse of variance
        consistency = 1.0 - np.var(similarities)

        return float(consistency)
