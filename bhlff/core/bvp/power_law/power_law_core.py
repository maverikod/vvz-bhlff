"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core power law analysis for BVP framework.

This module implements the core power law analysis functionality
for analyzing power law behavior in BVP envelope fields.
"""

import numpy as np
from typing import Dict, Any, List
import logging

from ...bvp import BVPCore
from .power_law_comparison import PowerLawComparison
from .power_law_optimization import PowerLawOptimization
from .power_law_statistics import PowerLawStatistics


class PowerLawCore:
    """
    Core power law analyzer for BVP framework.

    Physical Meaning:
        Provides core analysis of power law behavior in BVP
        envelope fields, coordinating specialized analysis modules.
    """

    def __init__(self, bvp_core: BVPCore = None):
        """
        Initialize power law analyzer.

        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

        # Analysis parameters
        self.statistical_significance = 0.05
        self.optimization_tolerance = 1e-6
        self.max_optimization_iterations = 100

        # Initialize specialized modules
        self.comparison = PowerLawComparison(bvp_core)
        self.optimization = PowerLawOptimization(bvp_core)
        self.statistics = PowerLawStatistics(bvp_core)

    def analyze_envelope_power_laws(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze power law behavior in envelope field.

        Physical Meaning:
            Analyzes power law behavior in the envelope field by
            identifying tail regions and fitting power laws to them.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            List[Dict[str, Any]]: List of power law analysis results for each region.
        """
        self.logger.info("Starting envelope power law analysis")

        # Identify tail regions
        tail_regions = self._identify_tail_regions(envelope)

        # Analyze each region
        results = []
        for region in tail_regions:
            region_result = self._analyze_region_power_law(envelope, region)
            results.append(region_result)

        self.logger.info(
            f"Envelope power law analysis completed: {len(results)} regions analyzed"
        )
        return results

    def analyze_power_law_tails(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze power law tails in BVP envelope field.

        Physical Meaning:
            Analyzes the power law decay of BVP envelope amplitude in the
            tail region, which characterizes the field's long-range behavior
            in homogeneous medium according to the 7D phase field theory.

        Mathematical Foundation:
            Computes power law decay A(r) ∝ r^(2β-3) in the tail region,
            where β is the fractional order and r is the radial distance
            from the field center.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - tail_slope: Power law exponent α
                - r_squared: R-squared value of the fit
                - power_law_range: Range of radial distances used
        """
        # Get power law analysis results
        power_law_results = self.analyze_envelope_power_laws(envelope)

        if not power_law_results:
            return {"tail_slope": 0.0, "r_squared": 0.0, "power_law_range": [0.0, 0.0]}

        # Use the first (most significant) result
        first_result = power_law_results[0]

        return {
            "tail_slope": first_result.get("slope", 0.0),
            "r_squared": first_result.get("r_squared", 0.0),
            "power_law_range": first_result.get("range", [0.0, 0.0]),
        }

    def _identify_tail_regions(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify tail regions in the envelope field.

        Physical Meaning:
            Identifies regions in the envelope field that exhibit
            power law behavior, typically in the tails of the distribution.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            List[Dict[str, Any]]: List of identified tail regions.
        """
        tail_regions = []

        # Analyze each dimension
        for dim in range(envelope.ndim):
            # Create slice for this dimension
            dim_slice = np.take(envelope, envelope.shape[dim] // 2, axis=dim)

            # Find tail regions in this dimension
            dim_regions = self._find_dimension_tail_regions(dim_slice, dim)
            tail_regions.extend(dim_regions)

        return tail_regions

    def _find_dimension_tail_regions(
        self, dim_slice: np.ndarray, dimension: int
    ) -> List[Dict[str, Any]]:
        """
        Find tail regions in a specific dimension.

        Physical Meaning:
            Finds regions in a specific dimension that exhibit
            power law behavior based on amplitude thresholds.

        Args:
            dim_slice (np.ndarray): Slice of envelope field for this dimension.
            dimension (int): Dimension index.

        Returns:
            List[Dict[str, Any]]: List of tail regions in this dimension.
        """
        regions = []

        # Calculate amplitude threshold
        amplitude_threshold = 0.1 * np.max(np.abs(dim_slice))

        # Create mask for tail regions
        mask = np.abs(dim_slice) > amplitude_threshold

        # Find contiguous regions
        contiguous_regions = self._find_contiguous_regions(mask)

        # Convert to region dictionaries
        for region_indices in contiguous_regions:
            if len(region_indices) > 5:  # Minimum region size
                region = {
                    "dimension": dimension,
                    "indices": region_indices,
                    "start_index": min(region_indices),
                    "end_index": max(region_indices),
                    "size": len(region_indices),
                }
                regions.append(region)

        return regions

    def _find_contiguous_regions(self, mask: np.ndarray) -> List[List[int]]:
        """
        Find contiguous regions in a boolean mask.

        Physical Meaning:
            Finds contiguous regions of True values in a boolean mask,
            representing regions that satisfy the tail criteria.

        Args:
            mask (np.ndarray): Boolean mask.

        Returns:
            List[List[int]]: List of contiguous regions as lists of indices.
        """
        regions = []
        current_region = []

        for i, value in enumerate(mask):
            if np.any(value) if hasattr(value, "__len__") else value:
                current_region.append(i)
            else:
                if current_region:
                    regions.append(current_region)
                    current_region = []

        # Add final region if exists
        if current_region:
            regions.append(current_region)

        return regions

    def _analyze_region_power_law(
        self, envelope: np.ndarray, region: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze power law behavior in a specific region.

        Physical Meaning:
            Analyzes power law behavior in a specific region of the
            envelope field by fitting power law functions to the data.

        Args:
            envelope (np.ndarray): 7D envelope field data.
            region (Dict[str, Any]): Region information.

        Returns:
            Dict[str, Any]: Power law analysis results for this region.
        """
        # Extract region data
        region_data = self._extract_region_data(envelope, region)

        # Fit power law
        power_law_fit = self._fit_power_law(region_data)

        # Calculate fitting quality
        fitting_quality = self._calculate_fitting_quality(region_data, power_law_fit)

        return {
            "region_info": region,
            "power_law_fit": power_law_fit,
            "fitting_quality": fitting_quality,
            "region_data_summary": {
                "data_points": len(region_data.get("amplitudes", [])),
                "amplitude_range": [
                    np.min(region_data.get("amplitudes", [0])),
                    np.max(region_data.get("amplitudes", [0])),
                ],
                "distance_range": [
                    np.min(region_data.get("distances", [0])),
                    np.max(region_data.get("distances", [0])),
                ],
            },
        }

    def _extract_region_data(
        self, envelope: np.ndarray, region: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Extract data from a specific region.

        Physical Meaning:
            Extracts relevant data from a specific region of the
            envelope field for power law analysis.

        Args:
            envelope (np.ndarray): 7D envelope field data.
            region (Dict[str, Any]): Region information.

        Returns:
            Dict[str, np.ndarray]: Extracted region data.
        """
        dim = region["dimension"]
        indices = region["indices"]

        # Extract amplitudes and distances
        amplitudes = []
        distances = []

        for i, idx in enumerate(indices):
            # Create slice for this index
            slice_indices = [slice(None)] * envelope.ndim
            slice_indices[dim] = idx

            # Extract amplitude
            amplitude = np.abs(envelope[tuple(slice_indices)])
            amplitudes.append(amplitude)

            # Calculate distance from center
            center = envelope.shape[dim] // 2
            distance = abs(idx - center)
            distances.append(distance)

        return {"amplitudes": np.array(amplitudes), "distances": np.array(distances)}

    def _fit_power_law(self, region_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Fit power law to region data using full 7D BVP theory optimization.

        Physical Meaning:
            Fits a power law function to the region data using complete
            analytical methods based on 7D phase field theory principles.
            Implements full optimization with proper error handling and
            quality assessment.

        Mathematical Foundation:
            Uses scipy.optimize.curve_fit with proper bounds and constraints
            for accurate power law parameter estimation in 7D phase field theory.

        Args:
            region_data (Dict[str, np.ndarray]): Region data.

        Returns:
            Dict[str, float]: Power law fit parameters with full analysis.
        """
        try:
            from scipy.optimize import curve_fit

            amplitudes = region_data["amplitudes"]
            distances = region_data["distances"]

            # Avoid log of zero and ensure valid data
            valid_mask = (amplitudes > 0) & (distances > 0)
            if not np.any(valid_mask):
                return {
                    "exponent": 0.0,
                    "coefficient": 0.0,
                    "r_squared": 0.0,
                    "chi_squared": float("inf"),
                    "fitting_quality": 0.0,
                    "parameter_errors": [0.0, 0.0],
                }

            # Extract valid data
            valid_amplitudes = amplitudes[valid_mask]
            valid_distances = distances[valid_mask]

            if len(valid_distances) < 3:
                return {
                    "exponent": 0.0,
                    "coefficient": valid_amplitudes[0],
                    "r_squared": 0.0,
                    "chi_squared": float("inf"),
                    "fitting_quality": 0.0,
                    "parameter_errors": [0.0, 0.0],
                }

            # Define power law function for 7D BVP theory
            def power_law_func(r, amplitude, exponent):
                """Power law function for 7D phase field theory."""
                return amplitude * (r**exponent)

            # Initial parameter guesses based on 7D BVP theory
            initial_amplitude = np.mean(valid_amplitudes)
            initial_exponent = -2.0  # Typical 7D BVP exponent

            # Perform full curve fitting with proper bounds
            popt, pcov = curve_fit(
                power_law_func,
                valid_distances,
                valid_amplitudes,
                p0=[initial_amplitude, initial_exponent],
                maxfev=1000,
                bounds=([0.001, -10.0], [100.0, 0.0]),  # Reasonable bounds for 7D BVP
            )

            # Extract fitted parameters
            amplitude, exponent = popt

            # Compute comprehensive quality metrics
            r_squared = self._compute_r_squared_full(
                valid_distances, valid_amplitudes, popt, power_law_func
            )
            chi_squared = self._compute_chi_squared_full(
                valid_distances, valid_amplitudes, popt, power_law_func
            )
            reduced_chi_squared = chi_squared / (len(valid_distances) - 2)
            fitting_quality = self._compute_fitting_quality_full(pcov)

            # Compute parameter uncertainties
            parameter_errors = np.sqrt(np.diag(pcov))

            return {
                "exponent": float(exponent),
                "coefficient": float(amplitude),
                "r_squared": float(r_squared),
                "chi_squared": float(chi_squared),
                "reduced_chi_squared": float(reduced_chi_squared),
                "fitting_quality": float(fitting_quality),
                "parameter_errors": parameter_errors.tolist(),
                "covariance": pcov.tolist(),
            }

        except Exception as e:
            self.logger.error(f"Power law fitting failed: {e}")
            # Return default values with error indication
            return {
                "exponent": 0.0,
                "coefficient": 1.0,
                "r_squared": 0.0,
                "chi_squared": float("inf"),
                "fitting_quality": 0.0,
                "parameter_errors": [0.0, 0.0],
                "error": str(e),
            }

    def _calculate_fitting_quality(
        self, region_data: Dict[str, np.ndarray], power_law_fit: Dict[str, float]
    ) -> float:
        """
        Calculate comprehensive quality of power law fit using 7D BVP theory.

        Physical Meaning:
            Calculates comprehensive quality of the power law fit based on
            multiple statistical measures, parameter uncertainties, and
            physical constraints from 7D phase field theory.

        Mathematical Foundation:
            Combines R-squared, reduced chi-squared, parameter uncertainty,
            and physical constraints for robust quality assessment.

        Args:
            region_data (Dict[str, np.ndarray]): Region data.
            power_law_fit (Dict[str, float]): Power law fit parameters.

        Returns:
            float: Comprehensive fitting quality score (0-1).
        """
        try:
            # Extract quality metrics from fit results
            r_squared = power_law_fit.get("r_squared", 0.0)
            reduced_chi_squared = power_law_fit.get("reduced_chi_squared", float("inf"))
            parameter_errors = power_law_fit.get("parameter_errors", [0.0, 0.0])
            exponent = power_law_fit.get("exponent", 0.0)
            coefficient = power_law_fit.get("coefficient", 1.0)

            # Compute quality based on multiple factors
            quality_factors = []

            # R-squared contribution (higher is better)
            r_squared_quality = max(0.0, min(1.0, r_squared))
            quality_factors.append(r_squared_quality)

            # Reduced chi-squared contribution (closer to 1 is better)
            if reduced_chi_squared != float("inf"):
                chi_squared_quality = max(
                    0.0, min(1.0, 1.0 / (1.0 + abs(reduced_chi_squared - 1.0)))
                )
                quality_factors.append(chi_squared_quality)

            # Parameter uncertainty contribution (lower uncertainty is better)
            if len(parameter_errors) >= 2:
                amplitude_error = parameter_errors[0]
                exponent_error = parameter_errors[1]

                # Relative errors
                rel_amplitude_error = amplitude_error / max(abs(coefficient), 1e-10)
                rel_exponent_error = exponent_error / max(abs(exponent), 1e-10)

                # Uncertainty quality (lower relative error is better)
                uncertainty_quality = max(
                    0.0,
                    min(1.0, 1.0 / (1.0 + rel_amplitude_error + rel_exponent_error)),
                )
                quality_factors.append(uncertainty_quality)

            # Physical constraints for 7D BVP theory
            physical_quality = self._compute_physical_constraints_quality(
                exponent, coefficient
            )
            quality_factors.append(physical_quality)

            # Data point contribution
            data_points = len(region_data.get("amplitudes", []))
            data_quality = self._compute_data_points_quality(data_points)
            quality_factors.append(data_quality)

            # Compute weighted average of quality factors
            if quality_factors:
                quality = np.mean(quality_factors)
            else:
                quality = 0.0

            return max(0.0, min(1.0, quality))

        except Exception as e:
            self.logger.error(f"Quality calculation failed: {e}")
            return 0.0

    def _compute_r_squared_full(
        self, distances: np.ndarray, amplitudes: np.ndarray, popt: np.ndarray, func
    ) -> float:
        """
        Compute R-squared for power law fit using full 7D BVP theory.

        Physical Meaning:
            Computes R-squared coefficient of determination
            for power law fitting quality assessment in 7D phase field theory.

        Args:
            distances (np.ndarray): Distance values.
            amplitudes (np.ndarray): Amplitude values.
            popt (np.ndarray): Fitted parameters.
            func: Power law function.

        Returns:
            float: R-squared value.
        """
        try:
            # Compute predicted values
            predicted = func(distances, *popt)

            # Compute R-squared
            ss_res = np.sum((amplitudes - predicted) ** 2)
            ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)

            if ss_tot == 0:
                return 0.0

            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))

        except Exception as e:
            self.logger.error(f"R-squared computation failed: {e}")
            return 0.0

    def _compute_chi_squared_full(
        self, distances: np.ndarray, amplitudes: np.ndarray, popt: np.ndarray, func
    ) -> float:
        """
        Compute chi-squared statistic for power law fit using 7D BVP theory.

        Physical Meaning:
            Computes chi-squared statistic for goodness of fit
            assessment in power law analysis for 7D phase field theory.

        Args:
            distances (np.ndarray): Distance values.
            amplitudes (np.ndarray): Amplitude values.
            popt (np.ndarray): Fitted parameters.
            func: Power law function.

        Returns:
            float: Chi-squared value.
        """
        try:
            # Compute predicted values
            predicted = func(distances, *popt)

            # Compute chi-squared with proper error handling
            chi_squared = np.sum(
                ((amplitudes - predicted) / np.maximum(amplitudes, 1e-10)) ** 2
            )

            return float(chi_squared)

        except Exception as e:
            self.logger.error(f"Chi-squared computation failed: {e}")
            return float("inf")

    def _compute_fitting_quality_full(self, pcov: np.ndarray) -> float:
        """
        Compute fitting quality from covariance matrix using 7D BVP theory.

        Physical Meaning:
            Computes fitting quality based on parameter uncertainty
            from covariance matrix analysis for 7D phase field theory.

        Args:
            pcov (np.ndarray): Parameter covariance matrix.

        Returns:
            float: Fitting quality metric (0-1).
        """
        try:
            # Compute parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))

            # Compute relative uncertainties
            rel_errors = param_errors / np.maximum(np.abs(param_errors), 1e-10)

            # Quality based on uncertainty (lower is better)
            quality = 1.0 / (1.0 + np.mean(rel_errors))

            return max(0.0, min(1.0, quality))

        except Exception as e:
            self.logger.error(f"Fitting quality computation failed: {e}")
            return 0.0

    def _compute_physical_constraints_quality(
        self, exponent: float, coefficient: float
    ) -> float:
        """
        Compute quality based on physical constraints for 7D BVP theory.

        Physical Meaning:
            Evaluates physical constraints for power law parameters
            based on 7D phase field theory principles.

        Args:
            exponent (float): Power law exponent.
            coefficient (float): Power law coefficient.

        Returns:
            float: Physical constraints quality (0-1).
        """
        try:
            quality = 1.0

            # Check exponent bounds for 7D BVP theory
            if abs(exponent) > 10:  # Unrealistic exponent
                quality *= 0.5
            elif abs(exponent) > 5:  # Questionable exponent
                quality *= 0.8

            # Check coefficient bounds
            if coefficient <= 0:  # Invalid coefficient
                quality *= 0.0
            elif coefficient > 100:  # Unrealistic coefficient
                quality *= 0.7

            return max(0.0, min(1.0, quality))

        except Exception as e:
            self.logger.error(f"Physical constraints quality computation failed: {e}")
            return 0.0

    def _compute_data_points_quality(self, data_points: int) -> float:
        """
        Compute quality based on number of data points.

        Physical Meaning:
            Evaluates quality based on the number of data points
            available for power law fitting in 7D phase field theory.

        Args:
            data_points (int): Number of data points.

        Returns:
            float: Data points quality (0-1).
        """
        try:
            if data_points < 3:
                return 0.0
            elif data_points < 5:
                return 0.7
            elif data_points < 10:
                return 0.8
            elif data_points < 20:
                return 0.9
            else:
                return 1.0

        except Exception as e:
            self.logger.error(f"Data points quality computation failed: {e}")
            return 0.0
