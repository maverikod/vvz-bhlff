"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced power law analysis for BVP framework.

This module provides advanced functionality for power law analysis,
including statistical analysis, comparison, and optimization.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from ..bvp import BVPCore


class PowerLawAnalysis:
    """
    Advanced power law analyzer for BVP framework.

    Physical Meaning:
        Provides advanced analysis of power law behavior in BVP
        envelope fields, including statistical analysis, comparison
        between different regions, and optimization of power law fits.
    """

    def __init__(self, bvp_core: BVPCore = None):
        """
        Initialize advanced power law analyzer.

        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.statistical_significance = 0.05
        self.optimization_tolerance = 1e-6
        self.max_optimization_iterations = 100

    def compare_power_laws(self, envelope1: np.ndarray, envelope2: np.ndarray) -> Dict[str, Any]:
        """
        Compare power law behavior between two envelope fields.

        Physical Meaning:
            Compares power law characteristics between two envelope
            fields to analyze differences in their long-range behavior
            and decay patterns.

        Args:
            envelope1 (np.ndarray): First envelope field.
            envelope2 (np.ndarray): Second envelope field.

        Returns:
            Dict[str, Any]: Comparison results including:
                - exponent_differences: Differences in power law exponents
                - quality_comparison: Comparison of fitting quality
                - statistical_significance: Statistical significance of differences
        """
        self.logger.info("Starting power law comparison")
        
        # Analyze power laws for both envelopes
        results1 = self._analyze_envelope_power_laws(envelope1)
        results2 = self._analyze_envelope_power_laws(envelope2)
        
        # Compare exponents
        exponent_comparison = self._compare_exponents(results1, results2)
        
        # Compare quality
        quality_comparison = self._compare_quality(results1, results2)
        
        # Calculate statistical significance
        statistical_analysis = self._calculate_statistical_significance(results1, results2)
        
        results = {
            'exponent_differences': exponent_comparison,
            'quality_comparison': quality_comparison,
            'statistical_significance': statistical_analysis,
            'envelope1_results': results1,
            'envelope2_results': results2
        }
        
        self.logger.info("Power law comparison completed")
        return results

    def optimize_power_law_fits(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Optimize power law fits for better accuracy.

        Physical Meaning:
            Optimizes power law fits using advanced fitting techniques
            to improve accuracy and reliability of power law analysis.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Optimization results including:
                - optimized_exponents: Optimized power law exponents
                - optimization_quality: Quality of optimization
                - convergence_info: Convergence information
        """
        self.logger.info("Starting power law optimization")
        
        # Identify tail regions
        tail_regions = self._identify_tail_regions(envelope)
        
        # Optimize each tail region
        optimized_results = []
        for region in tail_regions:
            optimized_result = self._optimize_region_fit(envelope, region)
            optimized_results.append(optimized_result)
        
        # Calculate optimization quality
        optimization_quality = self._calculate_optimization_quality(optimized_results)
        
        # Analyze convergence
        convergence_info = self._analyze_convergence(optimized_results)
        
        results = {
            'optimized_exponents': [r['exponent'] for r in optimized_results],
            'optimization_quality': optimization_quality,
            'convergence_info': convergence_info,
            'optimized_results': optimized_results
        }
        
        self.logger.info("Power law optimization completed")
        return results

    def analyze_power_law_statistics(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Perform statistical analysis of power law behavior.

        Physical Meaning:
            Performs comprehensive statistical analysis of power law
            behavior, including distribution analysis, confidence
            intervals, and statistical tests.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Statistical analysis results including:
                - exponent_distribution: Distribution of power law exponents
                - confidence_intervals: Confidence intervals for exponents
                - statistical_tests: Results of statistical tests
        """
        self.logger.info("Starting power law statistical analysis")
        
        # Analyze power laws
        power_law_results = self._analyze_envelope_power_laws(envelope)
        
        # Analyze exponent distribution
        exponent_distribution = self._analyze_exponent_distribution(power_law_results)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(power_law_results)
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(power_law_results)
        
        results = {
            'exponent_distribution': exponent_distribution,
            'confidence_intervals': confidence_intervals,
            'statistical_tests': statistical_tests,
            'power_law_results': power_law_results
        }
        
        self.logger.info("Power law statistical analysis completed")
        return results

    def _analyze_envelope_power_laws(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze power laws for an envelope field.

        Args:
            envelope (np.ndarray): Envelope field data.

        Returns:
            List[Dict[str, Any]]: Power law analysis results.
        """
        # Identify tail regions
        tail_regions = self._identify_tail_regions(envelope)
        
        # Analyze each region
        results = []
        for region in tail_regions:
            region_result = self._analyze_region_power_law(envelope, region)
            results.append(region_result)
        
        return results

    def _identify_tail_regions(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify tail regions in the envelope field.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            List[Dict[str, Any]]: List of identified tail regions.
        """
        tail_regions = []
        
        # Analyze each dimension for tail regions
        for dim in range(envelope.ndim):
            # Extract 1D slice along dimension
            dim_slice = envelope.take(0, axis=dim)
            
            # Find tail regions in this dimension
            dim_tail_regions = self._find_dimension_tail_regions(dim_slice, dim)
            tail_regions.extend(dim_tail_regions)
        
        return tail_regions

    def _find_dimension_tail_regions(self, dim_slice: np.ndarray, dimension: int) -> List[Dict[str, Any]]:
        """
        Find tail regions in a specific dimension.

        Args:
            dim_slice (np.ndarray): 1D slice of the envelope field.
            dimension (int): Dimension index.

        Returns:
            List[Dict[str, Any]]: List of tail regions in this dimension.
        """
        tail_regions = []
        
        # Calculate threshold for tail identification
        max_amplitude = np.max(dim_slice)
        tail_threshold = max_amplitude * 0.1  # 10% threshold
        
        # Find regions below threshold
        below_threshold = dim_slice < tail_threshold
        
        # Find contiguous regions
        regions = self._find_contiguous_regions(below_threshold)
        
        # Filter regions by size
        for region in regions:
            if len(region) >= 10:  # Minimum 10 points
                tail_regions.append({
                    'dimension': dimension,
                    'start_index': region[0],
                    'end_index': region[-1],
                    'indices': region,
                    'amplitudes': dim_slice[region]
                })
        
        return tail_regions

    def _find_contiguous_regions(self, mask: np.ndarray) -> List[List[int]]:
        """
        Find contiguous regions in a boolean mask.

        Args:
            mask (np.ndarray): Boolean mask.

        Returns:
            List[List[int]]: List of contiguous regions.
        """
        regions = []
        current_region = []
        
        for i, value in enumerate(mask):
            if value:
                current_region.append(i)
            else:
                if current_region:
                    regions.append(current_region)
                    current_region = []
        
        # Add final region if it exists
        if current_region:
            regions.append(current_region)
        
        return regions

    def _analyze_region_power_law(self, envelope: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze power law behavior in a specific region.

        Args:
            envelope (np.ndarray): 7D envelope field data.
            region (Dict[str, Any]): Region to analyze.

        Returns:
            Dict[str, Any]: Region analysis results.
        """
        # Extract region data
        region_data = self._extract_region_data(envelope, region)
        
        # Fit power law
        power_law_fit = self._fit_power_law(region_data)
        
        # Calculate fitting quality
        fitting_quality = self._calculate_fitting_quality(region_data, power_law_fit)
        
        return {
            'exponent': power_law_fit['exponent'],
            'coefficient': power_law_fit['coefficient'],
            'fitting_quality': fitting_quality,
            'region_data': region_data
        }

    def _extract_region_data(self, envelope: np.ndarray, region: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract data for a specific region.

        Args:
            envelope (np.ndarray): 7D envelope field data.
            region (Dict[str, Any]): Region specification.

        Returns:
            Dict[str, np.ndarray]: Extracted region data.
        """
        dimension = region['dimension']
        indices = region['indices']
        
        # Extract data along the specified dimension
        region_data = envelope.take(indices, axis=dimension)
        
        # Calculate distances from center
        center = len(indices) // 2
        distances = np.abs(np.array(indices) - center)
        
        return {
            'amplitudes': region_data,
            'distances': distances,
            'indices': indices
        }

    def _fit_power_law(self, region_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Fit power law to region data.

        Args:
            region_data (Dict[str, np.ndarray]): Region data.

        Returns:
            Dict[str, float]: Power law fit parameters.
        """
        distances = region_data['distances']
        amplitudes = region_data['amplitudes']
        
        # Avoid log(0) by adding small offset
        distances = np.maximum(distances, 1e-10)
        amplitudes = np.maximum(amplitudes, 1e-10)
        
        # Linear fit in log space
        log_distances = np.log(distances)
        log_amplitudes = np.log(amplitudes)
        
        # Perform linear regression
        coeffs = np.polyfit(log_distances, log_amplitudes, 1)
        
        exponent = coeffs[0]
        coefficient = np.exp(coeffs[1])
        
        return {
            'exponent': exponent,
            'coefficient': coefficient
        }

    def _calculate_fitting_quality(self, region_data: Dict[str, np.ndarray], power_law_fit: Dict[str, float]) -> float:
        """
        Calculate quality of power law fit.

        Args:
            region_data (Dict[str, np.ndarray]): Region data.
            power_law_fit (Dict[str, float]): Power law fit parameters.

        Returns:
            float: Fitting quality (R² value).
        """
        distances = region_data['distances']
        amplitudes = region_data['amplitudes']
        
        # Calculate predicted values
        predicted = power_law_fit['coefficient'] * (distances ** power_law_fit['exponent'])
        
        # Calculate R²
        ss_res = np.sum((amplitudes - predicted) ** 2)
        ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return r_squared

    def _compare_exponents(self, results1: List[Dict[str, Any]], results2: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compare exponents between two sets of results.

        Args:
            results1 (List[Dict[str, Any]]): First set of results.
            results2 (List[Dict[str, Any]]): Second set of results.

        Returns:
            Dict[str, float]: Exponent comparison results.
        """
        exponents1 = [r['exponent'] for r in results1]
        exponents2 = [r['exponent'] for r in results2]
        
        if not exponents1 or not exponents2:
            return {
                'mean_difference': 0.0,
                'std_difference': 0.0,
                'correlation': 0.0
            }
        
        mean_diff = np.mean(exponents1) - np.mean(exponents2)
        std_diff = np.sqrt(np.var(exponents1) + np.var(exponents2))
        
        # Calculate correlation if same length
        if len(exponents1) == len(exponents2):
            correlation = np.corrcoef(exponents1, exponents2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'correlation': correlation
        }

    def _compare_quality(self, results1: List[Dict[str, Any]], results2: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compare fitting quality between two sets of results.

        Args:
            results1 (List[Dict[str, Any]]): First set of results.
            results2 (List[Dict[str, Any]]): Second set of results.

        Returns:
            Dict[str, float]: Quality comparison results.
        """
        qualities1 = [r['fitting_quality'] for r in results1]
        qualities2 = [r['fitting_quality'] for r in results2]
        
        if not qualities1 or not qualities2:
            return {
                'mean_difference': 0.0,
                'std_difference': 0.0,
                'correlation': 0.0
            }
        
        mean_diff = np.mean(qualities1) - np.mean(qualities2)
        std_diff = np.sqrt(np.var(qualities1) + np.var(qualities2))
        
        # Calculate correlation if same length
        if len(qualities1) == len(qualities2):
            correlation = np.corrcoef(qualities1, qualities2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'correlation': correlation
        }

    def _calculate_statistical_significance(self, results1: List[Dict[str, Any]], results2: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate statistical significance of differences.

        Args:
            results1 (List[Dict[str, Any]]): First set of results.
            results2 (List[Dict[str, Any]]): Second set of results.

        Returns:
            Dict[str, float]: Statistical significance results.
        """
        exponents1 = [r['exponent'] for r in results1]
        exponents2 = [r['exponent'] for r in results2]
        
        if not exponents1 or not exponents2:
            return {
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        
        # Perform t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(exponents1, exponents2)
        
        significant = p_value < self.statistical_significance
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant
        }

    def _optimize_region_fit(self, envelope: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize power law fit for a specific region.

        Args:
            envelope (np.ndarray): 7D envelope field data.
            region (Dict[str, Any]): Region to optimize.

        Returns:
            Dict[str, Any]: Optimization results.
        """
        # Extract region data
        region_data = self._extract_region_data(envelope, region)
        
        # Initial fit
        initial_fit = self._fit_power_law(region_data)
        
        # Optimize using iterative refinement
        optimized_fit = self._iterative_refinement(region_data, initial_fit)
        
        # Calculate optimization quality
        optimization_quality = self._calculate_fitting_quality(region_data, optimized_fit)
        
        return {
            'exponent': optimized_fit['exponent'],
            'coefficient': optimized_fit['coefficient'],
            'optimization_quality': optimization_quality,
            'iterations': optimized_fit.get('iterations', 0)
        }

    def _iterative_refinement(self, region_data: Dict[str, np.ndarray], initial_fit: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform iterative refinement of power law fit.

        Args:
            region_data (Dict[str, np.ndarray]): Region data.
            initial_fit (Dict[str, float]): Initial fit parameters.

        Returns:
            Dict[str, Any]: Refined fit parameters.
        """
        current_fit = initial_fit.copy()
        iterations = 0
        
        for i in range(self.max_optimization_iterations):
            # Calculate current quality
            current_quality = self._calculate_fitting_quality(region_data, current_fit)
            
            # Try small adjustments
            adjusted_fit = self._adjust_fit_parameters(current_fit)
            adjusted_quality = self._calculate_fitting_quality(region_data, adjusted_fit)
            
            # Accept if better
            if adjusted_quality > current_quality:
                current_fit = adjusted_fit
                iterations = i + 1
            else:
                break
        
        current_fit['iterations'] = iterations
        return current_fit

    def _adjust_fit_parameters(self, fit_params: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust fit parameters for optimization.

        Args:
            fit_params (Dict[str, float]): Current fit parameters.

        Returns:
            Dict[str, float]: Adjusted fit parameters.
        """
        # Small random adjustments
        adjustment_factor = 0.01
        
        adjusted_exponent = fit_params['exponent'] + np.random.normal(0, adjustment_factor)
        adjusted_coefficient = fit_params['coefficient'] * (1 + np.random.normal(0, adjustment_factor))
        
        return {
            'exponent': adjusted_exponent,
            'coefficient': adjusted_coefficient
        }

    def _calculate_optimization_quality(self, optimized_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall optimization quality.

        Args:
            optimized_results (List[Dict[str, Any]]): List of optimized results.

        Returns:
            Dict[str, float]: Optimization quality metrics.
        """
        if not optimized_results:
            return {
                'mean_quality': 0.0,
                'std_quality': 0.0,
                'mean_iterations': 0.0
            }
        
        qualities = [r['optimization_quality'] for r in optimized_results]
        iterations = [r.get('iterations', 0) for r in optimized_results]
        
        return {
            'mean_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'mean_iterations': np.mean(iterations)
        }

    def _analyze_convergence(self, optimized_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze convergence of optimization.

        Args:
            optimized_results (List[Dict[str, Any]]): List of optimized results.

        Returns:
            Dict[str, Any]: Convergence analysis results.
        """
        if not optimized_results:
            return {
                'converged_count': 0,
                'total_count': 0,
                'convergence_rate': 0.0
            }
        
        converged_count = sum(1 for r in optimized_results if r.get('iterations', 0) < self.max_optimization_iterations)
        total_count = len(optimized_results)
        convergence_rate = converged_count / total_count
        
        return {
            'converged_count': converged_count,
            'total_count': total_count,
            'convergence_rate': convergence_rate
        }

    def _analyze_exponent_distribution(self, power_law_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze distribution of power law exponents.

        Args:
            power_law_results (List[Dict[str, Any]]): List of power law results.

        Returns:
            Dict[str, float]: Exponent distribution analysis.
        """
        if not power_law_results:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        exponents = [r['exponent'] for r in power_law_results]
        exponents_array = np.array(exponents)
        
        return {
            'mean': np.mean(exponents_array),
            'std': np.std(exponents_array),
            'min': np.min(exponents_array),
            'max': np.max(exponents_array),
            'skewness': self._calculate_skewness(exponents_array),
            'kurtosis': self._calculate_kurtosis(exponents_array)
        }

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """
        Calculate skewness of data.

        Args:
            data (np.ndarray): Data array.

        Returns:
            float: Skewness value.
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """
        Calculate kurtosis of data.

        Args:
            data (np.ndarray): Data array.

        Returns:
            float: Kurtosis value.
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis

    def _calculate_confidence_intervals(self, power_law_results: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for power law results.

        Args:
            power_law_results (List[Dict[str, Any]]): List of power law results.

        Returns:
            Dict[str, Tuple[float, float]]: Confidence intervals.
        """
        if not power_law_results:
            return {
                'exponent_ci': (0.0, 0.0),
                'coefficient_ci': (0.0, 0.0)
            }
        
        exponents = [r['exponent'] for r in power_law_results]
        coefficients = [r['coefficient'] for r in power_law_results]
        
        # Calculate 95% confidence intervals
        exponent_ci = self._calculate_ci(exponents, 0.95)
        coefficient_ci = self._calculate_ci(coefficients, 0.95)
        
        return {
            'exponent_ci': exponent_ci,
            'coefficient_ci': coefficient_ci
        }

    def _calculate_ci(self, data: List[float], confidence: float) -> Tuple[float, float]:
        """
        Calculate confidence interval for data.

        Args:
            data (List[float]): Data list.
            confidence (float): Confidence level.

        Returns:
            Tuple[float, float]: Confidence interval bounds.
        """
        if not data:
            return (0.0, 0.0)
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        # Calculate critical value (simplified)
        alpha = 1 - confidence
        critical_value = 1.96  # Approximate for 95% CI
        
        margin_of_error = critical_value * (std / np.sqrt(len(data)))
        
        return (mean - margin_of_error, mean + margin_of_error)

    def _perform_statistical_tests(self, power_law_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform statistical tests on power law results.

        Args:
            power_law_results (List[Dict[str, Any]]): List of power law results.

        Returns:
            Dict[str, Any]: Statistical test results.
        """
        if not power_law_results:
            return {
                'normality_test': {'p_value': 1.0, 'is_normal': False},
                'outlier_test': {'outlier_count': 0, 'outlier_indices': []}
            }
        
        exponents = [r['exponent'] for r in power_law_results]
        
        # Normality test
        normality_test = self._test_normality(exponents)
        
        # Outlier test
        outlier_test = self._test_outliers(exponents)
        
        return {
            'normality_test': normality_test,
            'outlier_test': outlier_test
        }

    def _test_normality(self, data: List[float]) -> Dict[str, Any]:
        """
        Test normality of data.

        Args:
            data (List[float]): Data list.

        Returns:
            Dict[str, Any]: Normality test results.
        """
        if len(data) < 3:
            return {'p_value': 1.0, 'is_normal': False}
        
        # Simplified normality test
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        # Calculate skewness and kurtosis
        skewness = self._calculate_skewness(data_array)
        kurtosis = self._calculate_kurtosis(data_array)
        
        # Simple normality check
        is_normal = abs(skewness) < 0.5 and abs(kurtosis) < 0.5
        
        return {
            'p_value': 0.05 if is_normal else 0.01,
            'is_normal': is_normal,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def _test_outliers(self, data: List[float]) -> Dict[str, Any]:
        """
        Test for outliers in data.

        Args:
            data (List[float]): Data list.

        Returns:
            Dict[str, Any]: Outlier test results.
        """
        if len(data) < 3:
            return {'outlier_count': 0, 'outlier_indices': []}
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        # Identify outliers using 3-sigma rule
        outlier_threshold = 3 * std
        outliers = np.abs(data_array - mean) > outlier_threshold
        
        outlier_indices = np.where(outliers)[0].tolist()
        outlier_count = len(outlier_indices)
        
        return {
            'outlier_count': outlier_count,
            'outlier_indices': outlier_indices
        }
