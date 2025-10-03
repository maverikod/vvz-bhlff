"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core power law analysis for BVP framework.

This module provides core functionality for power law analysis,
including basic power law fitting and tail region analysis.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from ..bvp import BVPCore


class PowerLawCore:
    """
    Core power law analyzer for BVP framework.

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

        Physical Meaning:
            Sets up the analyzer with the BVP core for accessing
            field data and computational resources.

        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.tail_threshold = 0.1  # Threshold for tail region
        self.min_tail_points = 10  # Minimum points for tail analysis
        self.power_law_tolerance = 1e-3  # Tolerance for power law fitting

    def analyze_power_laws(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze power law behavior in the envelope field.

        Physical Meaning:
            Analyzes power law decay patterns in the BVP envelope field,
            identifying tail regions and calculating power law exponents
            that characterize the field's long-range behavior.

        Mathematical Foundation:
            Computes power law decay A(r) ∝ r^(2β-3) in the tail region,
            where β is the fractional order and r is the radial distance.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Power law analysis results including:
                - power_law_exponents: List of power law exponents
                - tail_regions: Identified tail regions
                - fitting_quality: Quality of power law fits
                - decay_rates: Decay rates in different regions
        """
        self.logger.info("Starting power law analysis")
        
        # Identify tail regions
        tail_regions = self._identify_tail_regions(envelope)
        
        # Analyze power law behavior in each tail region
        power_law_results = []
        for region in tail_regions:
            region_analysis = self._analyze_region_power_law(envelope, region)
            power_law_results.append(region_analysis)
        
        # Calculate overall power law characteristics
        overall_characteristics = self._calculate_overall_characteristics(power_law_results)
        
        results = {
            'power_law_exponents': [r['exponent'] for r in power_law_results],
            'tail_regions': tail_regions,
            'fitting_quality': [r['fitting_quality'] for r in power_law_results],
            'decay_rates': [r['decay_rate'] for r in power_law_results],
            'overall_characteristics': overall_characteristics,
            'region_analyses': power_law_results
        }
        
        self.logger.info(f"Power law analysis completed. Found {len(tail_regions)} tail regions")
        return results

    def analyze_power_law_tails(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze power law behavior specifically in tail regions.

        Physical Meaning:
            Focuses on analyzing power law decay in the tail regions
            of the envelope field, where the field exhibits long-range
            behavior characteristic of the 7D phase field theory.

        Mathematical Foundation:
            Analyzes A(r) ∝ r^(2β-3) decay in tail regions, where
            β is the fractional order parameter.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Tail analysis results including:
                - tail_power_laws: Power law fits for tail regions
                - tail_exponents: Power law exponents for tails
                - tail_quality: Quality of tail fits
                - tail_statistics: Statistical analysis of tails
        """
        self.logger.info("Starting tail region power law analysis")
        
        # Identify tail regions
        tail_regions = self._identify_tail_regions(envelope)
        
        # Analyze each tail region
        tail_analyses = []
        for region in tail_regions:
            tail_analysis = self._analyze_tail_region(envelope, region)
            tail_analyses.append(tail_analysis)
        
        # Calculate tail statistics
        tail_statistics = self._calculate_tail_statistics(tail_analyses)
        
        results = {
            'tail_power_laws': tail_analyses,
            'tail_exponents': [t['exponent'] for t in tail_analyses],
            'tail_quality': [t['quality'] for t in tail_analyses],
            'tail_statistics': tail_statistics
        }
        
        self.logger.info(f"Tail analysis completed. Analyzed {len(tail_regions)} tail regions")
        return results

    def _identify_tail_regions(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify tail regions in the envelope field.

        Physical Meaning:
            Identifies regions where the envelope field exhibits
            tail behavior, characterized by power law decay patterns.

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
        tail_threshold = max_amplitude * self.tail_threshold
        
        # Find regions below threshold
        below_threshold = dim_slice < tail_threshold
        
        # Find contiguous regions
        regions = self._find_contiguous_regions(below_threshold)
        
        # Filter regions by size
        for region in regions:
            if len(region) >= self.min_tail_points:
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
        
        # Calculate decay rate
        decay_rate = self._calculate_decay_rate(power_law_fit)
        
        return {
            'exponent': power_law_fit['exponent'],
            'coefficient': power_law_fit['coefficient'],
            'fitting_quality': fitting_quality,
            'decay_rate': decay_rate,
            'region_data': region_data
        }

    def _analyze_tail_region(self, envelope: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a specific tail region.

        Args:
            envelope (np.ndarray): 7D envelope field data.
            region (Dict[str, Any]): Tail region to analyze.

        Returns:
            Dict[str, Any]: Tail region analysis results.
        """
        # Extract tail data
        tail_data = self._extract_region_data(envelope, region)
        
        # Fit power law to tail
        power_law_fit = self._fit_power_law(tail_data)
        
        # Calculate quality of fit
        quality = self._calculate_fitting_quality(tail_data, power_law_fit)
        
        return {
            'exponent': power_law_fit['exponent'],
            'coefficient': power_law_fit['coefficient'],
            'quality': quality,
            'tail_data': tail_data
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

        Mathematical Foundation:
            Fits A(r) = C * r^α where C is coefficient and α is exponent.

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

    def _calculate_decay_rate(self, power_law_fit: Dict[str, float]) -> float:
        """
        Calculate decay rate from power law fit.

        Args:
            power_law_fit (Dict[str, float]): Power law fit parameters.

        Returns:
            float: Decay rate.
        """
        return abs(power_law_fit['exponent'])

    def _calculate_overall_characteristics(self, power_law_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall characteristics from power law results.

        Args:
            power_law_results (List[Dict[str, Any]]): List of power law results.

        Returns:
            Dict[str, float]: Overall characteristics.
        """
        if not power_law_results:
            return {
                'mean_exponent': 0.0,
                'std_exponent': 0.0,
                'mean_quality': 0.0,
                'mean_decay_rate': 0.0
            }
        
        exponents = [r['exponent'] for r in power_law_results]
        qualities = [r['fitting_quality'] for r in power_law_results]
        decay_rates = [r['decay_rate'] for r in power_law_results]
        
        return {
            'mean_exponent': np.mean(exponents),
            'std_exponent': np.std(exponents),
            'mean_quality': np.mean(qualities),
            'mean_decay_rate': np.mean(decay_rates)
        }

    def _calculate_tail_statistics(self, tail_analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate statistics for tail analyses.

        Args:
            tail_analyses (List[Dict[str, Any]]): List of tail analyses.

        Returns:
            Dict[str, float]: Tail statistics.
        """
        if not tail_analyses:
            return {
                'mean_exponent': 0.0,
                'std_exponent': 0.0,
                'mean_quality': 0.0,
                'tail_count': 0
            }
        
        exponents = [t['exponent'] for t in tail_analyses]
        qualities = [t['quality'] for t in tail_analyses]
        
        return {
            'mean_exponent': np.mean(exponents),
            'std_exponent': np.std(exponents),
            'mean_quality': np.mean(qualities),
            'tail_count': len(tail_analyses)
        }
