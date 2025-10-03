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
        
        self.logger.info(f"Envelope power law analysis completed: {len(results)} regions analyzed")
        return results

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
            dim_slice = np.take(envelope, envelope.shape[dim]//2, axis=dim)
            
            # Find tail regions in this dimension
            dim_regions = self._find_dimension_tail_regions(dim_slice, dim)
            tail_regions.extend(dim_regions)
        
        return tail_regions

    def _find_dimension_tail_regions(self, dim_slice: np.ndarray, dimension: int) -> List[Dict[str, Any]]:
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
                    'dimension': dimension,
                    'indices': region_indices,
                    'start_index': min(region_indices),
                    'end_index': max(region_indices),
                    'size': len(region_indices)
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
            if value:
                current_region.append(i)
            else:
                if current_region:
                    regions.append(current_region)
                    current_region = []
        
        # Add final region if exists
        if current_region:
            regions.append(current_region)
        
        return regions

    def _analyze_region_power_law(self, envelope: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
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
            'region_info': region,
            'power_law_fit': power_law_fit,
            'fitting_quality': fitting_quality,
            'region_data_summary': {
                'data_points': len(region_data.get('amplitudes', [])),
                'amplitude_range': [np.min(region_data.get('amplitudes', [0])), 
                                  np.max(region_data.get('amplitudes', [0]))],
                'distance_range': [np.min(region_data.get('distances', [0])), 
                                 np.max(region_data.get('distances', [0]))]
            }
        }

    def _extract_region_data(self, envelope: np.ndarray, region: Dict[str, Any]) -> Dict[str, np.ndarray]:
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
        dim = region['dimension']
        indices = region['indices']
        
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
        
        return {
            'amplitudes': np.array(amplitudes),
            'distances': np.array(distances)
        }

    def _fit_power_law(self, region_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Fit power law to region data.

        Physical Meaning:
            Fits a power law function to the region data to determine
            the power law exponent and coefficient.

        Args:
            region_data (Dict[str, np.ndarray]): Region data.

        Returns:
            Dict[str, float]: Power law fit parameters.
        """
        amplitudes = region_data['amplitudes']
        distances = region_data['distances']
        
        # Avoid log of zero
        valid_mask = (amplitudes > 0) & (distances > 0)
        if not np.any(valid_mask):
            return {'exponent': 0.0, 'coefficient': 0.0, 'r_squared': 0.0}
        
        log_amplitudes = np.log(amplitudes[valid_mask])
        log_distances = np.log(distances[valid_mask])
        
        # Linear fit in log space
        if len(log_distances) > 1:
            # Calculate slope (exponent) and intercept (log coefficient)
            slope, intercept = np.polyfit(log_distances, log_amplitudes, 1)
            
            # Calculate R-squared
            predicted = slope * log_distances + intercept
            ss_res = np.sum((log_amplitudes - predicted) ** 2)
            ss_tot = np.sum((log_amplitudes - np.mean(log_amplitudes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'exponent': slope,
                'coefficient': np.exp(intercept),
                'r_squared': r_squared
            }
        else:
            return {'exponent': 0.0, 'coefficient': amplitudes[0], 'r_squared': 0.0}

    def _calculate_fitting_quality(self, region_data: Dict[str, np.ndarray], power_law_fit: Dict[str, float]) -> float:
        """
        Calculate quality of power law fit.

        Physical Meaning:
            Calculates the quality of the power law fit based on
            statistical measures and physical constraints.

        Args:
            region_data (Dict[str, np.ndarray]): Region data.
            power_law_fit (Dict[str, float]): Power law fit parameters.

        Returns:
            float: Fitting quality score (0-1).
        """
        # Base quality from R-squared
        r_squared = power_law_fit.get('r_squared', 0.0)
        base_quality = max(0.0, r_squared)
        
        # Penalty for unrealistic exponents
        exponent = power_law_fit.get('exponent', 0.0)
        if abs(exponent) > 10:  # Unrealistic exponent
            base_quality *= 0.5
        
        # Bonus for reasonable number of data points
        data_points = len(region_data.get('amplitudes', []))
        if data_points > 20:
            base_quality *= 1.1
        elif data_points < 5:
            base_quality *= 0.7
        
        return min(1.0, base_quality)
