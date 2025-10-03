"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Beating frequency validation for Level C.

This module implements frequency validation functionality
for beating analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List
import logging

from bhlff.core.bvp import BVPCore


class BeatingValidationFrequencies:
    """
    Beating frequency validation for Level C.
    
    Physical Meaning:
        Provides frequency validation functionality for beating analysis
        in the 7D phase field.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """Initialize beating frequency validation."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        self.frequency_tolerance = 1e-6

    def validate_beating_frequencies(self, frequencies: List[float]) -> Dict[str, Any]:
        """
        Validate beating frequencies.
        
        Physical Meaning:
            Validates beating frequencies to ensure they
            are physically meaningful and consistent.
            
        Args:
            frequencies (List[float]): List of beating frequencies.
            
        Returns:
            Dict[str, Any]: Frequency validation results.
        """
        self.logger.info("Starting beating frequency validation")
        
        validation_result = {
            'frequencies_valid': True,
            'frequency_errors': [],
            'frequency_warnings': [],
            'frequency_metrics': {}
        }
        
        # Basic frequency validation
        if not frequencies:
            validation_result['frequency_errors'].append("Empty frequency list")
            validation_result['frequencies_valid'] = False
            return validation_result
        
        # Validate each frequency
        for i, freq in enumerate(frequencies):
            if not isinstance(freq, (int, float)):
                validation_result['frequency_errors'].append(f"Invalid frequency type at index {i}")
                validation_result['frequencies_valid'] = False
            elif freq <= 0:
                validation_result['frequency_errors'].append(f"Non-positive frequency at index {i}")
                validation_result['frequencies_valid'] = False
        
        # Calculate frequency metrics
        if frequencies:
            validation_result['frequency_metrics'] = {
                'mean_frequency': float(np.mean(frequencies)),
                'std_frequency': float(np.std(frequencies)),
                'min_frequency': float(np.min(frequencies)),
                'max_frequency': float(np.max(frequencies)),
                'frequency_count': len(frequencies)
            }
        
        self.logger.info(f"Beating frequency validation completed: {'PASSED' if validation_result['frequencies_valid'] else 'FAILED'}")
        return validation_result
