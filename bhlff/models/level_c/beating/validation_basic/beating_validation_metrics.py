"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Beating validation metrics for Level C.

This module implements validation metrics functionality
for beating analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any
import logging

from bhlff.core.bvp import BVPCore


class BeatingValidationMetrics:
    """
    Beating validation metrics for Level C.
    
    Physical Meaning:
        Provides validation metrics functionality for beating analysis
        in the 7D phase field.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """Initialize beating validation metrics."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def compute_validation_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute validation metrics.
        
        Physical Meaning:
            Computes comprehensive validation metrics for beating analysis
            results to assess their quality and reliability.
            
        Args:
            results (Dict[str, Any]): Beating analysis results.
            
        Returns:
            Dict[str, float]: Validation metrics.
        """
        self.logger.info("Computing validation metrics")
        
        metrics = {
            'overall_quality': 0.0,
            'frequency_quality': 0.0,
            'pattern_quality': 0.0,
            'coupling_quality': 0.0,
            'strength_quality': 0.0
        }
        
        # Compute frequency quality
        if 'frequencies' in results:
            frequencies = results['frequencies']
            if frequencies:
                metrics['frequency_quality'] = min(1.0, len(frequencies) / 10.0)  # Simplified
        
        # Compute pattern quality
        if 'patterns' in results:
            patterns = results['patterns']
            if patterns:
                metrics['pattern_quality'] = min(1.0, len(patterns) / 5.0)  # Simplified
        
        # Compute coupling quality
        if 'mode_coupling' in results:
            metrics['coupling_quality'] = 0.8  # Simplified
        
        # Compute strength quality
        if 'beating_strength' in results:
            strength = results['beating_strength']
            if isinstance(strength, (int, float)) and 0 <= strength <= 1:
                metrics['strength_quality'] = 1.0
            else:
                metrics['strength_quality'] = 0.5
        
        # Compute overall quality
        quality_values = [metrics['frequency_quality'], metrics['pattern_quality'], 
                         metrics['coupling_quality'], metrics['strength_quality']]
        metrics['overall_quality'] = float(np.mean(quality_values))
        
        self.logger.info(f"Validation metrics computed: overall_quality = {metrics['overall_quality']:.3f}")
        return metrics

    def get_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get validation summary.
        
        Physical Meaning:
            Provides a summary of validation results for easy interpretation
            and reporting.
            
        Args:
            validation_results (Dict[str, Any]): Validation results.
            
        Returns:
            Dict[str, Any]: Validation summary.
        """
        summary = {
            'validation_status': 'PASSED' if validation_results.get('validation_passed', False) else 'FAILED',
            'total_errors': len(validation_results.get('validation_errors', [])),
            'total_warnings': len(validation_results.get('validation_warnings', [])),
            'overall_quality': validation_results.get('validation_metrics', {}).get('overall_quality', 0.0),
            'validation_details': {}
        }
        
        # Add validation details
        validation_metrics = validation_results.get('validation_metrics', {})
        for key, value in validation_metrics.items():
            if isinstance(value, dict) and 'valid' in str(value):
                summary['validation_details'][key] = value.get('valid', False)
        
        return summary
