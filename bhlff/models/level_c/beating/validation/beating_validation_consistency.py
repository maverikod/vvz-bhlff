"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Consistency validation for beating analysis.

This module implements consistency validation functionality
for beating analysis results.
"""

import numpy as np
from typing import Dict, Any
import logging

from bhlff.core.bvp import BVPCore


class BeatingValidationConsistency:
    """
    Consistency validation for beating analysis.

    Physical Meaning:
        Provides consistency validation functionality for
        beating analysis results.
    """

    def __init__(self, bvp_core: BVPCore):
        """Initialize consistency validation analyzer."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def validate_consistency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate consistency of beating analysis results.

        Physical Meaning:
            Validates the internal consistency of beating analysis
            results to ensure they are physically reasonable.

        Args:
            results (Dict[str, Any]): Beating analysis results to validate.

        Returns:
            Dict[str, Any]: Consistency validation results.
        """
        consistency_results = {}
        
        # Check frequency-pattern consistency
        if 'beating_frequencies' in results and 'interference_patterns' in results:
            freq_pattern_consistency = self._check_frequency_pattern_consistency(
                results['beating_frequencies'],
                results['interference_patterns']
            )
            consistency_results['frequency_pattern_consistency'] = freq_pattern_consistency
        
        # Check coupling-frequency consistency
        if 'mode_coupling' in results and 'beating_frequencies' in results:
            coupling_freq_consistency = self._check_coupling_frequency_consistency(
                results['mode_coupling'],
                results['beating_frequencies']
            )
            consistency_results['coupling_frequency_consistency'] = coupling_freq_consistency
        
        # Check overall consistency
        overall_consistency = self._compute_overall_consistency(consistency_results)
        consistency_results['overall_consistency'] = overall_consistency
        
        return consistency_results

    def _check_frequency_pattern_consistency(self, frequencies: list, patterns: list) -> Dict[str, Any]:
        """Check consistency between frequencies and patterns."""
        if not frequencies or not patterns:
            return {'consistent': False, 'reason': 'Missing data', 'confidence': 0.0}
        
        # Simple consistency check
        freq_count = len(frequencies)
        pattern_count = len(patterns)
        
        # Reasonable ratio check
        ratio = freq_count / pattern_count if pattern_count > 0 else 0
        consistent = 0.5 <= ratio <= 2.0  # Reasonable range
        
        confidence = 1.0 - abs(ratio - 1.0) if consistent else 0.0
        
        return {
            'consistent': consistent,
            'frequency_count': freq_count,
            'pattern_count': pattern_count,
            'ratio': ratio,
            'confidence': confidence
        }

    def _check_coupling_frequency_consistency(self, coupling: Dict[str, Any], frequencies: list) -> Dict[str, Any]:
        """Check consistency between coupling and frequencies."""
        if not coupling or not frequencies:
            return {'consistent': False, 'reason': 'Missing data', 'confidence': 0.0}
        
        coupling_strength = coupling.get('coupling_strength', 0.0)
        freq_count = len(frequencies)
        
        # Strong coupling should correlate with more frequencies
        expected_freq_count = int(coupling_strength * 10)  # Simplified relationship
        freq_diff = abs(freq_count - expected_freq_count)
        
        consistent = freq_diff <= 3  # Allow some tolerance
        confidence = max(0.0, 1.0 - freq_diff / 10.0)
        
        return {
            'consistent': consistent,
            'coupling_strength': coupling_strength,
            'frequency_count': freq_count,
            'expected_frequency_count': expected_freq_count,
            'frequency_difference': freq_diff,
            'confidence': confidence
        }

    def _compute_overall_consistency(self, consistency_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall consistency metrics."""
        consistency_scores = []
        for key, result in consistency_results.items():
            if isinstance(result, dict) and 'confidence' in result:
                consistency_scores.append(result['confidence'])
        
        if not consistency_scores:
            return {
                'overall_consistency': 0.0,
                'consistency_passed': False,
                'number_of_checks': 0
            }
        
        overall_consistency = np.mean(consistency_scores)
        consistency_passed = overall_consistency > 0.5
        
        return {
            'overall_consistency': overall_consistency,
            'consistency_passed': consistency_passed,
            'number_of_checks': len(consistency_scores),
            'consistency_std': np.std(consistency_scores) if len(consistency_scores) > 1 else 0.0
        }
