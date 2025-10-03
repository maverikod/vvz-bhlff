"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Beating pattern validation for Level C.

This module implements pattern validation functionality
for beating analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List
import logging

from bhlff.core.bvp import BVPCore


class BeatingValidationPatterns:
    """
    Beating pattern validation for Level C.
    
    Physical Meaning:
        Provides pattern validation functionality for beating analysis
        in the 7D phase field.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """Initialize beating pattern validation."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def validate_interference_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate interference patterns.
        
        Physical Meaning:
            Validates interference patterns to ensure they
            are physically meaningful and consistent.
            
        Args:
            patterns (List[Dict[str, Any]]): List of interference patterns.
            
        Returns:
            Dict[str, Any]: Pattern validation results.
        """
        self.logger.info("Starting interference pattern validation")
        
        validation_result = {
            'patterns_valid': True,
            'pattern_errors': [],
            'pattern_warnings': [],
            'pattern_metrics': {}
        }
        
        # Basic pattern validation
        if not patterns:
            validation_result['pattern_errors'].append("Empty pattern list")
            validation_result['patterns_valid'] = False
            return validation_result
        
        # Validate each pattern
        for i, pattern in enumerate(patterns):
            pattern_validation = self._validate_single_pattern(pattern)
            if not pattern_validation.get('pattern_valid', True):
                validation_result['pattern_errors'].extend(pattern_validation.get('pattern_errors', []))
                validation_result['patterns_valid'] = False
        
        # Calculate pattern metrics
        validation_result['pattern_metrics'] = {
            'pattern_count': len(patterns),
            'valid_patterns': sum(1 for p in patterns if self._validate_single_pattern(p).get('pattern_valid', True))
        }
        
        self.logger.info(f"Interference pattern validation completed: {'PASSED' if validation_result['patterns_valid'] else 'FAILED'}")
        return validation_result

    def _validate_single_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single interference pattern."""
        validation_result = {
            'pattern_valid': True,
            'pattern_errors': [],
            'pattern_warnings': []
        }
        
        # Basic pattern validation
        if not isinstance(pattern, dict):
            validation_result['pattern_errors'].append("Pattern must be a dictionary")
            validation_result['pattern_valid'] = False
            return validation_result
        
        # Check required fields
        required_fields = ['amplitude', 'phase', 'frequency']
        for field in required_fields:
            if field not in pattern:
                validation_result['pattern_errors'].append(f"Missing required field: {field}")
                validation_result['pattern_valid'] = False
        
        return validation_result
