"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Main basic beating validation for Level C.

This module implements the main basic validation functionality
for beating analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List
import logging

from bhlff.core.bvp import BVPCore
from .beating_validation_frequencies import BeatingValidationFrequencies
from .beating_validation_patterns import BeatingValidationPatterns
from .beating_validation_metrics import BeatingValidationMetrics


class BeatingValidationBasicMain:
    """
    Main basic validation utilities for beating analysis.
    
    Physical Meaning:
        Provides main basic validation functions for beating analysis,
        coordinating specialized modules for different aspects of validation.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize beating validation analyzer.
        
        Physical Meaning:
            Sets up the basic validation analyzer with all necessary components
            for comprehensive validation of beating analysis results.
        
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        
        # Initialize validation parameters
        self.validation_threshold = 1e-6
        self.quality_threshold = 0.1
        self.error_tolerance = 1e-3
        
        # Initialize specialized modules
        self.frequency_validation = BeatingValidationFrequencies(bvp_core)
        self.pattern_validation = BeatingValidationPatterns(bvp_core)
        self.metrics_validation = BeatingValidationMetrics(bvp_core)
    
    def validate_beating_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate beating analysis results.
        
        Physical Meaning:
            Validates beating analysis results to ensure they
            are physically meaningful and mathematically consistent.
            
        Mathematical Foundation:
            Performs comprehensive validation including:
            - Frequency validation
            - Pattern validation
            - Mode coupling validation
            - Strength validation
            
        Args:
            results (Dict[str, Any]): Beating analysis results to validate.
            
        Returns:
            Dict[str, Any]: Validation results including:
                - validation_passed: Boolean indicating if validation passed
                - validation_errors: List of validation errors
                - validation_warnings: List of validation warnings
                - validation_metrics: Validation quality metrics
                - validation_summary: Summary of validation results
                
        Raises:
            ValueError: If results are invalid or missing required fields.
        """
        if not results:
            raise ValueError("Results cannot be empty")
        
        self.logger.info("Starting basic beating analysis validation")
        
        validation_results = {
            'validation_passed': True,
            'validation_errors': [],
            'validation_warnings': [],
            'validation_metrics': {},
            'validation_summary': {}
        }
        
        # Validate frequencies
        if 'frequencies' in results:
            freq_validation = self.frequency_validation.validate_beating_frequencies(results['frequencies'])
            validation_results['validation_metrics']['frequency_validation'] = freq_validation
        
        # Validate patterns
        if 'patterns' in results:
            pattern_validation = self.pattern_validation.validate_interference_patterns(results['patterns'])
            validation_results['validation_metrics']['pattern_validation'] = pattern_validation
        
        # Validate mode coupling
        if 'mode_coupling' in results:
            coupling_validation = self._validate_mode_coupling(results['mode_coupling'])
            validation_results['validation_metrics']['coupling_validation'] = coupling_validation
        
        # Validate beating strength
        if 'beating_strength' in results:
            strength_validation = self._validate_beating_strength(results['beating_strength'])
            validation_results['validation_metrics']['strength_validation'] = strength_validation
        
        # Compute overall validation metrics
        validation_metrics = self.metrics_validation.compute_validation_metrics(results)
        validation_results['validation_metrics'].update(validation_metrics)
        
        # Get validation summary
        validation_summary = self.metrics_validation.get_validation_summary(validation_results)
        validation_results['validation_summary'] = validation_summary
        
        # Determine overall validation status
        validation_results['validation_passed'] = len(validation_results['validation_errors']) == 0
        
        self.logger.info(f"Basic beating analysis validation completed: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        return validation_results
    
    def _validate_mode_coupling(self, coupling: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mode coupling results.
        
        Physical Meaning:
            Validates mode coupling analysis results to ensure
            they are physically meaningful.
            
        Args:
            coupling (Dict[str, Any]): Mode coupling results.
            
        Returns:
            Dict[str, Any]: Mode coupling validation results.
        """
        validation_result = {
            'coupling_valid': True,
            'coupling_errors': [],
            'coupling_warnings': []
        }
        
        # Basic coupling validation
        if 'coupling_strength' in coupling:
            strength = coupling['coupling_strength']
            if not isinstance(strength, (int, float)) or strength < 0:
                validation_result['coupling_errors'].append("Invalid coupling strength")
                validation_result['coupling_valid'] = False
        
        return validation_result
    
    def _validate_beating_strength(self, strength: float) -> Dict[str, Any]:
        """
        Validate beating strength.
        
        Physical Meaning:
            Validates beating strength to ensure it is
            physically meaningful.
            
        Args:
            strength (float): Beating strength value.
            
        Returns:
            Dict[str, Any]: Beating strength validation results.
        """
        validation_result = {
            'strength_valid': True,
            'strength_errors': [],
            'strength_warnings': []
        }
        
        # Basic strength validation
        if not isinstance(strength, (int, float)):
            validation_result['strength_errors'].append("Invalid strength type")
            validation_result['strength_valid'] = False
        elif strength < 0:
            validation_result['strength_errors'].append("Negative strength value")
            validation_result['strength_valid'] = False
        elif strength > 1.0:
            validation_result['strength_warnings'].append("Strength > 1.0 may be unrealistic")
        
        return validation_result
