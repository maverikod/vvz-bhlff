"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic beating validation utilities for Level C.

This module implements basic validation functions for beating
analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class BeatingValidationBasic:
    """
    Basic validation utilities for beating analysis.
    
    Physical Meaning:
        Provides basic validation functions for beating analysis,
        including result validation, quality assessment,
        and error analysis.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize beating validation analyzer.
        
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        
        # Initialize validation parameters
        self.validation_threshold = 1e-6
        self.quality_threshold = 0.1
        self.error_tolerance = 1e-3
    
    def validate_beating_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate beating analysis results.
        
        Physical Meaning:
            Validates beating analysis results to ensure they
            are physically meaningful and mathematically consistent.
            
        Args:
            results (Dict[str, Any]): Beating analysis results.
            
        Returns:
            Dict[str, Any]: Validation results including:
                - is_valid: Whether results are valid
                - validation_errors: List of validation errors
                - quality_score: Quality score of results
        """
        self.logger.info("Validating beating analysis results")
        
        validation_errors = []
        quality_scores = []
        
        # Validate beating frequencies
        if 'beating_frequencies' in results:
            freq_validation = self._validate_beating_frequencies(results['beating_frequencies'])
            if not freq_validation['is_valid']:
                validation_errors.extend(freq_validation['errors'])
            quality_scores.append(freq_validation['quality_score'])
        
        # Validate interference patterns
        if 'interference_patterns' in results:
            pattern_validation = self._validate_interference_patterns(results['interference_patterns'])
            if not pattern_validation['is_valid']:
                validation_errors.extend(pattern_validation['errors'])
            quality_scores.append(pattern_validation['quality_score'])
        
        # Validate mode coupling
        if 'mode_coupling' in results:
            coupling_validation = self._validate_mode_coupling(results['mode_coupling'])
            if not coupling_validation['is_valid']:
                validation_errors.extend(coupling_validation['errors'])
            quality_scores.append(coupling_validation['quality_score'])
        
        # Validate beating strength
        if 'beating_strength' in results:
            strength_validation = self._validate_beating_strength(results['beating_strength'])
            if not strength_validation['is_valid']:
                validation_errors.extend(strength_validation['errors'])
            quality_scores.append(strength_validation['quality_score'])
        
        # Compute overall validation
        is_valid = len(validation_errors) == 0
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        validation_results = {
            'is_valid': is_valid,
            'validation_errors': validation_errors,
            'quality_score': overall_quality,
            'component_validations': {
                'frequencies': freq_validation if 'beating_frequencies' in results else None,
                'patterns': pattern_validation if 'interference_patterns' in results else None,
                'coupling': coupling_validation if 'mode_coupling' in results else None,
                'strength': strength_validation if 'beating_strength' in results else None
            }
        }
        
        self.logger.info(f"Validation completed: is_valid = {is_valid}, quality = {overall_quality}")
        return validation_results
    
    def validate_beating_frequencies(self, frequencies: List[float]) -> Dict[str, Any]:
        """
        Validate beating frequencies.
        
        Physical Meaning:
            Validates beating frequencies to ensure they are
            physically meaningful and within expected ranges.
            
        Args:
            frequencies (List[float]): List of beating frequencies.
            
        Returns:
            Dict[str, Any]: Validation results.
        """
        errors = []
        quality_score = 1.0
        
        # Check for empty frequencies
        if not frequencies:
            errors.append("No beating frequencies found")
            quality_score = 0.0
            return {'is_valid': False, 'errors': errors, 'quality_score': quality_score}
        
        # Check for negative frequencies
        negative_freqs = [f for f in frequencies if f < 0]
        if negative_freqs:
            errors.append(f"Negative frequencies found: {negative_freqs}")
            quality_score *= 0.5
        
        # Check for zero frequencies
        zero_freqs = [f for f in frequencies if f == 0]
        if zero_freqs:
            errors.append(f"Zero frequencies found: {zero_freqs}")
            quality_score *= 0.8
        
        # Check for extremely large frequencies
        large_freqs = [f for f in frequencies if f > 1e6]
        if large_freqs:
            errors.append(f"Extremely large frequencies found: {large_freqs}")
            quality_score *= 0.7
        
        # Check for duplicate frequencies
        if len(frequencies) != len(set(frequencies)):
            errors.append("Duplicate frequencies found")
            quality_score *= 0.9
        
        # Check frequency distribution
        freq_std = np.std(frequencies)
        freq_mean = np.mean(frequencies)
        if freq_mean > 0:
            cv = freq_std / freq_mean
            if cv > 2.0:
                errors.append(f"High coefficient of variation: {cv}")
                quality_score *= 0.8
        
        is_valid = len(errors) == 0
        return {'is_valid': is_valid, 'errors': errors, 'quality_score': quality_score}
    
    def validate_interference_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate interference patterns.
        
        Physical Meaning:
            Validates interference patterns to ensure they
            are physically meaningful and well-formed.
            
        Args:
            patterns (List[Dict[str, Any]]): List of interference patterns.
            
        Returns:
            Dict[str, Any]: Validation results.
        """
        errors = []
        quality_score = 1.0
        
        # Check for empty patterns
        if not patterns:
            errors.append("No interference patterns found")
            quality_score = 0.0
            return {'is_valid': False, 'errors': errors, 'quality_score': quality_score}
        
        # Validate each pattern
        for i, pattern in enumerate(patterns):
            pattern_validation = self._validate_single_pattern(pattern)
            if not pattern_validation['is_valid']:
                errors.append(f"Pattern {i}: {pattern_validation['errors']}")
                quality_score *= 0.8
        
        # Check pattern diversity
        pattern_types = [p.get('type', 'unknown') for p in patterns]
        unique_types = set(pattern_types)
        if len(unique_types) < 2:
            errors.append("Low pattern diversity")
            quality_score *= 0.9
        
        # Check pattern strengths
        strengths = [p.get('strength', 0) for p in patterns]
        if strengths:
            weak_patterns = [s for s in strengths if s < 0.01]
            if len(weak_patterns) > len(strengths) / 2:
                errors.append("Too many weak patterns")
                quality_score *= 0.8
        
        is_valid = len(errors) == 0
        return {'is_valid': is_valid, 'errors': errors, 'quality_score': quality_score}
    
    def validate_mode_coupling(self, coupling: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mode coupling results.
        
        Physical Meaning:
            Validates mode coupling results to ensure they
            are physically meaningful and consistent.
            
        Args:
            coupling (Dict[str, Any]): Mode coupling results.
            
        Returns:
            Dict[str, Any]: Validation results.
        """
        errors = []
        quality_score = 1.0
        
        # Check required fields
        required_fields = ['coupling_strength', 'coupling_mechanisms', 'mode_interactions']
        for field in required_fields:
            if field not in coupling:
                errors.append(f"Missing required field: {field}")
                quality_score *= 0.5
        
        # Validate coupling strength
        if 'coupling_strength' in coupling:
            strength = coupling['coupling_strength']
            if not isinstance(strength, (int, float)):
                errors.append("Coupling strength must be numeric")
                quality_score *= 0.5
            elif strength < 0:
                errors.append("Negative coupling strength")
                quality_score *= 0.7
            elif strength > 1e6:
                errors.append("Extremely large coupling strength")
                quality_score *= 0.8
        
        # Validate coupling mechanisms
        if 'coupling_mechanisms' in coupling:
            mechanisms = coupling['coupling_mechanisms']
            if not isinstance(mechanisms, list):
                errors.append("Coupling mechanisms must be a list")
                quality_score *= 0.5
            elif not mechanisms:
                errors.append("No coupling mechanisms found")
                quality_score *= 0.8
        
        # Validate mode interactions
        if 'mode_interactions' in coupling:
            interactions = coupling['mode_interactions']
            if not isinstance(interactions, dict):
                errors.append("Mode interactions must be a dictionary")
                quality_score *= 0.5
        
        is_valid = len(errors) == 0
        return {'is_valid': is_valid, 'errors': errors, 'quality_score': quality_score}
    
    def validate_beating_strength(self, strength: float) -> Dict[str, Any]:
        """
        Validate beating strength.
        
        Physical Meaning:
            Validates beating strength to ensure it is
            physically meaningful and within expected ranges.
            
        Args:
            strength (float): Beating strength value.
            
        Returns:
            Dict[str, Any]: Validation results.
        """
        errors = []
        quality_score = 1.0
        
        # Check if strength is numeric
        if not isinstance(strength, (int, float)):
            errors.append("Beating strength must be numeric")
            quality_score = 0.0
            return {'is_valid': False, 'errors': errors, 'quality_score': quality_score}
        
        # Check for negative strength
        if strength < 0:
            errors.append("Negative beating strength")
            quality_score *= 0.5
        
        # Check for extremely large strength
        if strength > 1e6:
            errors.append("Extremely large beating strength")
            quality_score *= 0.7
        
        # Check for zero strength
        if strength == 0:
            errors.append("Zero beating strength")
            quality_score *= 0.8
        
        # Check for NaN or infinite values
        if not np.isfinite(strength):
            errors.append("Non-finite beating strength")
            quality_score = 0.0
        
        is_valid = len(errors) == 0
        return {'is_valid': is_valid, 'errors': errors, 'quality_score': quality_score}
    
    def _validate_single_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single interference pattern.
        
        Args:
            pattern (Dict[str, Any]): Interference pattern to validate.
            
        Returns:
            Dict[str, Any]: Validation results.
        """
        errors = []
        quality_score = 1.0
        
        # Check required fields
        required_fields = ['type', 'strength']
        for field in required_fields:
            if field not in pattern:
                errors.append(f"Missing required field: {field}")
                quality_score *= 0.5
        
        # Validate pattern type
        if 'type' in pattern:
            pattern_type = pattern['type']
            valid_types = ['spatial', 'temporal', 'phase', 'interference']
            if pattern_type not in valid_types:
                errors.append(f"Invalid pattern type: {pattern_type}")
                quality_score *= 0.7
        
        # Validate pattern strength
        if 'strength' in pattern:
            strength = pattern['strength']
            if not isinstance(strength, (int, float)):
                errors.append("Pattern strength must be numeric")
                quality_score *= 0.5
            elif strength < 0:
                errors.append("Negative pattern strength")
                quality_score *= 0.7
            elif strength > 1:
                errors.append("Pattern strength > 1")
                quality_score *= 0.8
        
        # Validate pattern data
        if 'pattern' in pattern:
            pattern_data = pattern['pattern']
            if not isinstance(pattern_data, np.ndarray):
                errors.append("Pattern data must be numpy array")
                quality_score *= 0.5
            elif pattern_data.size == 0:
                errors.append("Empty pattern data")
                quality_score *= 0.8
        
        is_valid = len(errors) == 0
        return {'is_valid': is_valid, 'errors': errors, 'quality_score': quality_score}
    
    def compute_validation_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute validation metrics for beating analysis results.
        
        Physical Meaning:
            Computes quantitative metrics for assessing the
            quality and reliability of beating analysis results.
            
        Args:
            results (Dict[str, Any]): Beating analysis results.
            
        Returns:
            Dict[str, float]: Validation metrics.
        """
        metrics = {}
        
        # Compute frequency metrics
        if 'beating_frequencies' in results:
            frequencies = results['beating_frequencies']
            if frequencies:
                metrics['frequency_count'] = len(frequencies)
                metrics['frequency_mean'] = np.mean(frequencies)
                metrics['frequency_std'] = np.std(frequencies)
                metrics['frequency_range'] = np.max(frequencies) - np.min(frequencies)
        
        # Compute pattern metrics
        if 'interference_patterns' in results:
            patterns = results['interference_patterns']
            if patterns:
                metrics['pattern_count'] = len(patterns)
                strengths = [p.get('strength', 0) for p in patterns]
                if strengths:
                    metrics['pattern_strength_mean'] = np.mean(strengths)
                    metrics['pattern_strength_std'] = np.std(strengths)
        
        # Compute coupling metrics
        if 'mode_coupling' in results:
            coupling = results['mode_coupling']
            if 'coupling_strength' in coupling:
                metrics['coupling_strength'] = coupling['coupling_strength']
            if 'coupling_mechanisms' in coupling:
                metrics['mechanism_count'] = len(coupling['coupling_mechanisms'])
        
        # Compute strength metrics
        if 'beating_strength' in results:
            metrics['beating_strength'] = results['beating_strength']
        
        return metrics
    
    def get_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get validation summary.
        
        Args:
            validation_results (Dict[str, Any]): Validation results.
            
        Returns:
            Dict[str, Any]: Validation summary.
        """
        summary = {
            'overall_valid': validation_results['is_valid'],
            'quality_score': validation_results['quality_score'],
            'error_count': len(validation_results['validation_errors']),
            'component_count': len(validation_results['component_validations']),
            'validation_status': 'PASS' if validation_results['is_valid'] else 'FAIL'
        }
        
        return summary
