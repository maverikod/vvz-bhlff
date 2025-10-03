"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic advanced beating validation utilities for Level C.

This module implements basic advanced validation functions for beating
analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class BeatingValidationAdvancedBasic:
    """
    Basic advanced validation utilities for beating analysis.
    
    Physical Meaning:
        Provides basic advanced validation functions for beating analysis,
        including statistical analysis and comparison between results.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize basic advanced beating validation analyzer.
        
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        
        # Advanced validation parameters
        self.statistical_significance = 0.05
        self.comparison_tolerance = 1e-3
        self.optimization_tolerance = 1e-6
        self.max_optimization_iterations = 100
    
    def validate_with_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate beating analysis results with statistical analysis.
        
        Physical Meaning:
            Performs comprehensive statistical validation of beating
            analysis results to ensure their reliability and accuracy.
            
        Args:
            results (Dict[str, Any]): Beating analysis results to validate.
            
        Returns:
            Dict[str, Any]: Statistical validation results.
        """
        self.logger.info("Starting statistical validation of beating analysis results")
        
        validation_results = {}
        
        # Validate beating frequencies
        if 'beating_frequencies' in results:
            freq_validation = self._validate_beating_frequencies(results['beating_frequencies'])
            validation_results['frequency_validation'] = freq_validation
        
        # Validate interference patterns
        if 'interference_patterns' in results:
            pattern_validation = self._validate_interference_patterns(results['interference_patterns'])
            validation_results['pattern_validation'] = pattern_validation
        
        # Validate mode coupling
        if 'mode_coupling' in results:
            coupling_validation = self._validate_mode_coupling(results['mode_coupling'])
            validation_results['coupling_validation'] = coupling_validation
        
        # Overall statistical validation
        overall_validation = self._compute_overall_statistical_validation(validation_results)
        validation_results['overall_validation'] = overall_validation
        
        self.logger.info("Statistical validation completed")
        return validation_results
    
    def compare_analysis_results(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two sets of beating analysis results.
        
        Physical Meaning:
            Compares two sets of beating analysis results to identify
            differences, similarities, and consistency between them.
            
        Args:
            results1 (Dict[str, Any]): First set of analysis results.
            results2 (Dict[str, Any]): Second set of analysis results.
            
        Returns:
            Dict[str, Any]: Comparison results.
        """
        self.logger.info("Comparing beating analysis results")
        
        comparison_results = {}
        
        # Compare beating frequencies
        if 'beating_frequencies' in results1 and 'beating_frequencies' in results2:
            freq_comparison = self._compare_beating_frequencies(
                results1['beating_frequencies'],
                results2['beating_frequencies']
            )
            comparison_results['frequency_comparison'] = freq_comparison
        
        # Compare interference patterns
        if 'interference_patterns' in results1 and 'interference_patterns' in results2:
            pattern_comparison = self._compare_interference_patterns(
                results1['interference_patterns'],
                results2['interference_patterns']
            )
            comparison_results['pattern_comparison'] = pattern_comparison
        
        # Compare mode coupling
        if 'mode_coupling' in results1 and 'mode_coupling' in results2:
            coupling_comparison = self._compare_mode_coupling(
                results1['mode_coupling'],
                results2['mode_coupling']
            )
            comparison_results['coupling_comparison'] = coupling_comparison
        
        # Overall comparison
        overall_comparison = self._compute_overall_comparison(comparison_results)
        comparison_results['overall_comparison'] = overall_comparison
        
        self.logger.info("Analysis results comparison completed")
        return comparison_results
    
    def validate_analysis_consistency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate consistency of beating analysis results.
        
        Physical Meaning:
            Validates the internal consistency of beating analysis results
            to ensure they are physically meaningful and mathematically sound.
            
        Args:
            results (Dict[str, Any]): Beating analysis results to validate.
            
        Returns:
            Dict[str, Any]: Consistency validation results.
        """
        self.logger.info("Validating analysis consistency")
        
        consistency_results = {}
        
        # Validate frequency consistency
        if 'beating_frequencies' in results:
            freq_consistency = self._validate_frequency_consistency(results['beating_frequencies'])
            consistency_results['frequency_consistency'] = freq_consistency
        
        # Validate pattern consistency
        if 'interference_patterns' in results:
            pattern_consistency = self._validate_pattern_consistency(results['interference_patterns'])
            consistency_results['pattern_consistency'] = pattern_consistency
        
        # Validate coupling consistency
        if 'mode_coupling' in results:
            coupling_consistency = self._validate_coupling_consistency(results['mode_coupling'])
            consistency_results['coupling_consistency'] = coupling_consistency
        
        # Overall consistency
        overall_consistency = self._compute_overall_consistency(consistency_results)
        consistency_results['overall_consistency'] = overall_consistency
        
        self.logger.info("Consistency validation completed")
        return consistency_results
    
    def _validate_beating_frequencies(self, frequencies: List[float]) -> Dict[str, Any]:
        """
        Validate beating frequencies.
        
        Args:
            frequencies (List[float]): List of beating frequencies.
            
        Returns:
            Dict[str, Any]: Frequency validation results.
        """
        validation = {}
        
        if not frequencies:
            validation['error'] = 'No frequencies to validate'
            return validation
        
        freq_array = np.array(frequencies)
        
        # Basic validation
        validation['count'] = len(frequencies)
        validation['mean'] = np.mean(freq_array)
        validation['std'] = np.std(freq_array)
        validation['min'] = np.min(freq_array)
        validation['max'] = np.max(freq_array)
        
        # Statistical validation
        validation['is_finite'] = np.all(np.isfinite(freq_array))
        validation['is_positive'] = np.all(freq_array > 0)
        validation['has_reasonable_range'] = np.all(freq_array < 1e6)
        
        # Physical validation
        validation['physical_reasonableness'] = self._check_frequency_physical_reasonableness(frequencies)
        
        return validation
    
    def _validate_interference_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate interference patterns.
        
        Args:
            patterns (List[Dict[str, Any]]): List of interference patterns.
            
        Returns:
            Dict[str, Any]: Pattern validation results.
        """
        validation = {}
        
        if not patterns:
            validation['error'] = 'No patterns to validate'
            return validation
        
        # Basic validation
        validation['count'] = len(patterns)
        validation['types'] = [p.get('type', 'unknown') for p in patterns]
        validation['type_diversity'] = len(set(validation['types']))
        
        # Strength validation
        strengths = [p.get('strength', 0) for p in patterns]
        validation['strength_mean'] = np.mean(strengths)
        validation['strength_std'] = np.std(strengths)
        validation['strength_is_finite'] = np.all(np.isfinite(strengths))
        validation['strength_is_positive'] = np.all(np.array(strengths) >= 0)
        
        # Pattern validation
        validation['pattern_validity'] = self._check_pattern_validity(patterns)
        
        return validation
    
    def _validate_mode_coupling(self, coupling: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mode coupling results.
        
        Args:
            coupling (Dict[str, Any]): Mode coupling results.
            
        Returns:
            Dict[str, Any]: Coupling validation results.
        """
        validation = {}
        
        # Coupling strength validation
        if 'coupling_strength' in coupling:
            strength = coupling['coupling_strength']
            validation['strength'] = {
                'value': strength,
                'is_finite': np.isfinite(strength),
                'is_positive': strength >= 0,
                'is_reasonable': 0 <= strength <= 1e6
            }
        
        # Coupling mechanisms validation
        if 'coupling_mechanisms' in coupling:
            mechanisms = coupling['coupling_mechanisms']
            validation['mechanisms'] = {
                'count': len(mechanisms),
                'types': list(set(mechanisms)),
                'type_diversity': len(set(mechanisms))
            }
        
        # Mode interactions validation
        if 'mode_interactions' in coupling:
            interactions = coupling['mode_interactions']
            validation['interactions'] = self._validate_mode_interactions(interactions)
        
        return validation
    
    def _compare_beating_frequencies(self, freq1: List[float], freq2: List[float]) -> Dict[str, Any]:
        """
        Compare two sets of beating frequencies.
        
        Args:
            freq1 (List[float]): First set of frequencies.
            freq2 (List[float]): Second set of frequencies.
            
        Returns:
            Dict[str, Any]: Frequency comparison results.
        """
        if not freq1 or not freq2:
            return {'error': 'Cannot compare empty frequency lists'}
        
        freq1_array = np.array(freq1)
        freq2_array = np.array(freq2)
        
        comparison = {
            'count_difference': len(freq1) - len(freq2),
            'mean_difference': np.mean(freq1_array) - np.mean(freq2_array),
            'std_difference': np.std(freq1_array) - np.std(freq2_array),
            'min_difference': np.min(freq1_array) - np.min(freq2_array),
            'max_difference': np.max(freq1_array) - np.max(freq2_array)
        }
        
        # Correlation analysis
        if len(freq1) == len(freq2):
            correlation = np.corrcoef(freq1_array, freq2_array)[0, 1]
            comparison['correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            comparison['correlation'] = 0.0
        
        # Statistical significance
        comparison['statistically_significant'] = self._check_statistical_significance(freq1_array, freq2_array)
        
        return comparison
    
    def _compare_interference_patterns(self, patterns1: List[Dict[str, Any]], patterns2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare two sets of interference patterns.
        
        Args:
            patterns1 (List[Dict[str, Any]]): First set of patterns.
            patterns2 (List[Dict[str, Any]]): Second set of patterns.
            
        Returns:
            Dict[str, Any]: Pattern comparison results.
        """
        comparison = {
            'count_difference': len(patterns1) - len(patterns2)
        }
        
        # Compare pattern types
        types1 = [p.get('type', 'unknown') for p in patterns1]
        types2 = [p.get('type', 'unknown') for p in patterns2]
        
        type_comparison = {
            'types1': set(types1),
            'types2': set(types2),
            'common_types': set(types1) & set(types2),
            'unique_to_1': set(types1) - set(types2),
            'unique_to_2': set(types2) - set(types1)
        }
        comparison['type_comparison'] = type_comparison
        
        # Compare pattern strengths
        strengths1 = [p.get('strength', 0) for p in patterns1]
        strengths2 = [p.get('strength', 0) for p in patterns2]
        
        if strengths1 and strengths2:
            strength_comparison = {
                'mean_difference': np.mean(strengths1) - np.mean(strengths2),
                'std_difference': np.std(strengths1) - np.std(strengths2)
            }
            comparison['strength_comparison'] = strength_comparison
        
        return comparison
    
    def _compare_mode_coupling(self, coupling1: Dict[str, Any], coupling2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two mode coupling results.
        
        Args:
            coupling1 (Dict[str, Any]): First coupling results.
            coupling2 (Dict[str, Any]): Second coupling results.
            
        Returns:
            Dict[str, Any]: Coupling comparison results.
        """
        comparison = {}
        
        # Compare coupling strengths
        if 'coupling_strength' in coupling1 and 'coupling_strength' in coupling2:
            strength1 = coupling1['coupling_strength']
            strength2 = coupling2['coupling_strength']
            comparison['strength_difference'] = strength1 - strength2
            comparison['strength_ratio'] = strength1 / strength2 if strength2 != 0 else float('inf')
        
        # Compare coupling mechanisms
        if 'coupling_mechanisms' in coupling1 and 'coupling_mechanisms' in coupling2:
            mechanisms1 = set(coupling1['coupling_mechanisms'])
            mechanisms2 = set(coupling2['coupling_mechanisms'])
            comparison['mechanism_comparison'] = {
                'common_mechanisms': mechanisms1 & mechanisms2,
                'unique_to_1': mechanisms1 - mechanisms2,
                'unique_to_2': mechanisms2 - mechanisms1
            }
        
        return comparison
    
    def _compute_overall_statistical_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall statistical validation metrics.
        
        Args:
            validation_results (Dict[str, Any]): Component validation results.
            
        Returns:
            Dict[str, Any]: Overall validation metrics.
        """
        overall_validation = {
            'component_count': len(validation_results),
            'validation_quality': 0.0
        }
        
        # Compute validation quality score
        quality_scores = []
        for component, validation in validation_results.items():
            if component != 'overall_validation':
                quality_score = self._compute_component_quality_score(validation)
                quality_scores.append(quality_score)
        
        if quality_scores:
            overall_validation['validation_quality'] = np.mean(quality_scores)
        
        return overall_validation
    
    def _compute_overall_comparison(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall comparison metrics.
        
        Args:
            comparison_results (Dict[str, Any]): Component comparison results.
            
        Returns:
            Dict[str, Any]: Overall comparison metrics.
        """
        overall_comparison = {
            'component_count': len(comparison_results),
            'similarity_score': 0.0
        }
        
        # Compute similarity score
        similarity_scores = []
        for component, comparison in comparison_results.items():
            if component != 'overall_comparison':
                similarity_score = self._compute_component_similarity_score(comparison)
                similarity_scores.append(similarity_score)
        
        if similarity_scores:
            overall_comparison['similarity_score'] = np.mean(similarity_scores)
        
        return overall_comparison
    
    def _compute_overall_consistency(self, consistency_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall consistency metrics.
        
        Args:
            consistency_results (Dict[str, Any]): Component consistency results.
            
        Returns:
            Dict[str, Any]: Overall consistency metrics.
        """
        overall_consistency = {
            'component_count': len(consistency_results),
            'consistency_score': 0.0
        }
        
        # Compute consistency score
        consistency_scores = []
        for component, consistency in consistency_results.items():
            if component != 'overall_consistency':
                consistency_score = self._compute_component_consistency_score(consistency)
                consistency_scores.append(consistency_score)
        
        if consistency_scores:
            overall_consistency['consistency_score'] = np.mean(consistency_scores)
        
        return overall_consistency
    
    def _check_frequency_physical_reasonableness(self, frequencies: List[float]) -> bool:
        """
        Check if frequencies are physically reasonable.
        
        Args:
            frequencies (List[float]): List of frequencies.
            
        Returns:
            bool: True if frequencies are physically reasonable.
        """
        if not frequencies:
            return False
        
        freq_array = np.array(frequencies)
        
        # Check for reasonable frequency range
        reasonable_range = np.all(freq_array > 0) and np.all(freq_array < 1e6)
        
        # Check for reasonable distribution
        reasonable_distribution = np.std(freq_array) < np.mean(freq_array) * 10
        
        return reasonable_range and reasonable_distribution
    
    def _check_pattern_validity(self, patterns: List[Dict[str, Any]]) -> bool:
        """
        Check if patterns are valid.
        
        Args:
            patterns (List[Dict[str, Any]]): List of patterns.
            
        Returns:
            bool: True if patterns are valid.
        """
        if not patterns:
            return False
        
        # Check that all patterns have required fields
        required_fields = ['type', 'strength']
        for pattern in patterns:
            if not all(field in pattern for field in required_fields):
                return False
        
        # Check that strengths are reasonable
        strengths = [p.get('strength', 0) for p in patterns]
        if not np.all(np.isfinite(strengths)):
            return False
        
        return True
    
    def _validate_mode_interactions(self, interactions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mode interactions.
        
        Args:
            interactions (Dict[str, Any]): Mode interactions data.
            
        Returns:
            Dict[str, Any]: Interaction validation results.
        """
        validation = {}
        
        # Validate interaction count
        if 'interaction_count' in interactions:
            count = interactions['interaction_count']
            validation['count'] = {
                'value': count,
                'is_finite': np.isfinite(count),
                'is_positive': count >= 0,
                'is_reasonable': count < 1000
            }
        
        # Validate interaction strength
        if 'interaction_strength' in interactions:
            strength = interactions['interaction_strength']
            validation['strength'] = {
                'value': strength,
                'is_finite': np.isfinite(strength),
                'is_positive': strength >= 0,
                'is_reasonable': strength < 1e6
            }
        
        return validation
    
    def _check_statistical_significance(self, data1: np.ndarray, data2: np.ndarray) -> bool:
        """
        Check statistical significance of difference between two datasets.
        
        Args:
            data1 (np.ndarray): First dataset.
            data2 (np.ndarray): Second dataset.
            
        Returns:
            bool: True if difference is statistically significant.
        """
        # Simplified statistical significance test
        mean_diff = np.abs(np.mean(data1) - np.mean(data2))
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        
        if pooled_std == 0:
            return mean_diff > self.comparison_tolerance
        
        t_statistic = mean_diff / pooled_std
        return t_statistic > 2.0  # Simplified threshold
    
    def _compute_component_quality_score(self, validation: Dict[str, Any]) -> float:
        """
        Compute quality score for a validation component.
        
        Args:
            validation (Dict[str, Any]): Validation results.
            
        Returns:
            float: Quality score.
        """
        if 'error' in validation:
            return 0.0
        
        # Simple quality score based on validation results
        quality_indicators = []
        
        # Check for finite values
        if 'is_finite' in validation:
            quality_indicators.append(1.0 if validation['is_finite'] else 0.0)
        
        # Check for positive values where expected
        if 'is_positive' in validation:
            quality_indicators.append(1.0 if validation['is_positive'] else 0.0)
        
        # Check for reasonable range
        if 'has_reasonable_range' in validation:
            quality_indicators.append(1.0 if validation['has_reasonable_range'] else 0.0)
        
        # Check for physical reasonableness
        if 'physical_reasonableness' in validation:
            quality_indicators.append(1.0 if validation['physical_reasonableness'] else 0.0)
        
        return np.mean(quality_indicators) if quality_indicators else 0.5
    
    def _compute_component_similarity_score(self, comparison: Dict[str, Any]) -> float:
        """
        Compute similarity score for a comparison component.
        
        Args:
            comparison (Dict[str, Any]): Comparison results.
            
        Returns:
            float: Similarity score.
        """
        if 'error' in comparison:
            return 0.0
        
        # Simple similarity score based on comparison results
        similarity_indicators = []
        
        # Check correlation
        if 'correlation' in comparison:
            correlation = comparison['correlation']
            similarity_indicators.append(max(0.0, correlation))
        
        # Check statistical significance
        if 'statistically_significant' in comparison:
            significance = comparison['statistically_significant']
            similarity_indicators.append(0.8 if significance else 0.2)
        
        return np.mean(similarity_indicators) if similarity_indicators else 0.5
    
    def _compute_component_consistency_score(self, consistency: Dict[str, Any]) -> float:
        """
        Compute consistency score for a consistency component.
        
        Args:
            consistency (Dict[str, Any]): Consistency results.
            
        Returns:
            float: Consistency score.
        """
        if 'error' in consistency:
            return 0.0
        
        # Simple consistency score based on consistency results
        consistency_indicators = []
        
        # Check for finite values
        if 'is_finite' in consistency:
            consistency_indicators.append(1.0 if consistency['is_finite'] else 0.0)
        
        # Check for positive values where expected
        if 'is_positive' in consistency:
            consistency_indicators.append(1.0 if consistency['is_positive'] else 0.0)
        
        # Check for reasonable values
        if 'is_reasonable' in consistency:
            consistency_indicators.append(1.0 if consistency['is_reasonable'] else 0.0)
        
        return np.mean(consistency_indicators) if consistency_indicators else 0.5
    
    def _validate_frequency_consistency(self, frequencies: List[float]) -> Dict[str, Any]:
        """
        Validate frequency consistency.
        
        Args:
            frequencies (List[float]): List of frequencies.
            
        Returns:
            Dict[str, Any]: Frequency consistency results.
        """
        if not frequencies:
            return {'error': 'No frequencies to validate'}
        
        freq_array = np.array(frequencies)
        
        consistency = {
            'is_finite': np.all(np.isfinite(freq_array)),
            'is_positive': np.all(freq_array > 0),
            'is_reasonable': np.all(freq_array < 1e6),
            'has_consistent_scale': np.std(freq_array) < np.mean(freq_array) * 10
        }
        
        return consistency
    
    def _validate_pattern_consistency(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate pattern consistency.
        
        Args:
            patterns (List[Dict[str, Any]]): List of patterns.
            
        Returns:
            Dict[str, Any]: Pattern consistency results.
        """
        if not patterns:
            return {'error': 'No patterns to validate'}
        
        consistency = {
            'has_consistent_types': len(set(p.get('type', 'unknown') for p in patterns)) > 0,
            'has_consistent_strengths': True
        }
        
        # Check strength consistency
        strengths = [p.get('strength', 0) for p in patterns]
        if strengths:
            consistency['has_consistent_strengths'] = np.all(np.isfinite(strengths))
        
        return consistency
    
    def _validate_coupling_consistency(self, coupling: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate coupling consistency.
        
        Args:
            coupling (Dict[str, Any]): Coupling results.
            
        Returns:
            Dict[str, Any]: Coupling consistency results.
        """
        consistency = {}
        
        # Check coupling strength consistency
        if 'coupling_strength' in coupling:
            strength = coupling['coupling_strength']
            consistency['strength'] = {
                'is_finite': np.isfinite(strength),
                'is_positive': strength >= 0,
                'is_reasonable': strength < 1e6
            }
        
        # Check mechanisms consistency
        if 'coupling_mechanisms' in coupling:
            mechanisms = coupling['coupling_mechanisms']
            consistency['mechanisms'] = {
                'is_valid': isinstance(mechanisms, list) and len(mechanisms) > 0
            }
        
        return consistency

