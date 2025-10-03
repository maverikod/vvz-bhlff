"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Beating validation utilities for Level C.

This module implements validation functions for beating
analysis in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class BeatingValidationAnalyzer:
    """
    Validation utilities for beating analysis.
    
    Physical Meaning:
        Provides validation functions for beating analysis,
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
            Dict[str, Any]: Validation results.
        """
        self.logger.info("Validating beating analysis results")
        
        # Validate beating frequencies
        frequency_validation = self._validate_beating_frequencies(results.get('beating_frequencies', []))
        
        # Validate interference patterns
        pattern_validation = self._validate_interference_patterns(results.get('interference_patterns', []))
        
        # Validate mode coupling
        coupling_validation = self._validate_mode_coupling(results.get('mode_coupling', {}))
        
        # Validate beating strength
        strength_validation = self._validate_beating_strength(results.get('beating_strength', 0.0))
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(frequency_validation, pattern_validation, coupling_validation, strength_validation)
        
        validation_results = {
            'frequency_validation': frequency_validation,
            'pattern_validation': pattern_validation,
            'coupling_validation': coupling_validation,
            'strength_validation': strength_validation,
            'validation_score': validation_score,
            'is_valid': validation_score > self.quality_threshold
        }
        
        self.logger.info(f"Validation completed. Score: {validation_score}")
        return validation_results
    
    def _validate_beating_frequencies(self, beating_frequencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate beating frequencies.
        
        Physical Meaning:
            Validates beating frequencies to ensure they
            are physically meaningful and mathematically consistent.
            
        Args:
            beating_frequencies (List[Dict[str, Any]]): Beating frequency analysis results.
            
        Returns:
            Dict[str, Any]: Frequency validation results.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for empty frequencies
        if not beating_frequencies:
            validation_results['warnings'].append("No beating frequencies detected")
            return validation_results
        
        # Validate each frequency
        for i, freq in enumerate(beating_frequencies):
            # Check frequency value
            if freq['frequency'] < 0:
                validation_results['errors'].append(f"Negative frequency at index {i}")
                validation_results['is_valid'] = False
            
            # Check amplitude value
            if freq['amplitude'] < 0:
                validation_results['errors'].append(f"Negative amplitude at index {i}")
                validation_results['is_valid'] = False
            
            # Check for reasonable frequency range
            if freq['frequency'] > 1e6:  # Arbitrary upper limit
                validation_results['warnings'].append(f"Very high frequency at index {i}: {freq['frequency']}")
        
        return validation_results
    
    def _validate_interference_patterns(self, interference_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate interference patterns.
        
        Physical Meaning:
            Validates interference patterns to ensure they
            are physically meaningful and mathematically consistent.
            
        Args:
            interference_patterns (List[Dict[str, Any]]): Interference pattern analysis results.
            
        Returns:
            Dict[str, Any]: Pattern validation results.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for empty patterns
        if not interference_patterns:
            validation_results['warnings'].append("No interference patterns detected")
            return validation_results
        
        # Validate each pattern
        for i, pattern in enumerate(interference_patterns):
            # Check pattern type
            if pattern['type'] not in ['temporal', 'spatial', 'phase']:
                validation_results['errors'].append(f"Invalid pattern type at index {i}: {pattern['type']}")
                validation_results['is_valid'] = False
            
            # Check strength value
            if pattern['strength'] < 0 or pattern['strength'] > 1:
                validation_results['warnings'].append(f"Unusual pattern strength at index {i}: {pattern['strength']}")
        
        return validation_results
    
    def _validate_mode_coupling(self, mode_coupling: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mode coupling analysis.
        
        Physical Meaning:
            Validates mode coupling analysis to ensure it
            is physically meaningful and mathematically consistent.
            
        Args:
            mode_coupling (Dict[str, Any]): Mode coupling analysis results.
            
        Returns:
            Dict[str, Any]: Coupling validation results.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for empty coupling
        if not mode_coupling:
            validation_results['warnings'].append("No mode coupling analysis available")
            return validation_results
        
        # Validate coupling strength
        coupling_strength = mode_coupling.get('coupling_strength', 0.0)
        if coupling_strength < 0:
            validation_results['errors'].append(f"Negative coupling strength: {coupling_strength}")
            validation_results['is_valid'] = False
        
        # Validate coupling patterns
        coupling_patterns = mode_coupling.get('coupling_patterns', [])
        if coupling_patterns:
            for i, pattern in enumerate(coupling_patterns):
                # Check pattern structure
                if 'statistics' not in pattern:
                    validation_results['errors'].append(f"Missing statistics in coupling pattern {i}")
                    validation_results['is_valid'] = False
        
        return validation_results
    
    def _validate_beating_strength(self, beating_strength: float) -> Dict[str, Any]:
        """
        Validate beating strength.
        
        Physical Meaning:
            Validates beating strength to ensure it
            is physically meaningful and mathematically consistent.
            
        Args:
            beating_strength (float): Beating strength value.
            
        Returns:
            Dict[str, Any]: Strength validation results.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for negative strength
        if beating_strength < 0:
            validation_results['errors'].append(f"Negative beating strength: {beating_strength}")
            validation_results['is_valid'] = False
        
        # Check for very high strength
        if beating_strength > 1e6:  # Arbitrary upper limit
            validation_results['warnings'].append(f"Very high beating strength: {beating_strength}")
        
        return validation_results
    
    def _calculate_validation_score(self, frequency_validation: Dict[str, Any], 
                                  pattern_validation: Dict[str, Any], 
                                  coupling_validation: Dict[str, Any], 
                                  strength_validation: Dict[str, Any]) -> float:
        """
        Calculate overall validation score.
        
        Physical Meaning:
            Calculates an overall validation score based on
            individual validation results.
            
        Args:
            frequency_validation (Dict[str, Any]): Frequency validation results.
            pattern_validation (Dict[str, Any]): Pattern validation results.
            coupling_validation (Dict[str, Any]): Coupling validation results.
            strength_validation (Dict[str, Any]): Strength validation results.
            
        Returns:
            float: Overall validation score.
        """
        # Calculate score based on validation results
        score = 1.0
        
        # Deduct points for errors
        if not frequency_validation['is_valid']:
            score -= 0.3
        if not pattern_validation['is_valid']:
            score -= 0.3
        if not coupling_validation['is_valid']:
            score -= 0.2
        if not strength_validation['is_valid']:
            score -= 0.2
        
        # Deduct points for warnings
        total_warnings = (len(frequency_validation['warnings']) + 
                         len(pattern_validation['warnings']) + 
                         len(coupling_validation['warnings']) + 
                         len(strength_validation['warnings']))
        
        score -= total_warnings * 0.05
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score
    
    def assess_analysis_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of beating analysis results.
        
        Physical Meaning:
            Assesses the quality of beating analysis results
            based on various quality metrics.
            
        Args:
            results (Dict[str, Any]): Beating analysis results.
            
        Returns:
            Dict[str, Any]: Quality assessment results.
        """
        self.logger.info("Assessing analysis quality")
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(results)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_metrics)
        
        # Determine quality level
        quality_level = self._determine_quality_level(quality_score)
        
        quality_results = {
            'quality_metrics': quality_metrics,
            'quality_score': quality_score,
            'quality_level': quality_level,
            'recommendations': self._generate_quality_recommendations(quality_metrics)
        }
        
        self.logger.info(f"Quality assessment completed. Score: {quality_score}, Level: {quality_level}")
        return quality_results
    
    def _calculate_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate quality metrics for the analysis results.
        
        Physical Meaning:
            Calculates various quality metrics to assess
            the quality of the beating analysis.
            
        Args:
            results (Dict[str, Any]): Beating analysis results.
            
        Returns:
            Dict[str, float]: Quality metrics.
        """
        metrics = {}
        
        # Calculate frequency quality
        beating_frequencies = results.get('beating_frequencies', [])
        metrics['frequency_quality'] = self._calculate_frequency_quality(beating_frequencies)
        
        # Calculate pattern quality
        interference_patterns = results.get('interference_patterns', [])
        metrics['pattern_quality'] = self._calculate_pattern_quality(interference_patterns)
        
        # Calculate coupling quality
        mode_coupling = results.get('mode_coupling', {})
        metrics['coupling_quality'] = self._calculate_coupling_quality(mode_coupling)
        
        # Calculate strength quality
        beating_strength = results.get('beating_strength', 0.0)
        metrics['strength_quality'] = self._calculate_strength_quality(beating_strength)
        
        return metrics
    
    def _calculate_frequency_quality(self, beating_frequencies: List[Dict[str, Any]]) -> float:
        """
        Calculate frequency quality metric.
        
        Physical Meaning:
            Calculates a quality metric for beating frequencies
            based on their characteristics.
            
        Args:
            beating_frequencies (List[Dict[str, Any]]): Beating frequency analysis results.
            
        Returns:
            float: Frequency quality metric.
        """
        if not beating_frequencies:
            return 0.0
        
        # Calculate quality based on frequency distribution
        frequencies = [bf['frequency'] for bf in beating_frequencies]
        amplitudes = [bf['amplitude'] for bf in beating_frequencies]
        
        # Quality based on frequency spread and amplitude distribution
        freq_spread = np.std(frequencies) / np.mean(frequencies) if np.mean(frequencies) > 0 else 0
        amp_distribution = np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 0
        
        # Higher quality for moderate spread and distribution
        quality = 1.0 - abs(freq_spread - 0.5) - abs(amp_distribution - 0.5)
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_pattern_quality(self, interference_patterns: List[Dict[str, Any]]) -> float:
        """
        Calculate pattern quality metric.
        
        Physical Meaning:
            Calculates a quality metric for interference patterns
            based on their characteristics.
            
        Args:
            interference_patterns (List[Dict[str, Any]]): Interference pattern analysis results.
            
        Returns:
            float: Pattern quality metric.
        """
        if not interference_patterns:
            return 0.0
        
        # Calculate quality based on pattern strength and diversity
        strengths = [pattern['strength'] for pattern in interference_patterns]
        types = [pattern['type'] for pattern in interference_patterns]
        
        # Quality based on strength distribution and type diversity
        strength_quality = np.mean(strengths)
        type_diversity = len(set(types)) / len(types)
        
        quality = (strength_quality + type_diversity) / 2.0
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_coupling_quality(self, mode_coupling: Dict[str, Any]) -> float:
        """
        Calculate coupling quality metric.
        
        Physical Meaning:
            Calculates a quality metric for mode coupling
            based on its characteristics.
            
        Args:
            mode_coupling (Dict[str, Any]): Mode coupling analysis results.
            
        Returns:
            float: Coupling quality metric.
        """
        if not mode_coupling:
            return 0.0
        
        # Calculate quality based on coupling strength and patterns
        coupling_strength = mode_coupling.get('coupling_strength', 0.0)
        coupling_patterns = mode_coupling.get('coupling_patterns', [])
        
        # Quality based on coupling strength and pattern count
        strength_quality = min(1.0, coupling_strength)
        pattern_quality = min(1.0, len(coupling_patterns) / 10.0)  # Normalize to 10 patterns
        
        quality = (strength_quality + pattern_quality) / 2.0
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_strength_quality(self, beating_strength: float) -> float:
        """
        Calculate strength quality metric.
        
        Physical Meaning:
            Calculates a quality metric for beating strength
            based on its characteristics.
            
        Args:
            beating_strength (float): Beating strength value.
            
        Returns:
            float: Strength quality metric.
        """
        # Quality based on beating strength magnitude
        if beating_strength == 0:
            return 0.0
        
        # Higher quality for moderate strength values
        quality = 1.0 - abs(beating_strength - 0.5) / 0.5
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_quality_score(self, quality_metrics: Dict[str, float]) -> float:
        """
        Calculate overall quality score.
        
        Physical Meaning:
            Calculates an overall quality score based on
            individual quality metrics.
            
        Args:
            quality_metrics (Dict[str, float]): Quality metrics.
            
        Returns:
            float: Overall quality score.
        """
        # Calculate weighted average of quality metrics
        weights = {
            'frequency_quality': 0.3,
            'pattern_quality': 0.3,
            'coupling_quality': 0.2,
            'strength_quality': 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_metrics:
                total_score += quality_metrics[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """
        Determine quality level based on quality score.
        
        Physical Meaning:
            Determines the quality level based on the
            overall quality score.
            
        Args:
            quality_score (float): Overall quality score.
            
        Returns:
            str: Quality level.
        """
        if quality_score >= 0.8:
            return "Excellent"
        elif quality_score >= 0.6:
            return "Good"
        elif quality_score >= 0.4:
            return "Fair"
        elif quality_score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_quality_recommendations(self, quality_metrics: Dict[str, float]) -> List[str]:
        """
        Generate quality improvement recommendations.
        
        Physical Meaning:
            Generates recommendations for improving the
            quality of the beating analysis.
            
        Args:
            quality_metrics (Dict[str, float]): Quality metrics.
            
        Returns:
            List[str]: List of recommendations.
        """
        recommendations = []
        
        # Check frequency quality
        if quality_metrics.get('frequency_quality', 0.0) < 0.5:
            recommendations.append("Improve frequency analysis by increasing frequency resolution")
        
        # Check pattern quality
        if quality_metrics.get('pattern_quality', 0.0) < 0.5:
            recommendations.append("Improve pattern detection by adjusting interference thresholds")
        
        # Check coupling quality
        if quality_metrics.get('coupling_quality', 0.0) < 0.5:
            recommendations.append("Improve mode coupling analysis by increasing coupling sensitivity")
        
        # Check strength quality
        if quality_metrics.get('strength_quality', 0.0) < 0.5:
            recommendations.append("Improve beating strength calculation by optimizing strength metrics")
        
        return recommendations

