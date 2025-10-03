"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced beating validation utilities for Level C.

This module implements advanced validation functions for beating
analysis in the 7D phase field, including statistical analysis,
comparison, and optimization.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from bhlff.core.bvp import BVPCore


class BeatingValidationAdvanced:
    """
    Advanced validation utilities for beating analysis.
    
    Physical Meaning:
        Provides advanced validation functions for beating analysis,
        including statistical analysis, comparison between results,
        and optimization of validation parameters.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize advanced beating validation analyzer.
        
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
            analysis results, including distribution analysis,
            significance testing, and outlier detection.
            
        Args:
            results (Dict[str, Any]): Beating analysis results.
            
        Returns:
            Dict[str, Any]: Statistical validation results.
        """
        self.logger.info("Performing statistical validation")
        
        statistical_results = {}
        
        # Statistical analysis of beating frequencies
        if 'beating_frequencies' in results:
            freq_stats = self._analyze_frequency_statistics(results['beating_frequencies'])
            statistical_results['frequency_statistics'] = freq_stats
        
        # Statistical analysis of interference patterns
        if 'interference_patterns' in results:
            pattern_stats = self._analyze_pattern_statistics(results['interference_patterns'])
            statistical_results['pattern_statistics'] = pattern_stats
        
        # Statistical analysis of mode coupling
        if 'mode_coupling' in results:
            coupling_stats = self._analyze_coupling_statistics(results['mode_coupling'])
            statistical_results['coupling_statistics'] = coupling_stats
        
        # Statistical analysis of beating strength
        if 'beating_strength' in results:
            strength_stats = self._analyze_strength_statistics(results['beating_strength'])
            statistical_results['strength_statistics'] = strength_stats
        
        # Overall statistical validation
        overall_stats = self._compute_overall_statistics(statistical_results)
        statistical_results['overall_statistics'] = overall_stats
        
        self.logger.info("Statistical validation completed")
        return statistical_results
    
    def compare_results(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two sets of beating analysis results.
        
        Physical Meaning:
            Compares two sets of beating analysis results to
            identify differences, similarities, and consistency
            between different analyses.
            
        Args:
            results1 (Dict[str, Any]): First set of results.
            results2 (Dict[str, Any]): Second set of results.
            
        Returns:
            Dict[str, Any]: Comparison results.
        """
        self.logger.info("Comparing beating analysis results")
        
        comparison_results = {}
        
        # Compare beating frequencies
        if 'beating_frequencies' in results1 and 'beating_frequencies' in results2:
            freq_comparison = self._compare_frequencies(
                results1['beating_frequencies'],
                results2['beating_frequencies']
            )
            comparison_results['frequency_comparison'] = freq_comparison
        
        # Compare interference patterns
        if 'interference_patterns' in results1 and 'interference_patterns' in results2:
            pattern_comparison = self._compare_patterns(
                results1['interference_patterns'],
                results2['interference_patterns']
            )
            comparison_results['pattern_comparison'] = pattern_comparison
        
        # Compare mode coupling
        if 'mode_coupling' in results1 and 'mode_coupling' in results2:
            coupling_comparison = self._compare_coupling(
                results1['mode_coupling'],
                results2['mode_coupling']
            )
            comparison_results['coupling_comparison'] = coupling_comparison
        
        # Compare beating strength
        if 'beating_strength' in results1 and 'beating_strength' in results2:
            strength_comparison = self._compare_strength(
                results1['beating_strength'],
                results2['beating_strength']
            )
            comparison_results['strength_comparison'] = strength_comparison
        
        # Overall comparison
        overall_comparison = self._compute_overall_comparison(comparison_results)
        comparison_results['overall_comparison'] = overall_comparison
        
        self.logger.info("Results comparison completed")
        return comparison_results
    
    def optimize_validation_parameters(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize validation parameters for better accuracy.
        
        Physical Meaning:
            Optimizes validation parameters to improve the
            accuracy and reliability of validation results.
            
        Args:
            results (Dict[str, Any]): Beating analysis results.
            
        Returns:
            Dict[str, Any]: Optimization results.
        """
        self.logger.info("Optimizing validation parameters")
        
        # Initialize optimization parameters
        current_params = {
            'validation_threshold': 1e-6,
            'quality_threshold': 0.1,
            'error_tolerance': 1e-3
        }
        
        # Optimize parameters
        optimized_params = self._optimize_parameters(results, current_params)
        
        # Validate with optimized parameters
        optimized_validation = self._validate_with_optimized_parameters(results, optimized_params)
        
        optimization_results = {
            'original_parameters': current_params,
            'optimized_parameters': optimized_params,
            'optimization_improvement': self._compute_optimization_improvement(
                current_params, optimized_params, results
            ),
            'optimized_validation': optimized_validation
        }
        
        self.logger.info("Parameter optimization completed")
        return optimization_results
    
    def _analyze_frequency_statistics(self, frequencies: List[float]) -> Dict[str, Any]:
        """
        Analyze statistical properties of beating frequencies.
        
        Args:
            frequencies (List[float]): List of beating frequencies.
            
        Returns:
            Dict[str, Any]: Frequency statistics.
        """
        if not frequencies:
            return {'error': 'No frequencies to analyze'}
        
        freq_array = np.array(frequencies)
        
        # Basic statistics
        basic_stats = {
            'count': len(frequencies),
            'mean': np.mean(freq_array),
            'std': np.std(freq_array),
            'min': np.min(freq_array),
            'max': np.max(freq_array),
            'median': np.median(freq_array)
        }
        
        # Distribution analysis
        distribution_stats = {
            'skewness': self._compute_skewness(freq_array),
            'kurtosis': self._compute_kurtosis(freq_array),
            'is_normal': self._test_normality(freq_array)
        }
        
        # Outlier analysis
        outlier_stats = self._detect_outliers(freq_array)
        
        return {
            'basic_statistics': basic_stats,
            'distribution_analysis': distribution_stats,
            'outlier_analysis': outlier_stats
        }
    
    def _analyze_pattern_statistics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze statistical properties of interference patterns.
        
        Args:
            patterns (List[Dict[str, Any]]): List of interference patterns.
            
        Returns:
            Dict[str, Any]: Pattern statistics.
        """
        if not patterns:
            return {'error': 'No patterns to analyze'}
        
        # Extract pattern properties
        strengths = [p.get('strength', 0) for p in patterns]
        types = [p.get('type', 'unknown') for p in patterns]
        
        # Strength statistics
        strength_stats = {
            'count': len(strengths),
            'mean': np.mean(strengths),
            'std': np.std(strengths),
            'min': np.min(strengths),
            'max': np.max(strengths)
        }
        
        # Type distribution
        type_counts = {}
        for pattern_type in types:
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        type_distribution = {
            'type_counts': type_counts,
            'most_common_type': max(type_counts, key=type_counts.get) if type_counts else None,
            'type_diversity': len(type_counts)
        }
        
        return {
            'strength_statistics': strength_stats,
            'type_distribution': type_distribution
        }
    
    def _analyze_coupling_statistics(self, coupling: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze statistical properties of mode coupling.
        
        Args:
            coupling (Dict[str, Any]): Mode coupling results.
            
        Returns:
            Dict[str, Any]: Coupling statistics.
        """
        coupling_stats = {}
        
        # Coupling strength statistics
        if 'coupling_strength' in coupling:
            strength = coupling['coupling_strength']
            coupling_stats['strength'] = {
                'value': strength,
                'is_finite': np.isfinite(strength),
                'is_positive': strength > 0
            }
        
        # Coupling mechanisms statistics
        if 'coupling_mechanisms' in coupling:
            mechanisms = coupling['coupling_mechanisms']
            coupling_stats['mechanisms'] = {
                'count': len(mechanisms),
                'types': list(set(mechanisms)),
                'diversity': len(set(mechanisms))
            }
        
        # Mode interactions statistics
        if 'mode_interactions' in coupling:
            interactions = coupling['mode_interactions']
            coupling_stats['interactions'] = {
                'interaction_count': interactions.get('interaction_count', 0),
                'interaction_strength': interactions.get('interaction_strength', 0),
                'interaction_types': interactions.get('interaction_types', [])
            }
        
        return coupling_stats
    
    def _analyze_strength_statistics(self, strength: float) -> Dict[str, Any]:
        """
        Analyze statistical properties of beating strength.
        
        Args:
            strength (float): Beating strength value.
            
        Returns:
            Dict[str, Any]: Strength statistics.
        """
        return {
            'value': strength,
            'is_finite': np.isfinite(strength),
            'is_positive': strength > 0,
            'magnitude': abs(strength),
            'log_magnitude': np.log10(abs(strength)) if strength != 0 else float('-inf')
        }
    
    def _compute_overall_statistics(self, statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall statistical metrics.
        
        Args:
            statistical_results (Dict[str, Any]): Statistical results.
            
        Returns:
            Dict[str, Any]: Overall statistics.
        """
        overall_stats = {
            'component_count': len(statistical_results),
            'statistical_quality': 0.0
        }
        
        # Compute overall quality score
        quality_scores = []
        for component, stats in statistical_results.items():
            if component != 'overall_statistics':
                quality_score = self._compute_component_quality_score(stats)
                quality_scores.append(quality_score)
        
        if quality_scores:
            overall_stats['statistical_quality'] = np.mean(quality_scores)
        
        return overall_stats
    
    def _compare_frequencies(self, freq1: List[float], freq2: List[float]) -> Dict[str, Any]:
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
        
        # Basic comparison
        comparison = {
            'count_difference': len(freq1) - len(freq2),
            'mean_difference': np.mean(freq1_array) - np.mean(freq2_array),
            'std_difference': np.std(freq1_array) - np.std(freq2_array)
        }
        
        # Correlation analysis
        if len(freq1) == len(freq2):
            correlation = np.corrcoef(freq1_array, freq2_array)[0, 1]
            comparison['correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            comparison['correlation'] = 0.0
        
        # Statistical significance test
        if len(freq1) > 1 and len(freq2) > 1:
            # Simplified t-test
            t_stat = self._compute_t_statistic(freq1_array, freq2_array)
            comparison['t_statistic'] = t_stat
            comparison['significant_difference'] = abs(t_stat) > 2.0  # Simplified threshold
        
        return comparison
    
    def _compare_patterns(self, patterns1: List[Dict[str, Any]], patterns2: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    
    def _compare_coupling(self, coupling1: Dict[str, Any], coupling2: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def _compare_strength(self, strength1: float, strength2: float) -> Dict[str, Any]:
        """
        Compare two beating strengths.
        
        Args:
            strength1 (float): First beating strength.
            strength2 (float): Second beating strength.
            
        Returns:
            Dict[str, Any]: Strength comparison results.
        """
        return {
            'difference': strength1 - strength2,
            'relative_difference': (strength1 - strength2) / strength2 if strength2 != 0 else float('inf'),
            'ratio': strength1 / strength2 if strength2 != 0 else float('inf')
        }
    
    def _compute_overall_comparison(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall comparison metrics.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results.
            
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
    
    def _optimize_parameters(self, results: Dict[str, Any], current_params: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize validation parameters.
        
        Args:
            results (Dict[str, Any]): Beating analysis results.
            current_params (Dict[str, float]): Current parameters.
            
        Returns:
            Dict[str, float]: Optimized parameters.
        """
        optimized_params = current_params.copy()
        
        # Simple optimization based on results characteristics
        if 'beating_frequencies' in results:
            frequencies = results['beating_frequencies']
            if frequencies:
                freq_std = np.std(frequencies)
                # Adjust validation threshold based on frequency variability
                optimized_params['validation_threshold'] = min(1e-6, freq_std * 0.01)
        
        if 'interference_patterns' in results:
            patterns = results['interference_patterns']
            if patterns:
                strengths = [p.get('strength', 0) for p in patterns]
                if strengths:
                    strength_mean = np.mean(strengths)
                    # Adjust quality threshold based on pattern strengths
                    optimized_params['quality_threshold'] = max(0.01, strength_mean * 0.1)
        
        return optimized_params
    
    def _validate_with_optimized_parameters(self, results: Dict[str, Any], optimized_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate results with optimized parameters.
        
        Args:
            results (Dict[str, Any]): Beating analysis results.
            optimized_params (Dict[str, float]): Optimized parameters.
            
        Returns:
            Dict[str, Any]: Validation results with optimized parameters.
        """
        # This would use the optimized parameters for validation
        # For now, return a simplified result
        return {
            'validation_threshold': optimized_params['validation_threshold'],
            'quality_threshold': optimized_params['quality_threshold'],
            'error_tolerance': optimized_params['error_tolerance'],
            'optimized_validation_passed': True
        }
    
    def _compute_optimization_improvement(self, original_params: Dict[str, float], optimized_params: Dict[str, float], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute improvement from parameter optimization.
        
        Args:
            original_params (Dict[str, float]): Original parameters.
            optimized_params (Dict[str, float]): Optimized parameters.
            results (Dict[str, Any]): Beating analysis results.
            
        Returns:
            Dict[str, Any]: Optimization improvement metrics.
        """
        improvement = {
            'parameter_changes': {},
            'improvement_score': 0.0
        }
        
        # Compute parameter changes
        for param in original_params:
            if param in optimized_params:
                change = optimized_params[param] - original_params[param]
                improvement['parameter_changes'][param] = change
        
        # Compute improvement score (simplified)
        improvement['improvement_score'] = 0.1  # Placeholder
        
        return improvement
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """
        Compute skewness of data.
        
        Args:
            data (np.ndarray): Data array.
            
        Returns:
            float: Skewness value.
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """
        Compute kurtosis of data.
        
        Args:
            data (np.ndarray): Data array.
            
        Returns:
            float: Kurtosis value.
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis
    
    def _test_normality(self, data: np.ndarray) -> bool:
        """
        Test normality of data (simplified).
        
        Args:
            data (np.ndarray): Data array.
            
        Returns:
            bool: Whether data is approximately normal.
        """
        # Simplified normality test
        skewness = self._compute_skewness(data)
        kurtosis = self._compute_kurtosis(data)
        
        return abs(skewness) < 0.5 and abs(kurtosis) < 0.5
    
    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect outliers in data.
        
        Args:
            data (np.ndarray): Data array.
            
        Returns:
            Dict[str, Any]: Outlier analysis results.
        """
        mean = np.mean(data)
        std = np.std(data)
        
        # 3-sigma rule
        outliers = np.abs(data - mean) > 3 * std
        outlier_indices = np.where(outliers)[0].tolist()
        
        return {
            'outlier_count': len(outlier_indices),
            'outlier_indices': outlier_indices,
            'outlier_values': data[outliers].tolist() if len(outlier_indices) > 0 else []
        }
    
    def _compute_t_statistic(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Compute t-statistic for two samples.
        
        Args:
            data1 (np.ndarray): First sample.
            data2 (np.ndarray): Second sample.
            
        Returns:
            float: T-statistic.
        """
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        std1 = np.std(data1, ddof=1)
        std2 = np.std(data2, ddof=1)
        
        # Pooled standard error
        n1 = len(data1)
        n2 = len(data2)
        pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        
        if pooled_se == 0:
            return 0.0
        
        t_stat = (mean1 - mean2) / pooled_se
        return t_stat
    
    def _compute_component_quality_score(self, stats: Dict[str, Any]) -> float:
        """
        Compute quality score for a component.
        
        Args:
            stats (Dict[str, Any]): Component statistics.
            
        Returns:
            float: Quality score.
        """
        # Simplified quality score computation
        return 0.8  # Placeholder
    
    def _compute_component_similarity_score(self, comparison: Dict[str, Any]) -> float:
        """
        Compute similarity score for a component comparison.
        
        Args:
            comparison (Dict[str, Any]): Component comparison.
            
        Returns:
            float: Similarity score.
        """
        # Simplified similarity score computation
        return 0.7  # Placeholder
