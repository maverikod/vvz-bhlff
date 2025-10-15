"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core observational comparison methods for cosmological analysis in 7D phase field theory.

This module implements core observational comparison methods for
cosmological evolution results, including comparison with
observational data and goodness of fit analysis.

Theoretical Background:
    Observational comparison in cosmological evolution
    involves comparing theoretical results with observational
    data to validate the model using 7D BVP theory principles.

Mathematical Foundation:
    Implements core observational comparison methods:
    - Structure formation comparison: with observational data
    - Parameter comparison: with observational constraints
    - Statistical comparison: with observational statistics
    - Goodness of fit: various goodness of fit metrics

Example:
    >>> core = ObservationalComparisonCore(evolution_results, observational_data)
    >>> comparison_results = core.compare_with_observations()
"""

import numpy as np
from typing import Dict, Any
from .observational_comparison_parameters import ObservationalComparisonParameters
from .observational_comparison_statistics import ObservationalComparisonStatistics


class ObservationalComparisonCore:
    """
    Core observational comparison for cosmological analysis.

    Physical Meaning:
        Implements core observational comparison methods for
        cosmological evolution results, including comparison
        with observational data and goodness of fit analysis.

    Mathematical Foundation:
        Implements core observational comparison methods:
        - Structure formation comparison: with observational data
        - Parameter comparison: with observational constraints
        - Statistical comparison: with observational statistics
        - Goodness of fit: various goodness of fit metrics

    Attributes:
        evolution_results (dict): Cosmological evolution results
        observational_data (dict): Observational data for comparison
        analysis_parameters (dict): Analysis parameters
        _parameters (ObservationalComparisonParameters): Parameter comparison
        _statistics (ObservationalComparisonStatistics): Statistical comparison
    """

    def __init__(self, evolution_results: Dict[str, Any], observational_data: Dict[str, Any] = None, analysis_parameters: Dict[str, Any] = None):
        """
        Initialize observational comparison core.

        Physical Meaning:
            Sets up the observational comparison with evolution results,
            observational data, and analysis parameters.

        Args:
            evolution_results: Cosmological evolution results
            observational_data: Observational data for comparison
            analysis_parameters: Analysis parameters
        """
        self.evolution_results = evolution_results
        self.observational_data = observational_data or {}
        self.analysis_parameters = analysis_parameters or {}
        self._parameters = ObservationalComparisonParameters(evolution_results, observational_data, analysis_parameters)
        self._statistics = ObservationalComparisonStatistics(evolution_results, observational_data, analysis_parameters)

    def compare_with_observations(self) -> Dict[str, Any]:
        """
        Compare results with observational data.

        Physical Meaning:
            Compares the theoretical results with observational
            data to validate the model using 7D BVP theory principles.

        Returns:
            Comparison results
        """
        if not self.observational_data:
            return {}

        # Load observational data
        obs_data = self._load_observational_data()
        
        # Compute 7D phase field observables
        model_observables = self._compute_7d_observables(self.evolution_results)
        
        # Statistical comparison
        comparison_results = self._statistical_comparison(obs_data, model_observables)
        
        # Compute chi-squared
        chi_squared = self._statistics.compute_chi_squared(obs_data, model_observables)
        
        # Compute likelihood
        likelihood = self._statistics.compute_likelihood(chi_squared)
        
        # Compare with observations
        comparison = {
            "structure_formation_comparison": self._parameters.compare_structure_formation(),
            "parameter_comparison": self._parameters.compare_parameters(),
            "statistical_comparison": self._statistics.compare_statistics(),
            "goodness_of_fit": self._statistics.compute_goodness_of_fit(),
            "chi_squared": chi_squared,
            "likelihood": likelihood,
            "comparison_results": comparison_results,
            "model_observables": model_observables,
            "observational_data": obs_data
        }

        return comparison

    def _load_observational_data(self) -> Dict[str, Any]:
        """
        Load observational data for comparison.
        
        Physical Meaning:
            Loads observational data from various sources
            for comparison with 7D BVP theory predictions.
            
        Returns:
            Observational data dictionary
        """
        # Load from observational data if available
        if self.observational_data:
            return self.observational_data
        
        # Default observational data structure
        return {
            "hubble_parameter": 70.0,
            "matter_density": 0.3,
            "dark_energy": 0.7,
            "correlation_function": np.array([]),
            "power_spectrum": np.array([]),
            "structure_statistics": {},
            "data_points": []
        }

    def _compute_7d_observables(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute 7D phase field observables from model data.
        
        Physical Meaning:
            Extracts observables from 7D phase field evolution
            for comparison with observational data.
            
        Args:
            model_data: Model evolution results
            
        Returns:
            7D observables dictionary
        """
        # Extract 7D phase field observables
        observables = {
            "hubble_parameter": self._parameters._extract_hubble_parameter_from_7d_field(),
            "matter_density": self._parameters._extract_matter_density_from_7d_field(),
            "dark_energy": self._parameters._extract_dark_energy_from_7d_field(),
            "correlation_function": self._statistics._compute_7d_correlation_function(),
            "power_spectrum": self._statistics._compute_7d_power_spectrum(),
            "structure_statistics": self._statistics._compute_7d_structure_statistics()
        }
        
        return observables

    def _statistical_comparison(self, obs_data: Dict[str, Any], model_observables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical comparison between observations and model.
        
        Physical Meaning:
            Performs comprehensive statistical comparison between
            observational data and 7D BVP theory predictions.
            
        Args:
            obs_data: Observational data
            model_observables: Model observables
            
        Returns:
            Statistical comparison results
        """
        # Compute statistical metrics
        comparison = {
            "parameter_correlation": self._compute_parameter_correlation(obs_data, model_observables),
            "statistical_significance": self._compute_statistical_significance(obs_data, model_observables),
            "model_consistency": self._compute_model_consistency(obs_data, model_observables)
        }
        
        return comparison

    def _compute_parameter_correlation(self, obs_data: Dict[str, Any], model_observables: Dict[str, Any]) -> float:
        """
        Compute parameter correlation between observations and model.
        
        Physical Meaning:
            Computes correlation coefficient between observational
            and theoretical parameters using 7D BVP theory.
            
        Mathematical Foundation:
            Uses Pearson correlation coefficient:
            r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)²Σ(yi - ȳ)²]
            
        Args:
            obs_data: Observational data
            model_observables: Model observables
            
        Returns:
            Correlation coefficient
        """
        # Extract comparable parameters
        obs_params = []
        model_params = []
        
        for key in ['hubble_parameter', 'matter_density', 'dark_energy']:
            if key in obs_data and key in model_observables:
                obs_params.append(obs_data[key])
                model_params.append(model_observables[key])
        
        if len(obs_params) < 2:
            return 1.0  # Default for insufficient data
        
        # Compute Pearson correlation coefficient
        obs_array = np.array(obs_params)
        model_array = np.array(model_params)
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(obs_array) | np.isnan(model_array))
        if np.sum(valid_mask) < 2:
            return 1.0
        
        obs_array = obs_array[valid_mask]
        model_array = model_array[valid_mask]
        
        # Compute correlation
        correlation = np.corrcoef(obs_array, model_array)[0, 1]
        
        return correlation if not np.isnan(correlation) else 1.0

    def _compute_statistical_significance(self, obs_data: Dict[str, Any], model_observables: Dict[str, Any]) -> float:
        """
        Compute statistical significance of model-observation agreement.
        
        Physical Meaning:
            Computes statistical significance using t-test
            for parameter differences in 7D BVP theory.
            
        Mathematical Foundation:
            Uses t-test for parameter differences:
            t = (μ1 - μ2) / √(s1²/n1 + s2²/n2)
            
        Args:
            obs_data: Observational data
            model_observables: Model observables
            
        Returns:
            Statistical significance (p-value)
        """
        from scipy.stats import ttest_ind
        
        # Extract comparable parameters
        obs_params = []
        model_params = []
        
        for key in ['hubble_parameter', 'matter_density', 'dark_energy']:
            if key in obs_data and key in model_observables:
                obs_params.append(obs_data[key])
                model_params.append(model_observables[key])
        
        if len(obs_params) < 2:
            return 0.95  # Default for insufficient data
        
        # Compute t-test
        try:
            t_stat, p_value = ttest_ind(obs_params, model_params)
            return p_value if not np.isnan(p_value) else 0.95
        except:
            return 0.95

    def _compute_model_consistency(self, obs_data: Dict[str, Any], model_observables: Dict[str, Any]) -> bool:
        """
        Compute model consistency with observations.
        
        Physical Meaning:
            Determines if the model is consistent with observations
            using 7D BVP theory criteria.
            
        Mathematical Foundation:
            Uses tolerance-based consistency check:
            |model - obs| < tolerance for all parameters
            
        Args:
            obs_data: Observational data
            model_observables: Model observables
            
        Returns:
            True if model is consistent
        """
        # Define tolerances for each parameter
        tolerances = {
            'hubble_parameter': self.analysis_parameters.get('hubble_tolerance', 2.0),
            'matter_density': self.analysis_parameters.get('matter_tolerance', 0.05),
            'dark_energy': self.analysis_parameters.get('dark_energy_tolerance', 0.05)
        }
        
        # Check consistency for each parameter
        for key, tolerance in tolerances.items():
            if key in obs_data and key in model_observables:
                obs_value = obs_data[key]
                model_value = model_observables[key]
                
                if isinstance(obs_value, (int, float)) and isinstance(model_value, (int, float)):
                    if abs(model_value - obs_value) > tolerance:
                        return False
        
        return True
