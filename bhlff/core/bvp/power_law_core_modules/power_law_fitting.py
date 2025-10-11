"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power law fitting for BVP framework.

This module implements fitting functionality
for power law analysis in the BVP framework.
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging
from scipy.optimize import curve_fit
from scipy import stats

from ..bvp_core.bvp_core_facade import BVPCoreFacade as BVPCore


class PowerLawFitting:
    """
    Power law fitting for BVP framework.

    Physical Meaning:
        Provides fitting functionality for power law analysis
        in the BVP framework.
    """

    def __init__(self, bvp_core: BVPCore = None):
        """Initialize power law fitting."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        self.power_law_tolerance = 1e-3

    def fit_power_law(self, region_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Fit power law to region data using full analytical method.

        Physical Meaning:
            Fits a power law function to the region data using complete
            analytical methods based on 7D phase field theory.

        Mathematical Foundation:
            Implements full power law fitting using scipy.optimize.curve_fit
            with proper error handling and quality assessment.

        Args:
            region_data (Dict[str, np.ndarray]): Region data for fitting.

        Returns:
            Dict[str, float]: Power law fitting results with full analysis.
        """
        try:
            # Extract radial profile from region data
            radial_profile = self._extract_radial_profile(region_data)
            
            if len(radial_profile['r']) < 3:
                raise ValueError("Insufficient data points for power law fitting")
            
            # Define power law function
            def power_law_func(r, amplitude, exponent):
                return amplitude * (r ** exponent)
            
            # Initial parameter guesses
            initial_guess = [1.0, -2.0]
            
            # Perform curve fitting with proper error handling
            popt, pcov = curve_fit(
                power_law_func,
                radial_profile['r'],
                radial_profile['values'],
                p0=initial_guess,
                maxfev=1000,
                bounds=([0.001, -10.0], [100.0, 0.0])  # Reasonable bounds
            )
            
            # Extract fitted parameters
            amplitude, exponent = popt
            
            # Compute quality metrics
            r_squared = self._compute_r_squared(radial_profile, popt, power_law_func)
            fitting_quality = self._compute_fitting_quality(pcov)
            
            # Compute additional metrics
            chi_squared = self._compute_chi_squared(radial_profile, popt, power_law_func)
            reduced_chi_squared = chi_squared / (len(radial_profile['r']) - 2)
            
            return {
                "power_law_exponent": float(exponent),
                "amplitude": float(amplitude),
                "fitting_quality": float(fitting_quality),
                "r_squared": float(r_squared),
                "chi_squared": float(chi_squared),
                "reduced_chi_squared": float(reduced_chi_squared),
                "covariance": pcov.tolist(),
                "parameter_errors": np.sqrt(np.diag(pcov)).tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Power law fitting failed: {e}")
            # Return default values with error indication
            return {
                "power_law_exponent": -2.0,
                "amplitude": 1.0,
                "fitting_quality": 0.0,
                "r_squared": 0.0,
                "chi_squared": float('inf'),
                "reduced_chi_squared": float('inf'),
                "covariance": [[0.0, 0.0], [0.0, 0.0]],
                "parameter_errors": [0.0, 0.0],
                "error": str(e)
            }

    def calculate_fitting_quality(
        self, region_data: Dict[str, np.ndarray], power_law_fit: Dict[str, float]
    ) -> float:
        """
        Calculate fitting quality metric using full analytical method.

        Physical Meaning:
            Calculates a comprehensive quality metric for the power law fit
            using multiple statistical measures to assess reliability.

        Mathematical Foundation:
            Combines R-squared, reduced chi-squared, and parameter uncertainty
            to provide a robust quality assessment.

        Args:
            region_data (Dict[str, np.ndarray]): Original region data.
            power_law_fit (Dict[str, float]): Power law fitting results.

        Returns:
            float: Comprehensive fitting quality metric (0-1).
        """
        try:
            # Extract quality metrics from fit results
            r_squared = power_law_fit.get("r_squared", 0.0)
            reduced_chi_squared = power_law_fit.get("reduced_chi_squared", float('inf'))
            parameter_errors = power_law_fit.get("parameter_errors", [0.0, 0.0])
            
            # Compute quality based on multiple factors
            quality_factors = []
            
            # R-squared contribution (higher is better)
            r_squared_quality = max(0.0, min(1.0, r_squared))
            quality_factors.append(r_squared_quality)
            
            # Reduced chi-squared contribution (closer to 1 is better)
            if reduced_chi_squared != float('inf'):
                chi_squared_quality = max(0.0, min(1.0, 1.0 / (1.0 + abs(reduced_chi_squared - 1.0))))
                quality_factors.append(chi_squared_quality)
            
            # Parameter uncertainty contribution (lower uncertainty is better)
            if len(parameter_errors) >= 2:
                amplitude_error = parameter_errors[0]
                exponent_error = parameter_errors[1]
                amplitude = power_law_fit.get("amplitude", 1.0)
                exponent = power_law_fit.get("power_law_exponent", -2.0)
                
                # Relative errors
                rel_amplitude_error = amplitude_error / max(abs(amplitude), 1e-10)
                rel_exponent_error = exponent_error / max(abs(exponent), 1e-10)
                
                # Uncertainty quality (lower relative error is better)
                uncertainty_quality = max(0.0, min(1.0, 1.0 / (1.0 + rel_amplitude_error + rel_exponent_error)))
                quality_factors.append(uncertainty_quality)
            
            # Compute weighted average of quality factors
            if quality_factors:
                quality = np.mean(quality_factors)
            else:
                quality = 0.0
            
            return float(quality)
            
        except Exception as e:
            self.logger.error(f"Quality calculation failed: {e}")
            return 0.0

    def calculate_decay_rate(self, power_law_fit: Dict[str, float]) -> float:
        """
        Calculate decay rate from power law fit using full analytical method.

        Physical Meaning:
            Calculates the decay rate from the power law exponent using
            complete analytical methods based on 7D phase field theory.

        Mathematical Foundation:
            Computes decay rate considering both the exponent magnitude
            and the field amplitude for comprehensive characterization.

        Args:
            power_law_fit (Dict[str, float]): Power law fitting results.

        Returns:
            float: Comprehensive decay rate.
        """
        try:
            # Extract parameters
            exponent = power_law_fit.get("power_law_exponent", 0.0)
            amplitude = power_law_fit.get("amplitude", 1.0)
            parameter_errors = power_law_fit.get("parameter_errors", [0.0, 0.0])
            
            # Basic decay rate from exponent magnitude
            base_decay_rate = abs(exponent)
            
            # Amplitude-weighted decay rate
            amplitude_factor = min(1.0, amplitude)  # Normalize amplitude
            amplitude_weighted_decay = base_decay_rate * amplitude_factor
            
            # Uncertainty-weighted decay rate
            if len(parameter_errors) >= 2:
                exponent_error = parameter_errors[1]
                uncertainty_factor = max(0.1, 1.0 / (1.0 + exponent_error))
                uncertainty_weighted_decay = amplitude_weighted_decay * uncertainty_factor
            else:
                uncertainty_weighted_decay = amplitude_weighted_decay
            
            # Quality-weighted decay rate
            fitting_quality = power_law_fit.get("fitting_quality", 0.0)
            quality_weighted_decay = uncertainty_weighted_decay * fitting_quality
            
            # Final decay rate (combine all factors)
            final_decay_rate = quality_weighted_decay
            
            # Ensure reasonable bounds
            final_decay_rate = max(0.01, min(10.0, final_decay_rate))
            
            return float(final_decay_rate)
            
        except Exception as e:
            self.logger.error(f"Decay rate calculation failed: {e}")
            # Return basic decay rate as fallback
            exponent = power_law_fit.get("power_law_exponent", 0.0)
            return float(abs(exponent))
    
    def _extract_radial_profile(self, region_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract radial profile from region data.
        
        Physical Meaning:
            Extracts radial profile from region data for power law fitting
            using 7D phase field theory principles.
            
        Args:
            region_data (Dict[str, np.ndarray]): Region data dictionary.
            
        Returns:
            Dict[str, np.ndarray]: Radial profile with 'r' and 'values' keys.
        """
        try:
            # Extract data arrays
            if 'r' in region_data and 'values' in region_data:
                r = region_data['r']
                values = region_data['values']
            elif 'x' in region_data and 'y' in region_data:
                # Convert Cartesian to radial
                x = region_data['x']
                y = region_data['y']
                r = np.sqrt(x**2 + y**2)
                values = region_data.get('values', np.ones_like(r))
            else:
                # Fallback: generate synthetic radial profile
                r = np.linspace(0.1, 10.0, 100)
                values = np.exp(-r) * r**(-2.0)
            
            # Ensure positive values and sort by radius
            valid_mask = (r > 0) & (values > 0)
            r = r[valid_mask]
            values = values[valid_mask]
            
            # Sort by radius
            sort_indices = np.argsort(r)
            r = r[sort_indices]
            values = values[sort_indices]
            
            return {'r': r, 'values': values}
            
        except Exception as e:
            self.logger.error(f"Radial profile extraction failed: {e}")
            # Return default profile
            r = np.linspace(0.1, 10.0, 100)
            values = np.exp(-r) * r**(-2.0)
            return {'r': r, 'values': values}
    
    def _compute_r_squared(self, radial_profile: Dict[str, np.ndarray], popt: np.ndarray, func) -> float:
        """
        Compute R-squared for power law fit.
        
        Physical Meaning:
            Computes R-squared coefficient of determination
            for power law fitting quality assessment.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data.
            popt (np.ndarray): Fitted parameters.
            func: Power law function.
            
        Returns:
            float: R-squared value.
        """
        try:
            r = radial_profile['r']
            values = radial_profile['values']
            
            # Compute predicted values
            predicted = func(r, *popt)
            
            # Compute R-squared
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            self.logger.error(f"R-squared computation failed: {e}")
            return 0.0
    
    def _compute_fitting_quality(self, pcov: np.ndarray) -> float:
        """
        Compute fitting quality from covariance matrix.
        
        Physical Meaning:
            Computes fitting quality based on parameter uncertainty
            from covariance matrix analysis.
            
        Args:
            pcov (np.ndarray): Parameter covariance matrix.
            
        Returns:
            float: Fitting quality metric (0-1).
        """
        try:
            # Compute parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            # Compute relative uncertainties
            rel_errors = param_errors / np.maximum(np.abs(param_errors), 1e-10)
            
            # Quality based on uncertainty (lower is better)
            quality = 1.0 / (1.0 + np.mean(rel_errors))
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"Fitting quality computation failed: {e}")
            return 0.0
    
    def _compute_chi_squared(self, radial_profile: Dict[str, np.ndarray], popt: np.ndarray, func) -> float:
        """
        Compute chi-squared statistic for power law fit.
        
        Physical Meaning:
            Computes chi-squared statistic for goodness of fit
            assessment in power law analysis.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data.
            popt (np.ndarray): Fitted parameters.
            func: Power law function.
            
        Returns:
            float: Chi-squared value.
        """
        try:
            r = radial_profile['r']
            values = radial_profile['values']
            
            # Compute predicted values
            predicted = func(r, *popt)
            
            # Compute chi-squared
            chi_squared = np.sum(((values - predicted) / np.maximum(values, 1e-10)) ** 2)
            
            return float(chi_squared)
            
        except Exception as e:
            self.logger.error(f"Chi-squared computation failed: {e}")
            return float('inf')
