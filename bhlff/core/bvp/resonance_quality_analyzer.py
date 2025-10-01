"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Resonance quality factor analysis for BVP impedance analysis.

This module implements algorithms for calculating quality factors
of resonance peaks using Lorentzian fitting and FWHM analysis.

Physical Meaning:
    Estimates the quality factor Q by fitting a Lorentzian function
    to resonance peaks, representing the energy storage characteristics
    of the resonant system.

Mathematical Foundation:
    Q = f₀ / Δf where f₀ is the resonance frequency and
    Δf is the full width at half maximum (FWHM).

Example:
    >>> analyzer = ResonanceQualityAnalyzer(constants)
    >>> q_factor = analyzer.calculate_quality_factor(frequencies, magnitude, peak_idx)
"""

import numpy as np
from typing import List, Dict

from .bvp_constants import BVPConstants


class ResonanceQualityAnalyzer:
    """
    Quality factor analysis for resonance peaks.

    Physical Meaning:
        Calculates quality factors of resonance peaks using
        Lorentzian fitting and FWHM analysis.
    """

    def __init__(self, constants: BVPConstants):
        """
        Initialize quality analyzer.

        Args:
            constants (BVPConstants): BVP constants instance.
        """
        self.constants = constants

    def calculate_quality_factors(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_indices: List[int]
    ) -> List[float]:
        """
        Calculate quality factors for multiple peaks.

        Physical Meaning:
            Estimates quality factors for all detected resonance peaks
            using Lorentzian fitting and FWHM analysis.

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Signal magnitude.
            peak_indices (List[int]): Peak indices.

        Returns:
            List[float]: Quality factors for each peak.
        """
        quality_factors = []

        for peak_idx in peak_indices:
            q_factor = self.calculate_quality_factor(frequencies, magnitude, peak_idx)
            quality_factors.append(q_factor)

        return quality_factors

    def calculate_quality_factor(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_idx: int
    ) -> float:
        """
        Calculate quality factor using Lorentzian fitting.

        Physical Meaning:
            Estimates the quality factor Q by fitting a Lorentzian
            function to the resonance peak.

        Mathematical Foundation:
            Q = f₀ / Δf where f₀ is the resonance frequency and
            Δf is the full width at half maximum (FWHM).

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Signal magnitude.
            peak_idx (int): Peak index.

        Returns:
            float: Quality factor.
        """
        if peak_idx <= 0 or peak_idx >= len(frequencies) - 1:
            return self.constants.get_impedance_parameter("min_quality_factor")

        # Extract peak region around peak
        window = self.constants.get_impedance_parameter("peak_window_size")
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(frequencies), peak_idx + window + 1)

        peak_freqs = frequencies[start_idx:end_idx]
        peak_mags = magnitude[start_idx:end_idx]

        # Find half-maximum points
        peak_magnitude = magnitude[peak_idx]
        half_max = peak_magnitude / 2

        # Full Lorentzian fitting using scipy.optimize.curve_fit
        try:
            from scipy.optimize import curve_fit
            
            # Define Lorentzian function
            def lorentzian(f, f0, gamma, A, offset):
                """
                Lorentzian function: A * gamma^2 / ((f - f0)^2 + gamma^2) + offset
                where gamma = FWHM / 2
                """
                return A * gamma**2 / ((f - f0)**2 + gamma**2) + offset
            
            # Initial parameter estimates
            f0_guess = frequencies[peak_idx]
            gamma_guess = (frequencies[end_idx-1] - frequencies[start_idx]) / 4  # Rough estimate
            A_guess = peak_magnitude
            offset_guess = np.mean(magnitude[start_idx:end_idx])
            
            initial_guess = [f0_guess, gamma_guess, A_guess, offset_guess]
            
            # Parameter bounds
            bounds = (
                [frequencies[start_idx], 0, 0, 0],  # Lower bounds
                [frequencies[end_idx-1], np.inf, np.inf, np.inf]  # Upper bounds
            )
            
            # Perform curve fitting
            popt, pcov = curve_fit(
                lorentzian, 
                peak_freqs, 
                peak_mags, 
                p0=initial_guess,
                bounds=bounds,
                maxfev=1000
            )
            
            # Extract fitted parameters
            f0_fitted, gamma_fitted, A_fitted, offset_fitted = popt
            
            # Calculate quality factor from fitted parameters
            fwhm_fitted = 2 * gamma_fitted
            if fwhm_fitted > 0:
                q_factor = f0_fitted / fwhm_fitted
                
                # Apply quality factor bounds
                min_q = self.constants.get_impedance_parameter("min_quality_factor")
                max_q = self.constants.get_impedance_parameter("max_quality_factor")
                q_factor = max(min_q, min(max_q, q_factor))
                
                # Store fitting results for analysis
                self._last_fitting_results = {
                    "parameters": popt,
                    "covariance": pcov,
                    "f0": f0_fitted,
                    "gamma": gamma_fitted,
                    "A": A_fitted,
                    "offset": offset_fitted,
                    "fwhm": fwhm_fitted,
                    "q_factor": q_factor,
                    "fitting_quality": self._assess_fitting_quality(popt, pcov, peak_freqs, peak_mags)
                }
                
                return q_factor
            else:
                # Fallback if FWHM is invalid
                return self._fallback_quality_estimation(frequencies, magnitude, peak_idx)
                
        except (ImportError, RuntimeError, ValueError) as e:
            # Fallback if scipy not available or fitting fails
            return self._fallback_quality_estimation(frequencies, magnitude, peak_idx)

    def fit_lorentzian(
        self, frequencies: np.ndarray, magnitude: np.ndarray, peak_idx: int
    ) -> dict:
        """
        Fit Lorentzian function to resonance peak using full optimization.

        Physical Meaning:
            Fits a Lorentzian function to the resonance peak using
            full scipy.optimize.curve_fit optimization for accurate
            quality factor estimation.

        Mathematical Foundation:
            Lorentzian: L(f) = A * γ² / ((f - f₀)² + γ²) + offset
            where A is amplitude, f₀ is center frequency, γ = FWHM/2.

        Args:
            frequencies (np.ndarray): Frequency array.
            magnitude (np.ndarray): Signal magnitude.
            peak_idx (int): Peak index.

        Returns:
            dict: Lorentzian fit parameters with full optimization results.
        """
        if peak_idx <= 0 or peak_idx >= len(frequencies) - 1:
            return {"amplitude": 0, "center": 0, "fwhm": 0, "q_factor": 0, "fitting_quality": 0.0}

        # Extract peak region
        window = self.constants.get_impedance_parameter("peak_window_size")
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(frequencies), peak_idx + window + 1)

        peak_freqs = frequencies[start_idx:end_idx]
        peak_mags = magnitude[start_idx:end_idx]

        # Full Lorentzian fitting using scipy.optimize.curve_fit
        try:
            from scipy.optimize import curve_fit
            
            # Define Lorentzian function
            def lorentzian(f, f0, gamma, A, offset):
                """
                Lorentzian function: A * gamma^2 / ((f - f0)^2 + gamma^2) + offset
                where gamma = FWHM / 2
                """
                return A * gamma**2 / ((f - f0)**2 + gamma**2) + offset
            
            # Initial parameter estimates
            f0_guess = frequencies[peak_idx]
            gamma_guess = (frequencies[end_idx-1] - frequencies[start_idx]) / 4
            A_guess = magnitude[peak_idx]
            offset_guess = np.mean(magnitude[start_idx:end_idx])
            
            initial_guess = [f0_guess, gamma_guess, A_guess, offset_guess]
            
            # Parameter bounds
            bounds = (
                [frequencies[start_idx], 0, 0, 0],  # Lower bounds
                [frequencies[end_idx-1], np.inf, np.inf, np.inf]  # Upper bounds
            )
            
            # Perform curve fitting with full optimization
            popt, pcov = curve_fit(
                lorentzian, 
                peak_freqs, 
                peak_mags, 
                p0=initial_guess,
                bounds=bounds,
                maxfev=2000,  # Increased iterations for better convergence
                method='trf'  # Trust Region Reflective algorithm
            )
            
            # Extract fitted parameters
            f0_fitted, gamma_fitted, A_fitted, offset_fitted = popt
            
            # Calculate quality factor from fitted parameters
            fwhm_fitted = 2 * gamma_fitted
            q_factor = f0_fitted / fwhm_fitted if fwhm_fitted > 0 else 0
            
            # Apply quality factor bounds
            min_q = self.constants.get_impedance_parameter("min_quality_factor")
            max_q = self.constants.get_impedance_parameter("max_quality_factor")
            q_factor = max(min_q, min(max_q, q_factor))
            
            # Assess fitting quality
            fitting_quality = self._assess_fitting_quality(popt, pcov, peak_freqs, peak_mags)
            
            return {
                "amplitude": float(A_fitted),
                "center": float(f0_fitted),
                "fwhm": float(fwhm_fitted),
                "q_factor": float(q_factor),
                "gamma": float(gamma_fitted),
                "offset": float(offset_fitted),
                "fitting_quality": fitting_quality["fitting_quality_score"],
                "r_squared": fitting_quality["r_squared"],
                "chi_squared": fitting_quality["chi_squared"],
                "parameter_errors": fitting_quality["parameter_errors"],
                "covariance_matrix": pcov.tolist()
            }
            
        except (ImportError, RuntimeError, ValueError) as e:
            # Fallback to simplified estimation
            return self._fallback_lorentzian_estimation(frequencies, magnitude, peak_idx)
    
    def _assess_fitting_quality(self, popt: np.ndarray, pcov: np.ndarray, 
                               frequencies: np.ndarray, magnitude: np.ndarray) -> Dict[str, float]:
        """
        Assess the quality of the Lorentzian fitting.
        
        Physical Meaning:
            Evaluates the quality of the curve fitting by computing
            various statistical measures and goodness-of-fit metrics.
        """
        # Compute residuals
        f0, gamma, A, offset = popt
        fitted_values = A * gamma**2 / ((frequencies - f0)**2 + gamma**2) + offset
        residuals = magnitude - fitted_values
        
        # Compute R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((magnitude - np.mean(magnitude))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Compute parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        relative_errors = param_errors / np.abs(popt)
        
        # Compute chi-squared
        chi_squared = np.sum((residuals / np.std(residuals))**2)
        reduced_chi_squared = chi_squared / (len(frequencies) - len(popt))
        
        # Compute AIC (Akaike Information Criterion)
        aic = 2 * len(popt) + len(frequencies) * np.log(ss_res / len(frequencies))
        
        return {
            "r_squared": float(r_squared),
            "chi_squared": float(chi_squared),
            "reduced_chi_squared": float(reduced_chi_squared),
            "aic": float(aic),
            "parameter_errors": param_errors.tolist(),
            "relative_errors": relative_errors.tolist(),
            "fitting_quality_score": self._compute_fitting_quality_score(r_squared, reduced_chi_squared, relative_errors)
        }
    
    def _compute_fitting_quality_score(self, r_squared: float, reduced_chi_squared: float, 
                                     relative_errors: np.ndarray) -> float:
        """
        Compute overall fitting quality score.
        
        Physical Meaning:
            Combines multiple quality metrics into a single score
            for assessing the reliability of the fitting results.
        """
        # R-squared score (0-1, higher is better)
        r_squared_score = r_squared
        
        # Chi-squared score (0-1, closer to 1 is better)
        chi_squared_score = 1.0 / (1.0 + abs(reduced_chi_squared - 1.0))
        
        # Parameter uncertainty score (0-1, lower uncertainty is better)
        max_relative_error = np.max(relative_errors)
        uncertainty_score = 1.0 / (1.0 + max_relative_error)
        
        # Combined score
        quality_score = 0.4 * r_squared_score + 0.3 * chi_squared_score + 0.3 * uncertainty_score
        
        return float(quality_score)
    
    def _fallback_quality_estimation(self, frequencies: np.ndarray, magnitude: np.ndarray, 
                                   peak_idx: int) -> float:
        """
        Fallback quality estimation when fitting fails.
        
        Physical Meaning:
            Provides a simple quality factor estimation when
            the full Lorentzian fitting is not available or fails.
        """
        # Simple FWHM estimation
        peak_magnitude = magnitude[peak_idx]
        half_max = peak_magnitude / 2
        
        # Find half-maximum points
        left_idx = peak_idx
        right_idx = peak_idx
        
        # Find left half-maximum
        for i in range(peak_idx, 0, -1):
            if magnitude[i] <= half_max:
                left_idx = i
                break
        
        # Find right half-maximum
        for i in range(peak_idx, len(magnitude)):
            if magnitude[i] <= half_max:
                right_idx = i
                break
        
        # Calculate FWHM
        if right_idx > left_idx:
            fwhm = frequencies[right_idx] - frequencies[left_idx]
            if fwhm > 0:
                q_factor = frequencies[peak_idx] / fwhm
                min_q = self.constants.get_impedance_parameter("min_quality_factor")
                max_q = self.constants.get_impedance_parameter("max_quality_factor")
                return max(min_q, min(max_q, q_factor))
        
        # Final fallback: estimate from peak width
        min_q = self.constants.get_impedance_parameter("min_quality_factor")
        return max(min_q, peak_magnitude / np.mean(magnitude))
    
    def _fallback_lorentzian_estimation(self, frequencies: np.ndarray, magnitude: np.ndarray, 
                                      peak_idx: int) -> dict:
        """
        Fallback Lorentzian estimation when full optimization fails.
        
        Physical Meaning:
            Provides a simplified Lorentzian parameter estimation when
            the full scipy.optimize.curve_fit optimization is not available or fails.
        """
        # Extract peak region
        window = self.constants.get_impedance_parameter("peak_window_size")
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(frequencies), peak_idx + window + 1)

        peak_freqs = frequencies[start_idx:end_idx]
        peak_mags = magnitude[start_idx:end_idx]

        # Simple parameter estimation
        amplitude = magnitude[peak_idx]
        center = frequencies[peak_idx]
        
        # Estimate FWHM using half-maximum method
        half_max = amplitude / 2
        left_idx = peak_idx
        right_idx = peak_idx
        
        # Find left half-maximum
        for i in range(peak_idx, start_idx, -1):
            if magnitude[i] <= half_max:
                left_idx = i
                break
        
        # Find right half-maximum
        for i in range(peak_idx, end_idx):
            if magnitude[i] <= half_max:
                right_idx = i
                break
        
        # Calculate FWHM
        if right_idx > left_idx:
            fwhm = frequencies[right_idx] - frequencies[left_idx]
        else:
            fwhm = (frequencies[end_idx-1] - frequencies[start_idx]) / 4
        
        # Calculate gamma and Q factor
        gamma = fwhm / 2
        q_factor = center / fwhm if fwhm > 0 else 0
        
        # Apply bounds
        min_q = self.constants.get_impedance_parameter("min_quality_factor")
        max_q = self.constants.get_impedance_parameter("max_quality_factor")
        q_factor = max(min_q, min(max_q, q_factor))
        
        # Estimate offset
        offset = np.mean(magnitude[start_idx:end_idx])
        
        return {
            "amplitude": float(amplitude),
            "center": float(center),
            "fwhm": float(fwhm),
            "q_factor": float(q_factor),
            "gamma": float(gamma),
            "offset": float(offset),
            "fitting_quality": 0.5,  # Lower quality for fallback
            "r_squared": 0.0,  # No fitting performed
            "chi_squared": 0.0,
            "parameter_errors": [0.0, 0.0, 0.0, 0.0],
            "covariance_matrix": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        }
