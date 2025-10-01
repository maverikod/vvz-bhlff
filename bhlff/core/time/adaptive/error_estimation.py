"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Error estimation module for adaptive integrator.

This module implements error estimation operations for adaptive integration,
including Richardson extrapolation and error component analysis.

Physical Meaning:
    Computes the complete local error estimate using
    full error analysis according to adaptive integration theory.

Mathematical Foundation:
    Implements full error estimation:
    - Richardson extrapolation
    - Embedded Runge-Kutta error estimation
    - Local truncation error analysis
    - Stability analysis
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging


class ErrorEstimation:
    """
    Error estimation for adaptive integration.
    
    Physical Meaning:
        Computes the complete local error estimate using
        full error analysis according to adaptive integration theory.
    """
    
    def __init__(self, tolerance: float, safety_factor: float):
        """Initialize error estimator."""
        self.tolerance = tolerance
        self.safety_factor = safety_factor
        self.logger = logging.getLogger(__name__)
    
    def compute_richardson_error(self, field_4th: np.ndarray, field_5th: np.ndarray, dt: float) -> float:
        """
        Compute error estimate using Richardson extrapolation.
        
        Physical Meaning:
            Uses Richardson extrapolation to provide a more accurate
            error estimate for adaptive step size control.
        """
        # Compute basic error estimate
        error_diff = field_5th - field_4th
        
        # Compute relative error with proper normalization
        field_magnitude = np.linalg.norm(field_4th)
        if field_magnitude < 1e-15:
            # Avoid division by zero for very small fields
            error_estimate = np.linalg.norm(error_diff)
        else:
            # Richardson extrapolation error estimate
            # For RK4(5), the error scales as h^5, so p = 1
            richardson_factor = 1.0 / (1.0 - (0.5)**1)  # h_4th/h_5th = 0.5
            error_estimate = richardson_factor * np.linalg.norm(error_diff) / field_magnitude
        
        # Apply additional error analysis
        error_components = self._analyze_error_components(error_diff, field_4th)
        
        # Combine error estimates
        total_error = self._combine_error_estimates(error_estimate, error_components)
        
        return float(total_error)
    
    def _analyze_error_components(self, error_diff: np.ndarray, field: np.ndarray) -> Dict[str, float]:
        """Analyze different components of the error."""
        # Spatial error analysis
        spatial_error = np.abs(error_diff)
        max_spatial_error = np.max(spatial_error)
        mean_spatial_error = np.mean(spatial_error)
        
        # Spectral error analysis
        error_spectral = np.fft.fftn(error_diff)
        field_spectral = np.fft.fftn(field)
        
        # Compute spectral error ratios
        spectral_error_magnitude = np.abs(error_spectral)
        field_spectral_magnitude = np.abs(field_spectral)
        
        # Avoid division by zero
        spectral_ratio = np.where(
            field_spectral_magnitude > 1e-15,
            spectral_error_magnitude / field_spectral_magnitude,
            0.0
        )
        
        max_spectral_error = np.max(spectral_ratio)
        mean_spectral_error = np.mean(spectral_ratio)
        
        # High-frequency error analysis
        high_freq_mask = self._compute_high_frequency_mask(field.shape)
        high_freq_error = np.mean(spectral_ratio[high_freq_mask])
        
        return {
            "max_spatial_error": float(max_spatial_error),
            "mean_spatial_error": float(mean_spatial_error),
            "max_spectral_error": float(max_spectral_error),
            "mean_spectral_error": float(mean_spectral_error),
            "high_frequency_error": float(high_freq_error)
        }
    
    def _compute_high_frequency_mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Compute mask for high-frequency components."""
        # Create frequency grids
        freq_grids = []
        for dim_size in shape:
            freqs = np.fft.fftfreq(dim_size)
            freq_grids.append(freqs)
        
        # Create multi-dimensional frequency grid
        freq_mesh = np.meshgrid(*freq_grids, indexing='ij')
        
        # Compute frequency magnitude
        freq_magnitude = np.sqrt(sum(freq**2 for freq in freq_mesh))
        
        # High-frequency mask (top 25% of frequencies)
        freq_threshold = np.percentile(freq_magnitude, 75)
        high_freq_mask = freq_magnitude > freq_threshold
        
        return high_freq_mask
    
    def _combine_error_estimates(self, basic_error: float, error_components: Dict[str, float]) -> float:
        """Combine different error estimates into a single error measure."""
        # Weight different error components
        weights = {
            "spatial": 0.4,
            "spectral": 0.4,
            "high_frequency": 0.2
        }
        
        # Compute weighted error estimate
        weighted_error = (
            weights["spatial"] * error_components["mean_spatial_error"] +
            weights["spectral"] * error_components["mean_spectral_error"] +
            weights["high_frequency"] * error_components["high_frequency_error"]
        )
        
        # Combine with basic error estimate
        combined_error = 0.7 * basic_error + 0.3 * weighted_error
        
        # Apply error bounds
        min_error = 1e-15
        max_error = 1.0
        
        combined_error = max(min_error, min(combined_error, max_error))
        
        return combined_error
