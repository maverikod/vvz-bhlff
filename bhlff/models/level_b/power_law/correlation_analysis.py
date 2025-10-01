"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Correlation analysis module for power law analysis.

This module implements correlation function analysis for the 7D phase field theory,
including spatial correlation functions and correlation length calculations.

Physical Meaning:
    Analyzes spatial correlation functions in 7D space-time to understand
    the structure and coherence of the BVP field distribution.

Mathematical Foundation:
    Implements 7D correlation analysis:
    C(r) = ∫ a(x) a*(x+r) dV_7
    where integration preserves the 7D structure.
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

from ...core.bvp import BVPCore


class CorrelationAnalysis:
    """
    Correlation analysis for BVP field.
    
    Physical Meaning:
        Computes spatial correlation functions in 7D space-time
        to analyze field coherence and structure.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """Initialize correlation analyzer."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
    
    def compute_correlation_functions(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Compute full 7D spatial correlation functions.
        
        Physical Meaning:
            Computes the complete 7D spatial correlation function
            C(r) = ⟨a(x)a(x+r)⟩ for all 7 dimensions according to
            the 7D phase field theory.
            
        Mathematical Foundation:
            C(r) = ∫ a(x) a*(x+r) dV_7
            where integration is over all 7D space-time M₇.
        """
        amplitude = np.abs(envelope)
        
        # Compute full 7D correlation function
        correlation_7d = self._compute_7d_correlation_function(amplitude)
        
        # Compute correlation lengths in each dimension
        correlation_lengths = self._compute_7d_correlation_lengths(correlation_7d)
        
        # Analyze 7D correlation structure
        correlation_structure = self._analyze_7d_correlation_structure(correlation_7d)
        
        # Compute individual dimension correlations
        dimensional_correlations = self._compute_dimensional_correlations(amplitude)

        return {
            "spatial_correlation_7d": correlation_7d,
            "correlation_lengths": correlation_lengths,
            "correlation_structure": correlation_structure,
            "dimensional_correlations": dimensional_correlations,
        }
    
    def _compute_7d_correlation_function(self, amplitude: np.ndarray) -> np.ndarray:
        """Compute full 7D correlation function preserving dimensional structure."""
        # Initialize correlation function with same shape as input
        correlation_7d = np.zeros_like(amplitude)
        
        # Compute correlation for each dimension
        for dim in range(amplitude.ndim):
            # Compute correlation along this dimension
            correlation_dim = self._compute_dimension_correlation(amplitude, dim)
            correlation_7d += correlation_dim
        
        # Normalize by number of dimensions
        correlation_7d /= amplitude.ndim
        
        return correlation_7d
    
    def _compute_dimension_correlation(self, amplitude: np.ndarray, dim: int) -> np.ndarray:
        """Compute correlation along a specific dimension."""
        # Create shifted versions along the dimension
        correlation_dim = np.zeros_like(amplitude)
        
        # Compute correlation for different shifts
        for shift in range(min(amplitude.shape[dim], 10)):  # Limit shifts for efficiency
            # Create shifted array
            shifted = np.roll(amplitude, shift, axis=dim)
            
            # Compute correlation
            correlation_shift = amplitude * np.conj(shifted)
            correlation_dim += correlation_shift
        
        # Normalize by number of shifts
        correlation_dim /= min(amplitude.shape[dim], 10)
        
        return correlation_dim
    
    def _compute_7d_correlation_lengths(self, correlation_7d: np.ndarray) -> Dict[str, float]:
        """Compute correlation lengths in each dimension."""
        correlation_lengths = {}
        
        for dim in range(correlation_7d.ndim):
            # Compute correlation length along this dimension
            correlation_1d = np.mean(correlation_7d, axis=tuple(i for i in range(correlation_7d.ndim) if i != dim))
            
            # Find correlation length (where correlation drops to 1/e)
            max_corr = np.max(correlation_1d)
            target_corr = max_corr / np.e
            
            # Find first point below target
            correlation_length = 0
            for i, corr in enumerate(correlation_1d):
                if corr < target_corr:
                    correlation_length = i
                    break
            
            correlation_lengths[f"dim_{dim}"] = float(correlation_length)
        
        return correlation_lengths
    
    def _analyze_7d_correlation_structure(self, correlation_7d: np.ndarray) -> Dict[str, Any]:
        """Analyze 7D correlation structure."""
        # Compute anisotropy measures
        max_correlation = np.max(correlation_7d)
        mean_correlation = np.mean(correlation_7d)
        
        # Compute dimensional coupling
        dimensional_coupling = self._compute_dimensional_coupling(correlation_7d)
        
        # Compute correlation decay
        correlation_decay = self._compute_correlation_decay(correlation_7d)
        
        return {
            "max_correlation": float(max_correlation),
            "mean_correlation": float(mean_correlation),
            "dimensional_coupling": dimensional_coupling,
            "correlation_decay": correlation_decay,
            "anisotropy_measure": float(max_correlation / mean_correlation) if mean_correlation > 0 else 0.0
        }
    
    def _compute_dimensional_coupling(self, correlation_7d: np.ndarray) -> Dict[str, float]:
        """Compute coupling between different dimensions."""
        coupling = {}
        
        # Compute coupling between adjacent dimensions
        for dim1 in range(correlation_7d.ndim - 1):
            for dim2 in range(dim1 + 1, correlation_7d.ndim):
                # Compute cross-correlation between dimensions
                corr_1 = np.mean(correlation_7d, axis=tuple(i for i in range(correlation_7d.ndim) if i != dim1))
                corr_2 = np.mean(correlation_7d, axis=tuple(i for i in range(correlation_7d.ndim) if i != dim2))
                
                # Compute coupling strength
                coupling_strength = np.corrcoef(corr_1, corr_2)[0, 1]
                coupling[f"dim_{dim1}_dim_{dim2}"] = float(coupling_strength) if not np.isnan(coupling_strength) else 0.0
        
        return coupling
    
    def _compute_correlation_decay(self, correlation_7d: np.ndarray) -> Dict[str, float]:
        """Compute correlation decay characteristics."""
        # Compute radial correlation
        center = tuple(s // 2 for s in correlation_7d.shape)
        radial_correlation = self._compute_radial_correlation(correlation_7d, center)
        
        # Fit exponential decay
        if len(radial_correlation) > 1:
            # Find decay length
            max_corr = np.max(radial_correlation)
            target_corr = max_corr / np.e
            
            decay_length = 0
            for i, corr in enumerate(radial_correlation):
                if corr < target_corr:
                    decay_length = i
                    break
        else:
            decay_length = 0

        return {
            "decay_length": float(decay_length),
            "radial_correlation": radial_correlation.tolist()
        }
    
    def _compute_radial_correlation(self, correlation_7d: np.ndarray, center: Tuple[int, ...]) -> np.ndarray:
        """Compute radial correlation from center point."""
        # Create distance array
        distances = np.zeros(correlation_7d.shape)
        
        # Compute distances from center
        for i in range(correlation_7d.shape[0]):
            for j in range(correlation_7d.shape[1]):
                for k in range(correlation_7d.shape[2]):
                    if correlation_7d.ndim == 3:
                        distances[i, j, k] = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    else:
                        # For higher dimensions, compute distance appropriately
                        dist_sq = sum((idx - center[dim])**2 for dim, idx in enumerate([i, j, k]))
                        distances[i, j, k] = np.sqrt(dist_sq)
        
        # Compute radial correlation
        max_distance = int(np.max(distances))
        radial_correlation = np.zeros(max_distance + 1)
        
        for r in range(max_distance + 1):
            mask = (distances >= r - 0.5) & (distances < r + 0.5)
            if np.any(mask):
                radial_correlation[r] = np.mean(correlation_7d[mask])
            else:
                radial_correlation[r] = 0.0
        
        return radial_correlation
    
    def _compute_dimensional_correlations(self, amplitude: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute correlations for individual dimensions."""
        dimensional_correlations = {}
        
        for dim in range(amplitude.ndim):
            # Compute correlation along this dimension
            correlation_dim = self._compute_dimension_correlation(amplitude, dim)
            
            # Store as 1D correlation
            correlation_1d = np.mean(correlation_dim, axis=tuple(i for i in range(amplitude.ndim) if i != dim))
            dimensional_correlations[f"dim_{dim}"] = correlation_1d
        
        return dimensional_correlations
