"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power law analysis for Level B fundamental properties.

This module implements power law tail analysis for the 7D phase field theory,
validating the theoretical prediction A(r) ∝ r^(2β-3) in homogeneous medium.

Theoretical Background:
    In homogeneous medium, the Riesz operator L_β = μ(-Δ)^β + λ produces
    power law tails with exponent 2β-3, representing the fundamental
    behavior of fractional Laplacian in 7D space-time.

Example:
    >>> analyzer = LevelBPowerLawAnalyzer()
    >>> result = analyzer.analyze_power_law_tail(field, beta, center)
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


class LevelBPowerLawAnalyzer:
    """
    Power law analysis for Level B fundamental properties.
    
    Physical Meaning:
        Analyzes the power law behavior of the phase field in homogeneous
        medium, validating the theoretical prediction A(r) ∝ r^(2β-3)
        for the Riesz operator.
        
    Mathematical Foundation:
        In spectral space, the Riesz operator has symbol D(k) = μk^(2β) + λ.
        For λ=0, this leads to power law tails with exponent 2β-3 in
        real space, representing the fundamental decay behavior.
    """
    
    def __init__(self):
        """Initialize power law analyzer."""
        pass
    
    def analyze_power_law_tail(self, field: np.ndarray, beta: float, 
                              center: List[float], min_decades: float = 1.5) -> Dict[str, Any]:
        """
        Analyze power law tail A(r) ∝ r^(2β-3).
        
        Physical Meaning:
            Validates that the phase field exhibits power law decay
            A(r) ∝ r^(2β-3) in homogeneous medium, confirming the
            fundamental behavior of the Riesz operator.
            
        Mathematical Foundation:
            In spectral space, the Riesz operator has symbol D(k) = μk^(2β).
            For λ=0, this leads to power law tails with exponent 2β-3
            in real space.
            
        Args:
            field (np.ndarray): Phase field solution
            beta (float): Fractional order β ∈ (0,2)
            center (List[float]): Center of the defect [x, y, z]
            min_decades (float): Minimum decades for analysis
            
        Returns:
            Dict[str, Any]: Analysis results including slope, error, and quality metrics
        """
        # 1. Compute radial profile
        radial_profile = self._compute_radial_profile(field, center)
        
        # 2. Filter tail region (exclude core)
        r_core = self._estimate_core_radius(radial_profile)
        tail_mask = radial_profile['r'] > 2 * r_core
        r_tail = radial_profile['r'][tail_mask]
        A_tail = radial_profile['A'][tail_mask]
        
        # 3. Check sufficient range
        log_range = np.log10(r_tail.max() / r_tail.min()) if len(r_tail) > 1 else 0
        if log_range < min_decades:
            raise ValueError(f"Insufficient range: {log_range:.2f} < {min_decades}")
        
        # 4. Linear regression in log-log coordinates
        log_r = np.log(r_tail)
        log_A = np.log(np.abs(A_tail))
        
        # Exclude zero values
        valid_mask = np.isfinite(log_A) & (A_tail != 0)
        log_r_valid = log_r[valid_mask]
        log_A_valid = log_A[valid_mask]
        
        if len(log_r_valid) < 2:
            raise ValueError("Insufficient valid data points for regression")
        
        # Regression
        slope, intercept, r_squared, _, _ = stats.linregress(log_r_valid, log_A_valid)
        
        # 5. Compare with theoretical value
        theoretical_slope = 2 * beta - 3
        relative_error = abs(slope - theoretical_slope) / abs(theoretical_slope)
        
        # 6. Acceptance criteria
        passed = (
            r_squared >= 0.99 and           # High correlation
            relative_error <= 0.05 and      # Error ≤5%
            log_range >= min_decades        # Sufficient range
        )
        
        return {
            'slope': slope,
            'theoretical_slope': theoretical_slope,
            'relative_error': relative_error,
            'r_squared': r_squared,
            'log_range': log_range,
            'passed': passed,
            'radial_profile': radial_profile,
            'tail_data': {
                'r': r_tail,
                'A': A_tail,
                'log_r': log_r_valid,
                'log_A': log_A_valid
            }
        }
    
    def _compute_radial_profile(self, field: np.ndarray, center: List[float]) -> Dict[str, np.ndarray]:
        """
        Compute radial profile of the field.
        
        Physical Meaning:
            Computes the radial profile A(r) by averaging the field
            over spherical shells centered at the defect.
            
        Args:
            field (np.ndarray): 3D field array
            center (List[float]): Center coordinates [x, y, z]
            
        Returns:
            Dict[str, np.ndarray]: Radial profile with 'r' and 'A' arrays
        """
        # Get field shape (assuming 3D spatial dimensions)
        shape = field.shape[:3]  # Take first 3 dimensions
        
        # Create coordinate grids
        x = np.arange(shape[0])
        y = np.arange(shape[1])
        z = np.arange(shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Compute distances from center
        distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        
        # Get field amplitude
        amplitude = np.abs(field)
        
        # Create radial bins
        r_max = np.max(distances)
        r_bins = np.linspace(0, r_max, min(100, int(r_max)))
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        # Bin the data
        A_radial = []
        for i in range(len(r_bins) - 1):
            mask = (distances >= r_bins[i]) & (distances < r_bins[i + 1])
            if np.any(mask):
                A_radial.append(np.mean(amplitude[mask]))
            else:
                A_radial.append(0.0)
        
        return {
            'r': r_centers,
            'A': np.array(A_radial)
        }
    
    def _estimate_core_radius(self, radial_profile: Dict[str, np.ndarray]) -> float:
        """
        Estimate core radius from radial profile.
        
        Physical Meaning:
            Estimates the radius of the core region where the field
            amplitude is highest and most coherent.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data
            
        Returns:
            float: Estimated core radius
        """
        A = radial_profile['A']
        r = radial_profile['r']
        
        # Find maximum amplitude
        max_idx = np.argmax(A)
        max_amplitude = A[max_idx]
        
        # Find radius where amplitude drops to 10% of maximum
        threshold = 0.1 * max_amplitude
        below_threshold = A < threshold
        
        if np.any(below_threshold):
            # Find first radius below threshold
            core_idx = np.where(below_threshold)[0]
            if len(core_idx) > 0:
                return r[core_idx[0]]
        
        # If no threshold found, use 10% of maximum radius
        return 0.1 * r.max()
    
    def visualize_power_law_analysis(self, analysis_result: Dict[str, Any], 
                                   output_path: str = "power_law_analysis.png") -> None:
        """
        Visualize power law analysis results.
        
        Physical Meaning:
            Creates visualization of the power law analysis showing
            the radial profile, log-log fit, and quality metrics.
            
        Args:
            analysis_result (Dict[str, Any]): Results from analyze_power_law_tail
            output_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Radial profile
        radial_profile = analysis_result['radial_profile']
        ax1.plot(radial_profile['r'], radial_profile['A'], 'b-', linewidth=2, label='Radial Profile')
        ax1.set_xlabel('Radius r')
        ax1.set_ylabel('Amplitude A(r)')
        ax1.set_title('Radial Profile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Log-log fit
        tail_data = analysis_result['tail_data']
        ax2.scatter(tail_data['log_r'], tail_data['log_A'], alpha=0.6, s=20, label='Data')
        
        # Plot fitted line
        if len(tail_data['log_r']) > 1:
            slope = analysis_result['slope']
            intercept = analysis_result.get('intercept', 0)
            log_r_fit = np.linspace(tail_data['log_r'].min(), tail_data['log_r'].max(), 100)
            log_A_fit = slope * log_r_fit + intercept
            ax2.plot(log_r_fit, log_A_fit, 'r-', linewidth=2, 
                    label=f'Fit: slope={slope:.3f}')
        
        ax2.set_xlabel('log(r)')
        ax2.set_ylabel('log(A)')
        ax2.set_title(f'Power Law Fit (R²={analysis_result["r_squared"]:.3f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add text with results
        textstr = f'Slope: {analysis_result["slope"]:.3f}\n'
        textstr += f'Theoretical: {analysis_result["theoretical_slope"]:.3f}\n'
        textstr += f'Error: {analysis_result["relative_error"]:.1%}\n'
        textstr += f'R²: {analysis_result["r_squared"]:.3f}'
        
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_power_law_variations(self, field: np.ndarray, center: List[float], 
                               beta_range: List[float]) -> Dict[str, Any]:
        """
        Run power law analysis for different beta values.
        
        Physical Meaning:
            Analyzes power law behavior for different fractional orders
            β, validating the theoretical relationship A(r) ∝ r^(2β-3).
            
        Args:
            field (np.ndarray): Phase field solution
            center (List[float]): Center of the defect
            beta_range (List[float]): Range of β values to test
            
        Returns:
            Dict[str, Any]: Results for all β values
        """
        results = {}
        
        for beta in beta_range:
            try:
                result = self.analyze_power_law_tail(field, beta, center)
                results[f'beta_{beta}'] = result
            except Exception as e:
                results[f'beta_{beta}'] = {'error': str(e), 'passed': False}
        
        return results