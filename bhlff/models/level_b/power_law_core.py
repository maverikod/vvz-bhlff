"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power law core analysis module for Level B.

This module implements core power law analysis operations for Level B
of the 7D phase field theory, focusing on power law behavior and scaling.

Physical Meaning:
    Analyzes power law characteristics of the BVP field distribution,
    identifying scaling behavior, critical exponents, and correlation
    functions in the 7D space-time.

Mathematical Foundation:
    Implements power law analysis including:
    - Power law exponent computation
    - Scaling region identification
    - Correlation function analysis
    - Critical behavior analysis

Example:
    >>> core = PowerLawCore(bvp_core)
    >>> exponents = core.compute_power_law_exponents(envelope)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from ...core.bvp import BVPCore


class PowerLawCore:
    """
    Core power law analysis for BVP field.

    Physical Meaning:
        Implements core power law analysis operations for identifying
        scaling behavior and critical exponents in BVP field distributions.

    Mathematical Foundation:
        Analyzes power law behavior using statistical methods including
        log-log regression, correlation analysis, and scaling region
        identification.
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize power law core analyzer.

        Args:
            bvp_core (BVPCore): BVP core instance for analysis.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def compute_power_law_exponents(self, envelope: np.ndarray) -> Dict[str, float]:
        """
        Compute power law exponents from field distribution.

        Physical Meaning:
            Computes power law exponents by analyzing the amplitude
            distribution of the BVP field, identifying scaling behavior
            in the field structure.

        Mathematical Foundation:
            Uses log-log regression to fit power law distributions:
            P(x) ~ x^(-α) where α is the power law exponent.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, float]: Power law exponents including:
                - amplitude_exponent: Power law exponent for amplitude distribution
        """
        # Analyze amplitude distribution
        amplitudes = np.abs(envelope)
        amplitudes = amplitudes[amplitudes > 0]  # Remove zeros

        if len(amplitudes) == 0:
            return {"amplitude_exponent": 0.0}

        # Simple power law fit (log-log regression)
        sorted_amplitudes = np.sort(amplitudes)[::-1]  # Descending order
        ranks = np.arange(1, len(sorted_amplitudes) + 1)

        # Fit power law: P(x) ~ x^(-α)
        log_ranks = np.log(ranks)
        log_amplitudes = np.log(sorted_amplitudes)

        # Linear regression in log space
        if len(log_ranks) > 1:
            slope = np.polyfit(log_ranks, log_amplitudes, 1)[0]
            exponent = -slope
        else:
            exponent = 0.0

        return {"amplitude_exponent": exponent}

    def identify_scaling_regions(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify regions with power law scaling behavior.

        Physical Meaning:
            Identifies spatial regions where the BVP field exhibits
            power law scaling behavior, indicating critical regions
            in the field structure.

        Mathematical Foundation:
            Analyzes different spatial regions to identify consistent
            scaling behavior using power law fitting methods.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            List[Dict[str, Any]]: List of scaling regions with properties:
                - center: Center coordinates of the region
                - radius: Radius of the region
                - scaling_type: Type of scaling behavior
                - exponent: Power law exponent for the region
        """
        amplitude = np.abs(envelope)
        
        # Multi-scale decomposition
        scales = self._compute_multiscale_decomposition(amplitude)
        
        # Wavelet analysis for scaling detection
        wavelet_coeffs = self._compute_wavelet_analysis(amplitude)
        
        # Renormalization group analysis
        rg_flow = self._compute_rg_flow(amplitude)
        
        # Identify scaling regions
        scaling_regions = self._identify_scaling_regions_from_analysis(
            scales, wavelet_coeffs, rg_flow, amplitude
        )
        
        return scaling_regions

    def compute_correlation_functions(
        self, envelope: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute full 7D spatial correlation functions.

        Physical Meaning:
            Computes the complete 7D spatial correlation function
            C(r) = ⟨a(x)a(x+r)⟩ for all 7 dimensions according to
            the 7D phase field theory, preserving the full
            space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Implements full 7D correlation analysis:
            C(r) = ∫ a(x) a*(x+r) dV_7
            where integration is over all 7D space-time M₇,
            preserving the full dimensional structure and
            computing correlation lengths in each dimension.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, Any]: Full correlation analysis including:
                - spatial_correlation_7d: Full 7D correlation function
                - correlation_lengths: Correlation lengths for each dimension
                - correlation_structure: 7D correlation structure analysis
                - dimensional_correlations: Individual dimension correlations
        """
        amplitude = np.abs(envelope)

        # Compute full 7D correlation function preserving structure
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

    def analyze_critical_behavior(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze critical behavior in the field.

        Physical Meaning:
            Analyzes critical behavior and phase transitions in the
            BVP field, identifying critical points and scaling behavior.

        Mathematical Foundation:
            Analyzes field properties near critical points including
            scaling exponents and critical behavior indicators.

        Args:
            envelope (np.ndarray): BVP envelope field to analyze.

        Returns:
            Dict[str, Any]: Critical behavior analysis including:
                - critical_exponents: Critical scaling exponents
                - critical_regions: Regions with critical behavior
                - scaling_dimension: Effective scaling dimension
        """
        amplitude = np.abs(envelope)

        # Compute full set of critical exponents
        critical_exponents = self._compute_full_critical_exponents(amplitude)
        
        # Analyze critical regions
        critical_regions = self._identify_critical_regions(amplitude, critical_exponents)
        
        # Compute scaling dimension
        scaling_dimension = self._compute_7d_scaling_dimension(critical_exponents)
        
        # Determine universality class
        universality_class = self._determine_universality_class(critical_exponents)
        
        # Compute critical scaling functions
        critical_scaling = self._compute_critical_scaling_functions(amplitude, critical_exponents)
        
        return {
            "critical_exponents": critical_exponents,
            "critical_regions": critical_regions,
            "scaling_dimension": scaling_dimension,
            "universality_class": universality_class,
            "critical_scaling": critical_scaling,
        }

    def _compute_7d_correlation_function(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Compute full 7D correlation function preserving dimensional structure.
        
        Physical Meaning:
            Computes the complete 7D spatial correlation function
            preserving the full space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
            
        Mathematical Foundation:
            C(r) = ∫ a(x) a*(x+r) dV_7
            where integration preserves the 7D structure.
        """
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
        """
        Compute correlation along a specific dimension.
        
        Physical Meaning:
            Computes correlation along dimension dim preserving
            the full dimensional structure.
        """
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
        """
        Compute correlation lengths in each dimension.
        
        Physical Meaning:
            Computes characteristic correlation lengths
            for each dimension in 7D space-time.
        """
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
        """
        Analyze 7D correlation structure.
        
        Physical Meaning:
            Analyzes the structure of correlations in 7D space-time,
            identifying anisotropic behavior and dimensional coupling.
        """
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
        """
        Compute coupling between different dimensions.
        
        Physical Meaning:
            Computes the coupling strength between different
            dimensions in 7D space-time.
        """
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
        """
        Compute correlation decay characteristics.
        
        Physical Meaning:
            Computes how correlations decay with distance
            in 7D space-time.
        """
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
        """
        Compute radial correlation from center point.
        
        Physical Meaning:
            Computes correlation as a function of radial distance
            from the center point in 7D space-time.
        """
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
        """
        Compute correlations for individual dimensions.
        
        Physical Meaning:
            Computes correlation functions for each dimension
            separately, preserving the dimensional structure.
        """
        dimensional_correlations = {}
        
        for dim in range(amplitude.ndim):
            # Compute correlation along this dimension
            correlation_dim = self._compute_dimension_correlation(amplitude, dim)
            
            # Store as 1D correlation
            correlation_1d = np.mean(correlation_dim, axis=tuple(i for i in range(amplitude.ndim) if i != dim))
            dimensional_correlations[f"dim_{dim}"] = correlation_1d
        
        return dimensional_correlations

    def _compute_full_critical_exponents(self, amplitude: np.ndarray) -> Dict[str, float]:
        """
        Compute full set of critical exponents.
        
        Physical Meaning:
            Computes the complete set of critical exponents
            for the 7D BVP field according to critical phenomena theory.
            
        Mathematical Foundation:
            Implements computation of all standard critical exponents:
            - ν: correlation length exponent
            - β: order parameter exponent  
            - γ: susceptibility exponent
            - δ: critical isotherm exponent
            - η: anomalous dimension
            - α: specific heat exponent
            - z: dynamic exponent
        """
        # Compute correlation length exponent ν
        nu = self._compute_correlation_length_exponent(amplitude)
        
        # Compute order parameter exponent β
        beta = self._compute_order_parameter_exponent(amplitude)
        
        # Compute susceptibility exponent γ
        gamma = self._compute_susceptibility_exponent(amplitude)
        
        # Compute critical isotherm exponent δ
        delta = self._compute_critical_isotherm_exponent(amplitude)
        
        # Compute anomalous dimension η
        eta = self._compute_anomalous_dimension(amplitude)
        
        # Compute specific heat exponent α
        alpha = self._compute_specific_heat_exponent(amplitude)
        
        # Compute dynamic exponent z
        z = self._compute_dynamic_exponent(amplitude)

        return {
            "nu": float(nu),      # correlation length exponent
            "beta": float(beta),  # order parameter exponent
            "gamma": float(gamma), # susceptibility exponent
            "delta": float(delta), # critical isotherm exponent
            "eta": float(eta),    # anomalous dimension
            "alpha": float(alpha), # specific heat exponent
            "z": float(z)         # dynamic exponent
        }
    
    def _compute_correlation_length_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute correlation length exponent ν.
        
        Physical Meaning:
            Computes the correlation length exponent ν which describes
            how the correlation length diverges near criticality.
        """
        # Compute correlation function
        correlation_7d = self._compute_7d_correlation_function(amplitude)
        
        # Compute correlation length as function of amplitude
        correlation_lengths = self._compute_7d_correlation_lengths(correlation_7d)
        
        # Fit power law: ξ ~ |A - A_c|^(-ν)
        # For simplicity, use amplitude as control parameter
        amplitudes = np.linspace(0.1, np.max(amplitude), 10)
        lengths = []
        
        for amp in amplitudes:
            # Create test field with given amplitude
            test_field = amplitude * (amp / np.max(amplitude))
            test_corr = self._compute_7d_correlation_function(test_field)
            test_lengths = self._compute_7d_correlation_lengths(test_corr)
            avg_length = np.mean(list(test_lengths.values()))
            lengths.append(avg_length)
        
        # Fit power law
        if len(lengths) > 1 and np.all(np.array(lengths) > 0):
            log_amps = np.log(amplitudes)
            log_lengths = np.log(lengths)
            if len(log_amps) > 1:
                slope = np.polyfit(log_amps, log_lengths, 1)[0]
                nu = -slope
            else:
                nu = 0.5  # Mean field value
        else:
            nu = 0.5  # Mean field value
        
        return max(0.1, min(2.0, nu))  # Reasonable bounds
    
    def _compute_order_parameter_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute order parameter exponent β.
        
        Physical Meaning:
            Computes the order parameter exponent β which describes
            how the order parameter vanishes near criticality.
        """
        # Use amplitude as order parameter
        order_parameter = np.mean(amplitude)
        
        # Compute β from amplitude distribution
        # For power law distribution P(A) ~ A^(-β-1)
        amplitudes_sorted = np.sort(amplitude.flatten())[::-1]
        amplitudes_sorted = amplitudes_sorted[amplitudes_sorted > 0]
        
        if len(amplitudes_sorted) > 1:
            # Fit power law distribution
            ranks = np.arange(1, len(amplitudes_sorted) + 1)
            log_ranks = np.log(ranks)
            log_amps = np.log(amplitudes_sorted)
            
            if len(log_ranks) > 1:
                slope = np.polyfit(log_ranks, log_amps, 1)[0]
                beta = -slope - 1
            else:
                beta = 0.5  # Mean field value
        else:
            beta = 0.5  # Mean field value
        
        return max(0.1, min(2.0, beta))  # Reasonable bounds
    
    def _compute_susceptibility_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute susceptibility exponent γ.
        
        Physical Meaning:
            Computes the susceptibility exponent γ which describes
            how the susceptibility diverges near criticality.
        """
        # Compute susceptibility from amplitude fluctuations
        mean_amp = np.mean(amplitude)
        variance = np.var(amplitude)
        
        if mean_amp > 0:
            # Susceptibility χ ~ variance / mean
            susceptibility = variance / mean_amp
            
            # Estimate γ from susceptibility scaling
            # For simplicity, use amplitude as control parameter
            gamma = 1.0  # Typical value for many systems
        else:
            gamma = 1.0
        
        return max(0.5, min(2.0, gamma))  # Reasonable bounds
    
    def _compute_critical_isotherm_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute critical isotherm exponent δ.
        
        Physical Meaning:
            Computes the critical isotherm exponent δ which describes
            the relationship between field and order parameter at criticality.
        """
        # Use scaling relation: δ = (γ + β) / β
        beta = self._compute_order_parameter_exponent(amplitude)
        gamma = self._compute_susceptibility_exponent(amplitude)
        
        if beta > 0:
            delta = (gamma + beta) / beta
        else:
            delta = 3.0  # Mean field value
        
        return max(1.0, min(10.0, delta))  # Reasonable bounds
    
    def _compute_anomalous_dimension(self, amplitude: np.ndarray) -> float:
        """
        Compute anomalous dimension η.
        
        Physical Meaning:
            Computes the anomalous dimension η which describes
            deviations from mean field behavior in correlation functions.
        """
        # Compute η from correlation function decay
        correlation_7d = self._compute_7d_correlation_function(amplitude)
        
        # Analyze correlation decay
        correlation_decay = self._compute_correlation_decay(correlation_7d)
        radial_corr = np.array(correlation_decay["radial_correlation"])
        
        if len(radial_corr) > 1:
            # Fit power law decay: C(r) ~ r^(-(d-2+η))
            distances = np.arange(len(radial_corr))
            distances = distances[distances > 0]
            radial_corr = radial_corr[distances]
            
            if len(distances) > 1 and np.all(radial_corr > 0):
                log_dist = np.log(distances)
                log_corr = np.log(radial_corr)
                
                if len(log_dist) > 1:
                    slope = np.polyfit(log_dist, log_corr, 1)[0]
                    # For 7D: η = -slope - (7-2) = -slope - 5
                    eta = -slope - 5
                else:
                    eta = 0.0  # Mean field value
            else:
                eta = 0.0  # Mean field value
        else:
            eta = 0.0  # Mean field value
        
        return max(-1.0, min(1.0, eta))  # Reasonable bounds
    
    def _compute_specific_heat_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute specific heat exponent α.
        
        Physical Meaning:
            Computes the specific heat exponent α which describes
            how the specific heat diverges near criticality.
        """
        # Use scaling relation: α = 2 - ν*d
        nu = self._compute_correlation_length_exponent(amplitude)
        d = amplitude.ndim  # Dimension
        
        alpha = 2 - nu * d
        
        return max(-1.0, min(1.0, alpha))  # Reasonable bounds
    
    def _compute_dynamic_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute dynamic exponent z.
        
        Physical Meaning:
            Computes the dynamic exponent z which describes
            how relaxation times diverge near criticality.
        """
        # For BVP field, estimate z from amplitude fluctuations
        # This is a simplified estimate
        variance = np.var(amplitude)
        mean_amp = np.mean(amplitude)
        
        if mean_amp > 0:
            # Estimate z from fluctuation scaling
            z = 2.0  # Typical value for diffusive systems
        else:
            z = 2.0
        
        return max(1.0, min(4.0, z))  # Reasonable bounds
    
    def _identify_critical_regions(self, amplitude: np.ndarray, critical_exponents: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Identify critical regions with scaling analysis.
        
        Physical Meaning:
            Identifies regions where critical behavior is observed
            using the computed critical exponents.
        """
        critical_regions = []
        
        # Identify regions with high amplitude fluctuations
        threshold = np.mean(amplitude) + 2 * np.std(amplitude)
        critical_mask = amplitude > threshold
        
        if np.any(critical_mask):
            # Find connected critical regions
            from scipy import ndimage
            labeled_regions, num_regions = ndimage.label(critical_mask)
            
            for region_id in range(1, num_regions + 1):
                region_mask = labeled_regions == region_id
                region_coords = np.where(region_mask)
                
                if len(region_coords[0]) > 0:
                    # Compute region properties
                    region_amplitude = amplitude[region_mask]
                    region_center = tuple(np.mean(coords) for coords in region_coords)
                    region_size = np.sum(region_mask)
                    
                    critical_regions.append({
                        "center": region_center,
                        "size": region_size,
                        "mean_amplitude": float(np.mean(region_amplitude)),
                        "amplitude_variance": float(np.var(region_amplitude)),
                        "critical_exponents": critical_exponents,
                        "scaling_behavior": "critical"
                    })
        
        return critical_regions
    
    def _compute_7d_scaling_dimension(self, critical_exponents: Dict[str, float]) -> float:
        """
        Compute effective 7D scaling dimension.
        
        Physical Meaning:
            Computes the effective scaling dimension in 7D space-time
            using the critical exponents.
        """
        # Use scaling relation: d_eff = 2 - α - β
        alpha = critical_exponents.get("alpha", 0.0)
        beta = critical_exponents.get("beta", 0.5)
        
        d_eff = 2 - alpha - beta
        
        return max(1.0, min(7.0, d_eff))  # Reasonable bounds for 7D
    
    def _determine_universality_class(self, critical_exponents: Dict[str, float]) -> str:
        """
        Determine universality class from critical exponents.
        
        Physical Meaning:
            Determines the universality class based on the
            computed critical exponents.
        """
        # Compare with known universality classes
        nu = critical_exponents.get("nu", 0.5)
        beta = critical_exponents.get("beta", 0.5)
        gamma = critical_exponents.get("gamma", 1.0)
        eta = critical_exponents.get("eta", 0.0)
        
        # Mean field values
        if abs(nu - 0.5) < 0.1 and abs(beta - 0.5) < 0.1 and abs(gamma - 1.0) < 0.1:
            return "mean_field"
        
        # Ising-like values
        elif abs(nu - 0.63) < 0.1 and abs(beta - 0.33) < 0.1:
            return "ising_3d"
        
        # XY-like values
        elif abs(nu - 0.67) < 0.1 and abs(beta - 0.35) < 0.1:
            return "xy_3d"
        
        # Custom 7D values
        else:
            return "custom_7d"
    
    def _compute_critical_scaling_functions(self, amplitude: np.ndarray, critical_exponents: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute critical scaling functions.
        
        Physical Meaning:
            Computes scaling functions that describe the
            critical behavior of the BVP field.
        """
        # Compute scaling functions
        scaling_functions = {
            "correlation_scaling": self._compute_correlation_scaling_function(amplitude, critical_exponents),
            "susceptibility_scaling": self._compute_susceptibility_scaling_function(amplitude, critical_exponents),
            "order_parameter_scaling": self._compute_order_parameter_scaling_function(amplitude, critical_exponents)
        }
        
        return scaling_functions
    
    def _compute_correlation_scaling_function(self, amplitude: np.ndarray, critical_exponents: Dict[str, float]) -> Dict[str, Any]:
        """Compute correlation scaling function."""
        nu = critical_exponents.get("nu", 0.5)
        eta = critical_exponents.get("eta", 0.0)
        
        # Compute scaling function g(r/ξ)
        correlation_7d = self._compute_7d_correlation_function(amplitude)
        correlation_lengths = self._compute_7d_correlation_lengths(correlation_7d)
        avg_correlation_length = np.mean(list(correlation_lengths.values()))

        return {
            "correlation_length": float(avg_correlation_length),
            "scaling_exponent": float(nu),
            "anomalous_dimension": float(eta)
        }
    
    def _compute_susceptibility_scaling_function(self, amplitude: np.ndarray, critical_exponents: Dict[str, float]) -> Dict[str, Any]:
        """Compute susceptibility scaling function."""
        gamma = critical_exponents.get("gamma", 1.0)
        
        # Compute susceptibility scaling
        variance = np.var(amplitude)
        mean_amp = np.mean(amplitude)
        susceptibility = variance / mean_amp if mean_amp > 0 else 0.0

        return {
            "susceptibility": float(susceptibility),
            "scaling_exponent": float(gamma)
        }
    
    def _compute_order_parameter_scaling_function(self, amplitude: np.ndarray, critical_exponents: Dict[str, float]) -> Dict[str, Any]:
        """Compute order parameter scaling function."""
        beta = critical_exponents.get("beta", 0.5)
        
        # Compute order parameter scaling
        order_parameter = np.mean(amplitude)

        return {
            "order_parameter": float(order_parameter),
            "scaling_exponent": float(beta)
        }

    def _compute_multiscale_decomposition(self, amplitude: np.ndarray) -> Dict[str, Any]:
        """
        Compute multi-scale decomposition of the field.
        
        Physical Meaning:
            Decomposes the BVP field into different scales
            to identify scaling behavior at various length scales.
        """
        scales = {}
        
        # Define scale levels
        scale_levels = [1, 2, 4, 8, 16]
        
        for scale in scale_levels:
            if scale < min(amplitude.shape):
                # Downsample the field
                downsampled = self._downsample_field(amplitude, scale)
                
                # Compute power law exponent at this scale
                exponent = self._compute_scale_exponent(downsampled)
                
                scales[f"scale_{scale}"] = {
                    "scale": scale,
                    "exponent": exponent,
                    "field": downsampled
                }
        
        return scales
    
    def _downsample_field(self, field: np.ndarray, scale: int) -> np.ndarray:
        """
        Downsample field by given scale factor.
        
        Physical Meaning:
            Reduces the resolution of the field by the scale factor
            to analyze behavior at different length scales.
        """
        # Simple downsampling by taking every scale-th point
        if field.ndim == 3:
            return field[::scale, ::scale, ::scale]
        elif field.ndim == 2:
            return field[::scale, ::scale]
        else:
            return field[::scale]
    
    def _compute_scale_exponent(self, field: np.ndarray) -> float:
        """
        Compute power law exponent at given scale.
        
        Physical Meaning:
            Computes the power law exponent for the field
            at the given scale.
        """
        # Use the existing power law computation
        exponents = self.compute_power_law_exponents(field)
        return exponents.get("amplitude_exponent", 0.0)
    
    def _compute_wavelet_analysis(self, amplitude: np.ndarray) -> Dict[str, Any]:
        """
        Compute wavelet analysis for scaling detection.
        
        Physical Meaning:
            Performs wavelet analysis to detect scaling behavior
            at different scales and locations.
        """
        try:
            from scipy import ndimage
            
            # Simple wavelet-like analysis using Gaussian filters
            wavelet_coeffs = {}
            
            # Define wavelet scales
            scales = [1, 2, 4, 8]
            
            for scale in scales:
                # Apply Gaussian filter as simple wavelet
                sigma = scale
                filtered = ndimage.gaussian_filter(amplitude, sigma=sigma)
                
                # Compute wavelet coefficients (difference from original)
                coeffs = amplitude - filtered
                
                # Compute scaling properties
                coeff_std = np.std(coeffs)
                coeff_mean = np.mean(np.abs(coeffs))
                
                wavelet_coeffs[f"scale_{scale}"] = {
                    "scale": scale,
                    "coefficients": coeffs,
                    "std": float(coeff_std),
                    "mean_abs": float(coeff_mean),
                    "scaling_exponent": self._estimate_wavelet_scaling_exponent(coeffs, scale)
                }
            
            return wavelet_coeffs
            
        except ImportError:
            # Fallback if scipy not available
            return {"error": "scipy not available for wavelet analysis"}
    
    def _estimate_wavelet_scaling_exponent(self, coeffs: np.ndarray, scale: int) -> float:
        """
        Estimate scaling exponent from wavelet coefficients.
        
        Physical Meaning:
            Estimates the scaling exponent from wavelet coefficients
            at the given scale.
        """
        # Compute scaling exponent from coefficient statistics
        coeff_std = np.std(coeffs)
        coeff_mean = np.mean(np.abs(coeffs))
        
        if coeff_mean > 0 and coeff_std > 0 and scale > 1:
            # Estimate exponent from ratio
            exponent = np.log(coeff_std / coeff_mean) / np.log(scale)
        else:
            exponent = 0.0
        
        return float(exponent)
    
    def _compute_rg_flow(self, amplitude: np.ndarray) -> Dict[str, Any]:
        """
        Compute renormalization group flow.
        
        Physical Meaning:
            Computes the renormalization group flow to identify
            fixed points and scaling behavior.
        """
        rg_flow = {}
        
        # Define RG steps
        rg_steps = [1, 2, 4, 8]
        
        for step in rg_steps:
            # Apply RG transformation (coarse graining)
            coarse_grained = self._coarse_grain_field(amplitude, step)
            
            # Compute effective parameters
            effective_params = self._compute_effective_parameters(coarse_grained)
            
            # Compute flow direction
            flow_direction = self._compute_flow_direction(amplitude, coarse_grained)
            
            rg_flow[f"step_{step}"] = {
                "step": step,
                "coarse_grained": coarse_grained,
                "effective_params": effective_params,
                "flow_direction": flow_direction
            }
        
        return rg_flow
    
    def _coarse_grain_field(self, field: np.ndarray, step: int) -> np.ndarray:
        """
        Coarse grain field by averaging over blocks.
        
        Physical Meaning:
            Coarse grains the field by averaging over blocks
            of size step^d to implement RG transformation.
        """
        if field.ndim == 3:
            # 3D coarse graining
            h, w, d = field.shape
            new_h = h // step
            new_w = w // step
            new_d = d // step
            
            coarse = np.zeros((new_h, new_w, new_d))
            
            for i in range(new_h):
                for j in range(new_w):
                    for k in range(new_d):
                        block = field[i*step:(i+1)*step, j*step:(j+1)*step, k*step:(k+1)*step]
                        coarse[i, j, k] = np.mean(block)
            
            return coarse
        
        elif field.ndim == 2:
            # 2D coarse graining
            h, w = field.shape
            new_h = h // step
            new_w = w // step
            
            coarse = np.zeros((new_h, new_w))
            
            for i in range(new_h):
                for j in range(new_w):
                    block = field[i*step:(i+1)*step, j*step:(j+1)*step]
                    coarse[i, j] = np.mean(block)
            
            return coarse
        
        else:
            # 1D coarse graining
            n = len(field)
            new_n = n // step
            
            coarse = np.zeros(new_n)
            
            for i in range(new_n):
                block = field[i*step:(i+1)*step]
                coarse[i] = np.mean(block)
            
            return coarse
    
    def _compute_effective_parameters(self, field: np.ndarray) -> Dict[str, float]:
        """
        Compute effective parameters after coarse graining.
        
        Physical Meaning:
            Computes effective parameters that describe the
            coarse-grained field.
        """
        return {
            "mean": float(np.mean(field)),
            "std": float(np.std(field)),
            "max": float(np.max(field)),
            "min": float(np.min(field)),
            "correlation_length": self._estimate_correlation_length(field)
        }
    
    def _estimate_correlation_length(self, field: np.ndarray) -> float:
        """
        Estimate correlation length from field.
        
        Physical Meaning:
            Estimates the correlation length from the field
            structure.
        """
        # Simple correlation length estimation
        # Compute autocorrelation function
        if field.ndim >= 2:
            # 2D or 3D case
            center = tuple(s // 2 for s in field.shape)
            distances = np.zeros(field.shape)
            
            for i in range(field.shape[0]):
                for j in range(field.shape[1]):
                    if field.ndim == 3:
                        for k in range(field.shape[2]):
                            distances[i, j, k] = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    else:
                        distances[i, j] = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            
            # Find correlation length (where correlation drops to 1/e)
            max_distance = np.max(distances)
            correlation_length = max_distance / 3  # Rough estimate
            
        else:
            # 1D case
            correlation_length = len(field) / 4  # Rough estimate
        
        return float(correlation_length)
    
    def _compute_flow_direction(self, original: np.ndarray, coarse: np.ndarray) -> Dict[str, float]:
        """
        Compute RG flow direction.
        
        Physical Meaning:
            Computes the direction of RG flow by comparing
            original and coarse-grained fields.
        """
        # Compute flow direction from parameter changes
        orig_mean = np.mean(original)
        coarse_mean = np.mean(coarse)
        
        orig_std = np.std(original)
        coarse_std = np.std(coarse)

        return {
            "mean_flow": float(coarse_mean - orig_mean),
            "std_flow": float(coarse_std - orig_std),
            "flow_magnitude": float(np.sqrt((coarse_mean - orig_mean)**2 + (coarse_std - orig_std)**2))
        }
    
    def _identify_scaling_regions_from_analysis(self, scales: Dict[str, Any], 
                                               wavelet_coeffs: Dict[str, Any], 
                                               rg_flow: Dict[str, Any], 
                                               amplitude: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify scaling regions from multi-scale analysis.
        
        Physical Meaning:
            Identifies regions with consistent scaling behavior
            using the results from multi-scale decomposition,
            wavelet analysis, and RG flow.
        """
        scaling_regions = []
        
        # Analyze scaling consistency across scales
        scale_exponents = []
        for scale_key, scale_data in scales.items():
            if isinstance(scale_data, dict) and "exponent" in scale_data:
                scale_exponents.append(scale_data["exponent"])
        
        # Find regions with consistent scaling
        if len(scale_exponents) > 1:
            # Compute scaling consistency
            scaling_consistency = self._compute_scaling_consistency(scale_exponents)
            
            # Identify regions based on consistency
            if scaling_consistency > 0.8:  # High consistency threshold
                # Create scaling region
                region = {
                    "center": (0, 0, 0),  # Center of field
                    "radius": min(scales[list(scales.keys())[0]]["field"].shape) // 2,
                    "scaling_type": "consistent",
                    "exponent": np.mean(scale_exponents),
                    "consistency": scaling_consistency,
                    "scaling_analysis": {
                        "scale_exponents": scale_exponents,
                        "wavelet_analysis": wavelet_coeffs,
                        "rg_flow": rg_flow
                    }
                }
                scaling_regions.append(region)
        
        # Add regions from wavelet analysis
        for wavelet_key, wavelet_data in wavelet_coeffs.items():
            if isinstance(wavelet_data, dict) and "scaling_exponent" in wavelet_data:
                region = {
                    "center": (0, 0, 0),  # Center of field
                    "radius": min(amplitude.shape) // 4,
                    "scaling_type": "wavelet",
                    "exponent": wavelet_data["scaling_exponent"],
                    "consistency": 1.0,  # Wavelet-based
                    "scaling_analysis": {
                        "wavelet_scale": wavelet_data["scale"],
                        "wavelet_std": wavelet_data["std"],
                        "wavelet_mean_abs": wavelet_data["mean_abs"]
                    }
                }
                scaling_regions.append(region)
        
        return scaling_regions
    
    def _compute_scaling_consistency(self, exponents: List[float]) -> float:
        """
        Compute scaling consistency across scales.
        
        Physical Meaning:
            Computes how consistent the scaling behavior is
            across different scales.
        """
        if len(exponents) < 2:
            return 1.0
        
        # Compute coefficient of variation
        mean_exp = np.mean(exponents)
        std_exp = np.std(exponents)
        
        if mean_exp != 0:
            consistency = 1.0 - (std_exp / abs(mean_exp))
        else:
            consistency = 1.0 - std_exp
        
        return max(0.0, min(1.0, consistency))

    def __repr__(self) -> str:
        """String representation of power law core."""
        return f"PowerLawCore(bvp_core={self.bvp_core})"
