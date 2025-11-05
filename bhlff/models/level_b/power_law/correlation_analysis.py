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

from bhlff.core.bvp import BVPCore


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
        """
        Compute full 7D correlation function using spectral methods.

        Physical Meaning:
            Computes 7D spatial correlation function C(r) = ⟨a(x)a*(x+r)⟩
            using FFT-based spectral methods. Preserves full 7D structure
            for accurate correlation length estimation.

        Mathematical Foundation:
            Uses FFT-based autocorrelation: C(r) = FFT⁻¹[|FFT[a]|²]
            This is the most efficient method for 7D correlation computation
            and preserves all dimensional structure.

        Args:
            amplitude (np.ndarray): Field amplitude (7D structure).

        Returns:
            np.ndarray: 7D correlation function with same shape.
        """
        # Use FFT-based autocorrelation for 7D structure
        # C(r) = FFT⁻¹[|FFT[a]|²]
        try:
            from bhlff.utils.cuda_utils import get_global_backend, CUDABackend

            backend = get_global_backend()
            if isinstance(backend, CUDABackend):
                import cupy as cp

                # Transfer to GPU
                amp_gpu = backend.array(amplitude)
                # Compute FFT on GPU
                amp_fft = backend.fft(amp_gpu)
                # Compute power spectrum
                power_spectrum = cp.abs(amp_fft) ** 2
                # Compute inverse FFT
                correlation_7d_gpu = backend.ifft(power_spectrum)
                correlation_7d = backend.to_numpy(correlation_7d_gpu).real
                cp.cuda.Stream.null.synchronize()
            else:
                # CPU fallback with NumPy FFT
                amp_fft = np.fft.fftn(amplitude)
                power_spectrum = np.abs(amp_fft) ** 2
                correlation_7d = np.fft.ifftn(power_spectrum).real
        except Exception as e:
            self.logger.debug(f"FFT-based correlation failed: {e}, using fallback")
            # Fallback: compute correlation for each dimension
            correlation_7d = np.zeros_like(amplitude)
            # For 7D, explicitly handle all 7 dimensions
            expected_dims = 7
            if amplitude.ndim != expected_dims:
                self.logger.warning(
                    f"Expected 7D field, got {amplitude.ndim}D. "
                    f"Computing correlation for {amplitude.ndim}D structure."
                )

            # Compute correlation for each dimension
            for dim in range(amplitude.ndim):
                correlation_dim = self._compute_dimension_correlation(amplitude, dim)
                correlation_7d += correlation_dim

            # Normalize by number of dimensions
            correlation_7d /= amplitude.ndim

        return correlation_7d

    def _compute_dimension_correlation(
        self, amplitude: np.ndarray, dim: int
    ) -> np.ndarray:
        """
        Compute correlation along a specific dimension using vectorized operations.

        Physical Meaning:
            Computes 1D correlation function along dimension dim,
            preserving the full N-D structure of the field.

        Mathematical Foundation:
            C_dim(r) = (1/N_shift) Σ_shift ⟨a(x) a*(x+shift·e_dim)⟩
            where e_dim is unit vector along dimension dim.

        Args:
            amplitude (np.ndarray): Field amplitude.
            dim (int): Dimension index along which to compute correlation.

        Returns:
            np.ndarray: Correlation function along dimension dim.
        """
        # Use vectorized FFT-based approach for efficiency
        # For single dimension, compute FFT along that dimension
        try:
            # Compute FFT along the specified dimension
            amp_fft = np.fft.fftn(amplitude, axes=(dim,))
            power_spectrum = np.abs(amp_fft) ** 2
            correlation_dim = np.fft.ifftn(power_spectrum, axes=(dim,)).real
        except Exception:
            # Fallback: compute correlation using rolling shifts
            correlation_dim = np.zeros_like(amplitude)

            # Adaptive number of shifts based on dimension size
            max_shifts = min(amplitude.shape[dim], 20)
            if max_shifts < 2:
                # If dimension too small, return zero correlation
                return correlation_dim

            # Compute correlation for different shifts
            for shift in range(1, max_shifts):
                # Create shifted array
                shifted = np.roll(amplitude, shift, axis=dim)

                # Compute correlation
                correlation_shift = amplitude * np.conj(shifted)
                correlation_dim += correlation_shift

            # Normalize by number of shifts
            correlation_dim /= max_shifts - 1

        return correlation_dim

    def _compute_7d_correlation_lengths(
        self, correlation_7d: np.ndarray
    ) -> Dict[str, float]:
        """Compute correlation lengths in each dimension."""
        correlation_lengths = {}

        for dim in range(correlation_7d.ndim):
            # Compute correlation length along this dimension
            correlation_1d = np.mean(
                correlation_7d,
                axis=tuple(i for i in range(correlation_7d.ndim) if i != dim),
            )

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

    def _analyze_7d_correlation_structure(
        self, correlation_7d: np.ndarray
    ) -> Dict[str, Any]:
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
            "anisotropy_measure": (
                float(max_correlation / mean_correlation)
                if mean_correlation > 0
                else 0.0
            ),
        }

    def _compute_dimensional_coupling(
        self, correlation_7d: np.ndarray
    ) -> Dict[str, float]:
        """Compute coupling between different dimensions."""
        coupling = {}

        # Compute coupling between adjacent dimensions
        for dim1 in range(correlation_7d.ndim - 1):
            for dim2 in range(dim1 + 1, correlation_7d.ndim):
                # Compute cross-correlation between dimensions
                corr_1 = np.mean(
                    correlation_7d,
                    axis=tuple(i for i in range(correlation_7d.ndim) if i != dim1),
                )
                corr_2 = np.mean(
                    correlation_7d,
                    axis=tuple(i for i in range(correlation_7d.ndim) if i != dim2),
                )

                # Compute coupling strength
                coupling_strength = np.corrcoef(corr_1, corr_2)[0, 1]
                coupling[f"dim_{dim1}_dim_{dim2}"] = (
                    float(coupling_strength) if not np.isnan(coupling_strength) else 0.0
                )

        return coupling

    def _compute_correlation_decay(
        self, correlation_7d: np.ndarray
    ) -> Dict[str, float]:
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
            "radial_correlation": radial_correlation.tolist(),
        }

    def _compute_radial_correlation(
        self, correlation_7d: np.ndarray, center: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Compute radial correlation from center point using vectorized 7D operations.

        Physical Meaning:
            Computes radial correlation function C(r) by averaging correlation
            over all points at distance r from center. Works for any dimension
            including full 7D structure.

        Mathematical Foundation:
            For 7D space, distance from center is:
            r = √(Σᵢ (xᵢ - cᵢ)²) for i = 1..7
            Radial correlation: C_rad(r) = ⟨C(x)⟩_{|x-c|=r}

        Args:
            correlation_7d (np.ndarray): Correlation function (7D structure).
            center (Tuple[int, ...]): Center point coordinates.

        Returns:
            np.ndarray: Radial correlation function C_rad(r).
        """
        # Use vectorized distance computation for all dimensions
        # Create coordinate arrays for all dimensions
        coords = np.meshgrid(
            *[np.arange(s) for s in correlation_7d.shape], indexing="ij"
        )

        # Compute distance from center for all dimensions
        # Use vectorized operations: r = √(Σᵢ (xᵢ - cᵢ)²)
        dist_sq = np.zeros(correlation_7d.shape)
        for dim in range(correlation_7d.ndim):
            dist_sq += (coords[dim] - center[dim]) ** 2

        distances = np.sqrt(dist_sq)

        # Compute radial correlation by binning distances
        max_distance = int(np.ceil(np.max(distances)))
        radial_correlation = np.zeros(max_distance + 1)

        # Use vectorized binning for efficiency
        for r in range(max_distance + 1):
            # Create mask for distances within bin [r-0.5, r+0.5)
            mask = (distances >= r - 0.5) & (distances < r + 0.5)
            if np.any(mask):
                radial_correlation[r] = np.mean(correlation_7d[mask])
            else:
                radial_correlation[r] = 0.0

        return radial_correlation

    def _compute_dimensional_correlations(
        self, amplitude: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute correlations for individual dimensions."""
        dimensional_correlations = {}

        for dim in range(amplitude.ndim):
            # Compute correlation along this dimension
            correlation_dim = self._compute_dimension_correlation(amplitude, dim)

            # Store as 1D correlation
            correlation_1d = np.mean(
                correlation_dim,
                axis=tuple(i for i in range(amplitude.ndim) if i != dim),
            )
            dimensional_correlations[f"dim_{dim}"] = correlation_1d

        return dimensional_correlations
