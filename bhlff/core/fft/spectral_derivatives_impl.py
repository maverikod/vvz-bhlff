"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Spectral derivatives implementation for 7D BHLFF Framework.

This module provides the concrete implementation for spectral derivative operations
for the 7D phase field theory, including gradient, divergence, curl, and
higher-order derivatives with optimized performance for 7D computations.

Physical Meaning:
    Spectral derivatives implement mathematical differentiation operations
    in frequency space, providing efficient computation of derivatives
    for 7D phase field calculations with U(1)³ phase structure.

Mathematical Foundation:
    Implements spectral derivatives using the property that differentiation
    in real space corresponds to multiplication by ik in frequency space:
    - Gradient: ∇a → ik * â(k)
    - Divergence: ∇·a → ik · â(k)
    - Curl: ∇×a → ik × â(k)
    - Laplacian: Δa → -|k|² * â(k)

Example:
    >>> deriv = SpectralDerivatives(domain, precision="float64")
    >>> gradient = deriv.compute_gradient(field)
    >>> laplacian = deriv.compute_laplacian(field)
"""

import numpy as np
from typing import Any, Tuple, Dict, Optional
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain import Domain

from .spectral_derivatives_base import SpectralDerivativesBase


class SpectralDerivatives(SpectralDerivativesBase):
    """
    Spectral derivatives for 7D phase field calculations.

    Physical Meaning:
        Implements mathematical differentiation operations in 7D frequency space,
        providing efficient computation of derivatives for 7D phase field
        calculations with U(1)³ phase structure.

    Mathematical Foundation:
        Uses the property that differentiation in real space corresponds to
        multiplication by ik in frequency space for efficient computation.
    """

    def __init__(self, domain: Any, precision: str = "float64"):
        """
        Initialize spectral derivatives.

        Physical Meaning:
            Sets up the spectral derivative operations with the computational
            domain and numerical precision, pre-computing wave vectors
            for efficient derivative calculations.

        Args:
            domain (Domain): Computational domain for derivative operations.
            precision (str): Numerical precision for computations.
        """
        # Accept either Domain or an FFT backend carrying a domain
        backend = domain
        actual_domain = getattr(domain, "domain", domain)
        super().__init__(actual_domain, precision)
        # Preserve legacy API expected by tests
        self.fft_backend = backend

        self._wave_vectors = None
        self._k_magnitude_squared = None
        # If domain has shape, precompute; otherwise defer until first call
        if hasattr(actual_domain, "shape"):
            self._wave_vectors = self._compute_wave_vectors()
            self._k_magnitude_squared = self._compute_k_magnitude_squared()
            self.logger.info(
                f"SpectralDerivatives initialized for domain {actual_domain.shape}"
            )
        else:
            self.logger.info(
                "SpectralDerivatives initialized without domain.shape (deferred setup)"
            )

    # Legacy helper expected by tests: compute nth derivative along axis
    def compute_derivative(
        self, field: np.ndarray, axis: int, order: int = 1
    ) -> np.ndarray:
        """Compute nth derivative along a given axis using spectral method."""
        if not self.validate_field(field):
            raise ValueError("Invalid field for derivative computation")
        if order < 1:
            raise ValueError("Order must be >= 1")
        # Ensure wave vectors are available
        if self._wave_vectors is None or self._k_magnitude_squared is None:
            if hasattr(self.domain, "shape"):
                self._wave_vectors = self._compute_wave_vectors()
                self._k_magnitude_squared = self._compute_k_magnitude_squared()
            else:
                raise ValueError("Domain shape is required for spectral derivatives")
        if axis < 0 or axis >= len(self._wave_vectors):
            raise ValueError("Invalid axis for derivative computation")
        # Forward FFT (fallback allowed). Ensure input shape matches domain
        if hasattr(self.domain, "shape") and field.shape != tuple(self.domain.shape):
            raise ValueError("Field shape does not match domain shape")
        # Forward FFT
        try:
            from bhlff.core.fft.unified_spectral_operations import (
                UnifiedSpectralOperations,
            )

            spectral_ops = UnifiedSpectralOperations(
                self.domain, precision=self.precision
            )
            field_spectral = spectral_ops.forward_fft(field, normalization="physics")
        except Exception:
            # Ensure broadcasting shapes are handled
            field_spectral = np.fft.fftn(np.array(field, copy=False))
        # Apply (ik_axis)^order multiplier
        k_vec = self._wave_vectors[axis]
        deriv_spectral = (1j * k_vec) ** order * field_spectral
        # Inverse FFT
        try:
            result = spectral_ops.inverse_fft(deriv_spectral, normalization="physics")  # type: ignore[name-defined]
        except Exception:
            result = np.fft.ifftn(deriv_spectral)
        return result.real.astype(self.precision)

    # Legacy helper for mixed derivatives expected by tests
    def compute_mixed_derivative(
        self, field: np.ndarray, axes: Tuple[int, int], orders: Tuple[int, int]
    ) -> np.ndarray:
        if not self.validate_field(field):
            raise ValueError("Invalid field for mixed derivative computation")
        if self._wave_vectors is None or self._k_magnitude_squared is None:
            if hasattr(self.domain, "shape"):
                self._wave_vectors = self._compute_wave_vectors()
                self._k_magnitude_squared = self._compute_k_magnitude_squared()
            else:
                raise ValueError("Domain shape is required for spectral derivatives")
        ax1, ax2 = axes
        ord1, ord2 = orders
        if (
            ax1 < 0
            or ax1 >= len(self._wave_vectors)
            or ax2 < 0
            or ax2 >= len(self._wave_vectors)
        ):
            raise IndexError("axes out of range for wave vectors")
        # Forward FFT
        try:
            from bhlff.core.fft.unified_spectral_operations import (
                UnifiedSpectralOperations,
            )

            spectral_ops = UnifiedSpectralOperations(
                self.domain, precision=self.precision
            )
            field_spectral = spectral_ops.forward_fft(field, normalization="physics")
        except Exception:
            field_spectral = np.fft.fftn(field)
        # Apply multipliers
        k1 = self._wave_vectors[ax1]
        k2 = self._wave_vectors[ax2]
        deriv_spectral = (1j * k1) ** ord1 * (1j * k2) ** ord2 * field_spectral
        try:
            result = spectral_ops.inverse_fft(deriv_spectral, normalization="physics")  # type: ignore[name-defined]
        except Exception:
            result = np.fft.ifftn(deriv_spectral)
        return result.real.astype(self.precision)

    def compute_gradient(self, field: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute gradient of field in spectral space.

        Physical Meaning:
            Computes the gradient ∇a of the phase field in 7D space-time,
            representing the spatial and phase variations of the field.

        Mathematical Foundation:
            Gradient in spectral space: ∇a → ik * â(k)
            where k is the wave vector and â(k) is the spectral representation.

        Args:
            field (np.ndarray): Field to differentiate.

        Returns:
            Tuple[np.ndarray, ...]: Gradient components in each dimension.
        """
        if not self.validate_field(field):
            raise ValueError("Invalid field for gradient computation")

        # Transform to spectral space (fallback to np.fft if unified backend not applicable)
        try:
            from bhlff.core.fft.unified_spectral_operations import (
                UnifiedSpectralOperations,
            )

            if hasattr(self, "domain") and hasattr(self.domain, "shape"):
                spectral_ops = UnifiedSpectralOperations(
                    self.domain, precision=self.precision
                )
                field_spectral = spectral_ops.forward_fft(
                    field, normalization="physics"
                )
            else:
                raise Exception("No domain for unified backend")
        except Exception:
            field_spectral = np.fft.fftn(field)

        # Compute gradient components
        gradient_components = []
        for i, k_vec in enumerate(self._wave_vectors):
            gradient_spectral = 1j * k_vec * field_spectral
            try:
                gradient_component = spectral_ops.inverse_fft(gradient_spectral, normalization="physics")  # type: ignore[name-defined]
            except Exception:
                gradient_component = np.fft.ifftn(gradient_spectral)
            gradient_components.append(gradient_component.real.astype(self.precision))

        return tuple(gradient_components)

    def compute_divergence(self, field: np.ndarray) -> np.ndarray:
        """
        Compute divergence of vector field in spectral space.

        Physical Meaning:
            Computes the divergence ∇·a of the vector field in 7D space-time,
            representing the net flux of the field.

        Mathematical Foundation:
            Divergence in spectral space: ∇·a → ik · â(k)
            where k is the wave vector and â(k) is the spectral representation.

        Args:
            field (np.ndarray): Vector field to differentiate.

        Returns:
            np.ndarray: Divergence of the field.
        """
        if not self.validate_field(field):
            raise ValueError("Invalid field for divergence computation")

        # For scalar field, divergence is zero
        if len(field.shape) == len(self.domain.shape):
            return np.zeros_like(field)

        # Transform to spectral space (fallback allowed)
        try:
            from bhlff.core.fft.unified_spectral_operations import (
                UnifiedSpectralOperations,
            )

            if hasattr(self, "domain") and hasattr(self.domain, "shape"):
                spectral_ops = UnifiedSpectralOperations(
                    self.domain, precision=self.precision
                )
                field_spectral = spectral_ops.forward_fft(
                    field, normalization="physics"
                )
            else:
                raise Exception("No domain for unified backend")
        except Exception:
            field_spectral = np.fft.fftn(field)

        # Compute divergence
        divergence_spectral = np.zeros_like(field_spectral)
        for i, k_vec in enumerate(self._wave_vectors):
            if i < field.shape[-1]:  # Check if we have enough components
                divergence_spectral += 1j * k_vec * field_spectral[..., i]

        # Transform back to real space
        try:
            divergence = spectral_ops.inverse_fft(divergence_spectral, normalization="physics")  # type: ignore[name-defined]
        except Exception:
            divergence = np.fft.ifftn(divergence_spectral)
        return divergence.real.astype(self.precision)

    def compute_curl(self, field: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute curl of vector field in spectral space.

        Physical Meaning:
            Computes the curl ∇×a of the vector field in 7D space-time,
            representing the rotational component of the field.

        Mathematical Foundation:
            Curl in spectral space: ∇×a → ik × â(k)
            where k is the wave vector and â(k) is the spectral representation.

        Args:
            field (np.ndarray): Vector field to differentiate.

        Returns:
            Tuple[np.ndarray, ...]: Curl components in each dimension.
        """
        if not self.validate_field(field):
            raise ValueError("Invalid field for curl computation")

        # For scalar field, curl is zero
        if len(field.shape) == len(self.domain.shape):
            return tuple(np.zeros_like(field) for _ in range(len(self.domain.shape)))

        # Transform to spectral space (fallback allowed)
        try:
            from bhlff.core.fft.unified_spectral_operations import (
                UnifiedSpectralOperations,
            )

            if hasattr(self, "domain") and hasattr(self.domain, "shape"):
                spectral_ops = UnifiedSpectralOperations(
                    self.domain, precision=self.precision
                )
                field_spectral = spectral_ops.forward_fft(
                    field, normalization="physics"
                )
            else:
                raise Exception("No domain for unified backend")
        except Exception:
            field_spectral = np.fft.fftn(field)

        # Compute curl components
        curl_components = []
        for i in range(len(self.domain.shape)):
            curl_spectral = np.zeros_like(field_spectral[..., 0])

            # Compute curl using cross product in spectral space
            for j in range(len(self.domain.shape)):
                for k in range(len(self.domain.shape)):
                    if i != j and j != k and k != i:
                        # Levi-Civita symbol: ε_ijk
                        epsilon = (
                            1 if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else -1
                        )
                        if epsilon != 0 and k < field.shape[-1]:
                            curl_spectral += (
                                epsilon
                                * 1j
                                * self._wave_vectors[j]
                                * field_spectral[..., k]
                            )

            try:
                curl_component = spectral_ops.inverse_fft(curl_spectral, normalization="physics")  # type: ignore[name-defined]
            except Exception:
                curl_component = np.fft.ifftn(curl_spectral)
            curl_components.append(curl_component.real.astype(self.precision))

        return tuple(curl_components)

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of field in spectral space.

        Physical Meaning:
            Computes the Laplacian Δa of the phase field in 7D space-time,
            representing the second-order spatial variations of the field.

        Mathematical Foundation:
            Laplacian in spectral space: Δa → -|k|² * â(k)
            where |k|² is the squared magnitude of the wave vector.

        Args:
            field (np.ndarray): Field to differentiate.

        Returns:
            np.ndarray: Laplacian of the field.
        """
        if not self.validate_field(field):
            raise ValueError("Invalid field for Laplacian computation")

        # Transform to spectral space (fallback allowed)
        try:
            from bhlff.core.fft.unified_spectral_operations import (
                UnifiedSpectralOperations,
            )

            if hasattr(self, "domain") and hasattr(self.domain, "shape"):
                spectral_ops = UnifiedSpectralOperations(
                    self.domain, precision=self.precision
                )
                field_spectral = spectral_ops.forward_fft(
                    field, normalization="physics"
                )
            else:
                raise Exception("No domain for unified backend")
        except Exception:
            field_spectral = np.fft.fftn(field)

        # Compute Laplacian
        laplacian_spectral = -self._k_magnitude_squared * field_spectral

        # Transform back to real space
        try:
            laplacian = spectral_ops.inverse_fft(laplacian_spectral, normalization="physics")  # type: ignore[name-defined]
        except Exception:
            laplacian = np.fft.ifftn(laplacian_spectral)
        return laplacian.real.astype(self.precision)

    def compute_bi_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute bi-Laplacian (fourth-order derivative) of field.

        Physical Meaning:
            Computes the bi-Laplacian Δ²a of the phase field, representing
            fourth-order spatial variations of the field.

        Mathematical Foundation:
            Bi-Laplacian in spectral space: Δ²a → |k|⁴ * â(k)

        Args:
            field (np.ndarray): Field to differentiate.

        Returns:
            np.ndarray: Bi-Laplacian of the field.
        """
        if not self.validate_field(field):
            raise ValueError("Invalid field for bi-Laplacian computation")

        # Transform to spectral space (fallback allowed)
        try:
            from bhlff.core.fft.unified_spectral_operations import (
                UnifiedSpectralOperations,
            )

            if hasattr(self, "domain") and hasattr(self.domain, "shape"):
                spectral_ops = UnifiedSpectralOperations(
                    self.domain, precision=self.precision
                )
                field_spectral = spectral_ops.forward_fft(
                    field, normalization="physics"
                )
            else:
                raise Exception("No domain for unified backend")
        except Exception:
            field_spectral = np.fft.fftn(field)

        # Compute bi-Laplacian
        bi_laplacian_spectral = (self._k_magnitude_squared**2) * field_spectral

        # Transform back to real space
        try:
            bi_laplacian = spectral_ops.inverse_fft(bi_laplacian_spectral, normalization="physics")  # type: ignore[name-defined]
        except Exception:
            bi_laplacian = np.fft.ifftn(bi_laplacian_spectral)
        return bi_laplacian.real.astype(self.precision)

    def _compute_wave_vectors(self) -> Tuple[np.ndarray, ...]:
        """
        Compute wave vectors for the domain.

        Physical Meaning:
            Computes the wave vectors k for each dimension of the domain,
            representing the frequency components in spectral space.

        Returns:
            Tuple[np.ndarray, ...]: Wave vectors for each dimension.
        """
        wave_vectors = []

        num_dims = len(self.domain.shape)
        for axis, axis_size in enumerate(self.domain.shape):
            # Determine physical spacing per axis (Domain7DBVP-aware)
            if hasattr(self.domain, "N_spatial") and num_dims == 7:
                if axis < 3:  # spatial x,y,z
                    N_axis = self.domain.N_spatial
                    d_axis = self.domain.L_spatial / self.domain.N_spatial
                elif axis < 6:  # phase φ1,φ2,φ3 (period 2π)
                    N_axis = self.domain.N_phase
                    d_axis = 2 * np.pi / self.domain.N_phase
                else:  # time t
                    N_axis = self.domain.N_t
                    d_axis = self.domain.T / self.domain.N_t
            else:
                # Legacy fallback: assume first 3 spatial (L,N), next 3 phase (2π,N_phi), last time (T,N_t)
                if axis < 3 and hasattr(self.domain, "N") and hasattr(self.domain, "L"):
                    N_axis = getattr(self.domain, "N", axis_size)
                    d_axis = getattr(self.domain, "L", float(axis_size)) / max(1, N_axis)
                elif axis < 6 and hasattr(self.domain, "N_phi"):
                    N_axis = getattr(self.domain, "N_phi", axis_size)
                    d_axis = 2 * np.pi / max(1, N_axis)
                elif hasattr(self.domain, "N_t") and hasattr(self.domain, "T"):
                    N_axis = getattr(self.domain, "N_t", axis_size)
                    d_axis = getattr(self.domain, "T", float(axis_size)) / max(1, N_axis)
                else:
                    # As a last resort, assume unit spacing scaled to axis length
                    N_axis = axis_size
                    d_axis = 1.0

            # Compute 1D wave vector with physical scaling (radians per unit)
            k_1d = 2 * np.pi * np.fft.fftfreq(N_axis, d=d_axis)

            # Reshape for broadcasting across the full domain shape
            reshape_pattern = [1] * num_dims
            reshape_pattern[axis] = axis_size
            # If N_axis differs from axis_size (defensive), interpolate or trim/pad
            if N_axis != axis_size:
                # Simple safe handling: resample via slicing or padding zeros
                if N_axis > axis_size:
                    k_1d = k_1d[:axis_size]
                else:
                    pad = axis_size - N_axis
                    k_1d = np.pad(k_1d, (0, pad), mode="constant")
            k_axis = k_1d.reshape(reshape_pattern)

            wave_vectors.append(k_axis)

        return tuple(wave_vectors)

    def _compute_k_magnitude_squared(self) -> np.ndarray:
        """
        Compute squared magnitude of wave vectors.

        Physical Meaning:
            Computes |k|² for each point in the domain, representing
            the squared magnitude of the wave vector in spectral space.

        Returns:
            np.ndarray: Squared magnitude of wave vectors.
        """
        k_magnitude_squared = np.zeros(self.domain.shape)

        for k_vec in self._wave_vectors:
            k_magnitude_squared += k_vec**2  # broadcasting-safe

        return k_magnitude_squared

    def __repr__(self) -> str:
        """String representation of spectral derivatives."""
        return f"{self.__class__.__name__}(domain={self.domain.shape}, precision={self.precision})"
