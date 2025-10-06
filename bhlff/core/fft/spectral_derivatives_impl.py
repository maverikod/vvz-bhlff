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

    def __init__(self, domain: "Domain", precision: str = "float64"):
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
        super().__init__(domain, precision)

        # Pre-compute wave vectors for efficiency
        self._wave_vectors = self._compute_wave_vectors()
        self._k_magnitude_squared = self._compute_k_magnitude_squared()

        self.logger.info(f"SpectralDerivatives initialized for domain {domain.shape}")

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

        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Compute gradient components
        gradient_components = []
        for i, k_vec in enumerate(self._wave_vectors):
            gradient_spectral = 1j * k_vec * field_spectral
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

        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Compute divergence
        divergence_spectral = np.zeros_like(field_spectral)
        for i, k_vec in enumerate(self._wave_vectors):
            if i < field.shape[-1]:  # Check if we have enough components
                divergence_spectral += 1j * k_vec * field_spectral[..., i]

        # Transform back to real space
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

        # Transform to spectral space
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

        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Compute Laplacian
        laplacian_spectral = -self._k_magnitude_squared * field_spectral

        # Transform back to real space
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

        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Compute bi-Laplacian
        bi_laplacian_spectral = (self._k_magnitude_squared**2) * field_spectral

        # Transform back to real space
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

        for i, size in enumerate(self.domain.shape):
            # Compute wave numbers for this dimension
            k = np.fft.fftfreq(size, d=1.0 / size) * 2 * np.pi

            # Create meshgrid for this dimension
            k_mesh = np.zeros(self.domain.shape)
            for idx in np.ndindex(self.domain.shape):
                k_mesh[idx] = k[idx[i]]

            wave_vectors.append(k_mesh)

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
            k_magnitude_squared += k_vec**2

        return k_magnitude_squared

    def __repr__(self) -> str:
        """String representation of spectral derivatives."""
        return f"{self.__class__.__name__}(domain={self.domain.shape}, precision={self.precision})"
