"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Spectral derivatives implementation for 7D BHLFF Framework.

This module provides spectral derivative operations for the 7D phase field theory,
including gradient, divergence, curl, and higher-order derivatives with
optimized performance for 7D computations.

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


class SpectralDerivatives:
    """
    Spectral derivatives for 7D phase field calculations.

    Physical Meaning:
        Implements mathematical differentiation operations in 7D frequency space,
        providing efficient computation of derivatives for 7D phase field
        calculations with U(1)³ phase structure.

    Mathematical Foundation:
        Uses the property that differentiation in real space corresponds to
        multiplication by ik in frequency space, where k is the wave vector.

    Attributes:
        domain (Domain): Computational domain for the simulation.
        precision (str): Numerical precision for computations.
        _wave_vectors (Tuple[np.ndarray, ...]): Pre-computed wave vectors.
        _k_magnitude (np.ndarray): Pre-computed wave vector magnitudes.
    """

    def __init__(self, domain: "Domain", precision: str = "float64"):
        """
        Initialize spectral derivatives.

        Physical Meaning:
            Sets up the spectral derivatives calculator with the computational
            domain and numerical precision, pre-computing wave vectors for
            efficient derivative calculations.

        Args:
            domain (Domain): Computational domain with grid information.
            precision (str): Numerical precision ('float64' or 'float32').
        """
        self.domain = domain
        self.precision = precision
        self.logger = logging.getLogger(__name__)

        # Pre-compute wave vectors
        self._wave_vectors = self._compute_wave_vectors()
        self._k_magnitude = self._compute_k_magnitude()

        self.logger.info(f"SpectralDerivatives initialized for domain {domain.shape}")

    def compute_gradient(self, field: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute gradient of field in spectral space.

        Physical Meaning:
            Computes the gradient ∇a of the phase field in 7D space-time,
            representing the spatial and temporal rates of change of the
            phase field configuration.

        Mathematical Foundation:
            In spectral space: ∇a → ik * â(k)
            where k is the wave vector and â(k) is the spectral representation.

        Args:
            field (np.ndarray): Field to differentiate.

        Returns:
            Tuple[np.ndarray, ...]: Gradient components (∂a/∂x, ∂a/∂y, ∂a/∂z, ∂a/∂φ₁, ∂a/∂φ₂, ∂a/∂φ₃, ∂a/∂t).
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with domain {self.domain.shape}"
            )

        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Compute gradient components
        gradient_components = []
        for i, k_vec in enumerate(self._wave_vectors):
            gradient_spectral = 1j * k_vec * field_spectral
            gradient_component = np.fft.ifftn(gradient_spectral)
            gradient_components.append(gradient_component.real.astype(self.precision))

        return tuple(gradient_components)

    def compute_divergence(
        self, field_components: Tuple[np.ndarray, ...]
    ) -> np.ndarray:
        """
        Compute divergence of vector field in spectral space.

        Physical Meaning:
            Computes the divergence ∇·a of a vector field in 7D space-time,
            representing the net flux of the field through an infinitesimal
            volume element.

        Mathematical Foundation:
            In spectral space: ∇·a → ik · â(k)
            where k is the wave vector and â(k) is the spectral representation.

        Args:
            field_components (Tuple[np.ndarray, ...]): Vector field components.

        Returns:
            np.ndarray: Divergence field.
        """
        if len(field_components) != 7:
            raise ValueError(f"Expected 7 components, got {len(field_components)}")

        # Transform to spectral space
        field_spectral_components = [np.fft.fftn(comp) for comp in field_components]

        # Compute divergence in spectral space
        divergence_spectral = np.zeros_like(field_spectral_components[0])
        for i, (k_vec, field_spectral) in enumerate(
            zip(self._wave_vectors, field_spectral_components)
        ):
            divergence_spectral += 1j * k_vec * field_spectral

        # Transform back to real space
        divergence = np.fft.ifftn(divergence_spectral)
        return divergence.real.astype(self.precision)

    def compute_curl(
        self, field_components: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, ...]:
        """
        Compute curl of vector field in spectral space.

        Physical Meaning:
            Computes the curl ∇×a of a vector field in 7D space-time,
            representing the rotation of the field around each point.

        Mathematical Foundation:
            In spectral space: ∇×a → ik × â(k)
            where k is the wave vector and â(k) is the spectral representation.

        Args:
            field_components (Tuple[np.ndarray, ...]): Vector field components.

        Returns:
            Tuple[np.ndarray, ...]: Curl components.
        """
        if len(field_components) != 7:
            raise ValueError(f"Expected 7 components, got {len(field_components)}")

        # Transform to spectral space
        field_spectral_components = [np.fft.fftn(comp) for comp in field_components]

        # Compute curl in spectral space using cross product
        curl_components = []
        for i in range(7):
            curl_spectral = np.zeros_like(field_spectral_components[0])
            for j in range(7):
                for k in range(7):
                    if i != j and j != k and k != i:
                        # Levi-Civita symbol for 7D cross product
                        levi_civita = self._levi_civita_7d(i, j, k)
                        if levi_civita != 0:
                            curl_spectral += (
                                levi_civita
                                * 1j
                                * self._wave_vectors[j]
                                * field_spectral_components[k]
                            )

            curl_component = np.fft.ifftn(curl_spectral)
            curl_components.append(curl_component.real.astype(self.precision))

        return tuple(curl_components)

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of field in spectral space.

        Physical Meaning:
            Computes the Laplacian Δa of the phase field in 7D space-time,
            representing the sum of second partial derivatives in all dimensions.

        Mathematical Foundation:
            In spectral space: Δa → -|k|² * â(k)
            where |k|² is the squared magnitude of the wave vector.

        Args:
            field (np.ndarray): Field to differentiate.

        Returns:
            np.ndarray: Laplacian field.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with domain {self.domain.shape}"
            )

        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Compute Laplacian in spectral space
        laplacian_spectral = -self._k_magnitude**2 * field_spectral

        # Transform back to real space
        laplacian = np.fft.ifftn(laplacian_spectral)
        return laplacian.real.astype(self.precision)

    def compute_spectral_derivative(
        self, field: np.ndarray, order: int, direction: int
    ) -> np.ndarray:
        """
        Compute spectral derivative of specified order and direction.

        Physical Meaning:
            Computes the n-th order derivative of the phase field in the
            specified direction, representing the rate of change of the
            field configuration.

        Mathematical Foundation:
            In spectral space: ∂^n/∂x^n a → (ik)^n * â(k)
            where k is the wave vector component in the specified direction.

        Args:
            field (np.ndarray): Field to differentiate.
            order (int): Order of derivative.
            direction (int): Direction index (0-6 for 7D).

        Returns:
            np.ndarray: Derivative field.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with domain {self.domain.shape}"
            )

        if not 0 <= direction < 7:
            raise ValueError(f"Direction must be 0-6, got {direction}")

        if order < 0:
            raise ValueError(f"Order must be non-negative, got {order}")

        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Compute derivative in spectral space
        k_vec = self._wave_vectors[direction]
        derivative_spectral = (1j * k_vec) ** order * field_spectral

        # Transform back to real space
        derivative = np.fft.ifftn(derivative_spectral)
        return derivative.real.astype(self.precision)

    def _compute_wave_vectors(self) -> Tuple[np.ndarray, ...]:
        """
        Compute wave vectors for all 7 dimensions.

        Physical Meaning:
            Computes the wave vectors k for all 7 dimensions of the
            computational domain, representing the frequency components
            in spectral space.

        Mathematical Foundation:
            k = (2π/scale) * m, where m is the mode index and scale is
            the domain size in each dimension.

        Returns:
            Tuple[np.ndarray, ...]: Wave vectors for each dimension.
        """
        wave_vectors = []

        # Check if domain is Domain7DBVP or old Domain
        if hasattr(self.domain, "N_spatial"):
            # New Domain7DBVP structure
            # Spatial dimensions (x, y, z)
            for i in range(3):
                k = np.fft.fftfreq(
                    self.domain.N_spatial, self.domain.L_spatial / self.domain.N_spatial
                )
                k = k * 2 * np.pi / self.domain.L_spatial
                k = np.broadcast_to(k.reshape(-1, 1, 1, 1, 1, 1, 1), self.domain.shape)
                wave_vectors.append(k)

            # Phase dimensions (φ₁, φ₂, φ₃)
            for i in range(3):
                k = np.fft.fftfreq(self.domain.N_phase, 2 * np.pi / self.domain.N_phase)
                k = k * 2 * np.pi / (2 * np.pi)
                k = np.broadcast_to(k.reshape(1, 1, 1, -1, 1, 1, 1), self.domain.shape)
                wave_vectors.append(k)

            # Time dimension (t)
            k = np.fft.fftfreq(self.domain.N_t, self.domain.T / self.domain.N_t)
            k = k * 2 * np.pi / self.domain.T
            k = np.broadcast_to(k.reshape(1, 1, 1, 1, 1, 1, -1), self.domain.shape)
            wave_vectors.append(k)
        else:
            # Old Domain structure
            # Spatial dimensions (x, y, z)
            for i in range(3):
                k = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
                k = k * 2 * np.pi / self.domain.L
                k = np.broadcast_to(k.reshape(-1, 1, 1, 1, 1, 1, 1), self.domain.shape)
                wave_vectors.append(k)

            # Phase dimensions (φ₁, φ₂, φ₃)
            for i in range(3):
                k = np.fft.fftfreq(self.domain.N_phi, 2 * np.pi / self.domain.N_phi)
                k = k * 2 * np.pi / (2 * np.pi)
                k = np.broadcast_to(k.reshape(1, 1, 1, -1, 1, 1, 1), self.domain.shape)
                wave_vectors.append(k)

            # Time dimension (t)
            k = np.fft.fftfreq(self.domain.N_t, self.domain.T / self.domain.N_t)
            k = k * 2 * np.pi / self.domain.T
            k = np.broadcast_to(k.reshape(1, 1, 1, 1, 1, 1, -1), self.domain.shape)
            wave_vectors.append(k)

        return tuple(wave_vectors)

    def _compute_k_magnitude(self) -> np.ndarray:
        """
        Compute magnitude of wave vectors.

        Physical Meaning:
            Computes the magnitude |k| of the wave vectors, representing
            the spatial frequency of the field components.

        Mathematical Foundation:
            |k|² = k_x² + k_y² + k_z² + k_φ₁² + k_φ₂² + k_φ₃² + k_t²

        Returns:
            np.ndarray: Wave vector magnitudes.
        """
        if hasattr(self.domain, "N_spatial"):
            # New Domain7DBVP structure
            k_magnitude_squared = np.zeros(self.domain.shape)
            for k_vec in self._wave_vectors:
                k_magnitude_squared += k_vec**2

            return np.sqrt(k_magnitude_squared)
        else:
            # Old Domain structure
            k_magnitude_squared = np.zeros(self.domain.shape)
            for k_vec in self._wave_vectors:
                k_magnitude_squared += k_vec**2

            return np.sqrt(k_magnitude_squared)

    def _levi_civita_7d(self, i: int, j: int, k: int) -> int:
        """
        Compute Levi-Civita symbol for 7D cross product.

        Physical Meaning:
            Computes the Levi-Civita symbol εᵢⱼₖ for 7D cross product,
            representing the sign of the permutation of indices.

        Mathematical Foundation:
            εᵢⱼₖ = +1 if (i,j,k) is even permutation
            εᵢⱼₖ = -1 if (i,j,k) is odd permutation
            εᵢⱼₖ = 0 if any indices are equal

        Args:
            i, j, k (int): Indices for Levi-Civita symbol.

        Returns:
            int: Levi-Civita symbol value.
        """
        if i == j or j == k or k == i:
            return 0

        # Check if permutation is even or odd
        indices = [i, j, k]
        inversions = 0
        for m in range(3):
            for n in range(m + 1, 3):
                if indices[m] > indices[n]:
                    inversions += 1

        return 1 if inversions % 2 == 0 else -1
