"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Spectral operations implementation for 7D BHLFF Framework.

This module provides spectral operations for the 7D phase field theory,
including FFT operations with optimized performance for 7D computations.

Physical Meaning:
    Spectral operations implement mathematical operations in frequency space,
    providing efficient computation of FFT operations for 7D phase field
    calculations with U(1)³ phase structure.

Mathematical Foundation:
    Implements spectral operations including FFT operations for efficient
    computation in 7D frequency space:
    - 7D FFT: â(k_x, k_φ, k_t) = F[a(x, φ, t)]
    - Physics normalization: â(m) = Σ_x a(x) e^(-i k(m)·x) Δ^7
    - Orthogonal normalization: â(m) = (1/√N) Σ_x a(x) e^(-i k(m)·x)

Example:
    >>> ops = SpectralOperations(domain, precision="float64")
    >>> spectral_field = ops.forward_fft(field, 'physics')
    >>> real_field = ops.inverse_fft(spectral_field, 'physics')
"""

import numpy as np
from typing import Any, Tuple, Dict, Optional
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain import Domain
    from .fft_backend import FFTBackend
    from .spectral_derivatives import SpectralDerivatives
    from .spectral_filtering import SpectralFiltering


class SpectralOperations:
    """
    Spectral operations for 7D phase field calculations.

    Physical Meaning:
        Implements mathematical operations in 7D frequency space, providing
        efficient computation of FFT operations for 7D phase field calculations
        with U(1)³ phase structure.

    Mathematical Foundation:
        Implements FFT operations with proper normalization for 7D computations:
        - Physics normalization: â(m) = Σ_x a(x) e^(-i k(m)·x) Δ^7
        - Orthogonal normalization: â(m) = (1/√N) Σ_x a(x) e^(-i k(m)·x)
        where Δ^7 = (dx^3) * (dphi^3) * dt is the 7D volume element.

    Attributes:
        domain (Domain): Computational domain for the simulation.
        precision (str): Numerical precision for computations.
        _fft_backend (FFTBackend): FFT computation backend.
        _derivatives (SpectralDerivatives): Spectral derivatives calculator.
        _filtering (SpectralFiltering): Spectral filtering calculator.
    """

    def __init__(self, domain: "Domain", precision: str = "float64"):
        """
        Initialize spectral operations.

        Physical Meaning:
            Sets up the spectral operations calculator with the computational
            domain and numerical precision, initializing FFT backend and
            specialized calculators for derivatives and filtering.

        Args:
            domain (Domain): Computational domain with grid information.
            precision (str): Numerical precision ('float64' or 'float32').
        """
        self.domain = domain
        self.precision = precision
        self.logger = logging.getLogger(__name__)

        # Initialize FFT backend
        self._fft_backend = None  # Will be initialized when needed

        # Initialize specialized calculators
        self._derivatives = None  # Lazy initialization
        self._filtering = None  # Lazy initialization

        self.logger.info(f"SpectralOperations initialized for domain {domain.shape}")

    def forward_fft(
        self, field: np.ndarray, normalization: str = "ortho"
    ) -> np.ndarray:
        """
        Compute forward FFT of field.

        Physical Meaning:
            Transforms the phase field from real space to frequency space,
            representing the field in terms of its frequency components.

        Mathematical Foundation:
            - Physics normalization: â(m) = Σ_x a(x) e^(-i k(m)·x) Δ^7
            - Orthogonal normalization: â(m) = (1/√N) Σ_x a(x) e^(-i k(m)·x)
            where Δ^7 = (dx^3) * (dphi^3) * dt is the 7D volume element.

        Args:
            field (np.ndarray): Field to transform a(x,φ,t).
            normalization (str): Normalization type ('physics' or 'ortho').

        Returns:
            np.ndarray: Spectral field â(k_x, k_φ, k_t).

        Raises:
            ValueError: If field shape is incompatible with domain or
                normalization type is unsupported.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with domain {self.domain.shape}"
            )

        if normalization == "ortho":
            # Use orthogonal normalization
            field_spectral = np.fft.fftn(field, norm="ortho")
        elif normalization == "physics":
            # Use physics normalization
            field_spectral = np.fft.fftn(field)
            # Apply physics normalization factor
            if hasattr(self.domain, "N_spatial"):
                # New Domain7DBVP structure
                dx = self.domain.L_spatial / self.domain.N_spatial
                dphi = 2 * np.pi / self.domain.N_phase
                dt = self.domain.T / self.domain.N_t
            else:
                # Old Domain structure
                dx = self.domain.L / self.domain.N
                dphi = 2 * np.pi / self.domain.N_phi
                dt = self.domain.T / self.domain.N_t
            normalization_factor = (dx**3) * (dphi**3) * dt
            field_spectral *= normalization_factor
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        return field_spectral.astype(np.complex128)

    def inverse_fft(
        self, spectral_field: np.ndarray, normalization: str = "ortho"
    ) -> np.ndarray:
        """
        Compute inverse FFT of spectral field.

        Physical Meaning:
            Transforms the phase field from frequency space back to real space,
            reconstructing the field from its frequency components.

        Mathematical Foundation:
            - Physics normalization: a(x) = (1/Δ^7) Σ_m â(m) e^(i k(m)·x)
            - Orthogonal normalization: a(x) = (1/√N) Σ_m â(m) e^(i k(m)·x)
            where Δ^7 = (dx^3) * (dphi^3) * dt is the 7D volume element.

        Args:
            spectral_field (np.ndarray): Spectral field â(k_x, k_φ, k_t).
            normalization (str): Normalization type ('physics' or 'ortho').

        Returns:
            np.ndarray: Real field a(x,φ,t).

        Raises:
            ValueError: If spectral field shape is incompatible with domain or
                normalization type is unsupported.
        """
        if spectral_field.shape != self.domain.shape:
            raise ValueError(
                f"Spectral field shape {spectral_field.shape} incompatible with domain {self.domain.shape}"
            )

        if normalization == "ortho":
            # Use orthogonal normalization
            field = np.fft.ifftn(spectral_field, norm="ortho")
        elif normalization == "physics":
            # Use physics normalization
            if hasattr(self.domain, "N_spatial"):
                # New Domain7DBVP structure
                dx = self.domain.L_spatial / self.domain.N_spatial
                dphi = 2 * np.pi / self.domain.N_phase
                dt = self.domain.T / self.domain.N_t
            else:
                # Old Domain structure
                dx = self.domain.L / self.domain.N
                dphi = 2 * np.pi / self.domain.N_phi
                dt = self.domain.T / self.domain.N_t
            normalization_factor = (dx**3) * (dphi**3) * dt
            field = np.fft.ifftn(spectral_field / normalization_factor)
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        return field.astype(np.complex128)

    def get_derivatives(self) -> "SpectralDerivatives":
        """
        Get spectral derivatives calculator.

        Physical Meaning:
            Returns the spectral derivatives calculator for computing
            derivatives in frequency space.

        Returns:
            SpectralDerivatives: Spectral derivatives calculator.
        """
        if self._derivatives is None:
            from .spectral_derivatives import SpectralDerivatives

            self._derivatives = SpectralDerivatives(self.domain, self.precision)
        return self._derivatives

    def get_filtering(self) -> "SpectralFiltering":
        """
        Get spectral filtering calculator.

        Physical Meaning:
            Returns the spectral filtering calculator for applying
            filters in frequency space.

        Returns:
            SpectralFiltering: Spectral filtering calculator.
        """
        if self._filtering is None:
            from .spectral_filtering import SpectralFiltering

            self._filtering = SpectralFiltering(self.domain, self.precision)
        return self._filtering

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
        return self.get_derivatives().compute_spectral_derivative(
            field, order, direction
        )

    def apply_spectral_filter(
        self, field: np.ndarray, filter_type: str, **kwargs
    ) -> np.ndarray:
        """
        Apply spectral filter of specified type.

        Physical Meaning:
            Applies a spectral filter of the specified type to the field,
            providing a unified interface for different filtering operations.

        Args:
            field (np.ndarray): Field to filter.
            filter_type (str): Type of filter ('low_pass', 'high_pass', 'band_pass', 'gaussian').
            **kwargs: Additional arguments for the specific filter type.

        Returns:
            np.ndarray: Filtered field.
        """
        return self.get_filtering().apply_spectral_filter(field, filter_type, **kwargs)

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
