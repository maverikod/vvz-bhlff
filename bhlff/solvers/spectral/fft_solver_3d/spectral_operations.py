"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

3D spectral operations for FFT solver.

This module implements 3D-specific spectral operations including
derivatives and Laplacian computation using FFT methods.

Physical Meaning:
    Computes spectral derivatives and Laplacian of fields using FFT
    methods for efficient computation in 3D space.

Mathematical Foundation:
    Implements spectral operations:
    - Spectral derivative: ∂ⁿu/∂xⁿ = IFFT((ik)ⁿ * FFT(u))
    - Spectral Laplacian: ∇²u = IFFT(-k² * FFT(u))

Example:
    >>> ops = SpectralOperations3D(domain, fft_backend)
    >>> derivative = ops.compute_derivative(field, order=1, axis=0)
    >>> laplacian = ops.compute_laplacian(field)
"""

import numpy as np
from typing import Dict, Any

from ....core.domain import Domain
from ....core.fft import FFTBackend


class SpectralOperations3D:
    """
    3D spectral operations for FFT solver.

    Physical Meaning:
        Computes spectral derivatives and Laplacian of fields using FFT
        methods for efficient computation in 3D space.

    Mathematical Foundation:
        Implements spectral operations:
        - Spectral derivative: ∂ⁿu/∂xⁿ = IFFT((ik)ⁿ * FFT(u))
        - Spectral Laplacian: ∇²u = IFFT(-k² * FFT(u))

    Attributes:
        domain (Domain): 3D computational domain.
        fft_backend (FFTBackend): FFT backend for operations.
    """

    def __init__(self, domain: Domain, fft_backend: FFTBackend) -> None:
        """
        Initialize 3D spectral operations.

        Physical Meaning:
            Sets up the 3D spectral operations with domain and
            FFT backend for efficient computation.

        Args:
            domain (Domain): 3D computational domain.
            fft_backend (FFTBackend): FFT backend for operations.
        """
        self.domain = domain
        self.fft_backend = fft_backend

    def compute_derivative(
        self, field: np.ndarray, order: int = 1, axis: int = 0
    ) -> np.ndarray:
        """
        Compute spectral derivative.

        Physical Meaning:
            Computes the spectral derivative of the field using FFT
            methods for efficient computation.

        Mathematical Foundation:
            Computes spectral derivative: ∂ⁿu/∂xⁿ = IFFT((ik)ⁿ * FFT(u))
            where k is the wave vector and n is the derivative order.

        Args:
            field (np.ndarray): Field to differentiate.
            order (int): Derivative order.
            axis (int): Axis along which to differentiate.

        Returns:
            np.ndarray: Spectral derivative.
        """
        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Get wave vectors for the specified axis
        if axis == 0:
            k = np.fft.fftfreq(self.domain.N, self.domain.dx)
            K = k[:, np.newaxis, np.newaxis]
        elif axis == 1:
            k = np.fft.fftfreq(self.domain.N, self.domain.dx)
            K = k[np.newaxis, :, np.newaxis]
        elif axis == 2:
            k = np.fft.fftfreq(self.domain.N, self.domain.dx)
            K = k[np.newaxis, np.newaxis, :]
        else:
            raise ValueError(f"Invalid axis {axis} for 3D domain")

        # Apply derivative operator in spectral space
        derivative_spectral = (1j * K) ** order * field_spectral

        # Transform back to real space
        derivative = np.fft.ifftn(derivative_spectral)

        return derivative.real

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute spectral Laplacian.

        Physical Meaning:
            Computes the spectral Laplacian of the field using FFT
            methods for efficient computation.

        Mathematical Foundation:
            Computes spectral Laplacian: ∇²u = IFFT(-k² * FFT(u))
            where k² is the squared wave vector magnitude.

        Args:
            field (np.ndarray): Field to compute Laplacian of.

        Returns:
            np.ndarray: Spectral Laplacian.
        """
        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Get wave vectors for all 3 dimensions
        kx = np.fft.fftfreq(self.domain.N, self.domain.dx)
        ky = np.fft.fftfreq(self.domain.N, self.domain.dx)
        kz = np.fft.fftfreq(self.domain.N, self.domain.dx)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k_squared = KX**2 + KY**2 + KZ**2

        # Apply Laplacian operator in spectral space
        laplacian_spectral = -k_squared * field_spectral

        # Transform back to real space
        laplacian = np.fft.ifftn(laplacian_spectral)

        return laplacian.real

    def __repr__(self) -> str:
        """String representation of 3D spectral operations."""
        return f"SpectralOperations3D(domain={self.domain})"
