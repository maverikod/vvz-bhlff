"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Spectral operations implementation.

This module provides spectral operations for the 7D phase field theory,
including spectral derivatives and spectral filtering.

Physical Meaning:
    Spectral operations implement mathematical operations in frequency space,
    providing efficient computation of derivatives and filtering operations
    for phase field calculations.

Mathematical Foundation:
    Implements spectral operations including spectral derivatives and
    spectral filtering for efficient computation in frequency space.

Example:
    >>> ops = SpectralOperations(domain, fft_backend)
    >>> spectral_derivative = ops.spectral_derivative(field, order=2)
    >>> filtered_field = ops.spectral_filter(field, filter_type="low_pass")
"""

import numpy as np
from typing import Any, Tuple

from ..domain import Domain
from .fft_backend import FFTBackend


class SpectralOperations:
    """
    Spectral operations for phase field calculations.

    Physical Meaning:
        Implements mathematical operations in frequency space, providing
        efficient computation of derivatives, filtering, and other spectral
        operations for phase field calculations.

    Mathematical Foundation:
        Spectral operations work in frequency space where:
        - Derivatives become multiplication by powers of ik
        - Filtering becomes multiplication by filter functions
        - Convolutions become multiplication of spectra

    Attributes:
        domain (Domain): Computational domain.
        fft_backend (FFTBackend): FFT backend for transformations.
        _frequency_arrays (Tuple[np.ndarray, ...]): Frequency arrays.
    """

    def __init__(self, domain: Domain, fft_backend: FFTBackend) -> None:
        """
        Initialize spectral operations.

        Physical Meaning:
            Sets up spectral operations with the computational domain
            and FFT backend for efficient frequency space computations.

        Args:
            domain (Domain): Computational domain for spectral operations.
            fft_backend (FFTBackend): FFT backend for transformations.
        """
        self.domain = domain
        self.fft_backend = fft_backend
        self._frequency_arrays = self.fft_backend.get_frequency_arrays()

    def spectral_derivative(
        self, field: np.ndarray, order: int = 1, axis: int = 0
    ) -> np.ndarray:
        """
        Compute spectral derivative.

        Physical Meaning:
            Computes the derivative of the field using spectral methods,
            which is more accurate than finite difference methods.

        Mathematical Foundation:
            Spectral derivative: ∂^n/∂x^n a(x) = IFFT((ik)^n * FFT(a(x)))
            where k is the frequency and n is the derivative order.

        Args:
            field (np.ndarray): Input field a(x).
            order (int): Derivative order (default: 1).
            axis (int): Axis along which to compute derivative (default: 0).

        Returns:
            np.ndarray: Spectral derivative of the field.

        Raises:
            ValueError: If field shape is incompatible with domain.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        # Transform to spectral space
        field_spectral = self.fft_backend.fft(field)

        # Get frequency array for the specified axis
        if axis >= self.domain.dimensions:
            raise ValueError(
                f"Axis {axis} out of range for {self.domain.dimensions}D domain"
            )

        k = self._frequency_arrays[axis]

        # Create frequency multiplier for derivative
        if self.domain.dimensions == 1:
            k_multiplier = (1j * k) ** order
        elif self.domain.dimensions == 2:
            if axis == 0:
                k_multiplier = (1j * k[:, np.newaxis]) ** order
            else:  # axis == 1
                k_multiplier = (1j * k[np.newaxis, :]) ** order
        else:  # 3D
            if axis == 0:
                k_multiplier = (1j * k[:, np.newaxis, np.newaxis]) ** order
            elif axis == 1:
                k_multiplier = (1j * k[np.newaxis, :, np.newaxis]) ** order
            else:  # axis == 2
                k_multiplier = (1j * k[np.newaxis, np.newaxis, :]) ** order

        # Apply spectral derivative
        derivative_spectral = k_multiplier * field_spectral

        # Transform back to real space
        derivative = self.fft_backend.ifft(derivative_spectral)

        return derivative.real

    def spectral_gradient(self, field: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute spectral gradient.

        Physical Meaning:
            Computes the gradient of the field using spectral methods,
            providing all partial derivatives in all dimensions.

        Mathematical Foundation:
            Spectral gradient: ∇a(x) = IFFT(ik * FFT(a(x)))
            where k is the wave vector.

        Args:
            field (np.ndarray): Input field a(x).

        Returns:
            Tuple[np.ndarray, ...]: Gradient components in each dimension.

        Raises:
            ValueError: If field shape is incompatible with domain.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        gradient_components = []

        for axis in range(self.domain.dimensions):
            gradient_component = self.spectral_derivative(field, order=1, axis=axis)
            gradient_components.append(gradient_component)

        return tuple(gradient_components)

    def spectral_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute spectral Laplacian.

        Physical Meaning:
            Computes the Laplacian of the field using spectral methods,
            providing the sum of second partial derivatives.

        Mathematical Foundation:
            Spectral Laplacian: Δa(x) = IFFT(-|k|² * FFT(a(x)))
            where |k|² is the squared magnitude of the wave vector.

        Args:
            field (np.ndarray): Input field a(x).

        Returns:
            np.ndarray: Spectral Laplacian of the field.

        Raises:
            ValueError: If field shape is incompatible with domain.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        # Transform to spectral space
        field_spectral = self.fft_backend.fft(field)

        # Compute |k|²
        if self.domain.dimensions == 1:
            k = self._frequency_arrays[0]
            k_squared = k**2
        elif self.domain.dimensions == 2:
            kx, ky = self._frequency_arrays
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k_squared = KX**2 + KY**2
        else:  # 3D
            kx, ky, kz = self._frequency_arrays
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            k_squared = KX**2 + KY**2 + KZ**2

        # Apply spectral Laplacian
        laplacian_spectral = -k_squared * field_spectral

        # Transform back to real space
        laplacian = self.fft_backend.ifft(laplacian_spectral)

        return laplacian.real

    def spectral_filter(
        self,
        field: np.ndarray,
        filter_type: str = "low_pass",
        cutoff: float = 0.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Apply spectral filter to field.

        Physical Meaning:
            Applies a spectral filter to the field, removing or enhancing
            specific frequency components.

        Mathematical Foundation:
            Spectral filtering: a_filtered(x) = IFFT(H(k) * FFT(a(x)))
            where H(k) is the filter function in frequency space.

        Args:
            field (np.ndarray): Input field a(x).
            filter_type (str): Type of filter ("low_pass", "high_pass",
                "band_pass", "gaussian").
            cutoff (float): Cutoff frequency for the filter.
            **kwargs: Additional filter parameters.

        Returns:
            np.ndarray: Filtered field.

        Raises:
            ValueError: If field shape is incompatible with domain or
                filter_type is unsupported.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        valid_filter_types = ["low_pass", "high_pass", "band_pass", "gaussian"]
        if filter_type not in valid_filter_types:
            raise ValueError(
                f"Unsupported filter type: {filter_type}"
            )

        # Transform to spectral space
        field_spectral = self.fft_backend.fft(field)

        # Create filter function
        filter_function = self._create_filter_function(filter_type, cutoff, **kwargs)

        # Apply filter
        filtered_spectral = filter_function * field_spectral

        # Transform back to real space
        filtered_field = self.fft_backend.ifft(filtered_spectral)

        return filtered_field.real

    def _create_filter_function(
        self, filter_type: str, cutoff: float, **kwargs: Any
    ) -> np.ndarray:
        """
        Create filter function in frequency space.

        Physical Meaning:
            Creates the filter function H(k) in frequency space for
            spectral filtering operations.

        Mathematical Foundation:
            Creates filter functions such as:
            - Low pass: H(k) = 1 if |k| < k_c, 0 otherwise
            - High pass: H(k) = 0 if |k| < k_c, 1 otherwise
            - Gaussian: H(k) = exp(-|k|²/(2σ²))

        Args:
            filter_type (str): Type of filter.
            cutoff (float): Cutoff frequency.
            **kwargs: Additional filter parameters.

        Returns:
            np.ndarray: Filter function in frequency space.
        """
        # Compute |k|
        if self.domain.dimensions == 1:
            k = self._frequency_arrays[0]
            k_magnitude = np.abs(k)
        elif self.domain.dimensions == 2:
            kx, ky = self._frequency_arrays
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k_magnitude = np.sqrt(KX**2 + KY**2)
        else:  # 3D
            kx, ky, kz = self._frequency_arrays
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

        # Create filter function based on type
        if filter_type == "low_pass":
            filter_function = np.where(k_magnitude < cutoff, 1.0, 0.0)
        elif filter_type == "high_pass":
            filter_function = np.where(k_magnitude > cutoff, 1.0, 0.0)
        elif filter_type == "band_pass":
            low_cutoff = kwargs.get("low_cutoff", cutoff * 0.5)
            high_cutoff = kwargs.get("high_cutoff", cutoff * 1.5)
            filter_function = np.where(
                (k_magnitude > low_cutoff) & (k_magnitude < high_cutoff), 1.0, 0.0
            )
        elif filter_type == "gaussian":
            sigma = kwargs.get("sigma", cutoff)
            filter_function = np.exp(-(k_magnitude**2) / (2 * sigma**2))

        return filter_function

    def spectral_convolution(
        self, field1: np.ndarray, field2: np.ndarray
    ) -> np.ndarray:
        """
        Compute spectral convolution.

        Physical Meaning:
            Computes the convolution of two fields using spectral methods,
            which is more efficient than spatial convolution.

        Mathematical Foundation:
            Spectral convolution: (f * g)(x) = IFFT(FFT(f) * FFT(g))
            where * denotes convolution.

        Args:
            field1 (np.ndarray): First field f(x).
            field2 (np.ndarray): Second field g(x).

        Returns:
            np.ndarray: Convolution result (f * g)(x).

        Raises:
            ValueError: If field shapes are incompatible with domain.
        """
        shapes_match = (
            field1.shape == self.domain.shape and field2.shape == self.domain.shape
        )
        if not shapes_match:
            raise ValueError(
                f"Field shapes {field1.shape}, {field2.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        # Transform to spectral space
        field1_spectral = self.fft_backend.fft(field1)
        field2_spectral = self.fft_backend.fft(field2)

        # Compute convolution in spectral space
        convolution_spectral = field1_spectral * field2_spectral

        # Transform back to real space
        convolution = self.fft_backend.ifft(convolution_spectral)

        return convolution.real

    def __repr__(self) -> str:
        """String representation of spectral operations."""
        return (
            f"SpectralOperations(domain={self.domain}, "
            f"fft_backend={self.fft_backend})"
        )
