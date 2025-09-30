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
from .spectral_derivatives import SpectralDerivatives
from .spectral_filtering import SpectralFiltering


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
        self._derivatives = SpectralDerivatives(domain, fft_backend)
        self._filtering = SpectralFiltering(domain, fft_backend)

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
        return self._derivatives.spectral_derivative(field, order, axis)

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
        return self._derivatives.spectral_gradient(field)

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
        return self._derivatives.spectral_laplacian(field)

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
        return self._filtering.spectral_filter(field, filter_type, cutoff, **kwargs)

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
        return self._filtering.spectral_convolution(field1, field2)

    def compute_derivative(self, field: np.ndarray, order: int = 1, axis: int = 0) -> np.ndarray:
        """
        Compute derivative of field using spectral methods.
        
        Args:
            field (np.ndarray): Input field.
            order (int): Order of derivative.
            axis (int): Axis along which to compute derivative.
            
        Returns:
            np.ndarray: Derivative of the field.
        """
        return self._derivatives.compute_derivative(field, order, axis)
    
    def compute_gradient(self, field: np.ndarray) -> np.ndarray:
        """
        Compute gradient of scalar field.
        
        Args:
            field (np.ndarray): Scalar field.
            
        Returns:
            np.ndarray: Gradient vector field.
        """
        return self._derivatives.compute_gradient(field)
    
    def compute_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute divergence of vector field.
        
        Args:
            vector_field (np.ndarray): Vector field.
            
        Returns:
            np.ndarray: Divergence scalar field.
        """
        return self._derivatives.compute_divergence(vector_field)
    
    def compute_curl(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute curl of vector field.
        
        Args:
            vector_field (np.ndarray): Vector field.
            
        Returns:
            np.ndarray: Curl vector field.
        """
        return self._derivatives.compute_curl(vector_field)

    def __repr__(self) -> str:
        """String representation of spectral operations."""
        return (
            f"SpectralOperations(domain={self.domain}, "
            f"fft_backend={self.fft_backend})"
        )
