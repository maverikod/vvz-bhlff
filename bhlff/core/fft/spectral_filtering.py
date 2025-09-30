"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Spectral filtering implementation.

This module provides spectral filtering operations for the 7D phase field theory,
including various filter types and spectral convolution operations.

Physical Meaning:
    Spectral filtering implements mathematical filtering operations in frequency space,
    providing efficient computation of filtered fields and convolution operations
    for phase field calculations.

Mathematical Foundation:
    Implements spectral filtering where:
    - Filtering becomes multiplication by filter functions
    - Convolutions become multiplication of spectra
    - Various filter types: low-pass, high-pass, band-pass, gaussian

Example:
    >>> filter_ops = SpectralFiltering(domain, fft_backend)
    >>> filtered_field = filter_ops.spectral_filter(field, filter_type="low_pass")
    >>> convolution = filter_ops.spectral_convolution(field1, field2)
"""

import numpy as np
from typing import Any

from ..domain import Domain
from .fft_backend import FFTBackend


class SpectralFiltering:
    """
    Spectral filtering operations for phase field calculations.

    Physical Meaning:
        Implements mathematical filtering operations in frequency space,
        providing efficient computation of filtered fields and convolution
        operations for phase field calculations.

    Mathematical Foundation:
        Spectral filtering works in frequency space where:
        - Filtering becomes multiplication by filter functions
        - Convolutions become multiplication of spectra
        - Various filter types: low-pass, high-pass, band-pass, gaussian
    """

    def __init__(self, domain: Domain, fft_backend: FFTBackend):
        """
        Initialize spectral filtering.

        Physical Meaning:
            Sets up spectral filtering operations with the computational domain
            and FFT backend for efficient frequency space computations.

        Args:
            domain (Domain): Computational domain for spectral operations.
            fft_backend (FFTBackend): FFT backend for transformations.
        """
        self.domain = domain
        self.fft_backend = fft_backend
        self._frequency_arrays = self.fft_backend.get_frequency_arrays()

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
            raise ValueError(f"Unsupported filter type: {filter_type}")

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
            KX, KY = np.meshgrid(kx, ky, indexing="ij")
            k_magnitude = np.sqrt(KX**2 + KY**2)
        else:  # 3D
            kx, ky, kz = self._frequency_arrays
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
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

    def apply_low_pass_filter(self, field: np.ndarray, cutoff: float) -> np.ndarray:
        """
        Apply low-pass filter to field.
        
        Args:
            field (np.ndarray): Input field.
            cutoff (float): Cutoff frequency.
            
        Returns:
            np.ndarray: Filtered field.
        """
        return self.spectral_filter(field, filter_type="low_pass", cutoff=cutoff)
    
    def apply_high_pass_filter(self, field: np.ndarray, cutoff: float) -> np.ndarray:
        """
        Apply high-pass filter to field.
        
        Args:
            field (np.ndarray): Input field.
            cutoff (float): Cutoff frequency.
            
        Returns:
            np.ndarray: Filtered field.
        """
        return self.spectral_filter(field, filter_type="high_pass", cutoff=cutoff)
    
    def apply_band_pass_filter(self, field: np.ndarray, low_cutoff: float, high_cutoff: float) -> np.ndarray:
        """
        Apply band-pass filter to field.
        
        Args:
            field (np.ndarray): Input field.
            low_cutoff (float): Low cutoff frequency.
            high_cutoff (float): High cutoff frequency.
            
        Returns:
            np.ndarray: Filtered field.
        """
        return self.spectral_filter(field, filter_type="band_pass", 
                                  low_cutoff=low_cutoff, high_cutoff=high_cutoff)
    
    def apply_gaussian_filter(self, field: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian filter to field.
        
        Args:
            field (np.ndarray): Input field.
            sigma (float): Standard deviation of Gaussian.
            
        Returns:
            np.ndarray: Filtered field.
        """
        return self.spectral_filter(field, filter_type="gaussian", sigma=sigma)
