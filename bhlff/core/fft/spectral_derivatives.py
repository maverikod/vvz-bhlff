"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Spectral derivatives implementation.

This module provides spectral derivative operations for the 7D phase field theory,
including spectral derivatives, gradient, and Laplacian computations.

Physical Meaning:
    Spectral derivatives implement mathematical derivative operations in frequency space,
    providing efficient and accurate computation of derivatives for phase field calculations.

Mathematical Foundation:
    Implements spectral derivatives where:
    - Derivatives become multiplication by powers of ik
    - Gradient becomes multiplication by ik vector
    - Laplacian becomes multiplication by -|k|²

Example:
    >>> deriv_ops = SpectralDerivatives(domain, fft_backend)
    >>> derivative = deriv_ops.spectral_derivative(field, order=2)
    >>> gradient = deriv_ops.spectral_gradient(field)
    >>> laplacian = deriv_ops.spectral_laplacian(field)
"""

import numpy as np
from typing import Tuple

from ..domain import Domain
from .fft_backend import FFTBackend


class SpectralDerivatives:
    """
    Spectral derivative operations for phase field calculations.

    Physical Meaning:
        Implements mathematical derivative operations in frequency space,
        providing efficient and accurate computation of derivatives
        for phase field calculations.

    Mathematical Foundation:
        Spectral derivatives work in frequency space where:
        - Derivatives become multiplication by powers of ik
        - Gradient becomes multiplication by ik vector
        - Laplacian becomes multiplication by -|k|²
    """

    def __init__(self, domain: Domain, fft_backend: FFTBackend):
        """
        Initialize spectral derivatives.

        Physical Meaning:
            Sets up spectral derivative operations with the computational domain
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
            KX, KY = np.meshgrid(kx, ky, indexing="ij")
            k_squared = KX**2 + KY**2
        else:  # 3D
            kx, ky, kz = self._frequency_arrays
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
            k_squared = KX**2 + KY**2 + KZ**2

        # Apply spectral Laplacian
        laplacian_spectral = -k_squared * field_spectral

        # Transform back to real space
        laplacian = self.fft_backend.ifft(laplacian_spectral)

        return laplacian.real

    def compute_first_derivative(self, field: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Compute first derivative of field.
        
        Args:
            field (np.ndarray): Input field.
            axis (int): Axis along which to compute derivative.
            
        Returns:
            np.ndarray: First derivative of the field.
        """
        return self.spectral_derivative(field, order=1, axis=axis)
    
    def compute_second_derivative(self, field: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Compute second derivative of field.
        
        Args:
            field (np.ndarray): Input field.
            axis (int): Axis along which to compute derivative.
            
        Returns:
            np.ndarray: Second derivative of the field.
        """
        return self.spectral_derivative(field, order=2, axis=axis)
    
    def compute_nth_derivative(self, field: np.ndarray, order: int, axis: int = 0) -> np.ndarray:
        """
        Compute nth derivative of field.
        
        Args:
            field (np.ndarray): Input field.
            order (int): Order of derivative.
            axis (int): Axis along which to compute derivative.
            
        Returns:
            np.ndarray: Nth derivative of the field.
        """
        return self.spectral_derivative(field, order=order, axis=axis)
    
    def compute_mixed_derivative(self, field: np.ndarray, orders: list) -> np.ndarray:
        """
        Compute mixed derivative of field.
        
        Args:
            field (np.ndarray): Input field.
            orders (list): List of derivative orders for each axis.
            
        Returns:
            np.ndarray: Mixed derivative of the field.
        """
        result = field.copy()
        for axis, order in enumerate(orders):
            if order > 0:
                result = self.spectral_derivative(result, order=order, axis=axis)
        return result
    
    def compute_gradient(self, field: np.ndarray) -> np.ndarray:
        """
        Compute gradient of scalar field.
        
        Args:
            field (np.ndarray): Scalar field.
            
        Returns:
            np.ndarray: Gradient vector field.
        """
        gradients = []
        for axis in range(field.ndim):
            gradients.append(self.spectral_derivative(field, order=1, axis=axis))
        return np.stack(gradients, axis=-1)
    
    def compute_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute divergence of vector field.
        
        Args:
            vector_field (np.ndarray): Vector field.
            
        Returns:
            np.ndarray: Divergence scalar field.
        """
        divergence = np.zeros_like(vector_field[..., 0])
        for axis in range(vector_field.shape[-1]):
            divergence += self.spectral_derivative(vector_field[..., axis], order=1, axis=axis)
        return divergence
    
    def compute_curl(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute curl of vector field.
        
        Args:
            vector_field (np.ndarray): Vector field.
            
        Returns:
            np.ndarray: Curl vector field.
        """
        if vector_field.shape[-1] != 3:
            raise ValueError("Curl requires 3D vector field")
        
        curl = np.zeros_like(vector_field)
        # curl_x = ∂F_z/∂y - ∂F_y/∂z
        curl[..., 0] = (self.spectral_derivative(vector_field[..., 2], order=1, axis=1) - 
                       self.spectral_derivative(vector_field[..., 1], order=1, axis=2))
        # curl_y = ∂F_x/∂z - ∂F_z/∂x
        curl[..., 1] = (self.spectral_derivative(vector_field[..., 0], order=1, axis=2) - 
                       self.spectral_derivative(vector_field[..., 2], order=1, axis=0))
        # curl_z = ∂F_y/∂x - ∂F_x/∂y
        curl[..., 2] = (self.spectral_derivative(vector_field[..., 1], order=1, axis=0) - 
                       self.spectral_derivative(vector_field[..., 0], order=1, axis=1))
        return curl
