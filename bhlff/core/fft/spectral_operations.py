"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Spectral operations implementation for 7D BHLFF Framework.

This module provides spectral operations for the 7D phase field theory,
including spectral derivatives, filtering, and FFT operations with
optimized performance for 7D computations.

Physical Meaning:
    Spectral operations implement mathematical operations in frequency space,
    providing efficient computation of derivatives and filtering operations
    for 7D phase field calculations with U(1)³ phase structure.

Mathematical Foundation:
    Implements spectral operations including spectral derivatives and
    spectral filtering for efficient computation in 7D frequency space:
    - 7D FFT: â(k_x, k_φ, k_t) = F[a(x, φ, t)]
    - Spectral derivatives: ∂^n/∂x^n a → (ik)^n * â(k)
    - Spectral filtering: a_filtered = F^{-1}[H(k) * â(k)]

Example:
    >>> ops = SpectralOperations(domain, precision="float64")
    >>> spectral_derivative = ops.spectral_derivative(field, order=2)
    >>> filtered_field = ops.spectral_filter(field, filter_type="low_pass")
"""

import numpy as np
from typing import Any, Tuple, Dict, Optional
import logging

from typing import TYPE_CHECKING, Optional
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
        efficient computation of derivatives, filtering, and other spectral
        operations for 7D phase field calculations with U(1)³ phase structure.

    Mathematical Foundation:
        Spectral operations work in 7D frequency space where:
        - 7D FFT: â(k_x, k_φ, k_t) = F[a(x, φ, t)]
        - Derivatives become multiplication by powers of ik
        - Filtering becomes multiplication by filter functions
        - Convolutions become multiplication of spectra
        - 7D wave vector: |k|² = |k_x|² + |k_φ|² + k_t²

    Attributes:
        domain (Domain): Computational domain.
        precision (str): Numerical precision ('float64' or 'float32').
        fft_backend (Optional[FFTBackend]): FFT backend for transformations.
        _frequency_arrays (Tuple[np.ndarray, ...]): Frequency arrays.
        _wave_vectors (Tuple[np.ndarray, ...]): 7D wave vectors.
    """

    def __init__(self, domain: 'Domain', precision: str = "float64", 
                 fft_backend: Optional['FFTBackend'] = None) -> None:
        """
        Initialize spectral operations for 7D computations.

        Physical Meaning:
            Sets up spectral operations with the computational domain
            and precision for efficient 7D frequency space computations.

        Args:
            domain (Domain): Computational domain for spectral operations.
            precision (str): Numerical precision ('float64' or 'float32').
            fft_backend (Optional[FFTBackend]): FFT backend for transformations.
        """
        self.domain = domain
        self.precision = precision
        self.fft_backend = fft_backend
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        if fft_backend is not None:
            self._frequency_arrays = fft_backend.get_frequency_arrays()
            # TODO: Initialize derivatives and filtering when available
            self._derivatives = None
            self._filtering = None
        else:
            # Initialize without FFT backend for basic operations
            self._frequency_arrays = self._compute_frequency_arrays()
            self._derivatives = None
            self._filtering = None
        
        # Compute 7D wave vectors
        self._wave_vectors = self._compute_wave_vectors()
        
        self.logger.info(f"SpectralOperations initialized: domain={domain.shape}, precision={precision}")

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
        if self._derivatives is None:
            raise NotImplementedError("Spectral derivatives not yet implemented")
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
        if self._derivatives is None:
            raise NotImplementedError("Spectral derivatives not yet implemented")
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
        if self._derivatives is None:
            raise NotImplementedError("Spectral derivatives not yet implemented")
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
        if self._filtering is None:
            raise NotImplementedError("Spectral filtering not yet implemented")
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
        if self._filtering is None:
            raise NotImplementedError("Spectral filtering not yet implemented")
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
        if self._derivatives is None:
            raise NotImplementedError("Spectral derivatives not yet implemented")
        return self._derivatives.compute_derivative(field, order, axis)
    
    def compute_gradient(self, field: np.ndarray) -> np.ndarray:
        """
        Compute gradient of scalar field.
        
        Args:
            field (np.ndarray): Scalar field.
            
        Returns:
            np.ndarray: Gradient vector field.
        """
        if self._derivatives is None:
            raise NotImplementedError("Spectral derivatives not yet implemented")
        return self._derivatives.compute_gradient(field)
    
    def compute_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute divergence of vector field.
        
        Args:
            vector_field (np.ndarray): Vector field.
            
        Returns:
            np.ndarray: Divergence scalar field.
        """
        if self._derivatives is None:
            raise NotImplementedError("Spectral derivatives not yet implemented")
        return self._derivatives.compute_divergence(vector_field)
    
    def compute_curl(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute curl of vector field.
        
        Args:
            vector_field (np.ndarray): Vector field.
            
        Returns:
            np.ndarray: Curl vector field.
        """
        if self._derivatives is None:
            raise NotImplementedError("Spectral derivatives not yet implemented")
        return self._derivatives.compute_curl(vector_field)

    def forward_fft(self, field: np.ndarray, normalization: str = 'ortho') -> np.ndarray:
        """
        Forward FFT with proper normalization for 7D.
        
        Physical Meaning:
            Performs forward FFT transformation with specified normalization
            for 7D phase field computations.
            
        Args:
            field (np.ndarray): Input field in real space.
            normalization (str): Normalization type ('ortho' or 'physics').
            
        Returns:
            np.ndarray: Forward FFT result in spectral space.
        """
        if normalization == 'ortho':
            # Orthogonal normalization for testing
            return np.fft.fftn(field, norm='ortho')
        elif normalization == 'physics':
            # 7D physics normalization: Δ^7 = (dx^3) * (dphi^3) * dt
            dx = self.domain.L / self.domain.N
            dphi = (2 * np.pi) / self.domain.N_phi
            dt = self.domain.T / self.domain.N_t
            normalization_factor = (dx ** 3) * (dphi ** 3) * dt
            
            field_spectral = np.fft.fftn(field) * normalization_factor
            return field_spectral
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
    
    def inverse_fft(self, spectral_field: np.ndarray, normalization: str = 'ortho') -> np.ndarray:
        """
        Inverse FFT with proper normalization for 7D.
        
        Physical Meaning:
            Performs inverse FFT transformation with specified normalization
            for 7D phase field computations.
            
        Args:
            spectral_field (np.ndarray): Input field in spectral space.
            normalization (str): Normalization type ('ortho' or 'physics').
            
        Returns:
            np.ndarray: Inverse FFT result in real space.
        """
        if normalization == 'ortho':
            # Orthogonal normalization for testing
            return np.fft.ifftn(spectral_field, norm='ortho')
        elif normalization == 'physics':
            # 7D physics normalization: 1/(Δ^7) where Δ^7 = (dx^3) * (dphi^3) * dt
            dx = self.domain.L / self.domain.N
            dphi = (2 * np.pi) / self.domain.N_phi
            dt = self.domain.T / self.domain.N_t
            normalization_factor = (dx ** 3) * (dphi ** 3) * dt
            
            result = np.fft.ifftn(spectral_field) / normalization_factor
            return result
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
    
    def compute_wave_vectors(self) -> Tuple[np.ndarray, ...]:
        """
        Compute wave vectors for 7D domain.
        
        Physical Meaning:
            Computes the discrete wave vectors k = 2π/L * m for each
            dimension in the 7D domain.
            
        Returns:
            Tuple[np.ndarray, ...]: Wave vectors for each dimension.
        """
        return self._wave_vectors
    
    def compute_wave_vector_magnitude(self) -> np.ndarray:
        """
        Compute magnitude of 7D wave vectors |k|.
        
        Physical Meaning:
            Computes the magnitude of the 7D wave vector:
            |k|² = |k_x|² + |k_φ|² + k_t²
            
        Returns:
            np.ndarray: Wave vector magnitudes |k|.
        """
        # Create meshgrid of wave vectors
        K_mesh = np.meshgrid(*self._wave_vectors, indexing='ij')
        
        # Compute magnitude squared
        k_magnitude_squared = sum(K**2 for K in K_mesh)
        
        # Take square root
        k_magnitude = np.sqrt(k_magnitude_squared)
        
        return k_magnitude
    
    def energy_conservation_check(self, real_field: np.ndarray, 
                                spectral_field: np.ndarray) -> float:
        """
        Check energy conservation in FFT transformation.
        
        Physical Meaning:
            Verifies that energy is conserved during FFT transformation
            according to Parseval's theorem.
            
        Mathematical Foundation:
            Σ |a(x)|² = (1/N^7) Σ |â(k)|²
            
        Args:
            real_field (np.ndarray): Field in real space.
            spectral_field (np.ndarray): Field in spectral space.
            
        Returns:
            float: Relative error in energy conservation.
        """
        real_energy = np.sum(np.abs(real_field)**2)
        spectral_energy = np.sum(np.abs(spectral_field)**2) / np.prod(real_field.shape)
        
        return abs(real_energy - spectral_energy) / real_energy
    
    def _compute_frequency_arrays(self) -> Tuple[np.ndarray, ...]:
        """
        Compute frequency arrays for the domain.
        
        Physical Meaning:
            Computes the frequency arrays for each dimension
            in the 7D domain.
            
        Returns:
            Tuple[np.ndarray, ...]: Frequency arrays for each dimension.
        """
        frequency_arrays = []
        
        for n in self.domain.shape:
            # Compute frequency array
            freq = np.fft.fftfreq(n, d=1.0/n)
            frequency_arrays.append(freq)
        
        return tuple(frequency_arrays)
    
    def _compute_wave_vectors(self) -> Tuple[np.ndarray, ...]:
        """
        Compute wave vectors for 7D domain.
        
        Physical Meaning:
            Computes the discrete wave vectors k = 2π/L * m for each
            dimension in the 7D domain.
            
        Returns:
            Tuple[np.ndarray, ...]: Wave vectors for each dimension.
        """
        wave_vectors = []
        
        # For 7D domain: spatial dimensions use L, phase dimensions use 2π, time uses T
        dimensions = ['x', 'y', 'z', 'phi1', 'phi2', 'phi3', 't']
        scales = [self.domain.L, self.domain.L, self.domain.L, 
                 2*np.pi, 2*np.pi, 2*np.pi, self.domain.T]
        
        for i, (n, scale) in enumerate(zip(self.domain.shape, scales)):
            # Correct formula: k = (2π/scale) * m
            k = np.fft.fftfreq(n, d=scale/n)
            k *= 2 * np.pi  # k = (2π/scale) * m
            wave_vectors.append(k)
        
        return tuple(wave_vectors)
    
    def __repr__(self) -> str:
        """String representation of spectral operations."""
        return (
            f"SpectralOperations(domain={self.domain}, "
            f"precision={self.precision}, fft_backend={self.fft_backend})"
        )
