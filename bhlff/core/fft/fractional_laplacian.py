"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Fractional Laplacian Operator Implementation for 7D BHLFF Framework.

This module implements the fractional Laplacian operator (-Δ)^β for the 7D phase
field theory, providing efficient computation of non-local interactions in
spectral space.

Theoretical Background:
    The fractional Laplacian (-Δ)^β represents non-local interactions in the
    phase field, with β controlling the range of interactions from local (β→0)
    to long-range (β→2). In spectral space: (-Δ)^β f → |k|^(2β) * f̂(k).

Example:
    >>> laplacian = FractionalLaplacian(domain, beta=1.0)
    >>> result = laplacian.apply(field)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..domain import Domain


class FractionalLaplacian:
    """
    Fractional Laplacian operator (-Δ)^β implementation.
    
    Physical Meaning:
        Represents the fractional derivative operator that governs
        non-local interactions in the phase field, with β controlling
        the range of interactions from local (β→0) to long-range (β→2).
        
    Mathematical Foundation:
        In spectral space: (-Δ)^β f → |k|^(2β) * f̂(k)
        where k is the wave vector. The operator is defined as:
        (-Δ)^β f(x) = F^{-1}[|k|^(2β) * F[f](k)]
        
    Attributes:
        domain (Domain): Computational domain for the simulation.
        beta (float): Fractional order β ∈ (0,2).
        _spectral_coeffs (np.ndarray): Pre-computed spectral coefficients |k|^(2β).
        _wave_vectors (Tuple[np.ndarray, ...]): Wave vectors for each dimension.
    """
    
    def __init__(self, domain: 'Domain', beta: float, lambda_param: float = 0.0):
        """
        Initialize fractional Laplacian with order β.
        
        Physical Meaning:
            Sets up the fractional Laplacian operator with the specified
            fractional order, which determines the range of non-local
            interactions in the phase field.
            
        Args:
            domain (Domain): Computational domain with grid information.
            beta (float): Fractional order β ∈ (0,2).
                - β → 0: local interactions
                - β = 1: classical Laplacian
                - β → 2: long-range interactions
        """
        self.domain = domain
        self.beta = beta
        self.lambda_param = lambda_param
        self._validate_beta()
        
        # Pre-compute spectral coefficients
        self._setup_spectral_coefficients()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """
        Apply fractional Laplacian (-Δ)^β to field.
        
        Physical Meaning:
            Computes the fractional Laplacian of the field, representing
            non-local interactions with range determined by β.
            
        Mathematical Foundation:
            Applies the operator in spectral space:
            1. FFT: f̂(k) = F[f](k)
            2. Multiply: ĝ(k) = |k|^(2β) * f̂(k)
            3. IFFT: g(x) = F^{-1}[ĝ(k)]
            
        Args:
            field (np.ndarray): Input field f(x) in real space.
                
        Returns:
            np.ndarray: Result field g(x) = (-Δ)^β f(x) in real space.
                
        Raises:
            ValueError: If field has incompatible shape with domain.
        """
        if field.shape != self.domain.shape:
            raise ValueError(f"Field shape {field.shape} incompatible with domain shape {self.domain.shape}")
        
        # Transform to spectral space
        field_spectral = np.fft.fftn(field, norm='ortho')
        
        # Apply spectral coefficients
        result_spectral = field_spectral * self._spectral_coeffs
        
        # Transform back to real space
        result = np.fft.ifftn(result_spectral, norm='ortho')
        
        return result.real
    
    def get_spectral_coefficients(self) -> np.ndarray:
        """
        Get spectral coefficients |k|^(2β) for all wave vectors.
        
        Physical Meaning:
            Returns the spectral representation of the fractional
            Laplacian operator, which represents the strength of
            interactions at different wave numbers.
            
        Returns:
            np.ndarray: Spectral coefficients |k|^(2β).
        """
        return self._spectral_coeffs.copy()
    
    def handle_special_cases(self, k_magnitude: np.ndarray) -> np.ndarray:
        """
        Handle special cases: k=0, β→0, β→2.
        
        Physical Meaning:
            Handles edge cases in the fractional Laplacian computation
            to ensure numerical stability and physical correctness.
            
        Mathematical Foundation:
            - k=0: |k|^(2β) = 0 for β > 0
            - β→0: |k|^(2β) → 1 (identity operator)
            - β→2: |k|^(2β) → |k|^4 (biharmonic operator)
            
        Args:
            k_magnitude (np.ndarray): Wave vector magnitudes |k|.
            
        Returns:
            np.ndarray: Processed spectral coefficients.
        """
        # Handle k=0 mode
        k_zero_mask = (k_magnitude == 0)
        k_nonzero_mask = ~k_zero_mask
        
        # Initialize result
        result = np.zeros_like(k_magnitude)
        
        # Handle k=0 case: D(0) = λ (as per TЗ)
        result[k_zero_mask] = self.lambda_param
        
        # Handle k≠0 case
        if np.any(k_nonzero_mask):
            result[k_nonzero_mask] = k_magnitude[k_nonzero_mask] ** (2 * self.beta)
        
        # Handle overflow protection
        max_k = np.max(k_magnitude[k_nonzero_mask]) if np.any(k_nonzero_mask) else 0
        if max_k > 0 and 2 * self.beta * np.log(max_k) > 700:  # exp(700) ≈ float64 max
            self.logger.warning(f"Potential overflow in |k|^(2β) computation for β={self.beta}")
        
        return result
    
    def _validate_beta(self) -> None:
        """
        Validate fractional order β.
        
        Physical Meaning:
            Ensures the fractional order is within the physically
            meaningful range for the fractional Laplacian.
        """
        if not (0 < self.beta < 2):
            raise ValueError(f"Fractional order β must be in (0,2), got {self.beta}")
    
    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for fractional Laplacian.
        
        Physical Meaning:
            Pre-computes the spectral representation |k|^(2β) of the
            fractional Laplacian operator for efficient application.
        """
        # Compute wave vectors for each dimension
        self._wave_vectors = self._compute_wave_vectors()
        
        # Compute wave vector magnitudes
        k_magnitude = self._compute_wave_vector_magnitude()
        
        # Apply special case handling
        self._spectral_coeffs = self.handle_special_cases(k_magnitude)
    
    def _compute_wave_vectors(self) -> Tuple[np.ndarray, ...]:
        """
        Compute wave vectors for each dimension.
        
        Physical Meaning:
            Computes the discrete wave vectors k = 2π/L * m for each
            dimension, where m are the mode indices.
            
        Returns:
            Tuple[np.ndarray, ...]: Wave vectors for each dimension.
        """
        wave_vectors = []
        
        for i, n in enumerate(self.domain.shape):
            # Compute wave numbers
            k = np.fft.fftfreq(n, d=self.domain.L / n)
            k *= 2 * np.pi  # Convert to angular frequency
            wave_vectors.append(k)
        
        return tuple(wave_vectors)
    
    def _compute_wave_vector_magnitude(self) -> np.ndarray:
        """
        Compute magnitude of wave vectors |k|.
        
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
    
    def get_operator_info(self) -> Dict[str, Any]:
        """
        Get information about the fractional Laplacian operator.
        
        Physical Meaning:
            Returns detailed information about the operator configuration
            and its physical properties.
            
        Returns:
            Dict[str, Any]: Operator information including:
                - beta: Fractional order
                - domain_shape: Domain dimensions
                - max_wave_number: Maximum wave number
                - interaction_range: Effective interaction range
        """
        max_k = np.max(np.sqrt(sum(K**2 for K in np.meshgrid(*self._wave_vectors, indexing='ij'))))
        
        return {
            'beta': self.beta,
            'domain_shape': self.domain.shape,
            'max_wave_number': max_k,
            'interaction_range': 'local' if self.beta < 0.5 else 'intermediate' if self.beta < 1.5 else 'long_range',
            'operator_type': 'fractional_laplacian'
        }
