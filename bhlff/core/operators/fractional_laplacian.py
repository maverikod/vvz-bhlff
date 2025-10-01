"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Fractional Laplacian implementation.

This module implements the fractional Laplacian operator for the 7D phase
field theory, providing the core fractional derivative operator.

Physical Meaning:
    The fractional Laplacian (-Δ)^β represents the fractional derivative
    operator that governs non-local interactions in phase field configurations.

Mathematical Foundation:
    Implements the fractional Laplacian (-Δ)^β in spectral space:
    (-Δ)^β a = FFT^{-1}(|k|^(2β) * FFT(a))
    where |k| is the magnitude of the wave vector.

Example:
    >>> laplacian = FractionalLaplacian(domain, beta=1.5)
    >>> result = laplacian.apply(field)
"""

import numpy as np

# No additional typing imports needed

from ..domain import Domain


class FractionalLaplacian:
    """
    Fractional Laplacian operator for 7D phase field theory.

    Physical Meaning:
        Implements the fractional Laplacian (-Δ)^β that represents
        non-local interactions in phase field configurations.

    Mathematical Foundation:
        The fractional Laplacian (-Δ)^β is defined in spectral space as:
        (-Δ)^β a = FFT^{-1}(|k|^(2β) * FFT(a))
        where |k| is the magnitude of the wave vector and β ∈ (0,2).

    Attributes:
        domain (Domain): Computational domain.
        beta (float): Fractional order β ∈ (0,2).
        _spectral_coeffs (np.ndarray): Pre-computed spectral coefficients.
    """

    def __init__(self, domain: Domain, beta: float) -> None:
        """
        Initialize fractional Laplacian operator.

        Physical Meaning:
            Sets up the fractional Laplacian with the specified fractional
            order β for non-local phase field interactions.

        Args:
            domain (Domain): Computational domain for the operator.
            beta (float): Fractional order β ∈ (0,2).

        Raises:
            ValueError: If beta is not in valid range (0,2).
        """
        if not (0 < beta < 2):
            raise ValueError("Fractional order beta must be in (0,2)")

        self.domain = domain
        self.beta = beta
        self._spectral_coeffs: np.ndarray
        self._setup_spectral_coefficients()

    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for fractional Laplacian.

        Physical Meaning:
            Pre-computes the spectral representation |k|^(2β) of the
            fractional Laplacian for efficient application.

        Mathematical Foundation:
            Computes |k|^(2β) where |k| is the magnitude of the wave vector.
        """
        # Get 7D wave vectors for BVP theory
        kx = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        ky = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        kz = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        kphi1 = np.fft.fftfreq(self.domain.N_phi, 2 * np.pi / self.domain.N_phi)
        kphi2 = np.fft.fftfreq(self.domain.N_phi, 2 * np.pi / self.domain.N_phi)
        kphi3 = np.fft.fftfreq(self.domain.N_phi, 2 * np.pi / self.domain.N_phi)
        kt = np.fft.fftfreq(self.domain.N_t, self.domain.T / self.domain.N_t)

        # Create 7D meshgrids
        KX, KY, KZ, KPHI1, KPHI2, KPHI3, KT = np.meshgrid(
            kx, ky, kz, kphi1, kphi2, kphi3, kt, indexing="ij"
        )

        # Compute 7D wave vector magnitude
        k_magnitude = np.sqrt(
            KX**2 + KY**2 + KZ**2 + KPHI1**2 + KPHI2**2 + KPHI3**2 + KT**2
        )

        # Compute spectral coefficients |k|^(2β)
        self._spectral_coeffs = k_magnitude ** (2 * self.beta)

        # Handle k=0 mode (DC component) for 7D
        self._spectral_coeffs[0, 0, 0, 0, 0, 0, 0] = 0.0

    def apply(self, field: np.ndarray) -> np.ndarray:
        """
        Apply fractional Laplacian to field.

        Physical Meaning:
            Applies the fractional Laplacian (-Δ)^β to the field,
            computing the non-local fractional derivative.

        Mathematical Foundation:
            Computes (-Δ)^β a using spectral methods:
            (-Δ)^β a = FFT^{-1}(|k|^(2β) * FFT(a))

        Args:
            field (np.ndarray): Input field a(x).

        Returns:
            np.ndarray: Result of fractional Laplacian application (-Δ)^β a(x).

        Raises:
            ValueError: If field shape is incompatible with domain.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        # Transform to spectral space
        field_spectral = np.fft.fftn(field)

        # Apply spectral operator
        result_spectral = self._spectral_coeffs * field_spectral

        # Transform back to real space
        result = np.fft.ifftn(result_spectral)

        return result.real

    def get_spectral_coefficients(self) -> np.ndarray:
        """
        Get spectral coefficients of the fractional Laplacian.

        Physical Meaning:
            Returns the pre-computed spectral coefficients |k|^(2β) for
            the fractional Laplacian.

        Returns:
            np.ndarray: Spectral coefficients |k|^(2β).
        """
        return self._spectral_coeffs.copy()

    def get_fractional_order(self) -> float:
        """
        Get the fractional order of the Laplacian.

        Physical Meaning:
            Returns the fractional order β that determines the degree
            of non-locality in the operator.

        Returns:
            float: Fractional order β.
        """
        return self.beta

    def __repr__(self) -> str:
        """String representation of the fractional Laplacian."""
        return f"FractionalLaplacian(domain={self.domain}, beta={self.beta})"
