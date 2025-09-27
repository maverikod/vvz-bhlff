"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Domain class for BHLFF computational domain.

This module implements the computational domain for 7D phase field theory
simulations, providing grid management, coordinate systems, and boundary
condition handling.

Physical Meaning:
    The computational domain represents the 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    where phase field simulations are performed. It provides the spatial
    and temporal grid for numerical computations.

Mathematical Foundation:
    The domain implements periodic boundary conditions in a cubic region
    Ω = [0,L)³ with uniform grid spacing Δ = L/N, where N is the number
    of grid points per dimension.
"""

import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass


@dataclass
class Domain:
    """
    Computational domain for 7D phase field theory.

    Physical Meaning:
        Represents the computational space for phase field simulations
        in 7D space-time, providing grid management and coordinate systems
        for numerical computations.

    Mathematical Foundation:
        Implements a cubic domain Ω = [0,L)³ with periodic boundary
        conditions and uniform grid spacing Δ = L/N.

    Attributes:
        L (float): Domain size in each dimension.
        N (int): Number of grid points per dimension.
        dimensions (int): Number of spatial dimensions (typically 3).
        dx (float): Grid spacing Δ = L/N.
        shape (Tuple[int, ...]): Grid shape (N, N, N) for 3D.
        coordinates (np.ndarray): Grid coordinates.
    """

    L: float
    N: int
    dimensions: int = 3

    def __post_init__(self) -> None:
        """
        Initialize derived attributes after object creation.

        Physical Meaning:
            Computes grid spacing and coordinate arrays based on
            domain size and resolution parameters.
        """
        if self.L <= 0:
            raise ValueError("Domain size L must be positive")
        if self.N <= 0:
            raise ValueError("Number of grid points N must be positive")
        if self.dimensions not in [1, 2, 3]:
            raise ValueError("Dimensions must be 1, 2, or 3")

        self.dx = self.L / self.N
        self.shape = tuple([self.N] * self.dimensions)
        self._setup_coordinates()

    def _setup_coordinates(self) -> None:
        """
        Setup coordinate arrays for the domain.

        Physical Meaning:
            Creates coordinate arrays for grid points in the domain,
            used for source placement and field visualization.
        """
        if self.dimensions == 1:
            self.coordinates = np.linspace(0, self.L, self.N, endpoint=False)
        elif self.dimensions == 2:
            x = np.linspace(0, self.L, self.N, endpoint=False)
            y = np.linspace(0, self.L, self.N, endpoint=False)
            self.coordinates = np.meshgrid(x, y, indexing="ij")
        else:  # 3D
            x = np.linspace(0, self.L, self.N, endpoint=False)
            y = np.linspace(0, self.L, self.N, endpoint=False)
            z = np.linspace(0, self.L, self.N, endpoint=False)
            self.coordinates = np.meshgrid(x, y, z, indexing="ij")

    def get_wave_numbers(self) -> np.ndarray:
        """
        Get wave number arrays for FFT operations.

        Physical Meaning:
            Computes the wave number arrays k = (2π/L)m for FFT operations,
            where m ∈ ℤ³ are integer mode numbers.

        Mathematical Foundation:
            Wave numbers are defined as k = (2π/L)m for periodic boundary
            conditions, where m ∈ {-⌊N/2⌋, ..., ⌈N/2⌉-1}.

        Returns:
            np.ndarray: Wave number arrays for each dimension.
        """
        if self.dimensions == 1:
            return np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
        elif self.dimensions == 2:
            kx = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
            ky = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
            return np.meshgrid(kx, ky, indexing="ij")
        else:  # 3D
            kx = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
            ky = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
            kz = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
            return np.meshgrid(kx, ky, kz, indexing="ij")

    def get_center_index(self) -> Union[int, Tuple[int, ...]]:
        """
        Get the index of the domain center.

        Physical Meaning:
            Returns the grid index corresponding to the center of the domain,
            used for placing point sources and analyzing field patterns.

        Returns:
            Union[int, Tuple[int, ...]]: Center index or tuple of indices.
        """
        center = self.N // 2
        if self.dimensions == 1:
            return center
        else:
            return tuple([center] * self.dimensions)

    def get_volume(self) -> float:
        """
        Get the domain volume.

        Physical Meaning:
            Computes the total volume of the computational domain.

        Returns:
            float: Domain volume L^d where d is the number of dimensions.
        """
        return self.L**self.dimensions

    def get_grid_spacing(self) -> float:
        """
        Get the grid spacing.

        Physical Meaning:
            Returns the uniform grid spacing Δ = L/N.

        Returns:
            float: Grid spacing.
        """
        return self.dx

    def __repr__(self) -> str:
        """String representation of the domain."""
        return f"Domain(L={self.L}, N={self.N}, dimensions={self.dimensions})"
