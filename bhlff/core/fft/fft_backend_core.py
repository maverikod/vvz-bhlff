"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

FFT backend core implementation.

This module provides the core FFT backend for efficient spectral operations
in the 7D phase field theory.

Physical Meaning:
    FFT backend implements the computational engine for spectral methods,
    providing efficient transformation between real and frequency space
    for phase field calculations.

Mathematical Foundation:
    Implements Fast Fourier Transform operations for efficient computation
    of spectral methods in phase field equations.

Example:
    >>> backend = FFTBackend(domain, plan_type="MEASURE")
    >>> spectral_data = backend.fft(real_data)
    >>> real_data = backend.ifft(spectral_data)
"""

import numpy as np
from typing import Tuple, Dict, Any

from ..domain import Domain
from .fft_plan_manager import FFTPlanManager
from .fft_twiddle_computer import FFTTwiddleComputer
from .fft_butterfly_computer import FFTButterflyComputer


class FFTBackend:
    """
    FFT backend for spectral operations.

    Physical Meaning:
        Provides the computational backend for Fast Fourier Transform
        operations, enabling efficient spectral methods for phase field
        calculations.

    Mathematical Foundation:
        Implements FFT operations for transforming between real space
        and frequency space, enabling efficient computation of spectral
        methods in phase field equations.

    Attributes:
        domain (Domain): Computational domain.
        plan_type (str): FFT planning strategy.
        _fft_plans (Dict): Pre-computed FFT plans.
    """

    def __init__(
        self,
        domain: Domain,
        plan_type: str = "MEASURE",
        precision: str = "float64",
    ) -> None:
        """
        Initialize FFT backend.

        Physical Meaning:
            Sets up the FFT backend with specified planning strategy
            and precision for efficient spectral operations.

        Args:
            domain (Domain): Computational domain for FFT operations.
            plan_type (str): FFT planning strategy ("ESTIMATE", "MEASURE",
                "PATIENT", "EXHAUSTIVE").
            precision (str): Numerical precision ("float32", "float64").

        Raises:
            ValueError: If plan_type or precision is not supported.
        """
        self.domain = domain
        self.plan_type = plan_type
        self.precision = precision
        
        # Initialize component managers
        self._plan_manager = FFTPlanManager(domain, plan_type, precision)
        self._twiddle_computer = FFTTwiddleComputer(domain, precision)
        self._butterfly_computer = FFTButterflyComputer(domain)
        
        # Setup FFT plans and pre-compute factors
        self._plan_manager.setup_fft_plans()
        self._twiddle_computer.precompute_twiddle_factors()
        
        # Setup memory pools for efficient allocation
        self._setup_memory_pools()

    def _setup_memory_pools(self) -> None:
        """
        Setup memory pools for efficient allocation.
        
        Physical Meaning:
            Creates memory pools for efficient allocation and deallocation
            of temporary arrays during FFT operations.
        """
        self._memory_pools = {
            "temp_arrays": [],
            "workspace_arrays": [],
            "cache_size": 10,  # Number of arrays to keep in cache
        }

    def fft(self, real_data: np.ndarray) -> np.ndarray:
        """
        Compute forward FFT.

        Physical Meaning:
            Transforms real space data to frequency space using Fast
            Fourier Transform.

        Mathematical Foundation:
            Computes FFT: â(k) = FFT(a(x)) where a(x) is real space data
            and â(k) is frequency space data.

        Args:
            real_data (np.ndarray): Real space data a(x).

        Returns:
            np.ndarray: Frequency space data â(k).

        Raises:
            ValueError: If data shape is incompatible with domain.
        """
        if real_data.shape != self.domain.shape:
            raise ValueError(
                f"Data shape {real_data.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        # Compute forward FFT
        if self.domain.dimensions == 1:
            spectral_data = np.fft.fft(real_data)
        elif self.domain.dimensions == 2:
            spectral_data = np.fft.fft2(real_data)
        else:  # 3D
            spectral_data = np.fft.fftn(real_data)

        return spectral_data

    def ifft(self, spectral_data: np.ndarray) -> np.ndarray:
        """
        Compute inverse FFT.

        Physical Meaning:
            Transforms frequency space data back to real space using
            inverse Fast Fourier Transform.

        Mathematical Foundation:
            Computes IFFT: a(x) = IFFT(â(k)) where â(k) is frequency space
            data and a(x) is real space data.

        Args:
            spectral_data (np.ndarray): Frequency space data â(k).

        Returns:
            np.ndarray: Real space data a(x).

        Raises:
            ValueError: If data shape is incompatible with domain.
        """
        if spectral_data.shape != self.domain.shape:
            raise ValueError(
                f"Data shape {spectral_data.shape} incompatible with "
                f"domain shape {self.domain.shape}"
            )

        # Compute inverse FFT
        if self.domain.dimensions == 1:
            real_data = np.fft.ifft(spectral_data)
        elif self.domain.dimensions == 2:
            real_data = np.fft.ifft2(spectral_data)
        else:  # 3D
            real_data = np.fft.ifftn(spectral_data)

        return real_data

    def fft_shift(self, spectral_data: np.ndarray) -> np.ndarray:
        """
        Shift FFT data to center zero frequency.

        Physical Meaning:
            Shifts the FFT data so that zero frequency is at the center
            of the array, which is useful for visualization and analysis.

        Mathematical Foundation:
            Applies fftshift to move zero frequency to the center:
            â_shifted(k) = fftshift(â(k))

        Args:
            spectral_data (np.ndarray): Frequency space data â(k).

        Returns:
            np.ndarray: Shifted frequency space data â_shifted(k).
        """
        if self.domain.dimensions == 1:
            return np.fft.fftshift(spectral_data)
        elif self.domain.dimensions == 2:
            return np.fft.fftshift(spectral_data)
        else:  # 3D
            return np.fft.fftshift(spectral_data)

    def ifft_shift(self, spectral_data: np.ndarray) -> np.ndarray:
        """
        Inverse shift FFT data.

        Physical Meaning:
            Applies inverse fftshift to restore the original frequency
            ordering of the FFT data.

        Mathematical Foundation:
            Applies ifftshift to restore original frequency ordering:
            â(k) = ifftshift(â_shifted(k))

        Args:
            spectral_data (np.ndarray): Shifted frequency space data â_shifted(k).

        Returns:
            np.ndarray: Original frequency space data â(k).
        """
        if self.domain.dimensions == 1:
            return np.fft.ifftshift(spectral_data)
        elif self.domain.dimensions == 2:
            return np.fft.ifftshift(spectral_data)
        else:  # 3D
            return np.fft.ifftshift(spectral_data)

    def get_frequency_arrays(self) -> Tuple[np.ndarray, ...]:
        """
        Get frequency arrays for the domain.

        Physical Meaning:
            Returns the frequency arrays corresponding to the computational
            domain for spectral analysis.

        Mathematical Foundation:
            Computes frequency arrays using fftfreq:
            k = 2π * fftfreq(N, dx)

        Returns:
            Tuple[np.ndarray, ...]: Frequency arrays for each dimension.
        """
        dx = self.domain.dx

        if self.domain.dimensions == 1:
            kx = 2 * np.pi * np.fft.fftfreq(self.domain.N, dx)
            return (kx,)
        elif self.domain.dimensions == 2:
            kx = 2 * np.pi * np.fft.fftfreq(self.domain.N, dx)
            ky = 2 * np.pi * np.fft.fftfreq(self.domain.N, dx)
            return (kx, ky)
        else:  # 3D
            kx = 2 * np.pi * np.fft.fftfreq(self.domain.N, dx)
            ky = 2 * np.pi * np.fft.fftfreq(self.domain.N, dx)
            kz = 2 * np.pi * np.fft.fftfreq(self.domain.N, dx)
            return (kx, ky, kz)

    def get_plan_type(self) -> str:
        """
        Get the FFT plan type.

        Physical Meaning:
            Returns the FFT planning strategy being used.

        Returns:
            str: FFT plan type.
        """
        return self._plan_manager.get_plan_type()

    def get_precision(self) -> str:
        """
        Get the numerical precision.

        Physical Meaning:
            Returns the numerical precision being used for FFT operations.

        Returns:
            str: Numerical precision.
        """
        return self._plan_manager.get_precision()

    def __repr__(self) -> str:
        """String representation of the FFT backend."""
        return (
            f"FFTBackend(domain={self.domain}, "
            f"plan_type={self.plan_type}, precision={self.precision})"
        )
