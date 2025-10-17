"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unified spectral operations for 7D BHLFF Framework.

This module provides unified spectral operations combining functionality
from multiple FFT modules into a single comprehensive interface for
the 7D phase field theory.

Physical Meaning:
    Spectral operations implement mathematical operations in frequency space,
    providing efficient computation of FFT operations for 7D phase field
    calculations with U(1)³ phase structure.

Mathematical Foundation:
    Implements spectral operations including FFT operations for efficient
    computation in 7D frequency space:
    - 7D FFT: â(k_x, k_φ, k_t) = F[a(x, φ, t)]
    - Physics normalization: â(m) = Σ_x a(x) e^(-i k(m)·x) Δ^7
    - Orthogonal normalization: â(m) = (1/√N) Σ_x a(x) e^(-i k(m)·x)

Example:
    >>> ops = UnifiedSpectralOperations(domain, precision="float64")
    >>> spectral_field = ops.forward_fft(field, 'physics')
    >>> real_field = ops.inverse_fft(spectral_field, 'physics')
"""

import numpy as np
from typing import Any, Tuple, Dict, Optional
import logging

from typing import TYPE_CHECKING
from bhlff.utils.cuda_utils import get_global_backend

if TYPE_CHECKING:
    from ..domain import Domain


class UnifiedSpectralOperations:
    """
    Unified spectral operations for 7D phase field calculations.

    Physical Meaning:
        Implements mathematical operations in 7D frequency space, providing
        efficient computation of FFT operations for 7D phase field calculations
        with U(1)³ phase structure.

    Mathematical Foundation:
        Implements FFT operations with proper normalization for 7D computations:
        - Physics normalization: â(m) = Σ_x a(x) e^(-i k(m)·x) Δ^7
        - Orthogonal normalization: â(m) = (1/√N) Σ_x a(x) e^(-i k(m)·x)
        where Δ^7 = (dx^3) * (dphi^3) * dt is the 7D volume element.

    Attributes:
        domain (Domain): Computational domain for the simulation.
        precision (str): Numerical precision for computations.
        _fft_plans (Dict): Pre-computed FFT plans for efficiency.
    """

    def __init__(self, domain: "Domain", precision: str = "float64"):
        """
        Initialize unified spectral operations.

        Physical Meaning:
            Sets up the spectral operations calculator with the computational
            domain and numerical precision, initializing FFT backend and
            specialized calculators for derivatives and filtering.

        Args:
            domain (Domain): Computational domain with grid information.
            precision (str): Numerical precision ('float64' or 'float32').
        """
        self.domain = domain
        self.precision = precision
        self.logger = logging.getLogger(__name__)

        # Initialize CUDA backend for optimal performance
        self.backend = get_global_backend()
        self.logger.info(
            f"Using {type(self.backend).__name__} backend for spectral operations"
        )

        # Initialize FFT plans cache
        self._fft_plans = {}
        self._setup_fft_plans()

        self.logger.info(
            f"UnifiedSpectralOperations initialized for domain {domain.shape}"
        )

    def forward_fft(
        self, field: np.ndarray, normalization: str = "ortho"
    ) -> np.ndarray:
        """
        Compute forward FFT of field.

        Physical Meaning:
            Transforms the phase field from real space to frequency space,
            representing the field in terms of its frequency components.

        Mathematical Foundation:
            - Physics normalization: â(m) = Σ_x a(x) e^(-i k(m)·x) Δ^7
            - Orthogonal normalization: â(m) = (1/√N) Σ_x a(x) e^(-i k(m)·x)
            where Δ^7 = (dx^3) * (dphi^3) * dt is the 7D volume element.

        Args:
            field (np.ndarray): Field to transform a(x,φ,t).
            normalization (str): Normalization type ('physics' or 'ortho').

        Returns:
            np.ndarray: Spectral field â(k_x, k_φ, k_t).

        Raises:
            ValueError: If field shape is incompatible with domain or
                normalization type is unsupported.
        """
        if field.shape != self.domain.shape:
            raise ValueError(
                f"Field shape {field.shape} incompatible with domain {self.domain.shape}"
            )

        # Check if field is too large for GPU memory
        if self._is_field_too_large(field):
            self.logger.warning(f"Field too large for GPU memory, using CPU fallback")
            return self._forward_fft_cpu(field, normalization)
        
        # Use CUDA backend for FFT operations
        field_gpu = self.backend.array(field)

        if normalization == "ortho":
            # Use orthogonal normalization
            field_spectral_gpu = self.backend.fft(field_gpu)
            # Apply orthogonal normalization
            N_total = np.prod(self.domain.shape)
            field_spectral_gpu /= np.sqrt(N_total)
        elif normalization == "physics":
            # Use physics normalization
            field_spectral_gpu = self.backend.fft(field_gpu)
            # Apply physics normalization factor
            volume_element = self._compute_volume_element()
            field_spectral_gpu *= volume_element
        else:
            raise ValueError(f"Unsupported normalization type: {normalization}")

        # Convert back to CPU
        field_spectral = self.backend.to_numpy(field_spectral_gpu)
        return field_spectral
    
    def _is_field_too_large(self, field: np.ndarray) -> bool:
        """
        Check if field is too large for GPU memory.
        
        Physical Meaning:
            Large 7D fields can exceed GPU memory capacity,
            requiring CPU fallback for FFT operations.
            
        Args:
            field: Field to check.
            
        Returns:
            bool: True if field is too large for GPU.
        """
        try:
            # Estimate memory needed (4x field size for FFT)
            field_size_mb = field.nbytes / (1024**2)
            fft_memory_needed_mb = field_size_mb * 4
            
            # Check if we have enough GPU memory
            if hasattr(self.backend, 'get_memory_info'):
                memory_info = self.backend.get_memory_info()
                free_memory_mb = memory_info['free_memory'] / (1024**2)
                
                if fft_memory_needed_mb > free_memory_mb:
                    self.logger.warning(f"FFT requires {fft_memory_needed_mb:.1f} MB, "
                                      f"but only {free_memory_mb:.1f} MB available")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking field size: {e}")
            return True  # Fallback to CPU on error
    
    def _forward_fft_cpu(self, field: np.ndarray, normalization: str) -> np.ndarray:
        """
        CPU fallback for forward FFT.
        
        Physical Meaning:
            Performs FFT on CPU when GPU memory is insufficient,
            ensuring computation can continue with reduced performance.
            
        Args:
            field: Field to transform.
            normalization: Normalization type.
            
        Returns:
            np.ndarray: Spectral field.
        """
        import numpy.fft as np_fft
        
        # Perform FFT on CPU
        field_spectral = np_fft.fftn(field)
        
        if normalization == "ortho":
            # Apply orthogonal normalization
            N_total = np.prod(self.domain.shape)
            field_spectral /= np.sqrt(N_total)
        elif normalization == "physics":
            # Apply physics normalization factor
            volume_element = self._compute_volume_element()
            field_spectral *= volume_element
        
        return field_spectral
    
    def _inverse_fft_cpu(self, spectral_field: np.ndarray, normalization: str) -> np.ndarray:
        """
        CPU fallback for inverse FFT.
        
        Physical Meaning:
            Performs inverse FFT on CPU when GPU memory is insufficient,
            ensuring computation can continue with reduced performance.
            
        Args:
            spectral_field: Spectral field to transform.
            normalization: Normalization type.
            
        Returns:
            np.ndarray: Real space field.
        """
        import numpy.fft as np_fft
        
        if normalization == "ortho":
            # Apply orthogonal normalization
            N_total = np.prod(self.domain.shape)
            field_spectral = spectral_field * np.sqrt(N_total)
            field_real = np_fft.ifftn(field_spectral)
        elif normalization == "physics":
            # Apply physics normalization
            volume_element = self._compute_volume_element()
            field_spectral = spectral_field / volume_element
            field_real = np_fft.ifftn(field_spectral)
        
        return field_real.real

    def inverse_fft(
        self, spectral_field: np.ndarray, normalization: str = "ortho"
    ) -> np.ndarray:
        """
        Compute inverse FFT of spectral field.

        Physical Meaning:
            Transforms the spectral field from frequency space back to real space,
            reconstructing the original phase field.

        Mathematical Foundation:
            - Physics normalization: a(x) = (1/Δ^7) Σ_k â(k) e^(i k·x)
            - Orthogonal normalization: a(x) = (1/√N) Σ_k â(k) e^(i k·x)

        Args:
            spectral_field (np.ndarray): Spectral field â(k_x, k_φ, k_t).
            normalization (str): Normalization type ('physics' or 'ortho').

        Returns:
            np.ndarray: Real space field a(x,φ,t).

        Raises:
            ValueError: If spectral field shape is incompatible with domain or
                normalization type is unsupported.
        """
        if spectral_field.shape != self.domain.shape:
            raise ValueError(
                f"Spectral field shape {spectral_field.shape} incompatible with domain {self.domain.shape}"
            )

        # Check if field is too large for GPU memory
        if self._is_field_too_large(spectral_field):
            self.logger.warning(f"Spectral field too large for GPU memory, using CPU fallback")
            return self._inverse_fft_cpu(spectral_field, normalization)
        
        # Use CUDA backend for FFT operations
        spectral_field_gpu = self.backend.array(spectral_field)

        if normalization == "ortho":
            # Use orthogonal normalization
            field_spectral_gpu = spectral_field_gpu
            # Apply orthogonal normalization
            N_total = np.prod(self.domain.shape)
            field_spectral_gpu *= np.sqrt(N_total)
            field_real_gpu = self.backend.ifft(field_spectral_gpu)
        elif normalization == "physics":
            # Use physics normalization
            volume_element = self._compute_volume_element()
            field_spectral_gpu = spectral_field_gpu / volume_element
            field_real_gpu = self.backend.ifft(field_spectral_gpu)
        else:
            raise ValueError(f"Unsupported normalization type: {normalization}")

        # Convert back to CPU and return real part
        field_real = self.backend.to_numpy(field_real_gpu)
        return field_real.real

    def compute_spectral_derivatives(
        self, field: np.ndarray, order: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Compute spectral derivatives of field.

        Physical Meaning:
            Computes derivatives of the field using spectral methods,
            which are more accurate than finite difference methods
            for smooth fields.

        Mathematical Foundation:
            In spectral space: ∂/∂x → i k_x, ∂/∂y → i k_y, etc.
            Higher order derivatives: ∂ⁿ/∂xⁿ → (i k_x)ⁿ

        Args:
            field (np.ndarray): Field to differentiate.
            order (int): Order of derivative.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing derivatives in each direction.
        """
        # Transform to spectral space
        field_spectral = self.forward_fft(field, "ortho")

        # Get wave vectors
        k_vectors = self._get_wave_vectors()

        derivatives = {}
        for i, (axis, k_vec) in enumerate(
            zip(["x", "y", "z", "phi1", "phi2", "phi3", "t"], k_vectors)
        ):
            if i < len(field.shape):
                # Create wave vector grid for this axis
                k_grid = self._create_wave_vector_grid(k_vec, i, field.shape)

                # Compute derivative in spectral space
                derivative_spectral = (1j * k_grid) ** order * field_spectral

                # Transform back to real space
                derivatives[f"d{axis}"] = self.inverse_fft(derivative_spectral, "ortho")

        return derivatives

    def apply_spectral_filter(
        self, field: np.ndarray, filter_type: str, **kwargs
    ) -> np.ndarray:
        """
        Apply spectral filter to field.

        Physical Meaning:
            Applies various types of spectral filters to the field
            for noise reduction, smoothing, or feature extraction.

        Args:
            field (np.ndarray): Field to filter.
            filter_type (str): Type of filter ('lowpass', 'highpass', 'bandpass').
            **kwargs: Filter-specific parameters.

        Returns:
            np.ndarray: Filtered field.
        """
        # Transform to spectral space
        field_spectral = self.forward_fft(field, "ortho")

        # Apply filter
        if filter_type == "lowpass":
            cutoff = kwargs.get("cutoff", 0.5)
            filter_mask = self._create_lowpass_filter(cutoff)
        elif filter_type == "highpass":
            cutoff = kwargs.get("cutoff", 0.5)
            filter_mask = self._create_highpass_filter(cutoff)
        elif filter_type == "bandpass":
            low_cutoff = kwargs.get("low_cutoff", 0.3)
            high_cutoff = kwargs.get("high_cutoff", 0.7)
            filter_mask = self._create_bandpass_filter(low_cutoff, high_cutoff)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # Apply filter
        filtered_spectral = field_spectral * filter_mask

        # Transform back to real space
        return self.inverse_fft(filtered_spectral, "ortho")

    def compute_spectral_energy(self, field: np.ndarray) -> np.ndarray:
        """
        Compute spectral energy density.

        Physical Meaning:
            Computes the energy density in frequency space,
            representing the power spectrum of the field.

        Mathematical Foundation:
            E(k) = |â(k)|² where â(k) is the spectral field.

        Args:
            field (np.ndarray): Field to analyze.

        Returns:
            np.ndarray: Spectral energy density.
        """
        field_spectral = self.forward_fft(field, "ortho")
        return np.abs(field_spectral) ** 2

    def _setup_fft_plans(self) -> None:
        """
        Setup FFT plans for efficient computations.

        Physical Meaning:
            Pre-computes FFT plans to optimize the spectral
            transformations required for solving the phase field
            equation efficiently.
        """
        # For now, use numpy's built-in FFT which handles planning automatically
        # In a more advanced implementation, this could use FFTW or similar
        self._fft_plans = {
            "forward": "numpy_fftn",
            "inverse": "numpy_ifftn",
        }

    def _compute_volume_element(self) -> float:
        """
        Compute 7D volume element for physics normalization.

        Physical Meaning:
            Computes the volume element Δ^7 = (dx^3) * (dphi^3) * dt
            for proper physics normalization of FFT operations.

        Returns:
            float: Volume element.
        """
        # Simplified implementation - in practice this would depend on domain parameters
        dx = 1.0 / self.domain.shape[0] if len(self.domain.shape) > 0 else 1.0
        dy = 1.0 / self.domain.shape[1] if len(self.domain.shape) > 1 else dx
        dz = 1.0 / self.domain.shape[2] if len(self.domain.shape) > 2 else dy
        dphi1 = 2 * np.pi / self.domain.shape[3] if len(self.domain.shape) > 3 else 1.0
        dphi2 = 2 * np.pi / self.domain.shape[4] if len(self.domain.shape) > 4 else 1.0
        dphi3 = 2 * np.pi / self.domain.shape[5] if len(self.domain.shape) > 5 else 1.0
        dt = 1.0 / self.domain.shape[6] if len(self.domain.shape) > 6 else 1.0

        return dx * dy * dz * dphi1 * dphi2 * dphi3 * dt

    def _get_wave_vectors(self) -> list:
        """
        Get wave vectors for each dimension.

        Returns:
            list: List of wave vectors for each dimension.
        """
        k_vectors = []
        for i, n in enumerate(self.domain.shape):
            if i < 3:  # Spatial dimensions
                k = np.fft.fftfreq(n, 1.0 / n)
            elif i < 6:  # Phase dimensions
                k = np.fft.fftfreq(n, 2 * np.pi / n)
            else:  # Temporal dimension
                k = np.fft.fftfreq(n, 1.0 / n)
            k_vectors.append(k)
        return k_vectors

    def _create_wave_vector_grid(
        self, k_vec: np.ndarray, axis: int, shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Create wave vector grid for a specific axis.

        Args:
            k_vec (np.ndarray): Wave vector for the axis.
            axis (int): Axis index.
            shape (Tuple[int, ...]): Field shape.

        Returns:
            np.ndarray: Wave vector grid.
        """
        # Create meshgrid with wave vector in the specified axis
        indices = [slice(None)] * len(shape)
        indices[axis] = k_vec

        # Create grid
        grids = np.meshgrid(
            *[np.arange(s) if i != axis else k_vec for i, s in enumerate(shape)],
            indexing="ij",
        )
        return grids[axis]

    def _create_lowpass_filter(self, cutoff: float) -> np.ndarray:
        """Create lowpass filter mask."""
        # Simplified implementation
        k_vectors = self._get_wave_vectors()
        k_magnitude = np.sqrt(sum(k**2 for k in k_vectors))
        return np.where(k_magnitude <= cutoff, 1.0, 0.0)

    def _create_highpass_filter(self, cutoff: float) -> np.ndarray:
        """Create highpass filter mask."""
        # Simplified implementation
        k_vectors = self._get_wave_vectors()
        k_magnitude = np.sqrt(sum(k**2 for k in k_vectors))
        return np.where(k_magnitude >= cutoff, 1.0, 0.0)

    def _create_bandpass_filter(
        self, low_cutoff: float, high_cutoff: float
    ) -> np.ndarray:
        """Create bandpass filter mask."""
        # Simplified implementation
        k_vectors = self._get_wave_vectors()
        k_magnitude = np.sqrt(sum(k**2 for k in k_vectors))
        return np.where(
            (k_magnitude >= low_cutoff) & (k_magnitude <= high_cutoff), 1.0, 0.0
        )

    def get_spectral_info(self) -> Dict[str, Any]:
        """
        Get information about spectral operations.

        Physical Meaning:
            Returns information about the spectral operations setup
            for monitoring and analysis purposes.

        Returns:
            Dict[str, Any]: Spectral operations information.
        """
        return {
            "domain_shape": self.domain.shape,
            "precision": self.precision,
            "volume_element": self._compute_volume_element(),
            "fft_plans": self._fft_plans,
            "wave_vectors": [len(k) for k in self._get_wave_vectors()],
        }

    def __repr__(self) -> str:
        """String representation of spectral operations."""
        return (
            f"UnifiedSpectralOperations("
            f"domain={self.domain.shape}, "
            f"precision={self.precision})"
        )
