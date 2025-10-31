"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA-optimized compute processor for Level C computations.

This module provides CUDA-accelerated block-based processing for Level C
computations with automatic GPU memory management and vectorized operations.

Physical Meaning:
    Provides GPU-accelerated computations for Level C boundary and cell analysis,
    enabling efficient processing of large 7D phase field data with maximum
    performance through CUDA vectorization and optimized block processing.

Mathematical Foundation:
    Implements CUDA-accelerated computations with block-based processing:
    - Block size: optimized for 80% of available GPU memory
    - Vectorized operations: all array operations use GPU kernels
    - Memory-efficient: blocks processed sequentially to fit in GPU memory

Example:
    >>> processor = LevelCCUDAProcessor(bvp_core)
    >>> result = processor.compute_admittance_vectorized(field, source, frequencies)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    import cupyx.scipy.fft as cp_fft

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    cp_ndimage = None
    cp_fft = None

from bhlff.core.bvp import BVPCore
from bhlff.utils.cuda_utils import get_optimal_backend, CUDABackend, CPUBackend


class LevelCCUDAProcessor:
    """
    CUDA-optimized compute processor for Level C computations.

    Physical Meaning:
        Provides GPU-accelerated computations for Level C boundary and cell
        analysis with automatic memory management and vectorized operations,
        enabling efficient processing of large 7D phase field data.

    Mathematical Foundation:
        Implements CUDA-accelerated block-based processing:
        - Block size: optimized for 80% of available GPU memory
        - Vectorized operations: all array operations use GPU kernels
        - Memory-efficient: sequential block processing
    """

    def __init__(self, bvp_core: BVPCore, use_cuda: bool = True):
        """
        Initialize CUDA processor for Level C computations.

        Physical Meaning:
            Sets up GPU-accelerated computation system with automatic
            memory management and optimized block processing.

        Args:
            bvp_core (BVPCore): BVP core framework instance.
            use_cuda (bool): Whether to use CUDA acceleration.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        # Initialize backend
        if self.use_cuda:
            try:
                self.backend = get_optimal_backend()
                self.cuda_available = isinstance(self.backend, CUDABackend)
            except Exception as e:
                self.logger.warning(f"CUDA initialization failed: {e}, using CPU")
                self.backend = CPUBackend()
                self.cuda_available = False
        else:
            self.backend = CPUBackend()
            self.cuda_available = False

        # Compute optimal block size
        self.block_size = self._compute_optimal_block_size()
        self.logger.info(
            f"Level C CUDA processor initialized: "
            f"CUDA={self.cuda_available}, block_size={self.block_size}"
        )

    def _compute_optimal_block_size(self) -> int:
        """
        Compute optimal block size based on GPU memory (80% of available).

        Physical Meaning:
            Calculates block size to use 80% of available GPU memory,
            ensuring efficient memory usage while avoiding OOM errors.

        Returns:
            int: Optimal block size per dimension.
        """
        if not self.cuda_available:
            return 8  # Default CPU block size

        try:
            # Get GPU memory info
            if isinstance(self.backend, CUDABackend):
                mem_info = self.backend.get_memory_info()
                free_memory_bytes = mem_info["free_memory"]
                # Use 80% of free memory
                available_memory_bytes = int(free_memory_bytes * 0.8)
            else:
                return 8

            # Memory per element (complex128 = 16 bytes)
            bytes_per_element = 16

            # For 7D computations, we need space for:
            # - Input field: 1x
            # - FFT workspace: 3x (forward, intermediate, backward)
            # - Intermediate results: 2x (amplitude, phase)
            # - Boundary masks: 1x
            # Total overhead: ~7x
            overhead_factor = 7

            # Maximum elements per block
            max_elements = available_memory_bytes // (
                bytes_per_element * overhead_factor
            )

            # For 7D, calculate block size per dimension
            # Assuming roughly equal dimensions
            elements_per_dim = int(max_elements ** (1.0 / 7.0))

            # Ensure reasonable bounds (4 to 128)
            block_size = max(4, min(elements_per_dim, 128))

            self.logger.info(
                f"Optimal block size: {block_size} "
                f"(available GPU memory: {available_memory_bytes / 1e9:.2f} GB, "
                f"using 80%)"
            )

            return block_size

        except Exception as e:
            self.logger.warning(
                f"Failed to compute optimal block size: {e}, using default 8"
            )
            return 8

    def compute_admittance_vectorized(
        self,
        field: np.ndarray,
        source: np.ndarray,
        frequencies: np.ndarray,
        domain: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute admittance spectrum using vectorized CUDA operations.

        Physical Meaning:
            Computes admittance Y(ω) = I(ω)/V(ω) for all frequencies
            using GPU-accelerated vectorized operations.

        Mathematical Foundation:
            Y(ω) = ∫ a*(x) s(x) dV / ∫ |a(x)|² dV
            Computed for all frequencies simultaneously using vectorization.

        Args:
            field (np.ndarray): Field data.
            source (np.ndarray): Source field.
            frequencies (np.ndarray): Frequencies to compute.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            np.ndarray: Admittance values for each frequency (complex).
        """
        if self.cuda_available:
            return self._compute_admittance_cuda(field, source, frequencies, domain)
        else:
            return self._compute_admittance_cpu(field, source, frequencies, domain)

    def _compute_admittance_cuda(
        self,
        field: np.ndarray,
        source: np.ndarray,
        frequencies: np.ndarray,
        domain: Dict[str, Any],
    ) -> np.ndarray:
        """Compute admittance using CUDA acceleration."""
        # Transfer to GPU
        field_gpu = self.backend.array(field)
        source_gpu = self.backend.array(source)

        # Compute admittance for all frequencies using vectorized operations
        num_freqs = len(frequencies)
        admittances_gpu = self.backend.zeros(num_freqs, dtype=np.complex128)

        # Process in blocks if field is large
        if field.size > self.block_size**7:
            # Block-based processing
            for i, omega in enumerate(frequencies):
                admittance = self._compute_admittance_blocked_cuda(
                    field_gpu, source_gpu, omega, domain
                )
                admittances_gpu[i] = admittance
        else:
            # Process all frequencies at once
            admittances_gpu = self._compute_admittance_all_freqs_cuda(
                field_gpu, source_gpu, frequencies, domain
            )

        # Transfer back to CPU
        return self.backend.to_numpy(admittances_gpu)

    def _compute_admittance_blocked_cuda(
        self,
        field_gpu: "cp.ndarray",
        source_gpu: "cp.ndarray",
        omega: float,
        domain: Dict[str, Any],
    ) -> "cp.ndarray":
        """Compute admittance for single frequency using blocked CUDA processing."""
        # Compute field amplitude
        field_abs_sq = cp.abs(field_gpu) ** 2

        # Compute source-field correlation
        correlation = cp.conj(field_gpu) * source_gpu

        # Compute integrals using block-based reduction
        numerator = self._block_reduce_sum(correlation, self.block_size)
        denominator = self._block_reduce_sum(field_abs_sq, self.block_size)

        # Compute admittance
        if abs(denominator) > 1e-12:
            admittance = numerator / denominator
        else:
            admittance = cp.complex128(0.0)

        return admittance

    def _compute_admittance_all_freqs_cuda(
        self,
        field_gpu: "cp.ndarray",
        source_gpu: "cp.ndarray",
        frequencies: np.ndarray,
        domain: Dict[str, Any],
    ) -> "cp.ndarray":
        """Compute admittance for all frequencies using vectorized CUDA operations."""
        num_freqs = len(frequencies)

        # Compute field amplitude (shared for all frequencies)
        field_abs_sq = cp.abs(field_gpu) ** 2
        denominator = cp.sum(field_abs_sq)

        # For each frequency, compute numerator
        admittances = cp.zeros(num_freqs, dtype=cp.complex128)

        for i, omega in enumerate(frequencies):
            # Frequency-dependent field modulation
            phase_factor = cp.exp(1j * omega * domain.get("t", 0.0))
            field_modulated = field_gpu * phase_factor

            # Compute correlation
            correlation = cp.conj(field_modulated) * source_gpu
            numerator = cp.sum(correlation)

            # Compute admittance
            if abs(denominator) > 1e-12:
                admittances[i] = numerator / denominator

        return admittances

    def _compute_admittance_cpu(
        self,
        field: np.ndarray,
        source: np.ndarray,
        frequencies: np.ndarray,
        domain: Dict[str, Any],
    ) -> np.ndarray:
        """Compute admittance using CPU fallback."""
        num_freqs = len(frequencies)
        admittances = np.zeros(num_freqs, dtype=np.complex128)

        field_abs_sq = np.abs(field) ** 2
        denominator = np.sum(field_abs_sq)

        for i, omega in enumerate(frequencies):
            phase_factor = np.exp(1j * omega * domain.get("t", 0.0))
            field_modulated = field * phase_factor
            correlation = np.conj(field_modulated) * source
            numerator = np.sum(correlation)

            if abs(denominator) > 1e-12:
                admittances[i] = numerator / denominator

        return admittances

    def compute_radial_profile_vectorized(
        self,
        field: np.ndarray,
        center: np.ndarray,
        radii: np.ndarray,
        domain: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute radial profile using vectorized CUDA operations.

        Physical Meaning:
            Computes radial profile A(r) = (1/4π) ∫_S(r) |a(x)|² dS
            for all radii using GPU-accelerated vectorized operations.

        Mathematical Foundation:
            A(r) = (1/4π) ∫_S(r) |a(x)|² dS
            Computed for all radii simultaneously using vectorization.

        Args:
            field (np.ndarray): Field data.
            center (np.ndarray): Center point for radial profile.
            radii (np.ndarray): Radii to compute profile.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            np.ndarray: Radial profile amplitudes for each radius.
        """
        if self.cuda_available:
            return self._compute_radial_profile_cuda(field, center, radii, domain)
        else:
            return self._compute_radial_profile_cpu(field, center, radii, domain)

    def _compute_radial_profile_cuda(
        self,
        field: np.ndarray,
        center: np.ndarray,
        domain: Dict[str, Any],
        radii: np.ndarray,
    ) -> np.ndarray:
        """Compute radial profile using CUDA acceleration."""
        # Transfer to GPU
        field_gpu = self.backend.array(field)
        center_gpu = self.backend.array(center)
        radii_gpu = self.backend.array(radii)

        # Create coordinate arrays on GPU
        N = field.shape[0]
        L = domain.get("L", 1.0)
        x = cp.linspace(0, L, N)
        y = cp.linspace(0, L, N)
        z = cp.linspace(0, L, N)
        X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")

        # Compute distance from center (vectorized)
        dX = X - center_gpu[0]
        dY = Y - center_gpu[1]
        dZ = Z - center_gpu[2]
        distances = cp.sqrt(dX**2 + dY**2 + dZ**2)

        # Compute field amplitude squared
        field_abs_sq = cp.abs(field_gpu) ** 2

        # Compute profile for each radius
        num_radii = len(radii)
        amplitudes = cp.zeros(num_radii, dtype=cp.float64)

        shell_thickness = L / (2 * N)

        for i, r in enumerate(radii_gpu):
            # Create shell mask (vectorized)
            shell_mask = (distances >= (r - shell_thickness)) & (
                distances <= (r + shell_thickness)
            )

            # Compute amplitude in shell (vectorized reduction)
            shell_values = field_abs_sq[shell_mask]
            if shell_values.size > 0:
                amplitudes[i] = cp.sqrt(cp.mean(shell_values))
            else:
                amplitudes[i] = 0.0

        return self.backend.to_numpy(amplitudes)

    def _compute_radial_profile_cpu(
        self,
        field: np.ndarray,
        center: np.ndarray,
        radii: np.ndarray,
        domain: Dict[str, Any],
    ) -> np.ndarray:
        """Compute radial profile using CPU fallback."""
        N = field.shape[0]
        L = domain.get("L", 1.0)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        dX = X - center[0]
        dY = Y - center[1]
        dZ = Z - center[2]
        distances = np.sqrt(dX**2 + dY**2 + dZ**2)

        field_abs_sq = np.abs(field) ** 2
        num_radii = len(radii)
        amplitudes = np.zeros(num_radii)

        shell_thickness = L / (2 * N)

        for i, r in enumerate(radii):
            shell_mask = (distances >= (r - shell_thickness)) & (
                distances <= (r + shell_thickness)
            )
            shell_values = field_abs_sq[shell_mask]
            if shell_values.size > 0:
                amplitudes[i] = np.sqrt(np.mean(shell_values))
            else:
                amplitudes[i] = 0.0

        return amplitudes

    def _block_reduce_sum(self, array: "cp.ndarray", block_size: int) -> "cp.ndarray":
        """
        Compute sum using block-based reduction on GPU.

        Physical Meaning:
            Computes sum of large array using block-based reduction
            to fit within GPU memory constraints.

        Args:
            array (cp.ndarray): Array to sum.
            block_size (int): Block size for reduction.

        Returns:
            cp.ndarray: Sum value.
        """
        if array.size <= block_size**7:
            return cp.sum(array)

        # Block-based reduction
        total = cp.complex128(0.0)
        flat_array = array.flatten()

        num_blocks = (len(flat_array) + block_size**7 - 1) // (block_size**7)

        for i in range(num_blocks):
            start_idx = i * block_size**7
            end_idx = min(start_idx + block_size**7, len(flat_array))
            block = flat_array[start_idx:end_idx]
            total += cp.sum(block)

        return total

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if self.cuda_available and isinstance(self.backend, CUDABackend):
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                self.logger.warning(f"GPU memory cleanup failed: {e}")
