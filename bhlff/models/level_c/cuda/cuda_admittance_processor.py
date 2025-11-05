"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA-optimized admittance processor for Level C computations with 7D geometry.

This module provides CUDA-accelerated admittance computation functionality
for Level C boundary analysis with 7D phase field support, optimized block
processing, and GPU memory management preserving 7D structure.

Physical Meaning:
    Computes admittance Y(ω) = I(ω)/V(ω) for boundary analysis in 7D space-time
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, enabling efficient frequency-domain analysis of
    boundary effects while preserving the 7D geometric structure.

Mathematical Foundation:
    Implements admittance computation in 7D:
    Y(ω) = ∫ a*(x,φ,t) s(x,φ,t) dV₇ / ∫ |a(x,φ,t)|² dV₇
    where dV₇ = d³x d³φ dt is the 7D volume element.
    All operations preserve 7D structure with axis-wise reductions.

Theoretical Background:
    The 7D phase field theory operates in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, where
    spatial coordinates x ∈ ℝ³, phase coordinates φ ∈ 𝕋³, and time t ∈ ℝ.
    Block processing preserves this structure by computing optimal 7D tiling
    that uses 80% of GPU memory with proper axis-wise reductions.

Example:
    >>> processor = AdmittanceProcessor(backend, block_size)
    >>> admittances = processor.compute_vectorized(field, source, frequencies, domain)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class AdmittanceProcessor:
    """
    CUDA-optimized admittance processor for Level C computations with 7D geometry.

    Physical Meaning:
        Provides GPU-accelerated admittance computation for boundary analysis
        in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, preserving 7D geometric structure
        through axis-wise reductions and optimal block tiling.

    Mathematical Foundation:
        Computes admittance Y(ω) = I(ω)/V(ω) using vectorized operations with
        block-preserving reductions that maintain 7D structure. All operations
        use axis-wise reductions on GPU without flattening, ensuring geometric
        consistency in 7D space-time.

    Attributes:
        backend (CUDABackend): CUDA backend for GPU operations.
        block_size (int): Default block size per dimension.
        cuda_available (bool): Whether CUDA is available.
        logger (logging.Logger): Logger instance.
        _optimal_block_tiling (Tuple[int, ...]): Optimal 7D block tiling.
    """

    def __init__(self, backend: Any, block_size: int, cuda_available: bool):
        """
        Initialize admittance processor.

        Physical Meaning:
            Sets up GPU-accelerated admittance processor with 7D geometry support,
            computing optimal block tiling for 80% GPU memory usage.

        Args:
            backend (CUDABackend): CUDA backend for GPU operations.
            block_size (int): Default block size per dimension.
            cuda_available (bool): Whether CUDA is available.
        """
        self.backend = backend
        self.block_size = block_size
        self.cuda_available = cuda_available
        self.logger = logging.getLogger(__name__)
        self._optimal_block_tiling: Optional[Tuple[int, ...]] = None

    def compute_vectorized(
        self,
        field: np.ndarray,
        source: np.ndarray,
        frequencies: np.ndarray,
        domain: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute admittance spectrum using vectorized operations.

        Physical Meaning:
            Computes admittance Y(ω) = I(ω)/V(ω) for all frequencies
            using GPU-accelerated vectorized operations.

        Args:
            field (np.ndarray): Field data.
            source (np.ndarray): Source field.
            frequencies (np.ndarray): Frequencies to compute.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            np.ndarray: Admittance values for each frequency (complex).
        """
        # CUDA is required for Level C - no fallback
        if not self.cuda_available:
            raise RuntimeError("CUDA not available - Level C requires GPU")
        return self._compute_cuda(field, source, frequencies, domain)

    def _compute_cuda(
        self,
        field: np.ndarray,
        source: np.ndarray,
        frequencies: np.ndarray,
        domain: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute admittance using CUDA acceleration with 7D geometry preservation.

        Physical Meaning:
            Computes admittance Y(ω) = I(ω)/V(ω) in 7D space-time M₇,
            preserving geometric structure through block-preserving reductions.

        Mathematical Foundation:
            Performs axis-wise reductions on GPU without flattening, ensuring
            all operations maintain 7D structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Args:
            field (np.ndarray): 7D phase field a(x,φ,t).
            source (np.ndarray): 7D source field s(x,φ,t).
            frequencies (np.ndarray): Frequencies ω to compute.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            np.ndarray: Admittance values for each frequency (complex).
        """
        # CUDA is required - verify backend is CUDA
        if not self.cuda_available:
            raise RuntimeError("CUDA not available - Level C requires GPU")

        # Verify backend is CUDA (not CPU)
        from bhlff.utils.cuda_utils import CUDABackend

        if not isinstance(self.backend, CUDABackend):
            raise RuntimeError(
                f"Backend is not CUDA! Got {type(self.backend).__name__}. "
                f"Level C requires GPU acceleration."
            )

        self.logger.info(
            f"Computing admittance on GPU: field shape={field.shape}, "
            f"num_frequencies={len(frequencies)}, ndim={field.ndim}"
        )

        # Transfer to GPU
        field_gpu = self.backend.array(field)
        source_gpu = self.backend.array(source)

        # Verify arrays are on GPU
        if not isinstance(field_gpu, cp.ndarray):
            raise RuntimeError(f"Field not on GPU! Type: {type(field_gpu)}")
        if not isinstance(source_gpu, cp.ndarray):
            raise RuntimeError(f"Source not on GPU! Type: {type(source_gpu)}")

        # Synchronize to ensure GPU transfers complete
        cp.cuda.Stream.null.synchronize()

        self.logger.info(
            f"Arrays transferred to GPU: field={field_gpu.shape}, "
            f"source={source_gpu.shape}, ndim={field_gpu.ndim}"
        )

        # Compute optimal 7D block tiling for 80% GPU memory
        block_tiling = self._compute_optimal_7d_block_tiling(field_gpu)
        self._optimal_block_tiling = block_tiling

        # Check memory requirements before processing
        field_size_bytes = field_gpu.nbytes
        # Memory overhead: field + source + field_abs_sq + correlation + intermediates
        overhead_factor = 8  # Conservative for 7D operations
        required_memory = field_size_bytes * overhead_factor

        # Get available GPU memory (80% usage)
        mem_info = cp.cuda.runtime.memGetInfo()
        available_memory = mem_info[0]
        safe_memory = int(available_memory * 0.8)  # 80% as required

        if required_memory > safe_memory:
            self.logger.info(
                f"Field too large for direct processing: "
                f"{required_memory/1e9:.2f}GB required, "
                f"{safe_memory/1e9:.2f}GB available (80%). Using 7D block processing."
            )
            # Block-based processing preserving 7D structure
            num_freqs = len(frequencies)
            admittances_gpu = self.backend.zeros(num_freqs, dtype=np.complex128)
            for i, omega in enumerate(frequencies):
                admittance = self._compute_blocked_cuda(
                    field_gpu, source_gpu, omega, domain, block_tiling
                )
                admittances_gpu[i] = admittance
        else:
            # Process all frequencies at once with 7D structure
            num_freqs = len(frequencies)
            admittances_gpu = self._compute_all_freqs_cuda(
                field_gpu, source_gpu, frequencies, domain
            )

        # Synchronize before transfer
        cp.cuda.Stream.null.synchronize()

        # Transfer back to CPU
        return self.backend.to_numpy(admittances_gpu)

    def _compute_blocked_cuda(
        self,
        field_gpu: "cp.ndarray",
        source_gpu: "cp.ndarray",
        omega: float,
        domain: Dict[str, Any],
        block_tiling: Tuple[int, ...],
    ) -> "cp.ndarray":
        """
        Compute admittance for single frequency using 7D block-preserving CUDA processing.

        Physical Meaning:
            Computes admittance Y(ω) using block-based processing that preserves
            7D structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, performing axis-wise reductions
            on GPU without flattening.

        Mathematical Foundation:
            Processes 7D blocks with shape determined by optimal tiling,
            performing reductions along all axes to compute:
            numerator = Σ a*(x,φ,t) s(x,φ,t) over all 7D blocks
            denominator = Σ |a(x,φ,t)|² over all 7D blocks

        Args:
            field_gpu (cp.ndarray): 7D phase field on GPU.
            source_gpu (cp.ndarray): 7D source field on GPU.
            omega (float): Frequency ω.
            domain (Dict[str, Any]): Domain parameters.
            block_tiling (Tuple[int, ...]): Optimal 7D block tiling per dimension.

        Returns:
            cp.ndarray: Admittance value (complex scalar).
        """
        # Verify 7D structure
        if field_gpu.ndim != 7:
            raise ValueError(
                f"Expected 7D field, got {field_gpu.ndim}D. "
                f"Shape: {field_gpu.shape}"
            )

        if len(block_tiling) != 7:
            raise ValueError(
                f"Block tiling must have 7 dimensions, got {len(block_tiling)}"
            )

        self.logger.debug(
            f"7D block processing: field shape={field_gpu.shape}, "
            f"block tiling={block_tiling}"
        )

        # Initialize reduction accumulators on GPU
        numerator_sum = cp.complex128(0.0)
        denominator_sum = cp.complex128(0.0)

        # Get field shape
        shape = field_gpu.shape

        # Process 7D blocks with nested loops preserving structure
        for i0 in range(0, shape[0], block_tiling[0]):
            i0_end = min(i0 + block_tiling[0], shape[0])
            for i1 in range(0, shape[1], block_tiling[1]):
                i1_end = min(i1 + block_tiling[1], shape[1])
                for i2 in range(0, shape[2], block_tiling[2]):
                    i2_end = min(i2 + block_tiling[2], shape[2])
                    for i3 in range(0, shape[3], block_tiling[3]):
                        i3_end = min(i3 + block_tiling[3], shape[3])
                        for i4 in range(0, shape[4], block_tiling[4]):
                            i4_end = min(i4 + block_tiling[4], shape[4])
                            for i5 in range(0, shape[5], block_tiling[5]):
                                i5_end = min(i5 + block_tiling[5], shape[5])
                                for i6 in range(0, shape[6], block_tiling[6]):
                                    i6_end = min(i6 + block_tiling[6], shape[6])

                                    # Extract 7D block preserving structure
                                    field_block = field_gpu[
                                        i0:i0_end,
                                        i1:i1_end,
                                        i2:i2_end,
                                        i3:i3_end,
                                        i4:i4_end,
                                        i5:i5_end,
                                        i6:i6_end,
                                    ]
                                    source_block = source_gpu[
                                        i0:i0_end,
                                        i1:i1_end,
                                        i2:i2_end,
                                        i3:i3_end,
                                        i4:i4_end,
                                        i5:i5_end,
                                        i6:i6_end,
                                    ]

                                    # Apply frequency-dependent phase if needed
                                    if omega != 0.0:
                                        t_val = domain.get("t", 0.0)
                                        phase_factor = cp.exp(1j * omega * t_val)
                                        field_block = field_block * phase_factor

                                    # Compute field amplitude squared (GPU operation)
                                    field_abs_sq = cp.abs(field_block) ** 2

                                    # Compute source-field correlation (GPU operation)
                                    correlation = cp.conj(field_block) * source_block

                                    # Perform axis-wise reduction preserving structure
                                    # Use cp.sum() which performs reduction along all axes
                                    # without flattening, maintaining block geometry
                                    block_numerator = self._axis_wise_reduce(
                                        correlation, preserve_structure=True
                                    )
                                    block_denominator = self._axis_wise_reduce(
                                        field_abs_sq, preserve_structure=True
                                    )

                                    # Accumulate sums on GPU
                                    numerator_sum = numerator_sum + block_numerator
                                    denominator_sum = (
                                        denominator_sum + block_denominator
                                    )

                                    # Clean up intermediate arrays
                                    del field_abs_sq, correlation
                                    if omega != 0.0:
                                        del field_block

                                    # Periodic memory cleanup
                                    if (i6 // block_tiling[6]) % 8 == 0:
                                        cp.get_default_memory_pool().free_all_blocks()

        # Synchronize after all block operations
        cp.cuda.Stream.null.synchronize()

        # Compute admittance
        if abs(denominator_sum) > 1e-12:
            admittance = numerator_sum / denominator_sum
        else:
            admittance = cp.complex128(0.0)

        return admittance

    def _compute_all_freqs_cuda(
        self,
        field_gpu: "cp.ndarray",
        source_gpu: "cp.ndarray",
        frequencies: np.ndarray,
        domain: Dict[str, Any],
    ) -> "cp.ndarray":
        """
        Compute admittance for all frequencies using vectorized CUDA operations.

        Physical Meaning:
            Computes admittance Y(ω) for all frequencies simultaneously using
            vectorized GPU operations, preserving 7D structure with axis-wise
            reductions.

        Mathematical Foundation:
            Performs axis-wise reductions over all 7D dimensions without
            flattening, maintaining geometric structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Args:
            field_gpu (cp.ndarray): 7D phase field on GPU.
            source_gpu (cp.ndarray): 7D source field on GPU.
            frequencies (np.ndarray): Frequencies ω to compute.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            cp.ndarray: Admittance values for each frequency (complex array).
        """
        num_freqs = len(frequencies)

        # Compute field amplitude squared (shared for all frequencies)
        # Use axis-wise reduction preserving 7D structure
        field_abs_sq = cp.abs(field_gpu) ** 2
        denominator = self._axis_wise_reduce(field_abs_sq, preserve_structure=True)

        # For each frequency, compute numerator
        admittances = cp.zeros(num_freqs, dtype=cp.complex128)

        for i, omega in enumerate(frequencies):
            # Frequency-dependent field modulation
            t_val = domain.get("t", 0.0)
            phase_factor = cp.exp(1j * omega * t_val)
            field_modulated = field_gpu * phase_factor

            # Compute correlation
            correlation = cp.conj(field_modulated) * source_gpu

            # Use axis-wise reduction preserving 7D structure
            numerator = self._axis_wise_reduce(correlation, preserve_structure=True)

            # Compute admittance
            if abs(denominator) > 1e-12:
                admittances[i] = numerator / denominator

            # Clean up intermediate arrays
            del field_modulated, correlation

        # Synchronize before return
        cp.cuda.Stream.null.synchronize()

        return admittances

    def _compute_optimal_7d_block_tiling(
        self, field_gpu: "cp.ndarray"
    ) -> Tuple[int, ...]:
        """
        Compute optimal 7D block tiling for 80% GPU memory usage.

        Physical Meaning:
            Calculates optimal block size per dimension for 7D space-time
            M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, ensuring 80% GPU memory usage while
            preserving 7D geometric structure.

        Mathematical Foundation:
            For 7D array with shape (N₀, N₁, N₂, N₃, N₄, N₅, N₆):
            - Available memory: 80% of free GPU memory
            - Block size per dimension: (available_memory / overhead) ^ (1/7)
            - Ensures blocks fit in GPU memory while preserving 7D structure

        Args:
            field_gpu (cp.ndarray): 7D field array on GPU.

        Returns:
            Tuple[int, ...]: Optimal block tiling per dimension (7-tuple).
        """
        # Get GPU memory info
        mem_info = cp.cuda.runtime.memGetInfo()
        free_memory_bytes = mem_info[0]

        # Use 80% of free memory as required
        available_memory_bytes = int(free_memory_bytes * 0.8)

        # Memory per element (complex128 = 16 bytes)
        bytes_per_element = 16

        # Memory overhead for admittance computation:
        # - Input field: 1x
        # - Source field: 1x
        # - Field amplitude squared: 1x
        # - Correlation: 1x
        # - Intermediate operations: 3x
        # - Reduction buffers: 1x
        # Total: ~8x
        overhead_factor = 8

        # Maximum elements per 7D block
        max_elements_per_block = available_memory_bytes // (
            bytes_per_element * overhead_factor
        )

        # For 7D array, calculate block size per dimension
        if field_gpu.ndim == 7:
            # Compute block size per dimension: (max_elements)^(1/7)
            elements_per_dim = int(max_elements_per_block ** (1.0 / 7.0))

            # Get field shape
            shape = field_gpu.shape

            # Compute block tiling per dimension, ensuring it fits
            block_tiling = tuple(
                max(4, min(elements_per_dim, dim_size)) for dim_size in shape
            )

            self.logger.info(
                f"Optimal 7D block tiling: {block_tiling} "
                f"(available GPU memory: {available_memory_bytes / 1e9:.2f} GB, using 80%)"
            )

            return block_tiling
        else:
            # For non-7D arrays, use uniform block size
            uniform_size = int(max_elements_per_block ** (1.0 / field_gpu.ndim))
            block_tiling = tuple(
                max(4, min(uniform_size, dim_size)) for dim_size in field_gpu.shape
            )
            return block_tiling

    def _axis_wise_reduce(
        self, array: "cp.ndarray", preserve_structure: bool = True
    ) -> "cp.ndarray":
        """
        Perform axis-wise reduction on GPU preserving block structure.

        Physical Meaning:
            Computes sum over all axes of 7D array without flattening,
            preserving geometric structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Performs reduction along all axes: Σ_{all axes} a(x,φ,t)
            without flattening, maintaining 7D structure throughout.

        Args:
            array (cp.ndarray): Array to reduce (7D structure preserved).
            preserve_structure (bool): Whether to preserve structure (unused,
                kept for API consistency).

        Returns:
            cp.ndarray: Scalar sum value (complex).
        """
        # Use cp.sum() which performs reduction along all axes
        # without flattening internally when axes=None
        # This preserves the geometric structure in the reduction operation
        result = cp.sum(array)

        # Synchronize to ensure reduction completes
        cp.cuda.Stream.null.synchronize()

        return result
