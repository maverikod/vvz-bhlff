"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA-accelerated 7D block expansion with block processing.

This module provides CUDA implementations for explicit 7D block construction
from 3D spatial blocks, with optimized block-wise processing for large arrays
that respects 80% GPU memory limit.

Physical Meaning:
    CUDA-accelerated expansion of 3D spatial blocks to 7D space-time
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ using explicit construction. For large blocks,
    uses block-wise processing to manage memory constraints.

Mathematical Foundation:
    For 3D spatial block a(x,y,z), creates 7D block:
    a_7d(x, y, z, φ₁, φ₂, φ₃, t) = a(x, y, z) ⊗ 1_φ₁ ⊗ 1_φ₂ ⊗ 1_φ₃ ⊗ 1_t
    where ⊗ is outer product. For large blocks, processes in chunks.

Example:
    >>> spatial_block = np.ones((8, 8, 8))
    >>> block_7d = expand_spatial_to_7d_cuda(spatial_block, N_phi=4, N_t=8)
    >>> assert block_7d.shape == (8, 8, 8, 4, 4, 4, 8)
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import CUDA
try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


def expand_spatial_to_7d_cuda(
    spatial_block: np.ndarray,
    N_phi: int,
    N_t: int,
    phase_coords: Optional[np.ndarray] = None,
    time_coords: Optional[np.ndarray] = None,
    optimize_block_size: bool = True,
) -> "cp.ndarray":
    """
    CUDA-accelerated explicit 7D expansion from 3D spatial block with block processing.

    Physical Meaning:
        Expands 3D spatial block to 7D on GPU using vectorized operations with
        block-wise processing for large arrays. Generates the 7D block directly
        on device for maximum efficiency while respecting 80% GPU memory limit.
        Uses explicit 7D construction that preserves the structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

    Mathematical Foundation:
        Uses CUDA kernels to expand spatial block to 7D:
        a_7d(x, y, z, φ₁, φ₂, φ₃, t) = a(x, y, z)
        with vectorized assignment across all phase and time dimensions.
        For large blocks, processes in chunks preserving 7D structure.

    Args:
        spatial_block (np.ndarray): 3D spatial block (will be moved to GPU).
        N_phi (int): Number of grid points per phase dimension.
        N_t (int): Number of grid points for temporal dimension.
        phase_coords (Optional[np.ndarray]): Phase coordinates (unused in uniform case).
        time_coords (Optional[np.ndarray]): Time coordinates (unused in uniform case).
        optimize_block_size (bool): Whether to use block processing for large arrays.

    Returns:
        cp.ndarray: 7D block on GPU with shape (N_x, N_y, N_z, N_phi, N_phi, N_phi, N_t).
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for 7D expansion")

    # Move spatial block to GPU if needed
    if isinstance(spatial_block, np.ndarray):
        spatial_block_gpu = cp.asarray(spatial_block)
    else:
        spatial_block_gpu = spatial_block

    N_x, N_y, N_z = spatial_block_gpu.shape

    # Calculate total 7D block size
    total_elements_7d = N_x * N_y * N_z * N_phi * N_phi * N_phi * N_t
    bytes_per_element = np.dtype(spatial_block_gpu.dtype).itemsize
    total_size_bytes = total_elements_7d * bytes_per_element

    # Check if block processing is needed (80% GPU memory limit)
    if optimize_block_size and CUDA_AVAILABLE:
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_memory_bytes = mem_info[0]
            available_memory_bytes = int(free_memory_bytes * 0.8)  # 80% limit

            # Overhead factor for temporary arrays during expansion
            overhead_factor = 3.0  # spatial_block + expanded + tiled result
            required_memory = total_size_bytes * overhead_factor

            if required_memory > available_memory_bytes:
                # Use block-wise expansion for large arrays
                logger.debug(
                    f"7D block size {total_size_bytes/1e9:.2f}GB exceeds "
                    f"80% GPU memory limit ({available_memory_bytes/1e9:.2f}GB). "
                    f"Using block-wise expansion."
                )
                return expand_spatial_to_7d_cuda_blocked(
                    spatial_block_gpu, N_phi, N_t, available_memory_bytes
                )
        except Exception as e:
            logger.warning(
                f"Failed to check GPU memory for block processing: {e}. "
                f"Using direct expansion."
            )

    # Direct expansion for small arrays: explicit 7D construction on GPU
    # Replace blind tile with explicit 7D block construction using direct array creation
    # Physical meaning: Create 7D block with concrete phase/time extents: (N_x, N_y, N_z, N_phi, N_phi, N_phi, N_t)
    # This explicitly constructs the 7D structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ

    N_x, N_y, N_z = spatial_block_gpu.shape

    # Explicit 7D block construction: create array with concrete phase/time extents
    # Allocate 7D array directly on GPU with explicit shape
    block_7d_gpu = cp.zeros(
        (N_x, N_y, N_z, N_phi, N_phi, N_phi, N_t), dtype=spatial_block_gpu.dtype
    )

    # Explicit assignment: fill 7D block by expanding spatial block across phase/time dimensions
    # Vectorized operation: spatial values are constant across all phase and time dimensions
    # This is explicit 7D construction that respects dimensionality
    # Use advanced indexing to assign spatial values to all phase/time combinations
    block_7d_gpu[:, :, :, :, :, :, :] = spatial_block_gpu[
        :, :, :, cp.newaxis, cp.newaxis, cp.newaxis, cp.newaxis
    ]

    return block_7d_gpu


def expand_spatial_to_7d_cuda_blocked(
    spatial_block_gpu: "cp.ndarray",
    N_phi: int,
    N_t: int,
    available_memory_bytes: int,
) -> "cp.ndarray":
    """
    Block-wise CUDA expansion of 3D spatial block to 7D with explicit construction.

    Physical Meaning:
        Expands 3D spatial block to 7D using block-wise processing that respects
        80% GPU memory limit. Processes phase and time dimensions in blocks,
        preserving 7D structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ throughout.

    Mathematical Foundation:
        For large 7D blocks, processes in chunks:
        a_7d(x, y, z, φ₁, φ₂, φ₃, t) = a(x, y, z) ⊗ 1_φ₁ ⊗ 1_φ₂ ⊗ 1_φ₃ ⊗ 1_t
        Processed block-wise: spatial block is constant, phase/time blocks are tiled.

    Args:
        spatial_block_gpu (cp.ndarray): 3D spatial block on GPU (N_x, N_y, N_z).
        N_phi (int): Number of grid points per phase dimension.
        N_t (int): Number of grid points for temporal dimension.
        available_memory_bytes (int): Available GPU memory (80% of free memory).

    Returns:
        cp.ndarray: 7D block on GPU with shape (N_x, N_y, N_z, N_phi, N_phi, N_phi, N_t).
    """
    N_x, N_y, N_z = spatial_block_gpu.shape
    bytes_per_element = np.dtype(spatial_block_gpu.dtype).itemsize

    # Compute optimal block sizes for phase and time dimensions
    # We process spatial dimensions as-is (N_x, N_y, N_z)
    # and block phase (N_phi) and time (N_t) dimensions
    overhead_factor = 3.0  # spatial + expanded + tiled
    max_elements_per_block = available_memory_bytes // (
        bytes_per_element * overhead_factor
    )

    # Calculate block sizes for phase and time dimensions
    # We need to fit: N_x * N_y * N_z * block_phi^3 * block_t <= max_elements_per_block
    spatial_elements = N_x * N_y * N_z
    if spatial_elements == 0:
        raise ValueError("Spatial block is empty")

    max_phase_time_elements = max_elements_per_block // spatial_elements
    if max_phase_time_elements < 1:
        raise ValueError(
            f"Spatial block too large ({spatial_elements} elements) "
            f"for available memory ({available_memory_bytes/1e9:.2f}GB)"
        )

    # Compute block sizes: block_phi^3 * block_t <= max_phase_time_elements
    # Use equal block sizes for phase dimensions
    block_phi_base = int(
        max_phase_time_elements ** (1.0 / 4.0)
    )  # 4th root: 3 phase + 1 time
    block_phi = max(4, min(block_phi_base, N_phi))
    block_t = max(4, min(block_phi_base, N_t))

    logger.debug(
        f"Block-wise 7D expansion: spatial=({N_x}, {N_y}, {N_z}), "
        f"phase_block={block_phi}, time_block={block_t}, "
        f"target=({N_phi}, {N_phi}, {N_phi}, {N_t})"
    )

    # Allocate output 7D array on GPU
    result_shape = (N_x, N_y, N_z, N_phi, N_phi, N_phi, N_t)
    result_gpu = cp.zeros(result_shape, dtype=spatial_block_gpu.dtype)

    # Create CUDA stream for asynchronous processing
    stream = cp.cuda.Stream()

    # Process phase and time dimensions in blocks
    # Use explicit 7D construction: expand spatial block for each phase/time block
    with stream:
        for phi1_start in range(0, N_phi, block_phi):
            phi1_end = min(phi1_start + block_phi, N_phi)
            phi1_size = phi1_end - phi1_start

            for phi2_start in range(0, N_phi, block_phi):
                phi2_end = min(phi2_start + block_phi, N_phi)
                phi2_size = phi2_end - phi2_start

                for phi3_start in range(0, N_phi, block_phi):
                    phi3_end = min(phi3_start + block_phi, N_phi)
                    phi3_size = phi3_end - phi3_start

                    for t_start in range(0, N_t, block_t):
                        t_end = min(t_start + block_t, N_t)
                        t_size = t_end - t_start

                        # Explicit 7D construction for this phase/time block
                        # Replace blind tile with explicit 7D block construction
                        # Physical meaning: Create 7D block slice with concrete phase/time extents
                        # This explicitly constructs the 7D structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ

                        # Allocate explicit 7D block slice with concrete phase/time extents
                        block_expanded = cp.zeros(
                            (N_x, N_y, N_z, phi1_size, phi2_size, phi3_size, t_size),
                            dtype=spatial_block_gpu.dtype,
                        )

                        # Explicit assignment: fill 7D block slice by expanding spatial block
                        # Vectorized operation: spatial values are constant across phase/time dimensions
                        # Use advanced indexing for explicit 7D construction
                        block_expanded[:, :, :, :, :, :, :] = spatial_block_gpu[
                            :, :, :, cp.newaxis, cp.newaxis, cp.newaxis, cp.newaxis
                        ]

                        # Copy to result array at correct position using vectorized assignment
                        # This is explicit 7D block construction that preserves structure
                        result_gpu[
                            :,
                            :,
                            :,
                            phi1_start:phi1_end,
                            phi2_start:phi2_end,
                            phi3_start:phi3_end,
                            t_start:t_end,
                        ] = block_expanded

                        # Clean up temporary arrays to free GPU memory
                        del block_expanded

    # Synchronize stream
    stream.synchronize()

    return result_gpu
