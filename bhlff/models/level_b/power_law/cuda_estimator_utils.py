"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA utilities for block-aware critical exponent estimation.

This module provides CUDA-accelerated helper functions for computing
block statistics and backend management in 7D BVP analysis.

Physical Meaning:
    Efficient GPU-accelerated computation of block statistics while
    preserving 7D structure for critical exponent estimation.

Mathematical Foundation:
    Provides vectorized CUDA operations for mean, variance, and
    CCDF computation on blocks of 7D phase field data.
"""

from __future__ import annotations

from typing import Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_cuda_backend() -> Any:
    """
    Get CUDA backend if available.

    Returns:
        Any: CUDA backend instance or None if not available.
    """
    try:
        from bhlff.utils.cuda_utils import get_global_backend, CUDABackend

        backend = get_global_backend()
        if isinstance(backend, CUDABackend):
            return backend
        return None
    except Exception:
        return None


def compute_block_statistics_cuda(
    block_arr: np.ndarray, backend: Any
) -> Tuple[float, float]:
    """
    Compute block mean and variance using CUDA acceleration with memory management.

    Physical Meaning:
        Efficiently computes block statistics on GPU while preserving
        the 7D block structure. Checks GPU memory usage to ensure blocks
        do not exceed 80% of available memory. Essential for scalable
        processing of large 7D phase field data.

    Mathematical Foundation:
        Computes mean μ = ⟨A_block⟩ and variance σ² = ⟨(A - μ)²⟩
        using vectorized GPU operations. Memory usage is monitored
        to prevent GPU memory exhaustion.

    Args:
        block_arr (np.ndarray): Block array (can be 7D, preserves structure).
        backend (Any): CUDA backend instance.

    Returns:
        Tuple[float, float]: (mean, variance) computed on GPU.

    Raises:
        RuntimeError: If CUDA computation fails and CPU fallback unavailable.
    """
    import cupy as cp

    try:
        # Check memory requirements before transfer
        block_size_bytes = block_arr.nbytes
        mem_info = backend.get_memory_info()
        free_memory = mem_info.get("free_memory", 0)

        # Ensure block doesn't exceed 80% of free memory
        max_block_size = 0.8 * free_memory
        if block_size_bytes > max_block_size:
            logger.warning(
                f"Block size {block_size_bytes/1e6:.2f}MB exceeds "
                f"80% of free GPU memory ({max_block_size/1e6:.2f}MB). "
                f"Falling back to CPU."
            )
            return float(np.mean(block_arr)), float(np.var(block_arr))

        # Transfer to GPU
        block_gpu = cp.asarray(block_arr)

        # Compute statistics using vectorized operations
        mean_gpu = float(cp.mean(block_gpu))
        var_gpu = float(cp.var(block_gpu))

        # Synchronize to ensure computation completes
        cp.cuda.Stream.null.synchronize()

        # Clean up GPU memory
        del block_gpu
        cp.get_default_memory_pool().free_all_blocks()

        return mean_gpu, var_gpu
    except Exception as e:
        logger.warning(f"CUDA statistics computation failed: {e}, using CPU")
        return float(np.mean(block_arr)), float(np.var(block_arr))


def compute_global_mean_cuda(amplitude: np.ndarray, backend: Any) -> float:
    """
    Compute global mean using CUDA acceleration with memory management.

    Physical Meaning:
        Efficiently computes global critical amplitude A_c on GPU
        for use as reference point in scaling analysis. Handles
        large 7D arrays by checking memory constraints.

    Args:
        amplitude (np.ndarray): Full amplitude array (7D structure preserved).
        backend (Any): CUDA backend instance.

    Returns:
        float: Global mean computed on GPU (or CPU if memory insufficient).
    """
    import cupy as cp

    try:
        # Check memory requirements
        amp_size_bytes = amplitude.nbytes
        mem_info = backend.get_memory_info()
        free_memory = mem_info.get("free_memory", 0)

        # Ensure array doesn't exceed 80% of free memory
        max_array_size = 0.8 * free_memory
        if amp_size_bytes > max_array_size:
            logger.warning(
                f"Array size {amp_size_bytes/1e6:.2f}MB exceeds "
                f"80% of free GPU memory ({max_array_size/1e6:.2f}MB). "
                f"Using CPU computation."
            )
            return float(np.mean(amplitude))

        # Transfer to GPU
        amp_gpu = cp.asarray(amplitude)
        A_c = float(cp.mean(amp_gpu))
        cp.cuda.Stream.null.synchronize()

        # Clean up
        del amp_gpu
        cp.get_default_memory_pool().free_all_blocks()

        return A_c
    except Exception:
        return float(np.mean(amplitude))


def compute_ccdf_cuda(
    block_flat: np.ndarray, grid: np.ndarray, backend: Any
) -> np.ndarray:
    """
    Compute CCDF using CUDA acceleration with vectorized operations and memory management.

    Physical Meaning:
        Computes complementary cumulative distribution function
        P(>A) for each grid point using vectorized GPU operations.
        Essential for efficient tail analysis in β estimation.
        Checks memory usage to ensure operations stay within 80% GPU memory limit.

    Mathematical Foundation:
        CCDF(A) = P(>A) = (1/N) Σᵢ I(Aᵢ > A)
        where I is indicator function and N is block size.
        Uses broadcasting: v_gpu[None, :] > g_gpu[:, None] creates
        comparison matrix of size (grid_size, block_size).

    Args:
        block_flat (np.ndarray): Flattened block values (from block-wise processing).
        grid (np.ndarray): Amplitude grid for CCDF evaluation.
        backend (Any): CUDA backend instance.

    Returns:
        np.ndarray: CCDF values on grid.
    """
    import cupy as cp

    try:
        # Estimate memory requirements for broadcasting
        # v_gpu: (1, block_size), g_gpu: (grid_size, 1)
        # Comparison matrix: (grid_size, block_size)
        block_size_bytes = block_flat.nbytes
        grid_size_bytes = grid.nbytes
        # Broadcasting creates temporary matrix of size (grid_size, block_size)
        # Estimate: ~2x for comparison matrix + original arrays
        estimated_memory = 2 * (
            block_size_bytes * len(grid) + grid_size_bytes * len(block_flat)
        )

        mem_info = backend.get_memory_info()
        free_memory = mem_info.get("free_memory", 0)
        max_memory = 0.8 * free_memory

        if estimated_memory > max_memory:
            logger.debug(
                f"CCDF memory estimate {estimated_memory/1e6:.2f}MB exceeds "
                f"80% limit ({max_memory/1e6:.2f}MB). Using CPU."
            )
            # CPU fallback
            v = block_flat[None, :]
            g = grid[:, None]
            return (v > g).mean(axis=1)

        # Transfer to GPU
        v_gpu = cp.asarray(block_flat)
        g_gpu = cp.asarray(grid)

        # Vectorized CCDF: P(>A) for each grid point
        # Broadcasting: compare each grid value with all block values
        ccdf_gpu = (v_gpu[None, :] > g_gpu[:, None]).mean(axis=1)

        # Convert back to NumPy
        ccdf = cp.asnumpy(ccdf_gpu)
        cp.cuda.Stream.null.synchronize()

        # Clean up GPU memory
        del v_gpu, g_gpu, ccdf_gpu
        cp.get_default_memory_pool().free_all_blocks()

        return ccdf
    except Exception as e:
        logger.debug(f"CUDA CCDF computation failed, using CPU fallback: {e}")
        # CPU fallback
        v = block_flat[None, :]
        g = grid[:, None]
        return (v > g).mean(axis=1)
