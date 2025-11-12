"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Forward FFT blocked processing.

This module provides forward FFT blocked processing functions.
"""

from typing import Tuple
import numpy as np
import logging
import sys
from itertools import product

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from ..fft_gpu import forward_fft_gpu
from .blocked_tiling import compute_optimal_7d_block_tiling


def forward_fft_blocked(
    field: np.ndarray,
    normalization: str,
    domain_shape: Tuple[int, ...],
    gpu_memory_ratio: float,
) -> np.ndarray:
    """
    Perform forward FFT with 7D block processing.
    
    Physical Meaning:
        Computes forward FFT in spectral space for 7D phase field using
        block processing to fit GPU memory constraints (80% usage).
        For 7D fields, processes all 7 dimensions in blocks.
    """
    if len(field.shape) == 7 and len(domain_shape) == 7:
        # 7D block processing - process all dimensions in blocks
        return _forward_fft_blocked_7d(field, normalization, domain_shape, gpu_memory_ratio)
    else:
        # Non-7D: process only last dimension
        t_len = field.shape[-1]
        block_tiling = compute_optimal_7d_block_tiling(field.shape, gpu_memory_ratio)
        block = block_tiling[-1] if len(block_tiling) > 0 else t_len
        out = np.empty_like(field)
        start = 0
        while start < t_len:
            end = min(t_len, start + block)
            slab = field[..., start:end]
            slab_shape = tuple(list(domain_shape[:-1]) + [slab.shape[-1]])
            out[..., start:end] = forward_fft_gpu(slab, normalization, slab_shape)
            start = end
        return out


def _forward_fft_blocked_7d(
    field: np.ndarray,
    normalization: str,
    domain_shape: Tuple[int, ...],
    gpu_memory_ratio: float,
) -> np.ndarray:
    """
    Perform forward FFT with window-based processing for maximum GPU utilization.
    
    Physical Meaning:
        Computes forward FFT for 7D phase field using window-based processing:
        - Forms windows equal to 80% of GPU memory
        - Processes each window entirely on GPU (full FFT)
        - Writes result to swap and loads next window
        - For small fields, processes as single window
        
    This approach maximizes GPU utilization by processing large chunks
    instead of small blocks, reducing CPU/GPU transfers.
    """
    logger = logging.getLogger(__name__)
    
    field_size_mb = field.nbytes / (1024**2)
    logger.info(
        f"_forward_fft_blocked_7d: starting window processing for field {field.shape} "
        f"({field_size_mb:.2f}MB)"
    )
    
    # Calculate maximum window size using centralized utility function
    from bhlff.utils.cuda_utils import calculate_optimal_window_memory
    
    # FFT needs ~4x memory: input + output + temp arrays
    max_window_elements, actual_usage_gb, actual_usage_pct = calculate_optimal_window_memory(
        gpu_memory_ratio=gpu_memory_ratio,
        overhead_factor=4.0,  # FFT overhead: input + output + temp arrays
        logger=logger,
    )
    
    logger.info(
        f"FFT window calculation: max_window={max_window_elements/1e6:.1f}M elements, "
        f"expected usage={actual_usage_gb:.3f}GB ({actual_usage_pct:.1f}% of total GPU memory)"
    )
    sys.stdout.flush()
    
    # Calculate window size per dimension
    field_elements = np.prod(field.shape)
    
    # If field fits in window, process as single window
    if field_elements <= max_window_elements:
        logger.info(f"Field fits in single window, processing entirely on GPU")
        return forward_fft_gpu(field, normalization, domain_shape)
    
    # Calculate window size for spatial dimensions
    phase_temporal_size = np.prod(field.shape[3:])
    max_spatial_elements = max_window_elements // phase_temporal_size
    
    # Calculate window size per spatial dimension
    spatial_dims = field.shape[:3]
    elements_per_spatial_dim = int(max_spatial_elements ** (1.0 / 3.0))
    
    # Window size: ensure at least 32 per dimension for GPU efficiency
    window_size = tuple(
        max(32, min(elements_per_spatial_dim, dim))
        for dim in spatial_dims
    ) + field.shape[3:]  # Keep phase/temporal dimensions full
    
    window_elements = np.prod(window_size)
    window_size_mb = (window_elements * 16) / (1024**2)
    logger.info(
        f"Window size: {window_size} = {window_elements/1e6:.1f}M elements = {window_size_mb:.2f}MB"
    )
    
    # Calculate number of windows needed
    num_windows = tuple(
        (field.shape[i] + window_size[i] - 1) // window_size[i]
        for i in range(3)
    )
    total_windows = np.prod(num_windows)
    logger.info(
        f"Total windows: {total_windows} ({num_windows[0]}x{num_windows[1]}x{num_windows[2]})"
    )
    
    # Create output array - use transparent swap if input is memory-mapped
    if isinstance(field, np.memmap):
        from ..swap_manager import get_swap_manager
        swap_manager = get_swap_manager()
        out = swap_manager.create_swap_array(
            shape=field.shape,
            dtype=field.dtype,
            array_id=f"fft_forward_{id(field)}"
        )
    else:
        out = np.zeros_like(field)
    
    # Process windows
    window_idx = 0
    for wx, wy, wz in product(range(num_windows[0]), range(num_windows[1]), range(num_windows[2])):
        logger.info(f"[WINDOW {window_idx + 1}/{total_windows}] START: wx={wx}, wy={wy}, wz={wz}")
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Calculate window boundaries
        x_start = wx * window_size[0]
        x_end = min(x_start + window_size[0], field.shape[0])
        y_start = wy * window_size[1]
        y_end = min(y_start + window_size[1], field.shape[1])
        z_start = wz * window_size[2]
        z_end = min(z_start + window_size[2], field.shape[2])
        
        logger.info(
            f"[WINDOW {window_idx + 1}/{total_windows}] STEP 1: Extracting window "
            f"({x_start}:{x_end}, {y_start}:{y_end}, {z_start}:{z_end})"
        )
        sys.stdout.flush()
        
        # Extract window
        window = field[x_start:x_end, y_start:y_end, z_start:z_end, :, :, :, :]
        window_shape = window.shape
        window_size_mb = window.nbytes / (1024**2)
        
        logger.info(
            f"[WINDOW {window_idx + 1}/{total_windows}] STEP 1 COMPLETE: "
            f"shape={window_shape}, size={window_size_mb:.2f}MB"
        )
        sys.stdout.flush()
        
        # Check if window is large enough for parallel sub-window processing
        use_parallel_subwindows = (
            CUDA_AVAILABLE and 
            window_size_mb > 200 and 
            window_shape[6] >= 2
        )
        
        if use_parallel_subwindows:
            min_subwindow_mb = 100
            max_subwindows = max(2, int(window_size_mb / min_subwindow_mb))
            num_subwindows = min(max_subwindows, window_shape[6], 4)
            
            if num_subwindows >= 2:
                subwindow_size_t = (window_shape[6] + num_subwindows - 1) // num_subwindows
                streams = [cp.cuda.Stream() for _ in range(num_subwindows)]
                subwindow_results = []
                
                logger.info(
                    f"Window {window_idx + 1}: splitting into {num_subwindows} sub-windows "
                    f"for parallel processing via CUDA streams"
                )
                
                # Process sub-windows in parallel
                for stream_idx, stream in enumerate(streams):
                    t_start = stream_idx * subwindow_size_t
                    t_end = min(t_start + subwindow_size_t, window_shape[6])
                    
                    if t_start >= window_shape[6]:
                        break
                    
                    subwindow = window[:, :, :, :, :, :, t_start:t_end]
                    subwindow_shape = subwindow.shape
                    
                    subwindow_domain_shape = tuple(
                        list(domain_shape[:3]) + list(subwindow_shape[3:])
                    ) if len(domain_shape) == 7 else domain_shape
                    
                    # Process sub-window in stream
                    with stream:
                        subwindow_gpu = cp.asarray(subwindow)
                        subwindow_fft_gpu = forward_fft_gpu(
                            subwindow_gpu, normalization, subwindow_domain_shape
                        )
                        subwindow_results.append((t_start, t_end, subwindow_fft_gpu))
                
                # Synchronize all streams
                for stream in streams:
                    stream.synchronize()
                
                # Transfer results and assemble
                for t_start, t_end, subwindow_fft_gpu in subwindow_results:
                    subwindow_fft = cp.asnumpy(subwindow_fft_gpu)
                    out[x_start:x_end, y_start:y_end, z_start:z_end, :, :, :, t_start:t_end] = subwindow_fft
                    del subwindow_fft_gpu
                
                del subwindow_results
                
                # Free GPU memory
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.cuda.Stream.null.synchronize()
            else:
                use_parallel_subwindows = False
        
        if not use_parallel_subwindows:
            logger.info(f"[WINDOW {window_idx + 1}/{total_windows}] STEP 2: Processing single window on GPU")
            sys.stdout.flush()
            
            window_domain_shape = tuple(
                list(domain_shape[:3]) + list(window_shape[3:])
            ) if len(domain_shape) == 7 else domain_shape
            
            window_fft = forward_fft_gpu(window, normalization, window_domain_shape)
            
            logger.info(f"[WINDOW {window_idx + 1}/{total_windows}] STEP 3: Storing result")
            sys.stdout.flush()
            
            out[x_start:x_end, y_start:y_end, z_start:z_end, :, :, :, :] = window_fft
            
            logger.info(f"[WINDOW {window_idx + 1}/{total_windows}] STEP 3 COMPLETE: Result stored")
            sys.stdout.flush()
        
        # Free GPU memory after each window
        if CUDA_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            
            # Check memory after cleanup
            try:
                mem_info_after = cp.cuda.runtime.memGetInfo()
                free_gb_after = mem_info_after[0] / 1e9
                total_gb = mem_info_after[1] / 1e9
                used_gb = total_gb - free_gb_after
                if window_idx % 10 == 0 or window_idx == total_windows - 1:
                    logger.info(
                        f"Window {window_idx + 1}/{total_windows} completed, "
                        f"GPU memory: {used_gb:.2f}GB used / {total_gb:.2f}GB total "
                        f"({used_gb/total_gb*100:.1f}% used), {free_gb_after:.2f}GB free"
                    )
            except Exception:
                pass
        
        window_idx += 1
    
    logger.info(f"_forward_fft_blocked_7d: completed window processing ({total_windows} windows)")
    sys.stdout.flush()
    
    # Flush if memory-mapped
    if isinstance(out, np.memmap):
        logger.info(f"_forward_fft_blocked_7d: Flushing memory-mapped output")
        sys.stdout.flush()
        out.flush()
    
    logger.info(f"_forward_fft_blocked_7d: COMPLETE - all windows processed")
    sys.stdout.flush()
    return out

