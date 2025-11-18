"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Forward FFT blocked processing.

This module provides forward FFT blocked processing functions.
FFT is a global operation that requires all data simultaneously.
This implementation processes the entire field at once, using
swap/memory-mapped arrays for memory management with maximum block sizes.
"""

from typing import Tuple
import numpy as np
import logging
import sys

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
        swap/memory-mapped arrays for memory management. FFT is a global
        operation that requires all data simultaneously, so we process
        the entire field at once, using swap for large fields.
    """
    if len(field.shape) == 7 and len(domain_shape) == 7:
        # 7D block processing - process entire field with swap support
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
    Perform forward FFT for 7D field using swap/memory-mapped arrays.
    
    Physical Meaning:
        Computes forward FFT for entire 7D phase field. FFT is a global
        operation requiring all data simultaneously. This implementation:
        - Processes entire field at once on GPU (if memory allows)
        - Uses memory-mapped arrays for input/output if field is too large
        - Uses maximum block sizes for swap operations (80% GPU memory)
        - Supports streaming from FieldArray with iter_batches for swapped fields
        - Ensures correct normalization using full domain_shape
        
    Mathematical Foundation:
        FFT is computed as: F(k) = Σ f(x) * exp(-2πi k·x / N)
        This requires all spatial points simultaneously, so we cannot
        split the field into independent windows. Instead, we use swap
        to manage memory while processing the entire field. For swapped
        FieldArray, we stream blocks sequentially into pinned memory,
        then assemble for FFT to simulate a contiguous array.
    """
    logger = logging.getLogger(__name__)
    
    # Check if field is FieldArray with swap support
    field_array_obj = None
    if hasattr(field, 'is_swapped') and hasattr(field, 'iter_batches'):
        # FieldArray with streaming support
        field_array_obj = field
        field = field.array  # Extract underlying array
        logger.info(
            f"FieldArray detected: is_swapped={field_array_obj.is_swapped}, "
            f"shape={field.shape}"
        )
    
    field_size_mb = field.nbytes / (1024**2)
    logger.info(
        f"_forward_fft_blocked_7d: processing entire field {field.shape} "
        f"({field_size_mb:.2f}MB) with swap support"
    )
    sys.stdout.flush()
    
    # Calculate field memory requirements
    field_elements = np.prod(field.shape)
    bytes_per_element = 16  # complex128 = 16 bytes
    field_memory_bytes = field_elements * bytes_per_element
    # FFT overhead: input + output + temp arrays = 4x
    field_memory_with_overhead = field_memory_bytes * 4.0
    
    # Check available GPU memory
    use_swap = False
    if CUDA_AVAILABLE:
        try:
            from bhlff.utils.cuda_utils import get_global_backend
            backend = get_global_backend()
            if hasattr(backend, "get_memory_info"):
                mem_info = backend.get_memory_info()
                free_memory = mem_info.get("free_memory", 0)
                total_memory = mem_info.get("total_memory", 0)
                
                logger.info(
                    f"GPU memory: free={free_memory/1e9:.3f}GB, total={total_memory/1e9:.3f}GB, "
                    f"field with overhead={field_memory_with_overhead/1e9:.3f}GB"
                )
                
                # Use swap if field with overhead exceeds 80% of free memory
                if field_memory_with_overhead > free_memory * 0.8:
                    use_swap = True
                    logger.info(
                        f"Field with overhead ({field_memory_with_overhead/1e9:.3f}GB) exceeds "
                        f"80% of free GPU memory ({free_memory/1e9:.3f}GB), using swap"
                    )
                else:
                    logger.info(
                        f"Field fits in GPU memory, processing directly "
                        f"(field={field_memory_with_overhead/1e9:.3f}GB, free={free_memory/1e9:.3f}GB)"
                    )
        except Exception as e:
            logger.warning(f"Failed to check GPU memory: {e}, using swap for safety")
            use_swap = True
    
    # Create output array - use swap if needed
    output_dtype = np.complex128  # Always complex for FFT output
    
    if use_swap or isinstance(field, np.memmap):
        # Use memory-mapped arrays for large fields
        from ..swap_manager import get_swap_manager
        swap_manager = get_swap_manager()
        
        # Create memory-mapped output array
        out = swap_manager.create_swap_array(
            shape=field.shape,
            dtype=output_dtype,
            array_id=f"fft_forward_{id(field)}"
        )
        
        logger.info(
            f"Created memory-mapped output array: shape={out.shape}, dtype={out.dtype}"
        )
    else:
        # Use regular array for small fields
        out = np.zeros(field.shape, dtype=output_dtype)
        logger.info(
            f"Created regular output array: shape={out.shape}, dtype={out.dtype}"
        )
    
    # Verify output array properties
    if not np.iscomplexobj(out):
        raise ValueError(
            f"Output array must be complex for FFT, got dtype {out.dtype}"
        )
    if out.shape != field.shape:
        raise ValueError(
            f"Output array shape mismatch: expected {field.shape}, got {out.shape}"
        )
    
    # CRITICAL: FFT is a global operation - process entire field at once
    # Use swap/memory-mapped arrays for memory management, but process all data together
    # However, if field is too large, we need to handle OutOfMemoryError gracefully
    # by using memory-mapped arrays and processing in chunks if necessary
    try:
        logger.info("Processing entire field with FFT (global operation)")
        sys.stdout.flush()
        
        # For swapped FieldArray, use streaming batch iterator to load blocks
        # sequentially into pinned memory, then assemble for FFT
        if field_array_obj is not None and field_array_obj.is_swapped:
            logger.info("FieldArray is swapped, using streaming batch iterator")
            if not CUDA_AVAILABLE:
                raise RuntimeError(
                    "CUDA is required for streaming FFT with swapped FieldArray. "
                    "CPU fallback is not supported."
                )
            
            # Create CUDA stream for async transfers
            stream = cp.cuda.Stream()
            
            # Check if field fits in GPU memory (with overhead)
            try:
                from bhlff.utils.cuda_utils import get_global_backend
                backend = get_global_backend()
                if hasattr(backend, "get_memory_info"):
                    mem_info = backend.get_memory_info()
                    free_memory = mem_info.get("free_memory", 0)
                    available_memory = int(free_memory * gpu_memory_ratio)
                else:
                    available_memory = int(0.8 * 1024**3)  # 0.8 GB fallback
            except Exception:
                available_memory = int(0.8 * 1024**3)  # 0.8 GB fallback
            
            # Check if field fits in available memory
            if field_memory_with_overhead > available_memory:
                raise RuntimeError(
                    f"Field size {field_memory_with_overhead/1e9:.3f}GB exceeds "
                    f"available GPU memory {available_memory/1e9:.3f}GB. "
                    f"FFT requires all data simultaneously. "
                    f"Please reduce field size or increase GPU memory."
                )
            
            # Allocate regular numpy array for assembled field
            # This will be in CPU memory, then transferred to GPU
            assembled_field = np.zeros(field.shape, dtype=field.dtype)
            
            # Stream blocks from FieldArray and assemble in CPU memory
            logger.info("Streaming blocks from FieldArray and assembling")
            for batch_payload in field_array_obj.iter_batches(
                max_gpu_ratio=gpu_memory_ratio,
                use_cuda=False,  # Use CPU for assembly
                stream=None
            ):
                slices = batch_payload["slices"]
                cpu_block = batch_payload["cpu"]
                # Copy block to assembled field
                assembled_field[slices] = cpu_block
            
            logger.info("All blocks assembled, transferring to GPU")
            
            # Transfer assembled field to GPU using CUDA stream
            with stream:
                field_gpu = cp.asarray(assembled_field)
            stream.synchronize()
            
            # Process on GPU
            result = forward_fft_gpu(field_gpu, normalization, domain_shape)
            
            # Copy result back
            if isinstance(result, cp.ndarray):
                result = cp.asnumpy(result)
            
            # Free assembled field from CPU memory
            del assembled_field
            
        # For memory-mapped arrays, we need to load data in chunks
        # But FFT requires all data, so we need to load entire field to GPU
        # If it doesn't fit, we'll get OutOfMemoryError which we'll handle
        elif isinstance(field, np.memmap):
            logger.info("Input is memory-mapped, loading to GPU")
            # Verify input data before processing
            field_sum = np.sum(field)
            field_norm = np.linalg.norm(field)
            logger.debug(
                f"Memory-mapped input: shape={field.shape}, dtype={field.dtype}, "
                f"sum={field_sum:.6e}, norm={field_norm:.6e}"
            )
            
            # Try to load entire field - if it fails, we'll handle it
            try:
                # Load entire field to GPU
                if CUDA_AVAILABLE:
                    field_gpu = cp.asarray(field)
                    # Verify data was loaded correctly
                    if CUDA_AVAILABLE:
                        # For complex arrays, use abs() to get real magnitude
                        field_gpu_sum_val = cp.sum(field_gpu)
                        field_gpu_norm_val = cp.linalg.norm(field_gpu)
                        # Convert to float: use abs() for complex, direct for real
                        if np.iscomplexobj(field_gpu):
                            field_gpu_sum = float(abs(field_gpu_sum_val))
                            field_gpu_norm = float(abs(field_gpu_norm_val))
                        else:
                            field_gpu_sum = float(field_gpu_sum_val)
                            field_gpu_norm = float(field_gpu_norm_val)
                        logger.debug(
                            f"GPU input: sum={field_gpu_sum:.6e}, norm={field_gpu_norm:.6e}"
                        )
                        # Compare using abs() for complex numbers
                        field_sum_abs = abs(field_sum) if np.iscomplexobj(field) else field_sum
                        if abs(field_sum_abs - field_gpu_sum) > 1e-6:
                            logger.warning(
                                f"Data mismatch when loading to GPU: "
                                f"CPU sum={field_sum_abs:.6e}, GPU sum={field_gpu_sum:.6e}"
                            )
                else:
                    field_gpu = field
                
                # Process on GPU
                result = forward_fft_gpu(field_gpu, normalization, domain_shape)
                
                # Copy result back
                if CUDA_AVAILABLE and isinstance(result, cp.ndarray):
                    result = cp.asnumpy(result)
                    # Verify result after transfer
                    result_sum = np.sum(result)
                    result_norm = np.linalg.norm(result)
                    logger.debug(
                        f"Result after GPU->CPU transfer: sum={result_sum:.6e}, norm={result_norm:.6e}"
                    )
            except Exception as e:
                error_str = str(e)
                if "OutOfMemoryError" in error_str or "Out of memory" in error_str:
                    logger.warning(
                        f"Failed to load entire field to GPU: {e}. "
                        f"Field is too large for available GPU memory. "
                        f"FFT/IFFT requires all data simultaneously, so this field cannot be processed."
                    )
                    raise RuntimeError(
                        f"FFT failed: Field size {field_memory_with_overhead/1e9:.3f}GB is too large "
                        f"for available GPU memory. FFT/IFFT requires all data simultaneously. "
                        f"Please reduce field size or increase GPU memory."
                    ) from e
                else:
                    raise
        else:
            # Regular array - process directly
            result = forward_fft_gpu(field, normalization, domain_shape)
        
        # Copy result to output array
        # Use maximum block size for swap operations (80% GPU memory)
        if isinstance(out, np.memmap) or isinstance(result, np.memmap):
            # For memory-mapped arrays, copy in large blocks
            _copy_with_max_blocks(result, out, gpu_memory_ratio, logger)
        else:
            # For regular arrays, direct copy
            out[:] = result
        
        # Verify data integrity after copy
        if isinstance(result, np.ndarray) and isinstance(out, np.ndarray):
            if not np.allclose(result, out, rtol=1e-10, atol=1e-10):
                max_diff = np.max(np.abs(result - out))
                logger.error(
                    f"Data integrity check failed after copy: "
                    f"max difference = {max_diff:.2e}, "
                    f"result shape={result.shape}, out shape={out.shape}, "
                    f"result dtype={result.dtype}, out dtype={out.dtype}"
                )
                raise RuntimeError(
                    f"Data integrity check failed: max difference = {max_diff:.2e}"
                )
            else:
                logger.debug("Data integrity check passed after copy")
        
        logger.info("FFT completed successfully")
        sys.stdout.flush()
        
    except Exception as e:
        error_str = str(e)
        if "OutOfMemoryError" in error_str or "Out of memory" in error_str or "Insufficient GPU memory" in error_str:
            logger.error(
                f"FFT failed with memory error: {e}. "
                f"Field size: {field_memory_with_overhead/1e9:.3f}GB. "
                f"FFT/IFFT requires all data simultaneously, so this field cannot be processed with current GPU memory."
            )
            raise RuntimeError(
                f"FFT failed with memory error: {e}. "
                f"Cannot process field of size {field_memory_with_overhead/1e9:.3f}GB. "
                f"FFT/IFFT requires all data simultaneously. "
                f"Please reduce field size or increase GPU memory."
            ) from e
        else:
            raise
    
    # Flush if memory-mapped
    if isinstance(out, np.memmap):
        logger.info("Flushing memory-mapped output array")
        sys.stdout.flush()
        out.flush()
    
    # Final verification
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        logger.warning(
            f"Output array contains NaN or Inf values: "
            f"NaN count={np.sum(np.isnan(out))}, Inf count={np.sum(np.isinf(out))}"
        )
    
    if not np.iscomplexobj(out):
        raise ValueError(
            f"Final output array must be complex, got dtype {out.dtype}"
        )
    
    logger.info(
        f"_forward_fft_blocked_7d: COMPLETE - output shape={out.shape}, dtype={out.dtype}"
    )
    sys.stdout.flush()
    
    return out


def _copy_with_max_blocks(
    source: np.ndarray,
    target: np.ndarray,
    gpu_memory_ratio: float,
    logger: logging.Logger,
) -> None:
    """
    Copy array using maximum block sizes for swap operations.
    
    Physical Meaning:
        Copies data from source to target using maximum block sizes
        (80% GPU memory) for efficient swap operations. This maximizes
        throughput while respecting memory constraints.
        
    Args:
        source (np.ndarray): Source array (may be memory-mapped).
        target (np.ndarray): Target array (may be memory-mapped).
        gpu_memory_ratio (float): GPU memory ratio to use (default 0.8).
        logger (logging.Logger): Logger for debug messages.
    """
    if source.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: source {source.shape} != target {target.shape}"
        )
    
    # Calculate maximum block size (80% GPU memory)
    max_block_bytes = 0
    if CUDA_AVAILABLE:
        try:
            from bhlff.utils.cuda_utils import get_global_backend
            backend = get_global_backend()
            if hasattr(backend, "get_memory_info"):
                mem_info = backend.get_memory_info()
                free_memory = mem_info.get("free_memory", 0)
                max_block_bytes = int(free_memory * gpu_memory_ratio)
        except Exception:
            # Fallback to 1GB if cannot determine GPU memory
            max_block_bytes = 1024 * 1024 * 1024
    
    if max_block_bytes == 0:
        max_block_bytes = 1024 * 1024 * 1024  # 1GB fallback
    
    # Calculate block size in elements
    bytes_per_element = source.dtype.itemsize
    max_block_elements = max_block_bytes // bytes_per_element
    
    # For 7D arrays, copy in spatial blocks (first 3 dimensions)
    # Use maximum block size for spatial dimensions
    if len(source.shape) == 7:
        N_x, N_y, N_z = source.shape[:3]
        phase_temporal_size = np.prod(source.shape[3:])
        
        # Calculate spatial block size
        max_spatial_elements = max_block_elements // phase_temporal_size
        if max_spatial_elements < 1:
            max_spatial_elements = 1
        
        # Calculate block size per spatial dimension
        block_size_per_dim = int(max_spatial_elements ** (1.0 / 3.0))
        block_x = max(32, min(block_size_per_dim, N_x))
        block_y = max(32, min(block_size_per_dim, N_y))
        block_z = max(32, min(block_size_per_dim, N_z))
        
        logger.info(
            f"Copying with spatial blocks: ({block_x}, {block_y}, {block_z}) "
            f"= {block_x * block_y * block_z * phase_temporal_size / 1e6:.1f}M elements"
        )
        
        # Copy in spatial blocks
        for x_start in range(0, N_x, block_x):
            x_end = min(x_start + block_x, N_x)
            for y_start in range(0, N_y, block_y):
                y_end = min(y_start + block_y, N_y)
                for z_start in range(0, N_z, block_z):
                    z_end = min(z_start + block_z, N_z)
                    
                    # Copy block
                    source_block = source[x_start:x_end, y_start:y_end, z_start:z_end, :, :, :, :]
                    target[x_start:x_end, y_start:y_end, z_start:z_end, :, :, :, :] = source_block
                    
                    # Verify block was copied correctly
                    target_block = target[x_start:x_end, y_start:y_end, z_start:z_end, :, :, :, :]
                    if not np.allclose(source_block, target_block, rtol=1e-10, atol=1e-10):
                        max_diff = np.max(np.abs(source_block - target_block))
                        logger.warning(
                            f"Block copy verification failed at ({x_start}:{x_end}, {y_start}:{y_end}, {z_start}:{z_end}): "
                            f"max difference = {max_diff:.2e}"
                        )
                    
                    # Free GPU memory if using CUDA
                    if CUDA_AVAILABLE:
                        cp.get_default_memory_pool().free_all_blocks()
    else:
        # For non-7D arrays, copy directly
        target[:] = source
