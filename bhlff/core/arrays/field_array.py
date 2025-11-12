"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unified field array wrapper for transparent swap support.

This module provides a unified interface for large 7D phase field arrays
that can be stored in memory or on disk transparently, allowing seamless
integration between solvers, generators, and analyzers.

Physical Meaning:
    Provides transparent access to large 7D phase field arrays in space-time
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, automatically managing memory and disk storage
    to handle arrays that exceed available RAM.

Mathematical Foundation:
    Wraps numpy arrays and memory-mapped arrays, providing unified interface
    for 7D phase field operations while maintaining 7D structure and properties.
"""

import numpy as np
from typing import Optional, Union, Tuple, Any
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default swap threshold - can be overridden via environment variable
# For production, uses 80% of GPU memory; for testing, use small value
def _get_default_swap_threshold_gb() -> float:
    """
    Get default swap threshold based on GPU memory (80%) or environment variable.
    
    Physical Meaning:
        Determines swap threshold as 80% of available GPU memory for production,
        or uses environment variable for testing.
        
    Returns:
        float: Swap threshold in GB.
    """
    # Check environment variable first (for testing)
    env_threshold = os.getenv("BHLFF_SWAP_THRESHOLD_GB")
    if env_threshold is not None:
        return float(env_threshold)
    
    # For production, use 80% of GPU memory
    try:
        from ...utils.cuda_utils import get_global_backend, CUDABackend
        backend = get_global_backend()
        if isinstance(backend, CUDABackend):
            mem_info = backend.get_memory_info()
            free_memory = mem_info.get("free_memory", mem_info.get("total_memory", 0))
            # Use 80% of free GPU memory as threshold
            threshold_bytes = int(free_memory * 0.8)
            threshold_gb = threshold_bytes / 1e9
            logger.info(
                f"Using 80% of GPU memory as swap threshold: "
                f"{threshold_gb:.3f} GB (free GPU memory: {free_memory/1e9:.3f} GB)"
            )
            return threshold_gb
    except Exception as e:
        logger.warning(f"Could not determine GPU memory, using default 0.01 GB: {e}")
    
    # Fallback to small value for testing
    return 0.01


class FieldArray:
    """
    Unified field array wrapper with transparent swap support.
    
    Physical Meaning:
        Provides transparent access to large 7D phase field arrays that can
        be stored in memory or on disk, automatically managing swap operations
        for arrays exceeding available RAM.
        
    Mathematical Foundation:
        Wraps numpy arrays and memory-mapped arrays, providing unified interface
        for 7D phase field operations in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ while maintaining
        all array operations and properties.
        
    Attributes:
        _array (Union[np.ndarray, np.memmap]): Underlying array (in memory or on disk).
        _swap_id (Optional[str]): Swap file identifier if using disk storage.
        _swap_manager: Swap manager instance for disk operations.
    """
    
    def __init__(
        self,
        array: Optional[Union[np.ndarray, np.memmap]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: np.dtype = np.complex128,
        swap_threshold_gb: Optional[float] = None,
        swap_id: Optional[str] = None,
        swap_manager=None,
    ):
        """
        Initialize field array with transparent swap support.
        
        Physical Meaning:
            Creates a field array wrapper that automatically manages memory
            and disk storage based on array size, providing transparent access
            to large 7D phase field arrays.
            
        Args:
            array (Optional[Union[np.ndarray, np.memmap]]): Existing array to wrap.
                If None, creates new array.
            shape (Optional[Tuple[int, ...]]): Array shape for new array.
            dtype (np.dtype): Array data type (default: complex128).
            swap_threshold_gb (float): Threshold in GB for using swap (default: 10.0).
            swap_id (Optional[str]): Optional swap file identifier.
            swap_manager: Optional swap manager instance.
        """
        from ..fft.unified.swap_manager import get_swap_manager
        
        self._swap_manager = swap_manager or get_swap_manager()
        self._swap_id = swap_id
        # Use provided threshold or default based on GPU memory (80%)
        self._swap_threshold_gb = swap_threshold_gb if swap_threshold_gb is not None else _get_default_swap_threshold_gb()
        
        if array is not None:
            # Wrap existing array
            self._array = array
            # Check if should use swap
            if isinstance(array, np.memmap):
                self._swap_id = swap_id or f"field_{id(array)}"
            elif self._should_use_swap(array):
                logger.info(
                    f"Array size {array.nbytes/1e9:.3f}GB exceeds threshold "
                    f"{self._swap_threshold_gb:.3f}GB, converting to swap"
                )
                self._convert_to_swap()
        elif shape is not None:
            # Create new array
            if self._should_use_swap_for_shape(shape, dtype):
                estimated_size = np.prod(shape) * np.dtype(dtype).itemsize / 1e9
                logger.info(
                    f"Estimated size {estimated_size:.3f}GB exceeds threshold "
                    f"{self._swap_threshold_gb:.3f}GB, creating swap array"
                )
                self._array = self._swap_manager.create_swap_array(
                    shape=shape,
                    dtype=dtype,
                    array_id=swap_id or f"field_{id(self)}"
                )
                self._swap_id = swap_id or f"field_{id(self)}"
            else:
                self._array = np.zeros(shape, dtype=dtype)
        else:
            raise ValueError("Either array or shape must be provided")
    
    def _should_use_swap(self, array: np.ndarray) -> bool:
        """Check if array should use swap based on size."""
        size_gb = array.nbytes / 1e9
        return size_gb > self._swap_threshold_gb
    
    def _should_use_swap_for_shape(self, shape: Tuple[int, ...], dtype: np.dtype) -> bool:
        """Check if shape should use swap based on estimated size."""
        estimated_size = np.prod(shape) * np.dtype(dtype).itemsize
        size_gb = estimated_size / 1e9
        return size_gb > self._swap_threshold_gb
    
    def _convert_to_swap(self) -> None:
        """Convert in-memory array to swap."""
        if isinstance(self._array, np.memmap):
            return  # Already swapped
        
        logger.info(
            f"[FieldArray DEBUG] Converting to swap: shape={self._array.shape}, "
            f"size={self._array.nbytes/1e9:.3f}GB"
        )
        
        # Create swap array
        swap_id = self._swap_id or f"field_{id(self._array)}"
        logger.info(f"[FieldArray DEBUG] Creating swap array with id={swap_id}")
        swap_array = self._swap_manager.create_swap_array(
            shape=self._array.shape,
            dtype=self._array.dtype,
            array_id=swap_id
        )
        logger.info(f"[FieldArray DEBUG] Swap array created, copying data...")
        
        # Copy data to swap in chunks to avoid hanging
        array_size = self._array.size
        chunk_size = min(100_000_000, array_size)  # 100M elements per chunk
        for i in range(0, array_size, chunk_size):
            end = min(i + chunk_size, array_size)
            swap_array.flat[i:end] = self._array.flat[i:end]
            if i % (chunk_size * 10) == 0 or i == 0:
                logger.info(
                    f"[FieldArray DEBUG] Copied {i}/{array_size} elements "
                    f"({i/array_size*100:.1f}%)"
                )
        
        swap_array.flush()
        logger.info(f"[FieldArray DEBUG] Swap conversion complete")
        
        # Replace array
        self._array = swap_array
        self._swap_id = swap_id
    
    @property
    def array(self) -> Union[np.ndarray, np.memmap]:
        """
        Get underlying array.
        
        Physical Meaning:
            Returns the underlying array (in memory or on disk) for direct access
            when needed for operations that require numpy array interface.
            
        Returns:
            Union[np.ndarray, np.memmap]: Underlying array.
        """
        return self._array
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get array shape."""
        return self._array.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get array dtype."""
        return self._array.dtype
    
    @property
    def nbytes(self) -> int:
        """Get array size in bytes."""
        return self._array.nbytes
    
    @property
    def is_swapped(self) -> bool:
        """Check if array is stored on disk."""
        return isinstance(self._array, np.memmap)
    
    def __array__(self) -> np.ndarray:
        """NumPy array interface."""
        return self._array
    
    def __getitem__(self, key):
        """Array indexing."""
        return self._array[key]
    
    def __setitem__(self, key, value):
        """Array assignment."""
        self._array[key] = value
        if isinstance(self._array, np.memmap):
            self._array.flush()
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """NumPy ufunc support."""
        # Convert inputs to arrays
        arrays = []
        for inp in inputs:
            if isinstance(inp, FieldArray):
                arrays.append(inp._array)
            else:
                arrays.append(inp)
        
        # Apply ufunc
        result = getattr(ufunc, method)(*arrays, **kwargs)
        
        # Wrap result if it's an array
        if isinstance(result, np.ndarray):
            return FieldArray(array=result, swap_threshold_gb=self._swap_threshold_gb)
        return result
    
    def __mul__(self, other):
        """Multiplication operator."""
        if isinstance(other, FieldArray):
            result = self._array * other._array
        else:
            result = self._array * other
        return FieldArray(array=result, swap_threshold_gb=self._swap_threshold_gb)
    
    def __rmul__(self, other):
        """Right multiplication operator."""
        result = other * self._array
        return FieldArray(array=result, swap_threshold_gb=self._swap_threshold_gb)
    
    def __add__(self, other):
        """Addition operator."""
        if isinstance(other, FieldArray):
            result = self._array + other._array
        else:
            result = self._array + other
        return FieldArray(array=result, swap_threshold_gb=self._swap_threshold_gb)
    
    def __sub__(self, other):
        """Subtraction operator."""
        if isinstance(other, FieldArray):
            result = self._array - other._array
        else:
            result = self._array - other
        return FieldArray(array=result, swap_threshold_gb=self._swap_threshold_gb)
    
    def __truediv__(self, other):
        """Division operator."""
        if isinstance(other, FieldArray):
            result = self._array / other._array
        else:
            result = self._array / other
        return FieldArray(array=result, swap_threshold_gb=self._swap_threshold_gb)
    
    def to_memory(self) -> 'FieldArray':
        """
        Convert swapped array to memory.
        
        Physical Meaning:
            Converts array from disk storage to memory, useful for operations
            that require fast in-memory access.
            
        Returns:
            FieldArray: New FieldArray in memory.
        """
        if not self.is_swapped:
            return self
        
        # Load to memory
        array_memory = np.array(self._array)
        return FieldArray(
            array=array_memory,
            swap_threshold_gb=self._swap_threshold_gb
        )
    
    def to_swap(self) -> 'FieldArray':
        """
        Convert memory array to swap.
        
        Physical Meaning:
            Converts array from memory to disk storage, useful for freeing
            memory while keeping array accessible.
            
        Returns:
            FieldArray: New FieldArray on disk (or self if already swapped).
        """
        if self.is_swapped:
            return self
        
        # Convert to swap
        new_field = FieldArray(
            array=self._array,
            swap_threshold_gb=0.0,  # Force swap
            swap_id=self._swap_id
        )
        new_field._convert_to_swap()
        return new_field
    
    def save_pickle(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Save array to pickle file for fast loading.
        
        Physical Meaning:
            Saves array to pickle file for checkpointing and fast save/load
            operations, preserving array state.
            
        Args:
            filepath (Optional[Union[str, Path]]): Path to save pickle file.
                If None, uses swap manager directory.
                
        Returns:
            Path: Path to saved pickle file.
        """
        if filepath is None:
            filepath = self._swap_manager.swap_dir / f"pickle_{self._swap_id or id(self)}.pkl"
        
        return self._swap_manager.save_to_pickle(self._array, array_id=str(filepath.stem))
    
    @classmethod
    def load_pickle(
        cls,
        filepath: Union[str, Path],
        swap_threshold_gb: float = 10.0,
    ) -> 'FieldArray':
        """
        Load array from pickle file.
        
        Physical Meaning:
            Loads array from pickle file, restoring array state from checkpoint.
            
        Args:
            filepath (Union[str, Path]): Path to pickle file.
            swap_threshold_gb (float): Threshold for using swap.
            
        Returns:
            FieldArray: Loaded field array.
        """
        from ..fft.unified.swap_manager import get_swap_manager
        
        swap_manager = get_swap_manager()
        array = swap_manager.load_from_pickle(filepath)
        
        return cls(
            array=array,
            swap_threshold_gb=swap_threshold_gb
        )
    
    def cleanup(self) -> None:
        """
        Clean up swap files.
        
        Physical Meaning:
            Removes swap files associated with this array to free disk space.
        """
        if self._swap_id:
            self._swap_manager.cleanup(self._swap_id)
    
    def __repr__(self) -> str:
        """String representation."""
        swap_status = "swapped" if self.is_swapped else "memory"
        return (
            f"FieldArray(shape={self.shape}, dtype={self.dtype}, "
            f"size={self.nbytes/1e9:.2f}GB, storage={swap_status})"
        )
    
    def __del__(self):
        """Cleanup on deletion."""
        # Note: Don't auto-cleanup swap files as they might be shared
        # Explicit cleanup() call is preferred
        pass

