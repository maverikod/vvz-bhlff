"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA utilities for automatic GPU acceleration in BHLFF.

This module provides automatic CUDA detection and fallback to CPU
when CUDA is not available, ensuring optimal performance for
7D phase field calculations.

Physical Meaning:
    CUDA acceleration is critical for 7D phase field calculations
    due to the high computational complexity of spectral operations
    in 7D space-time. This module automatically detects and uses
    available GPU resources.

Example:
    >>> from bhlff.utils.cuda_utils import get_optimal_backend
    >>> backend = get_optimal_backend()
    >>> array = backend.zeros((64, 64, 64, 16, 16, 16, 100))
"""

import logging
import os
from typing import Optional, Union, Any
import numpy as np

# Try to import CUDA libraries
try:
    import cupy as cp
    import cupyx.scipy.fft as cp_fft

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    cp_fft = None

# Try to import NumPy FFT as fallback
try:
    import numpy.fft as np_fft

    NUMPY_FFT_AVAILABLE = True
except ImportError:
    NUMPY_FFT_AVAILABLE = False
    np_fft = None

logger = logging.getLogger(__name__)


class CUDABackend:
    """
    CUDA backend for GPU-accelerated computations.

    Physical Meaning:
        Provides GPU-accelerated array operations and FFT
        for 7D phase field calculations using CuPy.
    """

    def __init__(self):
        """Initialize CUDA backend."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.device = cp.cuda.Device()
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()

        logger.info(f"CUDA backend initialized on device {self.device.id}")

    def zeros(self, shape: tuple, dtype=np.complex128) -> "cp.ndarray":
        """Create zero array on GPU."""
        return cp.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple, dtype=np.complex128) -> "cp.ndarray":
        """Create ones array on GPU."""
        return cp.ones(shape, dtype=dtype)

    def array(self, array: np.ndarray) -> "cp.ndarray":
        """Convert numpy array to GPU array."""
        return cp.asarray(array)

    def to_numpy(self, array: "cp.ndarray") -> np.ndarray:
        """Convert GPU array to numpy array."""
        return cp.asnumpy(array)

    def fft(self, array: "cp.ndarray", axes: Optional[tuple] = None) -> "cp.ndarray":
        """Perform FFT on GPU."""
        return cp_fft.fftn(array, axes=axes)

    def ifft(self, array: "cp.ndarray", axes: Optional[tuple] = None) -> "cp.ndarray":
        """Perform inverse FFT on GPU."""
        return cp_fft.ifftn(array, axes=axes)

    def fftshift(
        self, array: "cp.ndarray", axes: Optional[tuple] = None
    ) -> "cp.ndarray":
        """Perform FFT shift on GPU."""
        return cp_fft.fftshift(array, axes=axes)

    def ifftshift(
        self, array: "cp.ndarray", axes: Optional[tuple] = None
    ) -> "cp.ndarray":
        """Perform inverse FFT shift on GPU."""
        return cp_fft.ifftshift(array, axes=axes)

    def get_memory_info(self) -> dict:
        """Get GPU memory information."""
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        return {
            "total_memory": self.device.mem_info[1],
            "free_memory": self.device.mem_info[0],
            "used_memory": self.device.mem_info[1] - self.device.mem_info[0],
            "mempool_used": mempool.used_bytes(),
            "mempool_total": mempool.total_bytes(),
            "pinned_used": pinned_mempool.n_free_blocks(),
            "pinned_total": pinned_mempool.n_free_blocks(),
        }


class CPUBackend:
    """
    CPU backend for computations when CUDA is not available.

    Physical Meaning:
        Provides CPU-based array operations and FFT
        for 7D phase field calculations using NumPy.
    """

    def __init__(self):
        """Initialize CPU backend."""
        if not NUMPY_FFT_AVAILABLE:
            raise RuntimeError("NumPy FFT not available")

        logger.info("CPU backend initialized")

    def zeros(self, shape: tuple, dtype=np.complex128) -> np.ndarray:
        """Create zero array on CPU."""
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple, dtype=np.complex128) -> np.ndarray:
        """Create ones array on CPU."""
        return np.ones(shape, dtype=dtype)

    def array(self, array: np.ndarray) -> np.ndarray:
        """Return array as-is (already on CPU)."""
        return array

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        """Return array as-is (already numpy)."""
        return array

    def fft(self, array: np.ndarray, axes: Optional[tuple] = None) -> np.ndarray:
        """Perform FFT on CPU."""
        return np_fft.fftn(array, axes=axes)

    def ifft(self, array: np.ndarray, axes: Optional[tuple] = None) -> np.ndarray:
        """Perform inverse FFT on CPU."""
        return np_fft.ifftn(array, axes=axes)

    def fftshift(self, array: np.ndarray, axes: Optional[tuple] = None) -> np.ndarray:
        """Perform FFT shift on CPU."""
        return np_fft.fftshift(array, axes=axes)

    def ifftshift(self, array: np.ndarray, axes: Optional[tuple] = None) -> np.ndarray:
        """Perform inverse FFT shift on CPU."""
        return np_fft.ifftshift(array, axes=axes)

    def get_memory_info(self) -> dict:
        """Get CPU memory information."""
        import psutil

        memory = psutil.virtual_memory()

        return {
            "total_memory": memory.total,
            "free_memory": memory.available,
            "used_memory": memory.used,
            "mempool_used": 0,
            "mempool_total": 0,
            "pinned_used": 0,
            "pinned_total": 0,
        }


def detect_cuda_availability() -> bool:
    """
    Detect if CUDA is available and working.

    Physical Meaning:
        Checks if CUDA is properly installed and functional
        for 7D phase field calculations.

    Returns:
        bool: True if CUDA is available and working.
    """
    if not CUDA_AVAILABLE:
        return False

    try:
        # Test basic CUDA operations
        test_array = cp.zeros((10, 10), dtype=cp.complex128)
        result = cp.fft.fft(test_array)
        cp.asnumpy(result)  # Test GPU->CPU transfer

        # Check if we can allocate reasonable memory
        test_large = cp.zeros((100, 100, 100), dtype=cp.complex128)
        del test_large

        return True
    except Exception as e:
        logger.warning(f"CUDA detection failed: {e}")
        return False


def get_optimal_backend() -> Union[CUDABackend, CPUBackend]:
    """
    Get the optimal backend for computations.

    Physical Meaning:
        Automatically selects the best available backend
        (CUDA GPU or CPU) for 7D phase field calculations
        based on availability and performance.

    Returns:
        Union[CUDABackend, CPUBackend]: Optimal backend for computations.
    """
    # Check environment variable override
    force_cpu = os.getenv("BHLFF_FORCE_CPU", "false").lower() == "true"
    force_cuda = os.getenv("BHLFF_FORCE_CUDA", "false").lower() == "true"

    if force_cpu:
        logger.info("Forcing CPU backend due to BHLFF_FORCE_CPU=true")
        return CPUBackend()

    if force_cuda and not CUDA_AVAILABLE:
        raise RuntimeError("CUDA forced but not available")

    # Try CUDA first if available
    if CUDA_AVAILABLE and detect_cuda_availability():
        try:
            backend = CUDABackend()
            logger.info("Using CUDA backend for optimal performance")
            return backend
        except Exception as e:
            logger.warning(f"CUDA backend failed: {e}, falling back to CPU")

    # Fall back to CPU
    logger.info("Using CPU backend")
    return CPUBackend()


def get_backend_info() -> dict:
    """
    Get information about the current backend.

    Physical Meaning:
        Provides detailed information about the computational
        backend being used for 7D phase field calculations.

    Returns:
        dict: Backend information including type, memory, and capabilities.
    """
    backend = get_optimal_backend()

    info = {
        "type": "CUDA" if isinstance(backend, CUDABackend) else "CPU",
        "cuda_available": CUDA_AVAILABLE,
        "numpy_fft_available": NUMPY_FFT_AVAILABLE,
        "memory_info": backend.get_memory_info(),
    }

    if isinstance(backend, CUDABackend):
        info["device_id"] = backend.device.id
        info["device_name"] = backend.device.name

    return info


# Global backend instance for automatic use
_global_backend: Optional[Union[CUDABackend, CPUBackend]] = None


def get_global_backend() -> Union[CUDABackend, CPUBackend]:
    """
    Get the global backend instance.

    Physical Meaning:
        Returns the global backend instance for use throughout
        the BHLFF framework, ensuring consistent GPU/CPU usage.

    Returns:
        Union[CUDABackend, CPUBackend]: Global backend instance.
    """
    global _global_backend
    if _global_backend is None:
        _global_backend = get_optimal_backend()
    return _global_backend


def reset_global_backend() -> None:
    """
    Reset the global backend instance.

    Physical Meaning:
        Resets the global backend to allow re-detection
        of optimal backend configuration.
    """
    global _global_backend
    _global_backend = None
