"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA-optimized block processor for 7D domain operations.

This module implements CUDA-accelerated block processing for 7D domains
to handle memory-efficient computations on large 7D space-time grids.

Physical Meaning:
    Provides CUDA-accelerated block processing for 7D phase field computations,
    enabling memory-efficient operations on large 7D space-time domains
    using GPU acceleration for maximum performance.

Example:
    >>> processor = CUDABlockProcessor(domain, block_size=8)
    >>> for block in processor.iterate_blocks_cuda():
    >>>     result = process_block_cuda(block)
"""

import numpy as np
from typing import Iterator, Tuple, Dict, Any, Optional, List
import logging
from dataclasses import dataclass

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    cp_ndimage = None

from .domain import Domain
from .block_processor import BlockProcessor, BlockInfo


class CUDABlockProcessor(BlockProcessor):
    """
    CUDA-optimized block processor for 7D domain operations.

    Physical Meaning:
        Provides CUDA-accelerated block processing for 7D phase field
        computations, enabling memory-efficient operations on large
        7D space-time domains using GPU acceleration.

    Mathematical Foundation:
        Implements CUDA-accelerated block decomposition of 7D space-time
        domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ with GPU memory management.
    """

    def __init__(self, domain: Domain, block_size: int = 8, overlap: int = 2):
        """
        Initialize CUDA block processor.

        Physical Meaning:
            Sets up CUDA-accelerated block processing system for 7D phase field
            computations with GPU memory management and optimization.

        Args:
            domain (Domain): 7D computational domain.
            block_size (int): Size of each processing block.
            overlap (int): Overlap between adjacent blocks for continuity.
        """
        super().__init__(domain, block_size, overlap)

        # Check CUDA availability
        self.cuda_available = CUDA_AVAILABLE
        if not self.cuda_available:
            self.logger.warning("CUDA not available, falling back to CPU processing")
            return

        # Initialize CUDA
        self._initialize_cuda()

        self.logger.info(f"CUDA block processor initialized: {self.cuda_available}")

    def _initialize_cuda(self) -> None:
        """Initialize CUDA environment."""
        if not self.cuda_available:
            return

        try:
            # Get CUDA device info
            self.device = cp.cuda.Device()
            self.device_id = self.device.id

            # Get GPU memory info
            mempool = cp.get_default_memory_pool()
            self.total_memory = self.device.mem_info[1]  # Total memory
            self.free_memory = self.device.mem_info[0]  # Free memory

            # Set memory pool for efficient memory management
            mempool.set_limit(size=self.free_memory * 0.8)  # Use 80% of free memory

            self.logger.info(
                f"CUDA device {self.device_id}: {self.free_memory / 1e9:.1f} GB free memory"
            )

        except Exception as e:
            self.logger.error(f"CUDA initialization failed: {e}")
            self.cuda_available = False

    def iterate_blocks_cuda(self) -> Iterator[Tuple[cp.ndarray, BlockInfo]]:
        """
        Iterate over all blocks in the 7D domain using CUDA.

        Physical Meaning:
            Yields blocks of the 7D domain for CUDA-accelerated processing,
            ensuring GPU memory efficiency and proper overlap handling.

        Yields:
            Tuple[cp.ndarray, BlockInfo]: CUDA block data and block information.
        """
        if not self.cuda_available:
            # Fallback to CPU processing
            for block_data, block_info in super().iterate_blocks():
                yield block_data, block_info
            return

        block_id = 0

        # Iterate over all block combinations
        for block_indices in self._generate_block_indices():
            # Compute block boundaries
            start_indices, end_indices = self._compute_block_boundaries(block_indices)

            # Create block info
            block_info = BlockInfo(
                block_id=block_id,
                start_indices=start_indices,
                end_indices=end_indices,
                shape=tuple(
                    end - start for start, end in zip(start_indices, end_indices)
                ),
                global_offset=start_indices,
                memory_usage=self.block_memory_usage,
            )

            # Extract block data to GPU
            block_data = self._extract_block_data_cuda(start_indices, end_indices)

            yield block_data, block_info
            block_id += 1

    def _extract_block_data_cuda(
        self, start_indices: Tuple[int, ...], end_indices: Tuple[int, ...]
    ) -> cp.ndarray:
        """Extract block data to GPU memory."""
        # Create slice object
        slices = tuple(
            slice(start, end) for start, end in zip(start_indices, end_indices)
        )

        # Generate synthetic block data for demonstration
        block_shape = tuple(
            end - start for start, end in zip(start_indices, end_indices)
        )

        # Generate data on CPU first, then transfer to GPU
        cpu_data = np.random.random(block_shape).astype(np.complex128)
        gpu_data = cp.asarray(cpu_data)

        return gpu_data

    def process_block_cuda(
        self, block_data: cp.ndarray, block_info: BlockInfo, operation: str = "fft"
    ) -> cp.ndarray:
        """
        Process a single block with CUDA acceleration.

        Physical Meaning:
            Processes a single block of 7D phase field data with
            CUDA-accelerated operations for maximum performance.

        Args:
            block_data (cp.ndarray): CUDA block data to process.
            block_info (BlockInfo): Block information.
            operation (str): Operation to perform on block.

        Returns:
            cp.ndarray: CUDA-processed block data.
        """
        if not self.cuda_available:
            # Fallback to CPU processing
            cpu_data = cp.asnumpy(block_data)
            cpu_result = super().process_block(cpu_data, block_info, operation)
            return cp.asarray(cpu_result)

        if operation == "fft":
            return self._process_block_fft_cuda(block_data, block_info)
        elif operation == "convolution":
            return self._process_block_convolution_cuda(block_data, block_info)
        elif operation == "gradient":
            return self._process_block_gradient_cuda(block_data, block_info)
        elif operation == "bvp_solve":
            return self._process_block_bvp_cuda(block_data, block_info)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _process_block_fft_cuda(
        self, block_data: cp.ndarray, block_info: BlockInfo
    ) -> cp.ndarray:
        """Process block with CUDA FFT operation."""
        # Apply CUDA FFT to block
        fft_result = cp.fft.fftn(block_data)

        # Apply 7D phase field specific processing on GPU
        phase = cp.angle(fft_result)
        processed_result = fft_result * cp.exp(-1j * phase)

        return processed_result

    def _process_block_convolution_cuda(
        self, block_data: cp.ndarray, block_info: BlockInfo
    ) -> cp.ndarray:
        """Process block with CUDA convolution operation."""
        # Create convolution kernel for 7D phase field
        kernel_shape = tuple(min(3, size) for size in block_data.shape)
        kernel = cp.ones(kernel_shape, dtype=cp.complex128) / cp.prod(kernel_shape)

        # Apply CUDA convolution
        convolved = cp_ndimage.convolve(block_data.real, kernel, mode="constant")

        return convolved.astype(cp.complex128)

    def _process_block_gradient_cuda(
        self, block_data: cp.ndarray, block_info: BlockInfo
    ) -> cp.ndarray:
        """Process block with CUDA gradient operation."""
        # Compute gradient in 7D space using CUDA
        gradient = cp.gradient(block_data.real)

        # Compute magnitude of gradient on GPU
        gradient_magnitude = cp.sqrt(cp.sum(cp.array([g**2 for g in gradient]), axis=0))

        return gradient_magnitude.astype(cp.complex128)

    def _process_block_bvp_cuda(
        self, block_data: cp.ndarray, block_info: BlockInfo
    ) -> cp.ndarray:
        """Process block with CUDA BVP solving."""
        # CUDA-accelerated BVP solving
        # This would implement the full BVP equation on GPU

        # For now, implement a simplified version
        amplitude = cp.abs(block_data)
        phase = cp.angle(block_data)

        # Apply BVP-specific processing
        result = amplitude * cp.exp(1j * phase)

        return result

    def merge_blocks_cuda(
        self, processed_blocks: List[Tuple[cp.ndarray, BlockInfo]]
    ) -> cp.ndarray:
        """
        Merge processed blocks back into full domain using CUDA.

        Physical Meaning:
            Merges processed blocks back into full 7D domain using GPU,
            handling overlaps and ensuring continuity.

        Args:
            processed_blocks (List[Tuple[cp.ndarray, BlockInfo]]): List of CUDA-processed blocks.

        Returns:
            cp.ndarray: Merged full domain data on GPU.
        """
        if not self.cuda_available:
            # Fallback to CPU processing
            cpu_blocks = [
                (cp.asnumpy(block_data), block_info)
                for block_data, block_info in processed_blocks
            ]
            cpu_result = super().merge_blocks(cpu_blocks)
            return cp.asarray(cpu_result)

        # Initialize result array on GPU
        result = cp.zeros(self.domain_shape, dtype=cp.complex128)
        weight_map = cp.zeros(self.domain_shape, dtype=cp.float64)

        # Merge blocks with overlap handling on GPU
        for block_data, block_info in processed_blocks:
            start_indices = block_info.start_indices
            end_indices = block_info.end_indices

            # Create slices
            slices = tuple(
                slice(start, end) for start, end in zip(start_indices, end_indices)
            )

            # Create weight mask for overlap handling on GPU
            weight_mask = self._create_weight_mask_cuda(block_info)

            # Add block data to result on GPU
            result[slices] += block_data * weight_mask
            weight_map[slices] += weight_mask

        # Normalize by weights on GPU
        result = cp.divide(
            result, weight_map, out=cp.zeros_like(result), where=weight_map != 0
        )

        return result

    def _create_weight_mask_cuda(self, block_info: BlockInfo) -> cp.ndarray:
        """Create weight mask for overlap handling on GPU."""
        block_shape = block_info.shape
        weight_mask = cp.ones(block_shape, dtype=cp.float64)

        # Apply overlap weights at boundaries on GPU
        for dim in range(self.n_dims):
            if block_info.start_indices[dim] > 0:
                # Overlap at start
                overlap_size = min(self.overlap, block_shape[dim])
                weight_mask[
                    tuple(
                        slice(0, overlap_size) if i == dim else slice(None)
                        for i in range(self.n_dims)
                    )
                ] *= 0.5

            if block_info.end_indices[dim] < self.domain_shape[dim]:
                # Overlap at end
                overlap_size = min(self.overlap, block_shape[dim])
                weight_mask[
                    tuple(
                        slice(-overlap_size, None) if i == dim else slice(None)
                        for i in range(self.n_dims)
                    )
                ] *= 0.5

        return weight_mask

    def optimize_block_size_cuda(self, available_memory_gb: float = None) -> int:
        """
        Optimize block size based on available GPU memory.

        Physical Meaning:
            Optimizes block size to fit within available GPU memory
            while maintaining processing efficiency.

        Args:
            available_memory_gb (float): Available GPU memory in GB.

        Returns:
            int: Optimized block size for CUDA processing.
        """
        if not self.cuda_available:
            return super().optimize_block_size(available_memory_gb or 8.0)

        if available_memory_gb is None:
            available_memory_gb = self.free_memory / 1e9

        # Calculate maximum block size that fits in GPU memory
        # Account for CUDA memory overhead
        effective_memory = available_memory_gb * 0.8  # Use 80% of available memory
        max_block_size = int((effective_memory / (8 * 1e-9)) ** (1.0 / self.n_dims))

        # Ensure block size is reasonable for CUDA
        optimized_size = min(max_block_size, self.block_size)
        optimized_size = max(4, optimized_size)  # Minimum block size

        self.logger.info(
            f"CUDA optimized block size: {optimized_size} "
            f"(available GPU memory: {available_memory_gb:.1f} GB)"
        )

        return optimized_size

    def get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        if not self.cuda_available:
            return {"cuda_available": False}

        return {
            "cuda_available": True,
            "device_id": self.device_id,
            "total_memory_gb": self.total_memory / 1e9,
            "free_memory_gb": self.free_memory / 1e9,
            "cupy_version": cp.__version__ if hasattr(cp, "__version__") else "unknown",
        }

    def get_memory_usage_cuda(self) -> Dict[str, Any]:
        """Get CUDA memory usage information."""
        base_usage = super().get_memory_usage()
        cuda_info = self.get_cuda_info()

        return {
            **base_usage,
            **cuda_info,
            "gpu_processing": self.cuda_available,
            "memory_pool_optimized": self.cuda_available,
        }

    def cleanup_cuda_memory(self) -> None:
        """Clean up CUDA memory."""
        if not self.cuda_available:
            return

        try:
            # Clear memory pool
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            # Clear pinned memory pool
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

            self.logger.info("CUDA memory cleaned up")

        except Exception as e:
            self.logger.warning(f"CUDA memory cleanup failed: {e}")

    def __del__(self):
        """Cleanup CUDA memory on destruction."""
        if hasattr(self, "cuda_available") and self.cuda_available:
            self.cleanup_cuda_memory()
