"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Enhanced block processor for 7D BVP data processing.

This module implements an enhanced block processing system for 7D BVP computations
with intelligent memory management, adaptive block sizing, and efficient data flow.

Physical Meaning:
    Provides intelligent block-based processing for 7D phase field computations,
    enabling memory-efficient operations on large 7D space-time domains with
    adaptive block sizing based on available memory and processing capabilities.

Mathematical Foundation:
    Implements adaptive block decomposition of 7D space-time domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    with intelligent memory management and processing optimization.

Example:
    >>> processor = EnhancedBlockProcessor(domain, config)
    >>> result = processor.process_7d_field(field, operation="bvp_solve")
"""

import numpy as np
import cupy as cp
from typing import Dict, Any, Optional, Tuple, List, Callable
import logging
import psutil
import gc
from dataclasses import dataclass
from enum import Enum

from .block_processor import BlockProcessor, BlockInfo
from .domain import Domain
from ...utils.memory_monitor import MemoryMonitor


class ProcessingMode(Enum):
    """Processing mode for block operations."""

    CPU_ONLY = "cpu_only"
    GPU_PREFERRED = "gpu_preferred"
    GPU_ONLY = "gpu_only"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingConfig:
    """Configuration for enhanced block processing."""

    mode: ProcessingMode = ProcessingMode.ADAPTIVE
    max_memory_usage: float = 0.8  # 80% of available memory
    min_block_size: int = 4
    max_block_size: int = 64
    overlap_ratio: float = 0.1  # 10% overlap between blocks
    batch_size: int = 4
    enable_memory_optimization: bool = True
    enable_adaptive_sizing: bool = True
    enable_parallel_processing: bool = True


class EnhancedBlockProcessor:
    """
    Enhanced block processor for 7D BVP data processing.

    Physical Meaning:
        Provides intelligent block-based processing for 7D phase field computations
        with adaptive memory management and processing optimization.

    Mathematical Foundation:
        Implements adaptive block decomposition with intelligent memory management
        for efficient 7D BVP computations.
    """

    def __init__(self, domain: Domain, config: ProcessingConfig = None):
        """
        Initialize enhanced block processor.

        Physical Meaning:
            Sets up enhanced block processing system with intelligent memory
            management and adaptive block sizing for 7D BVP computations.

        Args:
            domain (Domain): 7D computational domain.
            config (ProcessingConfig): Processing configuration.
        """
        self.domain = domain
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize memory monitoring
        self.memory_monitor = MemoryMonitor()

        # Initialize base block processor
        self.base_processor = BlockProcessor(
            domain, self._calculate_optimal_block_size()
        )

        # CUDA availability
        self.cuda_available = self._check_cuda_availability()

        # Processing statistics
        self.stats = {
            "blocks_processed": 0,
            "memory_peak_usage": 0.0,
            "processing_time": 0.0,
            "fallback_count": 0,
        }

        self.logger.info(
            f"Enhanced block processor initialized: "
            f"mode={self.config.mode.value}, "
            f"cuda_available={self.cuda_available}"
        )

    def process_7d_field(
        self, field: np.ndarray, operation: str = "bvp_solve", **kwargs
    ) -> np.ndarray:
        """
        Process 7D field using enhanced block processing.

        Physical Meaning:
            Processes 7D phase field using intelligent block decomposition
            with adaptive memory management and processing optimization.

        Args:
            field (np.ndarray): 7D phase field to process.
            operation (str): Processing operation to perform.
            **kwargs: Additional operation parameters.

        Returns:
            np.ndarray: Processed 7D field.
        """
        self.logger.info(
            f"Processing 7D field: shape={field.shape}, operation={operation}"
        )

        # Check memory requirements
        if not self._check_memory_requirements(field):
            self.logger.warning("Insufficient memory, using fallback processing")
            return self._fallback_processing(field, operation, **kwargs)

        # Choose processing strategy
        if self.config.mode == ProcessingMode.ADAPTIVE:
            strategy = self._choose_adaptive_strategy(field)
        else:
            strategy = self.config.mode

        # Process based on strategy
        if strategy == ProcessingMode.GPU_PREFERRED and self.cuda_available:
            return self._process_gpu_preferred(field, operation, **kwargs)
        elif strategy == ProcessingMode.GPU_ONLY and self.cuda_available:
            return self._process_gpu_only(field, operation, **kwargs)
        else:
            return self._process_cpu_optimized(field, operation, **kwargs)

    def _calculate_optimal_block_size(self) -> int:
        """
        Calculate optimal block size based on available memory and domain size.

        Physical Meaning:
            Calculates optimal block size to maximize processing efficiency
            while staying within memory constraints.

        Returns:
            int: Optimal block size.
        """
        # Get available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        usable_memory = available_memory * self.config.max_memory_usage

        # Calculate domain size
        domain_size = np.prod(self.domain.shape)
        element_size = 16  # complex128 = 16 bytes

        # Calculate maximum block size that fits in memory
        max_block_size = int(
            (usable_memory * 1024**3 / (element_size * 8))
            ** (1.0 / len(self.domain.shape))
        )

        # Apply constraints
        optimal_size = min(max_block_size, self.config.max_block_size)
        optimal_size = max(optimal_size, self.config.min_block_size)

        # Ensure it's reasonable for the domain
        for dim_size in self.domain.shape:
            optimal_size = min(optimal_size, dim_size)

        self.logger.info(
            f"Optimal block size: {optimal_size} "
            f"(available memory: {available_memory:.2f} GB)"
        )

        return optimal_size

    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            cp.cuda.Device(0).compute_capability
            return True
        except:
            return False

    def _check_memory_requirements(self, field: np.ndarray) -> bool:
        """
        Check if there's enough memory for processing.

        Physical Meaning:
            Verifies that available memory is sufficient for processing
            the 7D field with the current block configuration.
        """
        # Estimate memory requirements
        field_memory = field.nbytes / (1024**3)  # GB
        processing_memory = field_memory * 3  # 3x for processing overhead

        available_memory = psutil.virtual_memory().available / (1024**3)
        required_memory = processing_memory / self.config.max_memory_usage

        sufficient = available_memory >= required_memory

        if not sufficient:
            self.logger.warning(
                f"Insufficient memory: required={required_memory:.2f} GB, "
                f"available={available_memory:.2f} GB"
            )

        return sufficient

    def _choose_adaptive_strategy(self, field: np.ndarray) -> ProcessingMode:
        """
        Choose processing strategy based on field size and available resources.

        Physical Meaning:
            Intelligently selects the best processing strategy based on
            field characteristics and available computational resources.
        """
        field_size = field.nbytes / (1024**3)  # GB
        available_memory = psutil.virtual_memory().available / (1024**3)

        # If field is small, prefer GPU for speed
        if field_size < 0.5 and self.cuda_available:
            return ProcessingMode.GPU_PREFERRED

        # If field is large, prefer CPU for memory efficiency
        elif field_size > 2.0:
            return ProcessingMode.CPU_ONLY

        # Otherwise, use adaptive approach
        else:
            return ProcessingMode.ADAPTIVE

    def _process_gpu_preferred(
        self, field: np.ndarray, operation: str, **kwargs
    ) -> np.ndarray:
        """Process field with GPU preference and CPU fallback."""
        try:
            return self._process_gpu_blocks(field, operation, **kwargs)
        except Exception as e:
            self.logger.warning(f"GPU processing failed: {e}, falling back to CPU")
            self.stats["fallback_count"] += 1
            return self._process_cpu_optimized(field, operation, **kwargs)

    def _process_gpu_only(
        self, field: np.ndarray, operation: str, **kwargs
    ) -> np.ndarray:
        """Process field using GPU only."""
        if not self.cuda_available:
            raise RuntimeError("GPU processing requested but CUDA not available")

        return self._process_gpu_blocks(field, operation, **kwargs)

    def _process_cpu_optimized(
        self, field: np.ndarray, operation: str, **kwargs
    ) -> np.ndarray:
        """Process field using CPU with optimizations."""
        self.logger.info("Processing with CPU optimization")

        # Use base processor with optimized settings
        result = np.zeros_like(field, dtype=np.complex128)

        # Process in blocks
        for block_data, block_info in self.base_processor.iterate_blocks():
            processed_block = self._process_single_block_cpu(
                block_data, operation, **kwargs
            )
            self._merge_block_result(result, processed_block, block_info)

            # Memory cleanup
            if self.config.enable_memory_optimization:
                gc.collect()

        return result

    def _process_gpu_blocks(
        self, field: np.ndarray, operation: str, **kwargs
    ) -> np.ndarray:
        """Process field using GPU blocks."""
        self.logger.info("Processing with GPU blocks")

        # Transfer to GPU
        field_gpu = cp.asarray(field)
        result_gpu = cp.zeros_like(field_gpu, dtype=cp.complex128)

        try:
            # Process in blocks on GPU
            for block_data, block_info in self.base_processor.iterate_blocks():
                # Extract block on GPU
                block_gpu = self._extract_block_gpu(field_gpu, block_info)

                # Process block
                processed_block = self._process_single_block_gpu(
                    block_gpu, operation, **kwargs
                )

                # Merge result
                self._merge_block_result_gpu(result_gpu, processed_block, block_info)

            # Transfer back to CPU
            result = cp.asnumpy(result_gpu)

        finally:
            # Cleanup GPU memory
            del field_gpu, result_gpu
            cp.get_default_memory_pool().free_all_blocks()

        return result

    def _process_single_block_cpu(
        self, block_data: np.ndarray, operation: str, **kwargs
    ) -> np.ndarray:
        """Process a single block on CPU."""
        if operation == "bvp_solve":
            return self._solve_bvp_block_cpu(block_data, **kwargs)
        elif operation == "fft":
            return np.fft.fftn(block_data)
        elif operation == "ifft":
            return np.fft.ifftn(block_data)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _process_single_block_gpu(
        self, block_data: cp.ndarray, operation: str, **kwargs
    ) -> cp.ndarray:
        """Process a single block on GPU."""
        if operation == "bvp_solve":
            return self._solve_bvp_block_gpu(block_data, **kwargs)
        elif operation == "fft":
            return cp.fft.fftn(block_data)
        elif operation == "ifft":
            return cp.fft.ifftn(block_data)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _solve_bvp_block_cpu(self, block_data: np.ndarray, **kwargs) -> np.ndarray:
        """Solve BVP equation for a block on CPU."""
        # Simplified BVP solver for demonstration
        # In practice, this would implement the full BVP envelope equation
        return block_data * 0.5  # Placeholder implementation

    def _solve_bvp_block_gpu(self, block_data: cp.ndarray, **kwargs) -> cp.ndarray:
        """Solve BVP equation for a block on GPU."""
        # Simplified BVP solver for demonstration
        # In practice, this would implement the full BVP envelope equation
        return block_data * 0.5  # Placeholder implementation

    def _extract_block_gpu(
        self, field_gpu: cp.ndarray, block_info: BlockInfo
    ) -> cp.ndarray:
        """Extract block from GPU field."""
        start_indices = block_info.start_indices
        end_indices = block_info.end_indices

        slices = tuple(
            slice(start, end) for start, end in zip(start_indices, end_indices)
        )
        return field_gpu[slices]

    def _merge_block_result(
        self, result: np.ndarray, block_result: np.ndarray, block_info: BlockInfo
    ) -> None:
        """Merge block result into main result array."""
        start_indices = block_info.start_indices
        end_indices = block_info.end_indices

        slices = tuple(
            slice(start, end) for start, end in zip(start_indices, end_indices)
        )
        result[slices] = block_result

    def _merge_block_result_gpu(
        self, result_gpu: cp.ndarray, block_result: cp.ndarray, block_info: BlockInfo
    ) -> None:
        """Merge block result into main result array on GPU."""
        start_indices = block_info.start_indices
        end_indices = block_info.end_indices

        slices = tuple(
            slice(start, end) for start, end in zip(start_indices, end_indices)
        )
        result_gpu[slices] = block_result

    def _fallback_processing(
        self, field: np.ndarray, operation: str, **kwargs
    ) -> np.ndarray:
        """Fallback processing for memory-constrained situations."""
        self.logger.info("Using fallback processing due to memory constraints")

        # Use minimal block size
        original_block_size = self.base_processor.block_size
        self.base_processor.block_size = self.config.min_block_size

        try:
            result = self._process_cpu_optimized(field, operation, **kwargs)
        finally:
            # Restore original block size
            self.base_processor.block_size = original_block_size

        return result

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            "cuda_available": self.cuda_available,
            "current_block_size": self.base_processor.block_size,
            "memory_usage": self.memory_monitor.get_cpu_memory_usage(),
        }

    def optimize_for_field(self, field: np.ndarray) -> None:
        """
        Optimize processor settings for a specific field.

        Physical Meaning:
            Optimizes processor configuration based on field characteristics
            to maximize processing efficiency.
        """
        field_size = field.nbytes / (1024**3)  # GB

        # Adjust block size based on field size
        if field_size < 0.1:  # Small field
            self.base_processor.block_size = min(32, self.config.max_block_size)
        elif field_size < 1.0:  # Medium field
            self.base_processor.block_size = min(16, self.config.max_block_size)
        else:  # Large field
            self.base_processor.block_size = self.config.min_block_size

        self.logger.info(
            f"Optimized block size for field: {self.base_processor.block_size}"
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.cuda_available:
            cp.get_default_memory_pool().free_all_blocks()

        gc.collect()
        self.logger.info("Enhanced block processor cleaned up")
