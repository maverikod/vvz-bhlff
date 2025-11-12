"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Batched FFT operations for multiple fields with optimal GPU memory usage.

This module provides batched FFT operations for processing multiple fields
efficiently using GPU with automatic batch size calculation based on 80%
GPU memory usage.

Physical Meaning:
    Performs FFT operations on multiple 7D phase fields simultaneously,
    optimizing GPU memory usage by processing fields in batches that fit
    within 80% of available GPU memory.

Mathematical Foundation:
    For multiple fields {f₁, f₂, ..., fₙ}:
    - Batched forward FFT: {F₁, F₂, ..., Fₙ} = {FFT(f₁), FFT(f₂), ..., FFT(fₙ)}
    - Batched inverse FFT: {f₁, f₂, ..., fₙ} = {IFFT(F₁), IFFT(F₂), ..., IFFT(Fₙ)}
    - Batch size is calculated to use 80% of GPU memory for optimal performance

Example:
    >>> from bhlff.core.fft.batched_fft_operations import BatchedFFTOperations
    >>> batched_fft = BatchedFFTOperations(domain)
    >>> fields = [field1, field2, field3]
    >>> spectral_fields = batched_fft.batched_fftn(fields, normalization="ortho")
"""

import logging
from typing import List, Tuple, Optional, Union
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from ...utils.gpu_memory_monitor import GPUMemoryMonitor
from ..domain.optimal_block_size_calculator import OptimalBlockSizeCalculator

logger = logging.getLogger(__name__)


class BatchedFFTOperations:
    """
    Batched FFT operations for multiple fields with optimal GPU memory usage.
    
    Physical Meaning:
        Provides batched FFT operations for processing multiple 7D phase
        fields simultaneously, optimizing GPU memory usage by processing
        fields in batches that fit within 80% of available GPU memory.
        
    Mathematical Foundation:
        For multiple fields {f₁, f₂, ..., fₙ}:
        - Batched forward FFT: {F₁, F₂, ..., Fₙ} = {FFT(f₁), FFT(f₂), ..., FFT(fₙ)}
        - Batched inverse FFT: {f₁, f₂, ..., fₙ} = {IFFT(F₁), IFFT(F₂), ..., IFFT(Fₙ)}
        - Batch size is calculated to use 80% of GPU memory
        
    Attributes:
        domain_shape (Tuple[int, ...]): Shape of 7D domain.
        dtype (np.dtype): Data type for computations.
        gpu_memory_ratio (float): Fraction of GPU memory to use (default: 0.8).
        _block_size_calculator (OptimalBlockSizeCalculator): Calculator for optimal batch sizes.
        _gpu_memory_monitor (Optional[GPUMemoryMonitor]): GPU memory monitor.
    """
    
    def __init__(
        self,
        domain_shape: Tuple[int, ...],
        dtype: np.dtype = np.complex128,
        gpu_memory_ratio: float = 0.8,
    ):
        """
        Initialize batched FFT operations.
        
        Physical Meaning:
            Sets up batched FFT operations with optimal batch size calculation
            based on GPU memory availability.
            
        Args:
            domain_shape (Tuple[int, ...]): Shape of 7D domain.
            dtype (np.dtype): Data type for computations (default: complex128).
            gpu_memory_ratio (float): Fraction of GPU memory to use
                (default: 0.8 for 80% usage).
        """
        self.domain_shape = domain_shape
        self.dtype = dtype
        self.gpu_memory_ratio = gpu_memory_ratio
        
        # Initialize block size calculator for batch size calculation
        self._block_size_calculator = OptimalBlockSizeCalculator(
            gpu_memory_ratio=gpu_memory_ratio
        )
        
        # Initialize GPU memory monitor
        self._gpu_memory_monitor = None
        if CUDA_AVAILABLE:
            try:
                self._gpu_memory_monitor = GPUMemoryMonitor(
                    warning_threshold=0.75,
                    critical_threshold=0.9,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize GPUMemoryMonitor: {e}")
        
        logger.info(
            f"BatchedFFTOperations initialized for domain {domain_shape} "
            f"with GPU memory ratio {gpu_memory_ratio:.1%}"
        )
    
    def batched_fftn(
        self,
        fields: List[np.ndarray],
        normalization: str = "ortho",
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Perform batched forward FFT on multiple fields.
        
        Physical Meaning:
            Computes forward FFT for multiple 7D phase fields simultaneously,
            processing fields in batches that fit within GPU memory limits.
            
        Mathematical Foundation:
            For fields {f₁, f₂, ..., fₙ}:
            Returns {F₁, F₂, ..., Fₙ} where Fᵢ = FFT(fᵢ)
            
        Args:
            fields (List[np.ndarray]): List of 7D fields to transform.
            normalization (str): FFT normalization mode (default: "ortho").
            batch_size (Optional[int]): Batch size for processing.
                If None, automatically calculated based on GPU memory.
                
        Returns:
            List[np.ndarray]: List of spectral fields (FFT results).
            
        Raises:
            ValueError: If fields have incompatible shapes.
            RuntimeError: If GPU memory is insufficient.
        """
        if not fields:
            return []
        
        # Validate field shapes
        for i, field in enumerate(fields):
            if field.shape != self.domain_shape:
                raise ValueError(
                    f"Field {i} has shape {field.shape}, expected {self.domain_shape}"
                )
        
        # Calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(fields[0])
        
        # Check GPU memory before processing
        if CUDA_AVAILABLE and self._gpu_memory_monitor is not None:
            try:
                memory_info = self._gpu_memory_monitor.check_memory()
                if memory_info.get("usage_ratio", 0.0) > 0.9:
                    logger.warning(
                        f"GPU memory usage high: {memory_info.get('usage_ratio', 0.0):.1%}, "
                        f"proceeding with caution"
                    )
            except Exception as e:
                logger.warning(f"GPU memory check failed: {e}")
        
        # Process fields in batches
        results = []
        num_batches = (len(fields) + batch_size - 1) // batch_size
        
        logger.info(
            f"Processing {len(fields)} fields in {num_batches} batches "
            f"(batch_size={batch_size})"
        )
        
        for i in range(0, len(fields), batch_size):
            batch_num = i // batch_size + 1
            batch_fields = fields[i:i + batch_size]
            
            logger.debug(
                f"Processing batch {batch_num}/{num_batches}: "
                f"{len(batch_fields)} fields"
            )
            
            # Process batch
            if CUDA_AVAILABLE:
                batch_results = self._process_batch_cuda(
                    batch_fields, normalization, forward=True
                )
            else:
                batch_results = self._process_batch_cpu(
                    batch_fields, normalization, forward=True
                )
            
            results.extend(batch_results)
        
        return results
    
    def batched_ifftn(
        self,
        spectral_fields: List[np.ndarray],
        normalization: str = "ortho",
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Perform batched inverse FFT on multiple spectral fields.
        
        Physical Meaning:
            Computes inverse FFT for multiple 7D spectral fields simultaneously,
            processing fields in batches that fit within GPU memory limits.
            
        Mathematical Foundation:
            For spectral fields {F₁, F₂, ..., Fₙ}:
            Returns {f₁, f₂, ..., fₙ} where fᵢ = IFFT(Fᵢ)
            
        Args:
            spectral_fields (List[np.ndarray]): List of 7D spectral fields to transform.
            normalization (str): FFT normalization mode (default: "ortho").
            batch_size (Optional[int]): Batch size for processing.
                If None, automatically calculated based on GPU memory.
                
        Returns:
            List[np.ndarray]: List of real-space fields (IFFT results).
            
        Raises:
            ValueError: If fields have incompatible shapes.
            RuntimeError: If GPU memory is insufficient.
        """
        if not spectral_fields:
            return []
        
        # Validate field shapes
        for i, field in enumerate(spectral_fields):
            if field.shape != self.domain_shape:
                raise ValueError(
                    f"Spectral field {i} has shape {field.shape}, "
                    f"expected {self.domain_shape}"
                )
        
        # Calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(spectral_fields[0])
        
        # Check GPU memory before processing
        if CUDA_AVAILABLE and self._gpu_memory_monitor is not None:
            try:
                memory_info = self._gpu_memory_monitor.check_memory()
                if memory_info.get("usage_ratio", 0.0) > 0.9:
                    logger.warning(
                        f"GPU memory usage high: {memory_info.get('usage_ratio', 0.0):.1%}, "
                        f"proceeding with caution"
                    )
            except Exception as e:
                logger.warning(f"GPU memory check failed: {e}")
        
        # Process fields in batches
        results = []
        num_batches = (len(spectral_fields) + batch_size - 1) // batch_size
        
        logger.info(
            f"Processing {len(spectral_fields)} spectral fields in {num_batches} batches "
            f"(batch_size={batch_size})"
        )
        
        for i in range(0, len(spectral_fields), batch_size):
            batch_num = i // batch_size + 1
            batch_fields = spectral_fields[i:i + batch_size]
            
            logger.debug(
                f"Processing batch {batch_num}/{num_batches}: "
                f"{len(batch_fields)} fields"
            )
            
            # Process batch
            if CUDA_AVAILABLE:
                batch_results = self._process_batch_cuda(
                    batch_fields, normalization, forward=False
                )
            else:
                batch_results = self._process_batch_cpu(
                    batch_fields, normalization, forward=False
                )
            
            results.extend(batch_results)
        
        return results
    
    def _calculate_optimal_batch_size(self, sample_field: np.ndarray) -> int:
        """
        Calculate optimal batch size based on GPU memory.
        
        Physical Meaning:
            Computes maximum number of fields that can be processed
            simultaneously while staying within GPU memory limits.
            
        Args:
            sample_field (np.ndarray): Sample field for size calculation.
            
        Returns:
            int: Optimal batch size.
        """
        if not CUDA_AVAILABLE:
            # CPU fallback: use smaller batches
            return min(4, len(sample_field.shape))
        
        try:
            # Get GPU memory info
            mem_info = cp.cuda.runtime.memGetInfo()
            free_memory_bytes = mem_info[0]
            available_memory_bytes = int(free_memory_bytes * self.gpu_memory_ratio)
            
            # Memory per field (accounting for FFT workspace)
            bytes_per_field = sample_field.nbytes * 4  # 4x overhead for FFT workspace
            max_batch_size = max(1, available_memory_bytes // bytes_per_field)
            
            # Limit batch size to reasonable maximum
            max_batch_size = min(max_batch_size, 32)
            
            logger.debug(
                f"Optimal batch size: {max_batch_size} "
                f"(available memory: {available_memory_bytes / (1024**3):.2f} GB, "
                f"per field: {bytes_per_field / (1024**2):.2f} MB)"
            )
            
            return max_batch_size
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}, using default")
            return 4
    
    def _process_batch_cuda(
        self,
        batch_fields: List[np.ndarray],
        normalization: str,
        forward: bool,
    ) -> List[np.ndarray]:
        """
        Process batch of fields on GPU using CUDA.
        
        Physical Meaning:
            Performs FFT operations on batch of fields using GPU acceleration
            with vectorized operations for optimal performance.
            
        Args:
            batch_fields (List[np.ndarray]): Batch of fields to process.
            normalization (str): FFT normalization mode.
            forward (bool): True for forward FFT, False for inverse FFT.
            
        Returns:
            List[np.ndarray]: Processed fields.
        """
        # Convert to GPU arrays
        batch_gpu = [cp.asarray(field, dtype=self.dtype) for field in batch_fields]
        
        # Stack for vectorized processing
        stacked = cp.stack(batch_gpu)
        
        # Perform FFT
        if forward:
            result_stacked = cp.fft.fftn(
                stacked, axes=tuple(range(1, stacked.ndim)), norm=normalization
            )
        else:
            result_stacked = cp.fft.ifftn(
                stacked, axes=tuple(range(1, stacked.ndim)), norm=normalization
            )
        
        # Convert back to CPU and split
        results = [cp.asnumpy(result_stacked[i]) for i in range(len(batch_fields))]
        
        # Clean up GPU memory
        del batch_gpu, stacked, result_stacked
        cp.get_default_memory_pool().free_all_blocks()
        
        return results
    
    def _process_batch_cpu(
        self,
        batch_fields: List[np.ndarray],
        normalization: str,
        forward: bool,
    ) -> List[np.ndarray]:
        """
        Process batch of fields on CPU.
        
        Physical Meaning:
            Performs FFT operations on batch of fields using CPU with
            vectorized NumPy operations.
            
        Args:
            batch_fields (List[np.ndarray]): Batch of fields to process.
            normalization (str): FFT normalization mode.
            forward (bool): True for forward FFT, False for inverse FFT.
            
        Returns:
            List[np.ndarray]: Processed fields.
        """
        # Stack for vectorized processing
        stacked = np.stack(batch_fields)
        
        # Perform FFT
        if forward:
            result_stacked = np.fft.fftn(
                stacked, axes=tuple(range(1, stacked.ndim)), norm=normalization
            )
        else:
            result_stacked = np.fft.ifftn(
                stacked, axes=tuple(range(1, stacked.ndim)), norm=normalization
            )
        
        # Split back into individual fields
        results = [result_stacked[i] for i in range(len(batch_fields))]
        
        return results

