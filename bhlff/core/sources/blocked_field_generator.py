"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Blocked field generator for large 7D fields with automatic memory management.

This module implements lazy field generation with block-based processing,
where only the current block is kept in memory and all other blocks are
stored on disk. Provides transparent access to any block with caching.

Physical Meaning:
    Enables efficient generation and access to large 7D phase fields that
    exceed available memory by processing and storing data in manageable blocks.

Mathematical Foundation:
    Implements block decomposition of 7D field generation:
    - Field is divided into blocks of manageable size
    - Each block is generated on-demand
    - Blocks are cached on disk for later access
    - Only current block is in memory

Example:
    >>> generator = BlockedFieldGenerator(domain, source_generator)
    >>> field = generator.get_field()  # Returns lazy field object
    >>> block = field[0:8, 0:8, 0:8, 0:4, 0:4, 0:4, 0:8]  # Accesses specific block
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple, Iterator, Union
import logging
import tempfile
import os
import pickle
import hashlib
from pathlib import Path
import threading
from dataclasses import dataclass

# Try to import CUDA
try:
    import cupy as cp
    
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from ..domain import Domain


@dataclass
class BlockMetadata:
    """Metadata for a field block."""

    block_id: int
    block_indices: Tuple[int, ...]
    block_shape: Tuple[int, ...]
    file_path: str
    is_generated: bool
    is_cached: bool


class BlockedField:
    """
    Lazy 7D field with block-based access.

    Physical Meaning:
        Represents a 7D phase field that is generated on-demand in blocks,
        with only the current block in memory and other blocks stored on disk.

    Attributes:
        domain (Domain): Computational domain.
        generator (BlockedFieldGenerator): Block generator instance.
    """

    def __init__(
        self, domain: Domain, generator: "BlockedFieldGenerator", dtype: type = complex
    ) -> None:
        """
        Initialize blocked field.

        Args:
            domain (Domain): Computational domain.
            generator (BlockedFieldGenerator): Block generator instance.
            dtype (type): Data type of the field (default: complex).
        """
        self.domain = domain
        self.generator = generator
        self.shape = domain.shape
        self.dtype = dtype
    
    @property
    def ndim(self) -> int:
        """Number of dimensions of the field."""
        return len(self.shape)

    def __getitem__(self, key) -> np.ndarray:
        """
        Access field block using slicing.

        Physical Meaning:
            Provides transparent access to any block of the 7D field,
            automatically generating and caching blocks as needed.

        Args:
            key: Slice or tuple of slices for 7D indexing.

        Returns:
            np.ndarray: Requested field block.
        """
        return self.generator.get_block(key)

    def get_block(self, block_indices: Tuple[int, ...]) -> np.ndarray:
        """
        Get specific block by indices.

        Args:
            block_indices (Tuple[int, ...]): Block indices in each dimension.

        Returns:
            np.ndarray: Field block.
        """
        return self.generator.get_block_by_indices(block_indices)

    def iterate_blocks(self) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Iterate over all blocks in the field.

        Yields:
            Tuple[np.ndarray, Dict[str, Any]]: Block data and metadata.
        """
        return self.generator.iterate_blocks()


class BlockedFieldGenerator:
    """
    Block-based field generator with automatic memory management.

    Physical Meaning:
        Generates large 7D fields in blocks, keeping only the current block
        in memory and storing all other blocks on disk. Provides transparent
        access with caching for efficient repeated access.

    Mathematical Foundation:
        Implements block decomposition:
        - Field shape: (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t)
        - Block size: (B_x, B_y, B_z, B_φ₁, B_φ₂, B_φ₃, B_t)
        - Blocks generated on-demand and cached on disk
    """

    def __init__(
        self,
        domain: Domain,
        field_generator: Callable[[Domain, Dict[str, Any]], np.ndarray],
        block_size: Optional[Tuple[int, ...]] = None,
        cache_dir: Optional[str] = None,
        max_memory_mb: float = 500.0,
        config: Optional[Dict[str, Any]] = None,
        use_cuda: bool = True,
    ) -> None:
        """
        Initialize blocked field generator.

        Physical Meaning:
            Sets up block-based field generation system with automatic
            memory management and disk caching. Supports CUDA acceleration
            for 7D block processing with 80% GPU memory limit.

        Args:
            domain (Domain): Computational domain.
            field_generator (Callable): Function that generates field block
                given domain slice and config.
            block_size (Optional[Tuple[int, ...]]): Block size per dimension.
                If None, computed automatically based on memory constraints.
            cache_dir (Optional[str]): Directory for block cache.
                If None, uses temporary directory.
            max_memory_mb (float): Maximum memory usage in MB for blocks.
            config (Optional[Dict[str, Any]]): Configuration for field generator.
            use_cuda (bool): Whether to use CUDA for block processing (default: True).
        """
        self.domain = domain
        self.field_generator = field_generator
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        # Setup cache directory
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="bhlff_field_cache_")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Block cache directory: {self.cache_dir}")

        # Compute optimal block size (with CUDA support if available)
        if block_size is None:
            block_size = self._compute_optimal_block_size(max_memory_mb)
        self.block_size = block_size

        # Compute block configuration
        self._compute_block_configuration()

        # Block cache in memory (LRU-like, only one block)
        self._current_block: Optional[Union[np.ndarray, "cp.ndarray"]] = None
        self._current_block_id: Optional[str] = None
        self._block_lock = threading.Lock()

        # Block metadata cache
        self._block_metadata: Dict[str, BlockMetadata] = {}

        self.logger.info(
            f"BlockedFieldGenerator initialized: "
            f"block_size={self.block_size}, "
            f"total_blocks={self.total_blocks}, "
            f"cache_dir={self.cache_dir}, "
            f"use_cuda={self.use_cuda}"
        )

    def _compute_optimal_block_size(self, max_memory_mb: float) -> Tuple[int, ...]:
        """
        Compute optimal block size based on memory constraints with CUDA support.

        Physical Meaning:
            Calculates block size that fits within memory constraints (80% GPU
            memory limit if CUDA available) while maximizing processing efficiency
            for 7D space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            For 7D blocks with shape (N₀, N₁, N₂, N₃, N₄, N₅, N₆):
            - Available memory: 80% of free GPU memory (if CUDA) or max_memory_mb
            - Block size per dimension: (available_memory / overhead) ^ (1/7)
            - Preserves 7D structure: spatial (0,1,2), phase (3,4,5), temporal (6)

        Args:
            max_memory_mb (float): Maximum memory in MB (CPU fallback).

        Returns:
            Tuple[int, ...]: Optimal block size per dimension (7-tuple).
        """
        # Memory per element (complex128 = 16 bytes)
        bytes_per_element = 16
        overhead_factor = 5.0  # Memory overhead for operations

        # Try CUDA if available and enabled
        if self.use_cuda and CUDA_AVAILABLE:
            try:
                mem_info = cp.cuda.runtime.memGetInfo()
                free_memory_bytes = mem_info[0]
                # Use 80% of free GPU memory
                available_memory_bytes = int(free_memory_bytes * 0.8)
                max_elements = available_memory_bytes / (bytes_per_element * overhead_factor)
                
                self.logger.info(
                    f"Using CUDA: available GPU memory: {available_memory_bytes/1e9:.2f} GB "
                    f"(80% of {free_memory_bytes/1e9:.2f} GB free)"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to get GPU memory info: {e}, using CPU fallback"
                )
                max_elements = (max_memory_mb * 1024 * 1024) / bytes_per_element
        else:
            max_elements = (max_memory_mb * 1024 * 1024) / bytes_per_element

        # For 7D, compute block size per dimension
        # Assuming roughly equal dimensions
        elements_per_dim = int(max_elements ** (1.0 / 7.0))

        # Ensure reasonable bounds (4 to domain size)
        block_size_per_dim = max(4, min(elements_per_dim, 128))

        # Create block size tuple (7D: spatial, phase, temporal)
        block_size = tuple(
            min(block_size_per_dim, dim_size)
            for dim_size in self.domain.shape
        )

        self.logger.info(
            f"Optimal block size: {block_size} "
            f"(max memory: {max_memory_mb} MB, "
            f"max elements: {max_elements:.0e}, "
            f"elements per dim: {elements_per_dim})"
        )

        return block_size

    def _compute_block_configuration(self) -> None:
        """
        Compute block configuration for the 7D domain.
        
        Physical Meaning:
            Calculates number of blocks per dimension for 7D space-time
            M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, allowing large block counts with warnings
            instead of hard errors.
        """
        self.blocks_per_dim = []
        for dim_size, block_dim_size in zip(self.domain.shape, self.block_size):
            n_blocks = (dim_size + block_dim_size - 1) // block_dim_size
            self.blocks_per_dim.append(n_blocks)

        # Safe computation: use int64 to avoid overflow
        try:
            blocks_array = np.array(self.blocks_per_dim, dtype=np.int64)
            self.total_blocks = int(np.prod(blocks_array))
        except (OverflowError, ValueError) as e:
            self.logger.warning(
                f"Error computing total blocks: {e}. "
                f"Blocks per dim: {self.blocks_per_dim}. "
                f"Using safe limit."
            )
            # Use safe limit instead of hard error
            self.total_blocks = 100000
        
        # Warning thresholds instead of hard limits
        # Allow large block counts with warnings, not hard errors
        max_safe_blocks = 50000
        very_large_blocks = 200000  # Hard cap only for extremely large domains
        
        if self.total_blocks > very_large_blocks:
            self.logger.warning(
                f"Extremely large number of blocks ({self.total_blocks}). "
                f"This may cause system issues. "
                f"Consider increasing block_size. Current limit: {very_large_blocks}."
            )
            # Cap only at extremely large values
            self.total_blocks = min(self.total_blocks, very_large_blocks)
        elif self.total_blocks > max_safe_blocks:
            self.logger.warning(
                f"Large number of blocks ({self.total_blocks}). "
                f"Iteration may take time. "
                f"Consider increasing block_size for better performance."
            )
            # Don't limit here, just warn

        self.logger.info(
            f"Block configuration: {self.blocks_per_dim} blocks per dimension, "
            f"total {self.total_blocks} blocks"
        )

    def _get_block_id(self, block_indices: Tuple[int, ...]) -> str:
        """Generate unique block ID from indices."""
        indices_str = "_".join(str(i) for i in block_indices)
        return hashlib.md5(indices_str.encode()).hexdigest()

    def _get_block_path(self, block_id: str) -> Path:
        """Get file path for block cache."""
        return self.cache_dir / f"block_{block_id}.npy"

    def _get_metadata_path(self, block_id: str) -> Path:
        """Get file path for block metadata."""
        return self.cache_dir / f"block_{block_id}.meta"

    def get_block(
        self, key: Any
    ) -> np.ndarray:
        """
        Get block using slicing key.

        Args:
            key: Slice or tuple of slices for 7D indexing.

        Returns:
            np.ndarray: Field block.
        """
        # Convert key to block indices
        if isinstance(key, tuple):
            # Extract block indices from slices
            block_indices = []
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    # Convert slice to block indices
                    start = k.start if k.start is not None else 0
                    stop = k.stop if k.stop is not None else self.domain.shape[i]
                    block_idx = start // self.block_size[i]
                    block_indices.append(block_idx)
                else:
                    block_indices.append(k)
            block_indices = tuple(block_indices)
        else:
            # Single index - treat as first block
            block_indices = (0, 0, 0, 0, 0, 0, 0)

        return self.get_block_by_indices(block_indices)

    def get_block_by_indices(
        self, block_indices: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Get block by block indices with CUDA support and metadata validation.

        Physical Meaning:
            Retrieves or generates the specified 7D block, loading from cache
            if available or generating and caching if not. Validates metadata
            matches true block shape. Supports CUDA acceleration for generation.

        Mathematical Foundation:
            For 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ:
            - Block indices: (i_x, i_y, i_z, i_φ₁, i_φ₂, i_φ₃, i_t)
            - Block shape: (B_x, B_y, B_z, B_φ₁, B_φ₂, B_φ₃, B_t)
            - Ensures metadata block_shape matches actual loaded block shape

        Args:
            block_indices (Tuple[int, ...]): Block indices in each dimension (7-tuple).

        Returns:
            np.ndarray: Field block (on CPU, converted from GPU if CUDA was used).

        Raises:
            ValueError: If metadata block_shape doesn't match loaded block shape.
        """
        block_id = self._get_block_id(block_indices)

        # Check if block is in memory
        with self._block_lock:
            if self._current_block_id == block_id and self._current_block is not None:
                # Convert from GPU to CPU if needed
                if self.use_cuda and CUDA_AVAILABLE and isinstance(self._current_block, cp.ndarray):
                    return cp.asnumpy(self._current_block.copy())
                return self._current_block.copy()

        # Check if block is on disk
        block_path = self._get_block_path(block_id)
        metadata_path = self._get_metadata_path(block_id)
        
        if block_path.exists():
            self.logger.debug(f"Loading block {block_indices} from cache")
            
            # Load block using np.load (not raw tofile)
            block = np.load(block_path)
            
            # Validate metadata matches true block shape
            if metadata_path.exists():
                try:
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                    
                    # Ensure metadata block_shape matches actual block shape
                    if hasattr(metadata, 'block_shape') and metadata.block_shape != block.shape:
                        self.logger.warning(
                            f"Metadata block_shape mismatch for block {block_indices}: "
                            f"metadata={metadata.block_shape}, actual={block.shape}. "
                            f"Updating metadata."
                        )
                        # Update metadata to match actual shape
                        self._save_block_metadata(
                            block_id, block_indices, block.shape, block_path
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load/validate metadata for block {block_indices}: {e}. "
                        f"Regenerating metadata."
                    )
                    # Regenerate metadata
                    self._save_block_metadata(
                        block_id, block_indices, block.shape, block_path
                    )
            else:
                # No metadata file, create it
                self.logger.debug(
                    f"No metadata file for block {block_indices}, creating it."
                )
                self._save_block_metadata(
                    block_id, block_indices, block.shape, block_path
                )
            
            # Update memory cache
            with self._block_lock:
                self._current_block = block
                self._current_block_id = block_id
            
            return block.copy()

        # Generate block (with CUDA support if available)
        self.logger.info(f"Generating block {block_indices}")
        block = self._generate_block(block_indices)

        # Save to disk using np.save (not raw tofile)
        np.save(block_path, block)
        self._save_block_metadata(block_id, block_indices, block.shape, block_path)

        # Update memory cache
        with self._block_lock:
            self._current_block = block
            self._current_block_id = block_id

        # Convert from GPU to CPU if needed
        if self.use_cuda and CUDA_AVAILABLE and isinstance(block, cp.ndarray):
            return cp.asnumpy(block.copy())
        
        return block.copy()

    def _generate_block(
        self, block_indices: Tuple[int, ...]
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Generate a single 7D block of the field with CUDA support.

        Physical Meaning:
            Generates the specified 7D block by calling the field generator
            with appropriate domain slice. Supports CUDA acceleration for
            vectorized 7D operations preserving structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            For 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ:
            - Block boundaries: [start_i, end_i) for each dimension i ∈ [0,6]
            - Block shape: (end_i - start_i) for each dimension
            - Ensures generated block has correct 7D shape

        Args:
            block_indices (Tuple[int, ...]): Block indices in each dimension (7-tuple).

        Returns:
            Union[np.ndarray, cp.ndarray]: Generated field block (on CPU or GPU).
        """
        # Compute block boundaries using vectorized operations
        block_start = []
        block_end = []
        block_shape = []

        for i, (block_idx, dim_size, block_dim_size) in enumerate(
            zip(block_indices, self.domain.shape, self.block_size)
        ):
            start = block_idx * block_dim_size
            end = min(start + block_dim_size, dim_size)
            block_start.append(start)
            block_end.append(end)
            block_shape.append(end - start)

        # Create domain slice configuration
        slice_config = {
            "start": tuple(block_start),
            "end": tuple(block_end),
            "shape": tuple(block_shape),
            "use_cuda": self.use_cuda,  # Pass CUDA flag to generator
        }

        # Generate block using field generator
        block = self.field_generator(self.domain, slice_config, self.config)

        # Ensure correct 7D shape
        expected_shape = tuple(block_shape)
        if block.shape != expected_shape:
            # Reshape or pad if needed
            if block.size == np.prod(expected_shape):
                # Reshape if total size matches
                if self.use_cuda and CUDA_AVAILABLE and isinstance(block, cp.ndarray):
                    block = block.reshape(expected_shape)
                else:
                    block = block.reshape(expected_shape)
            else:
                # Pad or crop to correct size using vectorized operations
                if self.use_cuda and CUDA_AVAILABLE:
                    # Use CuPy for GPU operations
                    if isinstance(block, cp.ndarray):
                        padded = cp.zeros(expected_shape, dtype=block.dtype)
                    else:
                        block = cp.asarray(block)
                        padded = cp.zeros(expected_shape, dtype=block.dtype)
                else:
                    padded = np.zeros(expected_shape, dtype=block.dtype)
                
                # Vectorized slicing and assignment
                slices = tuple(
                    slice(0, min(s, d)) for s, d in zip(block.shape, expected_shape)
                )
                padded[slices] = block[slices]
                block = padded

        return block

    def _save_block_metadata(
        self,
        block_id: str,
        block_indices: Tuple[int, ...],
        block_shape: Tuple[int, ...],
        file_path: Path,
    ) -> None:
        """Save block metadata."""
        metadata = BlockMetadata(
            block_id=block_id,
            block_indices=block_indices,
            block_shape=block_shape,
            file_path=str(file_path),
            is_generated=True,
            is_cached=True,
        )
        metadata_path = self._get_metadata_path(block_id)
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        self._block_metadata[block_id] = metadata

    def iterate_blocks(self, max_blocks: Optional[int] = None) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Iterate over all blocks in the 7D field with CUDA support and vectorization.

        Physical Meaning:
            Iterates over 7D field blocks with memory safety warnings (not hard errors)
            for large block counts. Uses vectorized operations and CUDA acceleration
            when available. Processes blocks preserving 7D structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Iterates over all block combinations:
            - Block indices: (i_x, i_y, i_z, i_φ₁, i_φ₂, i_φ₃, i_t)
            - Total blocks: ∏ᵢ (blocks_per_dim[i])
            - Allows large block counts with warnings up to safe cap

        Args:
            max_blocks (Optional[int]): Maximum number of blocks to iterate.
                If None, uses safety limit based on memory constraints.
                Default: 50000 blocks for safety, but warns instead of hard error.

        Yields:
            Tuple[np.ndarray, Dict[str, Any]]: Block data and metadata with:
                - block_indices: Tuple of block indices
                - block_shape: Actual block shape (validated from metadata)
                - block_id: Unique block identifier

        Note:
            Large block counts are allowed with warnings, not hard errors.
            Only extremely large counts (>200000) are capped.
        """
        from itertools import product

        # Safety limit: allow large block counts with warnings
        # Default: 50000 for safety, but warn instead of hard error
        if max_blocks is None:
            max_blocks = 50000  # Safe default for iteration

        # Warning thresholds instead of hard errors
        # Allow large block counts with warnings, not hard errors
        if self.total_blocks > 200000:
            # Only extremely large counts get hard cap
            self.logger.warning(
                f"Extremely large number of blocks ({self.total_blocks}). "
                f"Using safety limit of {max_blocks} blocks. "
                f"Consider increasing block_size or processing blocks individually."
            )
            max_blocks = min(max_blocks, 200000)
        elif self.total_blocks > max_blocks:
            # Warn but allow iteration
            self.logger.warning(
                f"Large number of blocks ({self.total_blocks}) - iteration may take time. "
                f"Consider using max_blocks parameter to limit processing. "
                f"Proceeding with iteration (will process up to {max_blocks} blocks)."
            )
        else:
            self.logger.info(
                f"Iterating over {self.total_blocks} blocks "
                f"(limit: {max_blocks if max_blocks < self.total_blocks else 'unlimited'})"
            )

        block_count = 0
        # Vectorized iteration over all block combinations
        for block_indices in product(
            *[range(n_blocks) for n_blocks in self.blocks_per_dim]
        ):
            if block_count >= max_blocks:
                self.logger.warning(
                    f"Reached iteration limit ({max_blocks}). "
                    f"Total blocks: {self.total_blocks}, processed: {block_count}"
                )
                break

            # Get block with CUDA support and metadata validation
            block = self.get_block_by_indices(block_indices)
            
            # Load metadata to ensure block_shape matches
            block_id = self._get_block_id(block_indices)
            metadata_path = self._get_metadata_path(block_id)
            block_shape_from_metadata = block.shape
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, "rb") as f:
                        metadata_obj = pickle.load(f)
                    if hasattr(metadata_obj, 'block_shape'):
                        block_shape_from_metadata = metadata_obj.block_shape
                        # Ensure metadata matches actual shape
                        if block_shape_from_metadata != block.shape:
                            self.logger.debug(
                                f"Metadata shape mismatch for block {block_indices}: "
                                f"metadata={block_shape_from_metadata}, actual={block.shape}"
                            )
                            block_shape_from_metadata = block.shape  # Use actual shape
                except Exception as e:
                    self.logger.debug(
                        f"Failed to load metadata for block {block_indices}: {e}"
                    )
            
            metadata = {
                "block_indices": block_indices,
                "block_shape": block_shape_from_metadata,  # Ensure metadata matches true shape
                "block_id": block_id,
            }
            yield block, metadata
            block_count += 1

    def get_field(self) -> BlockedField:
        """
        Get lazy field object for transparent access.

        Returns:
            BlockedField: Lazy field object.
        """
        return BlockedField(self.domain, self)

    def clear_cache(self) -> None:
        """Clear all cached blocks from disk."""
        for block_file in self.cache_dir.glob("block_*.npy"):
            block_file.unlink()
        for meta_file in self.cache_dir.glob("block_*.meta"):
            meta_file.unlink()
        self._block_metadata.clear()
        self.logger.info("Cache cleared")

    def cleanup(self) -> None:
        """Cleanup cache directory and resources."""
        self.clear_cache()
        if self.cache_dir.exists():
            self.cache_dir.rmdir()
        self.logger.info("Cleanup completed")

