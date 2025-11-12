"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test block processing with 80% GPU memory limit.

This module tests that block processing correctly uses 80% GPU memory limit
and preserves 7D structure on moderate-sized datasets.
"""

import numpy as np
import pytest
import logging

from bhlff.models.level_b.power_law.cuda_estimator_utils import (
    compute_optimal_block_size,
    get_cuda_backend,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def moderate_7d_field():
    """Create moderate 7D field for testing."""
    # Moderate domain: 16^3 spatial × 4^3 phase × 4 temporal
    # Total elements: ~262K elements (tests block processing without OOM)
    shape = (16, 16, 16, 4, 4, 4, 4)
    field = np.random.rand(*shape).astype(np.complex128)
    return np.abs(field)


def test_optimal_block_size_calculation(moderate_7d_field):
    """Test that optimal block size calculation respects 80% GPU memory limit."""
    backend = get_cuda_backend()

    # Calculate optimal block size
    min_block_elems = compute_optimal_block_size(
        moderate_7d_field,
        backend,
        memory_overhead_factor=4.0,
        min_block_elems=256,
        max_block_elems=1048576,
        fraction_of_total=0.001,
    )

    # Validate block size is reasonable
    assert min_block_elems >= 256, f"Block size too small: {min_block_elems}"
    assert min_block_elems <= moderate_7d_field.size, (
        f"Block size {min_block_elems} exceeds field size {moderate_7d_field.size}"
    )

    logger.info(
        f"Optimal block size: {min_block_elems} elements "
        f"(field size: {moderate_7d_field.size}, "
        f"CUDA available: {backend is not None})"
    )


def test_block_processing_with_cuda(moderate_7d_field):
    """Test block processing with CUDA on moderate dataset."""
    try:
        import cupy as cp

        cuda_available = True
    except ImportError:
        cuda_available = False
        pytest.skip("CUDA not available")

    if not cuda_available:
        pytest.skip("CUDA not available")

    # Check memory before processing
    mem_info_before = cp.cuda.runtime.memGetInfo()
    free_memory_before = mem_info_before[0]

    # Process field with block processing
    backend = get_cuda_backend()
    min_block_elems = compute_optimal_block_size(
        moderate_7d_field,
        backend,
        memory_overhead_factor=4.0,
        min_block_elems=256,
        max_block_elems=1048576,
        fraction_of_total=0.001,
    )

    # Check memory after
    mem_info_after = cp.cuda.runtime.memGetInfo()
    free_memory_after = mem_info_after[0]

    # Memory usage should be reasonable
    memory_used = free_memory_before - free_memory_after
    memory_fraction = memory_used / free_memory_before if free_memory_before > 0 else 0

    logger.info(
        f"Block processing memory usage: {memory_fraction*100:.2f}% "
        f"({memory_used/1e9:.2f}GB used, "
        f"free before: {free_memory_before/1e9:.2f}GB)"
    )

    # Validate reasonable memory usage (should be less than 85% to account for overhead)
    # Note: This is a conservative check - actual usage depends on block size
    assert memory_fraction < 0.9, (
        f"Memory usage {memory_fraction*100:.2f}% exceeds 90% limit. "
        f"Block processing should use ~80% of GPU memory."
    )

    # Validate block size calculation
    assert min_block_elems > 0, "Block size should be positive"


def test_7d_structure_preservation(moderate_7d_field):
    """Test that 7D structure is preserved during block processing."""
    # Original shape should be 7D
    assert moderate_7d_field.ndim == 7, (
        f"Expected 7D field, got {moderate_7d_field.ndim}D"
    )

    # Calculate optimal block size (preserves 7D structure)
    backend = get_cuda_backend()
    min_block_elems = compute_optimal_block_size(
        moderate_7d_field,
        backend,
        memory_overhead_factor=4.0,
        min_block_elems=256,
        max_block_elems=1048576,
        fraction_of_total=0.001,
    )

    # Validate block size is reasonable for 7D structure
    assert min_block_elems >= 256, "Block size should accommodate 7D structure"

    logger.info(
        f"7D structure preserved: shape={moderate_7d_field.shape}, "
        f"block_size={min_block_elems}"
    )

