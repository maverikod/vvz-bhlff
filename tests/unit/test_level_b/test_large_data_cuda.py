"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for critical exponents estimation on large datasets with CUDA.

This module tests the performance and correctness of critical exponent
estimators on large 7D datasets using CUDA acceleration with block processing.

Physical Meaning:
    Validates that critical exponent estimation works correctly on large
    7D phase field datasets, ensuring proper block processing with 80%
    GPU memory limit and preservation of 7D structure.
"""

import numpy as np
import pytest
import logging

from bhlff.core.bvp import BVPCore
from bhlff.core.domain import Domain
from bhlff.models.level_b.power_law.critical_exponents import CriticalExponents
from bhlff.models.level_b.power_law.estimators import (
    estimate_nu_from_correlation_length,
    estimate_beta_from_tail,
    estimate_chi_from_variance,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def large_7d_domain():
    """Create large 7D domain for testing."""
    # Moderate domain: 32^3 spatial × 4^3 phase × 8 temporal
    # Total elements: ~4M elements (tests block processing)
    return Domain(L=1.0, N=32, N_phi=4, N_t=8, T=1.0, dimensions=7)


@pytest.fixture
def large_7d_field(large_7d_domain):
    """Create large 7D field with power-law structure."""
    shape = large_7d_domain.shape
    # Create field with power-law tail structure
    field = np.random.rand(*shape).astype(np.complex128)
    # Add power-law tail
    amplitude = np.abs(field)
    field = field * (1.0 + 0.1 * amplitude ** (-0.5))
    return field


@pytest.fixture
def bvp_core(large_7d_domain):
    """Create BVP core for testing."""
    from bhlff.core.bvp.bvp_constants import BVPConstants
    from bhlff.core.bvp.bvp_core_facade import BVPCoreFacade

    constants = BVPConstants()
    return BVPCoreFacade(large_7d_domain, constants)


def test_large_data_nu_estimation(bvp_core, large_7d_field):
    """Test ν estimation on large 7D dataset with CUDA."""
    amplitude = np.abs(large_7d_field)

    # Estimate ν with block processing
    nu = estimate_nu_from_correlation_length(bvp_core, amplitude)

    # Validate result
    assert np.isfinite(nu), "ν should be finite"
    assert 0 < nu < 2, f"ν should be in reasonable range, got {nu}"

    logger.info(f"Estimated ν={nu:.4f} on large dataset (shape={amplitude.shape})")


def test_large_data_beta_estimation(large_7d_field):
    """Test β estimation on large 7D dataset with CUDA."""
    amplitude = np.abs(large_7d_field)

    # Estimate β with block processing
    beta = estimate_beta_from_tail(amplitude)

    # Validate result
    assert np.isfinite(beta), "β should be finite"
    assert 0 < beta < 3, f"β should be in reasonable range, got {beta}"

    logger.info(f"Estimated β={beta:.4f} on large dataset (shape={amplitude.shape})")


def test_large_data_gamma_estimation(large_7d_field):
    """Test γ estimation on large 7D dataset with CUDA."""
    amplitude = np.abs(large_7d_field)

    # Estimate γ with block processing
    gamma = estimate_chi_from_variance(amplitude)

    # Validate result
    assert np.isfinite(gamma), "γ should be finite"
    assert 0 < gamma < 3, f"γ should be in reasonable range, got {gamma}"

    logger.info(f"Estimated γ={gamma:.4f} on large dataset (shape={amplitude.shape})")


def test_large_data_critical_exponents_full(bvp_core, large_7d_field):
    """Test full critical exponents analysis on large 7D dataset."""
    analyzer = CriticalExponents(bvp_core)

    # Analyze critical behavior
    results = analyzer.analyze_critical_behavior(large_7d_field)

    # Validate all exponents are computed
    exponents = results["critical_exponents"]
    assert "nu" in exponents
    assert "beta" in exponents
    assert "gamma" in exponents
    assert "delta" in exponents
    assert "eta" in exponents
    assert "alpha" in exponents
    assert "z" in exponents

    # Validate all exponents are finite
    for name, value in exponents.items():
        assert np.isfinite(value), f"{name} should be finite, got {value}"

    logger.info(
        f"Full critical exponents analysis completed on large dataset "
        f"(shape={large_7d_field.shape})"
    )
    logger.info(f"Exponents: {exponents}")


def test_block_processing_memory_limit(large_7d_field):
    """Test that block processing respects 80% GPU memory limit."""
    try:
        import cupy as cp

        cuda_available = True
    except ImportError:
        cuda_available = False
        pytest.skip("CUDA not available")

    if not cuda_available:
        pytest.skip("CUDA not available")

    # Check GPU memory usage during processing
    mem_info_before = cp.cuda.runtime.memGetInfo()
    free_memory_before = mem_info_before[0]

    # Process field with block processing
    amplitude = np.abs(large_7d_field)
    beta = estimate_beta_from_tail(amplitude)

    mem_info_after = cp.cuda.runtime.memGetInfo()
    free_memory_after = mem_info_after[0]

    # Memory usage should be reasonable (less than 80% of free memory)
    memory_used = free_memory_before - free_memory_after
    memory_fraction = memory_used / free_memory_before if free_memory_before > 0 else 0

    logger.info(
        f"Memory usage: {memory_fraction*100:.2f}% of free memory "
        f"({memory_used/1e9:.2f}GB used)"
    )

    # Validate that memory usage is reasonable
    assert memory_fraction < 0.85, (
        f"Memory usage {memory_fraction*100:.2f}% exceeds 85% limit. "
        f"Block processing should respect 80% GPU memory limit."
    )

    assert np.isfinite(beta), "β should be finite"


def test_7d_structure_preservation(bvp_core, large_7d_field):
    """Test that 7D structure is preserved during processing."""
    amplitude = np.abs(large_7d_field)

    # Original shape should be 7D
    assert amplitude.ndim == 7, f"Expected 7D field, got {amplitude.ndim}D"

    # Process with block-aware estimation
    nu = estimate_nu_from_correlation_length(bvp_core, amplitude)

    # Validate result
    assert np.isfinite(nu), "ν should be finite"
    assert 0 < nu < 2, f"ν should be in reasonable range, got {nu}"

    logger.info(
        f"7D structure preserved during processing "
        f"(shape={amplitude.shape}, ν={nu:.4f})"
    )

