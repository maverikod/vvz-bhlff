"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for MemoryKernel.

This module contains unit tests for the MemoryKernel class
in the 7D BVP framework, focusing on non-local temporal effects
in phase field dynamics.

Physical Meaning:
    Tests the memory kernel for non-local temporal effects in
    the 7D phase field dynamics.

Mathematical Foundation:
    Tests validate the memory kernel implementation for:
    - Non-local temporal coupling
    - Memory variable evolution
    - Relaxation dynamics
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.time import MemoryKernel
from bhlff.core.domain import Domain, Parameters


class TestMemoryKernel:
    """
    Unit tests for MemoryKernel.
    
    Physical Meaning:
        Tests the memory kernel for non-local temporal effects in
        the 7D phase field dynamics.
    """
    
    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)
    
    @pytest.fixture
    def memory_kernel(self, domain_7d):
        """Create memory kernel for testing."""
        return MemoryKernel(domain_7d, num_memory_vars=3)
    
    def test_initialization(self, memory_kernel, domain_7d):
        """
        Test memory kernel initialization.
        
        Physical Meaning:
            Validates that the memory kernel initializes correctly with
            the specified number of memory variables.
        """
        assert memory_kernel._initialized
        assert memory_kernel.domain == domain_7d
        assert memory_kernel.num_memory_vars == 3
        assert len(memory_kernel.memory_variables) == 3
        assert len(memory_kernel.relaxation_times) == 3
        assert len(memory_kernel.coupling_strengths) == 3
        
        # Check that all memory variables have correct shape
        for memory_var in memory_kernel.memory_variables:
            assert memory_var.shape == domain_7d.shape
            assert memory_var.dtype == np.complex128
    
    def test_memory_application(self, memory_kernel, domain_7d):
        """
        Test memory kernel application.
        
        Physical Meaning:
            Validates that the memory kernel correctly applies non-local
            temporal effects to the field.
        """
        # Create test field
        field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Apply memory kernel
        result = memory_kernel.apply(field, time=0.0)
        
        # Validate results
        assert result.shape == domain_7d.shape
        assert result.dtype == np.complex128
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_memory_evolution(self, memory_kernel, domain_7d):
        """
        Test memory variable evolution.
        
        Physical Meaning:
            Validates that memory variables evolve correctly according
            to their evolution equation.
        """
        # Create test field
        field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        dt = 0.01
        
        # Store initial memory variables
        initial_memory = [var.copy() for var in memory_kernel.memory_variables]
        
        # Evolve memory kernel
        memory_kernel.evolve(field, dt)
        
        # Check that memory variables changed
        for i, (initial, current) in enumerate(zip(initial_memory, memory_kernel.memory_variables)):
            assert not np.allclose(initial, current), f"Memory variable {i} did not evolve"
    
    def test_memory_reset(self, memory_kernel, domain_7d):
        """
        Test memory kernel reset.
        
        Physical Meaning:
            Validates that the memory kernel can be reset to clear
            all memory of past configurations.
        """
        # Evolve memory kernel to create non-zero values
        field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        memory_kernel.evolve(field, 0.01)
        
        # Check that memory variables are non-zero
        for memory_var in memory_kernel.memory_variables:
            assert not np.allclose(memory_var, 0)
        
        # Reset memory kernel
        memory_kernel.reset()
        
        # Check that all memory variables are zero
        for memory_var in memory_kernel.memory_variables:
            assert np.allclose(memory_var, 0)
    
    def test_relaxation_times(self, memory_kernel):
        """
        Test relaxation times.
        
        Physical Meaning:
            Validates that relaxation times are positive and properly
            configured for the memory kernel.
        """
        # Check that all relaxation times are positive
        for i, tau in enumerate(memory_kernel.relaxation_times):
            assert tau > 0, f"Relaxation time {i} must be positive"
        
        # Check that relaxation times are ordered (optional)
        for i in range(len(memory_kernel.relaxation_times) - 1):
            assert memory_kernel.relaxation_times[i] >= memory_kernel.relaxation_times[i + 1], \
                "Relaxation times should be ordered"
    
    def test_coupling_strengths(self, memory_kernel):
        """
        Test coupling strengths.
        
        Physical Meaning:
            Validates that coupling strengths are properly configured
            for the memory kernel.
        """
        # Check that all coupling strengths are finite
        for i, strength in enumerate(memory_kernel.coupling_strengths):
            assert np.isfinite(strength), f"Coupling strength {i} must be finite"
        
        # Check that at least one coupling strength is non-zero
        assert any(strength != 0 for strength in memory_kernel.coupling_strengths), \
            "At least one coupling strength must be non-zero"
    
    def test_memory_kernel_consistency(self, memory_kernel, domain_7d):
        """
        Test memory kernel consistency.
        
        Physical Meaning:
            Validates that the memory kernel maintains consistency
            between its internal state and external interface.
        """
        # Create test field
        field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Apply memory kernel multiple times
        result1 = memory_kernel.apply(field, time=0.0)
        result2 = memory_kernel.apply(field, time=0.0)
        
        # Results should be identical for same input
        np.testing.assert_allclose(result1, result2, rtol=1e-10)
    
    def test_memory_kernel_linearity(self, memory_kernel, domain_7d):
        """
        Test memory kernel linearity.
        
        Physical Meaning:
            Validates that the memory kernel behaves linearly
            for small field amplitudes.
        """
        # Create test fields
        field1 = np.random.randn(*domain_7d.shape).astype(np.complex128) * 0.1
        field2 = np.random.randn(*domain_7d.shape).astype(np.complex128) * 0.1
        combined_field = field1 + field2
        
        # Apply memory kernel
        result1 = memory_kernel.apply(field1, time=0.0)
        result2 = memory_kernel.apply(field2, time=0.0)
        result_combined = memory_kernel.apply(combined_field, time=0.0)
        
        # Check linearity (should hold for small amplitudes)
        expected = result1 + result2
        np.testing.assert_allclose(result_combined, expected, rtol=1e-6)
    
    def test_memory_kernel_time_dependence(self, memory_kernel, domain_7d):
        """
        Test memory kernel time dependence.
        
        Physical Meaning:
            Validates that the memory kernel correctly handles
            time-dependent effects.
        """
        # Create test field
        field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Apply memory kernel at different times
        result1 = memory_kernel.apply(field, time=0.0)
        result2 = memory_kernel.apply(field, time=1.0)
        
        # Results should be different for different times
        assert not np.allclose(result1, result2), \
            "Memory kernel should be time-dependent"
    
    def test_memory_kernel_validation(self, memory_kernel, domain_7d):
        """
        Test memory kernel validation.
        
        Physical Meaning:
            Validates that the memory kernel properly validates
            input parameters.
        """
        # Test valid field
        valid_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        result = memory_kernel.apply(valid_field, time=0.0)
        assert result is not None
        
        # Test invalid field shape
        invalid_field = np.random.randn(4, 4, 4).astype(np.complex128)
        with pytest.raises(ValueError, match="Field shape must match domain"):
            memory_kernel.apply(invalid_field, time=0.0)
        
        # Test invalid field type
        invalid_field = np.random.randn(*domain_7d.shape).astype(np.float64)
        with pytest.raises(ValueError, match="Field must be complex"):
            memory_kernel.apply(invalid_field, time=0.0)
        
        # Test invalid time
        with pytest.raises(ValueError, match="Time must be non-negative"):
            memory_kernel.apply(valid_field, time=-1.0)
