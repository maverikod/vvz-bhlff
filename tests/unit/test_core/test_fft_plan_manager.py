"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for FFTPlanManager class.

This module provides comprehensive unit tests for the FFTPlanManager class,
covering plan creation, management, and optimization.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.core.fft.fft_backend_core import FFTBackend
from bhlff.core.fft.fft_plan_manager import FFTPlanManager


class TestFFTPlanManager:
    """Comprehensive tests for FFTPlanManager class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(
            L=1.0,
            N=8,
            dimensions=7,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def fft_backend(self, domain):
        """Create FFT backend for testing."""
        return FFTBackend(domain)

    @pytest.fixture
    def plan_manager(self, fft_backend):
        """Create FFT plan manager for testing."""
        return FFTPlanManager(fft_backend)

    def test_plan_manager_initialization(self, plan_manager, fft_backend):
        """Test FFT plan manager initialization."""
        assert plan_manager.fft_backend == fft_backend

    def test_plan_manager_create_plan(self, plan_manager):
        """Test FFT plan creation."""
        # Create plan
        plan = plan_manager.create_plan()
        
        assert plan is not None

    def test_plan_manager_get_plan(self, plan_manager):
        """Test FFT plan retrieval."""
        # Get plan
        plan = plan_manager.get_plan()
        
        assert plan is not None

    def test_plan_manager_plan_caching(self, plan_manager):
        """Test FFT plan caching."""
        # Get plan twice
        plan1 = plan_manager.get_plan()
        plan2 = plan_manager.get_plan()
        
        # Should be the same plan (cached)
        assert plan1 is plan2

    def test_plan_manager_plan_optimization(self, plan_manager):
        """Test FFT plan optimization."""
        # Create optimized plan
        plan = plan_manager.create_optimized_plan()
        
        assert plan is not None

    def test_plan_manager_plan_validation(self, plan_manager):
        """Test FFT plan validation."""
        # Create plan
        plan = plan_manager.create_plan()
        
        # Validate plan
        is_valid = plan_manager.validate_plan(plan)
        
        assert isinstance(is_valid, bool)

    def test_plan_manager_plan_cleanup(self, plan_manager):
        """Test FFT plan cleanup."""
        # Create plan
        plan = plan_manager.create_plan()
        
        # Cleanup plan
        plan_manager.cleanup_plan(plan)
        
        # Should not raise errors
        assert True

    def test_plan_manager_plan_reset(self, plan_manager):
        """Test FFT plan reset."""
        # Create plan
        plan = plan_manager.create_plan()
        
        # Reset plan
        plan_manager.reset_plan(plan)
        
        # Should not raise errors
        assert True

    def test_plan_manager_plan_execution(self, plan_manager):
        """Test FFT plan execution."""
        # Create test field
        field = np.random.random(plan_manager.fft_backend.domain.shape)
        
        # Create plan
        plan = plan_manager.create_plan()
        
        # Execute plan
        result = plan_manager.execute_plan(plan, field)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == field.shape

    def test_plan_manager_plan_performance(self, plan_manager):
        """Test FFT plan performance."""
        # Create test field
        field = np.random.random(plan_manager.fft_backend.domain.shape)
        
        # Create plan
        plan = plan_manager.create_plan()
        
        # Measure performance
        start_time = time.time()
        result = plan_manager.execute_plan(plan, field)
        end_time = time.time()
        
        # Should be reasonable performance
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should be fast for small domain

    def test_plan_manager_plan_memory(self, plan_manager):
        """Test FFT plan memory usage."""
        # Create plan
        plan = plan_manager.create_plan()
        
        # Get memory usage
        memory_usage = plan_manager.get_plan_memory_usage(plan)
        
        assert isinstance(memory_usage, (int, float))
        assert memory_usage >= 0

    def test_plan_manager_plan_statistics(self, plan_manager):
        """Test FFT plan statistics."""
        # Create plan
        plan = plan_manager.create_plan()
        
        # Get statistics
        stats = plan_manager.get_plan_statistics(plan)
        
        assert isinstance(stats, dict)
        assert 'creation_time' in stats
        assert 'memory_usage' in stats
        assert 'optimization_level' in stats

    def test_plan_manager_plan_comparison(self, plan_manager):
        """Test FFT plan comparison."""
        # Create two plans
        plan1 = plan_manager.create_plan()
        plan2 = plan_manager.create_optimized_plan()
        
        # Compare plans
        comparison = plan_manager.compare_plans(plan1, plan2)
        
        assert isinstance(comparison, dict)
        assert 'performance_ratio' in comparison
        assert 'memory_ratio' in comparison

    def test_plan_manager_plan_serialization(self, plan_manager):
        """Test FFT plan serialization."""
        # Create plan
        plan = plan_manager.create_plan()
        
        # Serialize plan
        serialized = plan_manager.serialize_plan(plan)
        
        assert isinstance(serialized, (str, bytes, dict))

    def test_plan_manager_plan_deserialization(self, plan_manager):
        """Test FFT plan deserialization."""
        # Create plan
        plan = plan_manager.create_plan()
        
        # Serialize and deserialize
        serialized = plan_manager.serialize_plan(plan)
        deserialized = plan_manager.deserialize_plan(serialized)
        
        assert deserialized is not None

    def test_plan_manager_plan_validation_errors(self, plan_manager):
        """Test FFT plan validation error handling."""
        # Test with invalid plan
        invalid_plan = None
        
        with pytest.raises(ValueError):
            plan_manager.validate_plan(invalid_plan)
        
        with pytest.raises(ValueError):
            plan_manager.execute_plan(invalid_plan, np.array([1, 2, 3]))

    def test_plan_manager_plan_cleanup_errors(self, plan_manager):
        """Test FFT plan cleanup error handling."""
        # Test with invalid plan
        invalid_plan = None
        
        with pytest.raises(ValueError):
            plan_manager.cleanup_plan(invalid_plan)
        
        with pytest.raises(ValueError):
            plan_manager.reset_plan(invalid_plan)

    def test_plan_manager_plan_execution_errors(self, plan_manager):
        """Test FFT plan execution error handling."""
        # Create plan
        plan = plan_manager.create_plan()
        
        # Test with invalid field
        invalid_field = np.array([1, 2, 3])  # Wrong shape
        
        with pytest.raises(ValueError):
            plan_manager.execute_plan(plan, invalid_field)

    def test_plan_manager_plan_statistics_errors(self, plan_manager):
        """Test FFT plan statistics error handling."""
        # Test with invalid plan
        invalid_plan = None
        
        with pytest.raises(ValueError):
            plan_manager.get_plan_statistics(invalid_plan)
        
        with pytest.raises(ValueError):
            plan_manager.get_plan_memory_usage(invalid_plan)

    def test_plan_manager_plan_comparison_errors(self, plan_manager):
        """Test FFT plan comparison error handling."""
        # Test with invalid plans
        invalid_plan1 = None
        invalid_plan2 = None
        
        with pytest.raises(ValueError):
            plan_manager.compare_plans(invalid_plan1, invalid_plan2)
        
        with pytest.raises(ValueError):
            plan_manager.compare_plans(plan_manager.create_plan(), invalid_plan2)

    def test_plan_manager_plan_serialization_errors(self, plan_manager):
        """Test FFT plan serialization error handling."""
        # Test with invalid plan
        invalid_plan = None
        
        with pytest.raises(ValueError):
            plan_manager.serialize_plan(invalid_plan)
        
        with pytest.raises(ValueError):
            plan_manager.deserialize_plan(invalid_plan)
