"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for Level D field projection.

This module contains unit tests for field projection functionality,
including field projections onto EM/strong/weak interaction windows.

Physical Meaning:
    Tests verify that field projections correctly implement:
    - Field projections onto EM/strong/weak interaction windows
    - Projection validation and analysis

Example:
    >>> pytest tests/unit/test_level_d/test_level_d_field_projection.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import tempfile
import json
import os

from bhlff.models.level_d import (
    FieldProjection,
)
from bhlff.core.domain import Domain


class TestFieldProjection:
    """Test FieldProjection functionality."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=10.0, N=16, dimensions=7, N_phi=8, N_t=16, T=1.0)

    @pytest.fixture
    def parameters(self):
        """Create test parameters."""
        return {
            "projection_threshold": 0.1,
            "window_size": 2.0,
            "interaction_strength": 1.0,
        }

    @pytest.fixture
    def field_projection(self, domain, parameters):
        """Create FieldProjection instance."""
        return FieldProjection(domain, parameters)

    def test_initialization(self, domain, parameters):
        """Test FieldProjection initialization."""
        projection = FieldProjection(domain, parameters)
        
        assert projection.domain == domain
        assert projection.parameters == parameters

    def test_em_projection(self, field_projection):
        """Test EM interaction projection."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test EM projection
        em_projection = field_projection.project_em_interaction(field)
        assert em_projection is not None
        assert hasattr(em_projection, "projected_field")
        assert hasattr(em_projection, "interaction_strength")
        assert hasattr(em_projection, "projection_quality")

    def test_strong_projection(self, field_projection):
        """Test strong interaction projection."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test strong projection
        strong_projection = field_projection.project_strong_interaction(field)
        assert strong_projection is not None
        assert hasattr(strong_projection, "projected_field")
        assert hasattr(strong_projection, "interaction_strength")
        assert hasattr(strong_projection, "projection_quality")

    def test_weak_projection(self, field_projection):
        """Test weak interaction projection."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test weak projection
        weak_projection = field_projection.project_weak_interaction(field)
        assert weak_projection is not None
        assert hasattr(weak_projection, "projected_field")
        assert hasattr(weak_projection, "interaction_strength")
        assert hasattr(weak_projection, "projection_quality")

    def test_projection_validation(self, field_projection):
        """Test projection validation."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test EM projection validation
        em_projection = field_projection.project_em_interaction(field)
        is_valid = field_projection.validate_projection(em_projection)
        assert is_valid
        
        # Test with invalid projection
        invalid_projection = None
        is_valid = field_projection.validate_projection(invalid_projection)
        assert not is_valid

    def test_projection_analysis(self, field_projection):
        """Test projection analysis."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test EM projection analysis
        em_projection = field_projection.project_em_interaction(field)
        analysis = field_projection.analyze_projection(em_projection)
        assert isinstance(analysis, dict)
        assert "projection_efficiency" in analysis
        assert "interaction_quality" in analysis
        assert "field_coupling" in analysis

    def test_projection_comparison(self, field_projection):
        """Test projection comparison."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Create different projections
        em_projection = field_projection.project_em_interaction(field)
        strong_projection = field_projection.project_strong_interaction(field)
        
        # Test comparison
        comparison = field_projection.compare_projections(em_projection, strong_projection)
        assert isinstance(comparison, dict)
        assert "similarity" in comparison
        assert "difference_metrics" in comparison
        assert "comparison_score" in comparison

    def test_projection_optimization(self, field_projection):
        """Test projection optimization."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test EM projection optimization
        em_projection = field_projection.project_em_interaction(field)
        optimized_projection = field_projection.optimize_projection(em_projection)
        assert optimized_projection is not None
        assert hasattr(optimized_projection, "projected_field")
        assert hasattr(optimized_projection, "interaction_strength")
        assert hasattr(optimized_projection, "projection_quality")

    def test_projection_serialization(self, field_projection):
        """Test projection serialization."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test EM projection serialization
        em_projection = field_projection.project_em_interaction(field)
        serialized = field_projection.serialize_projection(em_projection)
        assert isinstance(serialized, dict)
        assert "projection_type" in serialized
        assert "projected_field" in serialized
        assert "interaction_strength" in serialized
        
        # Test deserialization
        deserialized = field_projection.deserialize_projection(serialized)
        assert deserialized is not None
        assert hasattr(deserialized, "projected_field")
        assert hasattr(deserialized, "interaction_strength")

    def test_projection_export(self, field_projection):
        """Test projection export."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test EM projection export
        em_projection = field_projection.project_em_interaction(field)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            field_projection.export_projection(em_projection, filename)
            assert os.path.exists(filename)
            
            # Verify file content
            with open(filename, 'r') as f:
                data = json.load(f)
            assert "projection_type" in data
            assert "projected_field" in data
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_projection_import(self, field_projection):
        """Test projection import."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Create and export a projection
        em_projection = field_projection.project_em_interaction(field)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            field_projection.export_projection(em_projection, filename)
            
            # Test import
            imported_projection = field_projection.import_projection(filename)
            assert imported_projection is not None
            assert hasattr(imported_projection, "projected_field")
            assert hasattr(imported_projection, "interaction_strength")
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_multi_interaction_projection(self, field_projection):
        """Test multi-interaction projection."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test multi-interaction projection
        multi_projection = field_projection.project_multi_interaction(field, ["em", "strong", "weak"])
        assert multi_projection is not None
        assert hasattr(multi_projection, "projections")
        assert hasattr(multi_projection, "interaction_types")
        assert len(multi_projection.projections) == 3

    def test_projection_statistics(self, field_projection):
        """Test projection statistics."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test projection statistics
        em_projection = field_projection.project_em_interaction(field)
        statistics = field_projection.compute_projection_statistics(em_projection)
        assert isinstance(statistics, dict)
        assert "mean_strength" in statistics
        assert "variance" in statistics
        assert "correlation_length" in statistics

    def test_projection_quality_metrics(self, field_projection):
        """Test projection quality metrics."""
        # Create test field
        field = np.random.random(field_projection.domain.shape) + 1j * np.random.random(field_projection.domain.shape)
        
        # Test quality metrics
        em_projection = field_projection.project_em_interaction(field)
        quality_metrics = field_projection.compute_quality_metrics(em_projection)
        assert isinstance(quality_metrics, dict)
        assert "projection_efficiency" in quality_metrics
        assert "field_fidelity" in quality_metrics
        assert "interaction_quality" in quality_metrics


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
