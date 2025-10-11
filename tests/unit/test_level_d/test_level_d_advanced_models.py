"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced unit tests for Level D models.

This module contains comprehensive unit tests for Level D advanced models,
including field projection, streamline analysis, and integration testing.

Physical Meaning:
    Tests verify that Level D models correctly implement:
    - Field projections onto EM/strong/weak interaction windows
    - Phase streamline analysis for topological structure
    - Integration testing of all Level D components

Example:
    >>> pytest tests/unit/test_level_d/test_level_d_advanced_models.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import tempfile
import json
import os

from bhlff.models.level_d import (
    LevelDModels,
    FieldProjection,
    StreamlineAnalyzer,
)
from bhlff.core.domain import Domain
from bhlff.models.base.abstract_models import AbstractLevelModels


class TestFieldProjection:
    """Test field projection functionality."""

    @pytest.fixture
    def test_field(self):
        """Create test field."""
        # Create a simple test field
        field = np.random.randn(32, 32, 32) + 1j * np.random.randn(32, 32, 32)
        return field

    @pytest.fixture
    def projection_params(self):
        """Create projection parameters."""
        return {
            "em": {
                "frequency_range": [0.1, 1.0],
                "amplitude_threshold": 0.1,
                "filter_type": "bandpass",
            },
            "strong": {
                "frequency_range": [1.0, 10.0],
                "q_threshold": 100,
                "filter_type": "high_q",
            },
            "weak": {
                "frequency_range": [0.01, 0.1],
                "q_threshold": 10,
                "filter_type": "chiral",
            },
        }

    def test_field_projection_initialization(self, test_field, projection_params):
        """Test field projection initialization."""
        projection = FieldProjection(test_field, projection_params)

        assert np.array_equal(projection.field, test_field)
        assert projection.projection_params == projection_params
        assert hasattr(projection, "_em_projector")
        assert hasattr(projection, "_strong_projector")
        assert hasattr(projection, "_weak_projector")
        assert hasattr(projection, "_signature_analyzer")

    def test_project_em_field(self, test_field, projection_params):
        """Test EM field projection."""
        projection = FieldProjection(test_field, projection_params)

        # Project EM field
        em_projection = projection.project_em_field(test_field)

        # Check that projection has correct shape
        assert em_projection.shape == test_field.shape
        assert np.all(np.isfinite(em_projection))

    def test_project_strong_field(self, test_field, projection_params):
        """Test strong field projection."""
        projection = FieldProjection(test_field, projection_params)

        # Project strong field
        strong_projection = projection.project_strong_field(test_field)

        # Check that projection has correct shape
        assert strong_projection.shape == test_field.shape
        assert np.all(np.isfinite(strong_projection))

    def test_project_weak_field(self, test_field, projection_params):
        """Test weak field projection."""
        projection = FieldProjection(test_field, projection_params)

        # Project weak field
        weak_projection = projection.project_weak_field(test_field)

        # Check that projection has correct shape
        assert weak_projection.shape == test_field.shape
        assert np.all(np.isfinite(weak_projection))

    def test_project_field_windows(self, test_field, projection_params):
        """Test field projection onto windows."""
        projection = FieldProjection(test_field, projection_params)

        # Project fields
        results = projection.project_field_windows(test_field)

        # Check results structure
        assert "em_projection" in results
        assert "strong_projection" in results
        assert "weak_projection" in results
        assert "signatures" in results

        # Check that all projections have correct shape
        assert results["em_projection"].shape == test_field.shape
        assert results["strong_projection"].shape == test_field.shape
        assert results["weak_projection"].shape == test_field.shape


class TestStreamlineAnalyzer:
    """Test streamline analyzer functionality."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=10.0, N=16, dimensions=7, N_phi=8, N_t=16, T=1.0)

    @pytest.fixture
    def parameters(self):
        """Create test parameters."""
        return {"num_streamlines": 50, "integration_steps": 500, "step_size": 0.01}

    @pytest.fixture
    def test_field(self, domain):
        """Create test field."""
        x = np.linspace(0, domain.L, domain.N)
        y = np.linspace(0, domain.L, domain.N)
        z = np.linspace(0, domain.L, domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        field = (
            np.sin(2 * np.pi * X / domain.L)
            * np.cos(2 * np.pi * Y / domain.L)
            * np.sin(2 * np.pi * Z / domain.L)
        )

        return field

    def test_streamline_analyzer_initialization(self, domain, parameters):
        """Test streamline analyzer initialization."""
        analyzer = StreamlineAnalyzer(domain, parameters)

        assert analyzer.domain == domain
        assert analyzer.parameters == parameters
        assert hasattr(analyzer, "_gradient_computer")
        assert hasattr(analyzer, "_streamline_tracer")
        assert hasattr(analyzer, "_topology_analyzer")

    def test_trace_phase_streamlines(self, domain, parameters, test_field):
        """Test phase streamline tracing."""
        analyzer = StreamlineAnalyzer(domain, parameters)

        # Define center point
        center = (5.0, 5.0, 5.0)

        # Trace streamlines
        results = analyzer.trace_phase_streamlines(test_field, center)

        # Check results structure
        assert "phase" in results
        assert "phase_gradient" in results
        assert "streamlines" in results
        assert "topology" in results

        # Check that phase has correct shape
        assert results["phase"].shape == test_field.shape

        # Check that gradient has correct shape
        expected_gradient_shape = test_field.shape + (3,)
        assert results["phase_gradient"].shape == expected_gradient_shape

    def test_analyze_streamlines(self, domain, parameters, test_field):
        """Test streamline analysis."""
        analyzer = StreamlineAnalyzer(domain, parameters)

        # Analyze streamlines
        results = analyzer.analyze_streamlines(test_field)

        # Check results structure
        assert "divergence_max" in results
        assert "divergence_mean" in results
        assert "curl_max" in results
        assert "curl_mean" in results
        assert "streamline_density" in results

        # Check that all values are finite
        for key, value in results.items():
            assert np.isfinite(value)


class TestLevelDIntegration:
    """Test Level D integration functionality."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=10.0, N=16, dimensions=7, N_phi=8, N_t=16, T=1.0)

    @pytest.fixture
    def parameters(self):
        """Create test parameters."""
        return {
            "jaccard_threshold": 0.8,
            "frequency_tolerance": 0.05,
            "mode_threshold": 0.1,
        }

    def test_level_d_integration(self, domain, parameters):
        """Test Level D integration."""
        # Create Level D models
        models = LevelDModels(domain, parameters)

        # Create test field
        x = np.linspace(0, domain.L, domain.N)
        y = np.linspace(0, domain.L, domain.N)
        z = np.linspace(0, domain.L, domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        test_field = (
            np.sin(2 * np.pi * X / domain.L)
            * np.cos(2 * np.pi * Y / domain.L)
            * np.sin(2 * np.pi * Z / domain.L)
        )

        # Test comprehensive analysis
        results = models.analyze_multimode_field(test_field)

        # Check that all components are present
        assert "superposition" in results
        assert "projections" in results
        assert "streamlines" in results

        # Check that results are valid
        assert results["field_shape"] == test_field.shape
        assert isinstance(results["analysis_parameters"], dict)
