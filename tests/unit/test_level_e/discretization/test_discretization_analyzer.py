"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for discretization analysis functionality.

This module tests the discretization analysis functionality
for Level E experiments in 7D phase field theory.

Physical Meaning:
    Tests ensure that discretization analysis correctly
    validates numerical stability for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/discretization/test_discretization_analyzer.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import DiscretizationAnalyzer


class TestDiscretizationAnalyzer:
    """Test discretization analysis functionality."""

    def test_initialization(self):
        """Test DiscretizationAnalyzer initialization."""
        grid_sizes = [32, 64, 128, 256]
        time_steps = [0.001, 0.01, 0.1, 1.0]

        analyzer = DiscretizationAnalyzer(grid_sizes, time_steps)

        assert analyzer.grid_sizes == grid_sizes
        assert analyzer.time_steps == time_steps
        assert analyzer.discretization_results is None
        assert analyzer.convergence_analysis is None

    def test_grid_convergence_analysis(self):
        """Test grid convergence analysis."""
        grid_sizes = [32, 64, 128, 256]
        time_steps = [0.001, 0.01, 0.1, 1.0]

        analyzer = DiscretizationAnalyzer(grid_sizes, time_steps)

        # Mock test data
        test_data = {
            "field_data": {
                32: np.random.rand(32, 32, 32),
                64: np.random.rand(64, 64, 64),
                128: np.random.rand(128, 128, 128),
                256: np.random.rand(256, 256, 256),
            },
            "reference_solution": np.random.rand(512, 512, 512),
        }

        # Test grid convergence
        results = analyzer.analyze_grid_convergence(test_data)

        assert results is not None
        assert "convergence_rate" in results
        assert "error_analysis" in results
        assert "optimal_grid_size" in results

        # Check convergence rate
        convergence_rate = results["convergence_rate"]
        assert isinstance(convergence_rate, float)
        assert convergence_rate > 0

        # Check error analysis
        error_analysis = results["error_analysis"]
        assert isinstance(error_analysis, dict)
        assert "grid_errors" in error_analysis
        assert "error_reduction" in error_analysis

        # Check optimal grid size
        optimal_grid_size = results["optimal_grid_size"]
        assert isinstance(optimal_grid_size, int)
        assert optimal_grid_size in grid_sizes

    def test_time_step_convergence_analysis(self):
        """Test time step convergence analysis."""
        grid_sizes = [32, 64, 128, 256]
        time_steps = [0.001, 0.01, 0.1, 1.0]

        analyzer = DiscretizationAnalyzer(grid_sizes, time_steps)

        # Mock test data
        test_data = {
            "time_series": {
                0.001: np.random.rand(1000),
                0.01: np.random.rand(100),
                0.1: np.random.rand(10),
                1.0: np.random.rand(1),
            },
            "reference_solution": np.random.rand(10000),
        }

        # Test time step convergence
        results = analyzer.analyze_time_step_convergence(test_data)

        assert results is not None
        assert "convergence_rate" in results
        assert "stability_analysis" in results
        assert "optimal_time_step" in results

        # Check convergence rate
        convergence_rate = results["convergence_rate"]
        assert isinstance(convergence_rate, float)
        assert convergence_rate > 0

        # Check stability analysis
        stability_analysis = results["stability_analysis"]
        assert isinstance(stability_analysis, dict)
        assert "stability_region" in stability_analysis
        assert "critical_time_step" in stability_analysis

        # Check optimal time step
        optimal_time_step = results["optimal_time_step"]
        assert isinstance(optimal_time_step, float)
        assert optimal_time_step in time_steps

    def test_numerical_stability_analysis(self):
        """Test numerical stability analysis."""
        grid_sizes = [32, 64, 128, 256]
        time_steps = [0.001, 0.01, 0.1, 1.0]

        analyzer = DiscretizationAnalyzer(grid_sizes, time_steps)

        # Mock test data
        test_data = {
            "field_data": {
                32: np.random.rand(32, 32, 32),
                64: np.random.rand(64, 64, 64),
                128: np.random.rand(128, 128, 128),
                256: np.random.rand(256, 256, 256),
            },
            "time_series": {
                0.001: np.random.rand(1000),
                0.01: np.random.rand(100),
                0.1: np.random.rand(10),
                1.0: np.random.rand(1),
            },
        }

        # Test numerical stability
        results = analyzer.analyze_numerical_stability(test_data)

        assert results is not None
        assert "stability_metrics" in results
        assert "instability_regions" in results
        assert "stability_recommendations" in results

        # Check stability metrics
        stability_metrics = results["stability_metrics"]
        assert isinstance(stability_metrics, dict)
        assert "cfl_condition" in stability_metrics
        assert "von_neumann_stability" in stability_metrics

        # Check instability regions
        instability_regions = results["instability_regions"]
        assert isinstance(instability_regions, list)
        assert len(instability_regions) >= 0

        # Check stability recommendations
        stability_recommendations = results["stability_recommendations"]
        assert isinstance(stability_recommendations, list)
        assert len(stability_recommendations) > 0

    def test_discretization_error_analysis(self):
        """Test discretization error analysis."""
        grid_sizes = [32, 64, 128, 256]
        time_steps = [0.001, 0.01, 0.1, 1.0]

        analyzer = DiscretizationAnalyzer(grid_sizes, time_steps)

        # Mock test data
        test_data = {
            "field_data": {
                32: np.random.rand(32, 32, 32),
                64: np.random.rand(64, 64, 64),
                128: np.random.rand(128, 128, 128),
                256: np.random.rand(256, 256, 256),
            },
            "reference_solution": np.random.rand(512, 512, 512),
        }

        # Test discretization error
        results = analyzer.analyze_discretization_error(test_data)

        assert results is not None
        assert "spatial_errors" in results
        assert "temporal_errors" in results
        assert "total_errors" in results

        # Check spatial errors
        spatial_errors = results["spatial_errors"]
        assert isinstance(spatial_errors, dict)
        assert "grid_errors" in spatial_errors
        assert "error_reduction" in spatial_errors

        # Check temporal errors
        temporal_errors = results["temporal_errors"]
        assert isinstance(temporal_errors, dict)
        assert "time_step_errors" in temporal_errors
        assert "error_reduction" in temporal_errors

        # Check total errors
        total_errors = results["total_errors"]
        assert isinstance(total_errors, dict)
        assert "combined_errors" in total_errors
        assert "error_ranking" in total_errors

    def test_discretization_report(self):
        """Test discretization report generation."""
        grid_sizes = [32, 64, 128, 256]
        time_steps = [0.001, 0.01, 0.1, 1.0]

        analyzer = DiscretizationAnalyzer(grid_sizes, time_steps)

        # Mock test data
        test_data = {
            "field_data": {
                32: np.random.rand(32, 32, 32),
                64: np.random.rand(64, 64, 64),
                128: np.random.rand(128, 128, 128),
                256: np.random.rand(256, 256, 256),
            },
            "time_series": {
                0.001: np.random.rand(1000),
                0.01: np.random.rand(100),
                0.1: np.random.rand(10),
                1.0: np.random.rand(1),
            },
        }

        # Test discretization report
        report = analyzer.generate_discretization_report(test_data)

        assert report is not None
        assert "summary" in report
        assert "detailed_analysis" in report
        assert "recommendations" in report

        # Check report content
        summary = report["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0

        detailed_analysis = report["detailed_analysis"]
        assert isinstance(detailed_analysis, dict)
        assert len(detailed_analysis) > 0

        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
