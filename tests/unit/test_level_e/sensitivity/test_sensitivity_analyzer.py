"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for sensitivity analysis functionality.

This module tests the sensitivity analysis functionality
for Level E experiments in 7D phase field theory.

Physical Meaning:
    Tests ensure that sensitivity analysis correctly
    identifies parameter sensitivity for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/sensitivity/test_sensitivity_analyzer.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import SensitivityAnalyzer


class TestSensitivityAnalyzer:
    """Test sensitivity analysis functionality."""

    def test_initialization(self):
        """Test SensitivityAnalyzer initialization."""
        parameter_ranges = {"beta": (0.6, 1.4), "mu": (0.5, 1.5), "eta": (0.0, 0.3)}

        analyzer = SensitivityAnalyzer(parameter_ranges)

        assert analyzer.parameter_ranges == parameter_ranges
        assert analyzer.sensitivity_results is None
        assert analyzer.parameter_importance is None

    def test_parameter_sensitivity_analysis(self):
        """Test parameter sensitivity analysis."""
        parameter_ranges = {"beta": (0.6, 1.4), "mu": (0.5, 1.5), "eta": (0.0, 0.3)}

        analyzer = SensitivityAnalyzer(parameter_ranges)

        # Mock test data
        test_data = {
            "beta": np.array([0.6, 0.8, 1.0, 1.2, 1.4]),
            "mu": np.array([0.5, 0.8, 1.0, 1.2, 1.5]),
            "eta": np.array([0.0, 0.1, 0.2, 0.3]),
        }

        # Test sensitivity analysis
        results = analyzer.analyze_parameter_sensitivity(test_data)

        assert results is not None
        assert "sensitivity_indices" in results
        assert "parameter_importance" in results
        assert "correlation_matrix" in results

        # Check sensitivity indices
        sensitivity_indices = results["sensitivity_indices"]
        assert len(sensitivity_indices) == 3  # beta, mu, eta
        assert "beta" in sensitivity_indices
        assert "mu" in sensitivity_indices
        assert "eta" in sensitivity_indices

        # Check parameter importance
        parameter_importance = results["parameter_importance"]
        assert len(parameter_importance) == 3
        assert all(0 <= importance <= 1 for importance in parameter_importance.values())

        # Check correlation matrix
        correlation_matrix = results["correlation_matrix"]
        assert correlation_matrix.shape == (3, 3)
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal should be 1

    def test_global_sensitivity_analysis(self):
        """Test global sensitivity analysis."""
        parameter_ranges = {"beta": (0.6, 1.4), "mu": (0.5, 1.5), "eta": (0.0, 0.3)}

        analyzer = SensitivityAnalyzer(parameter_ranges)

        # Mock test data
        test_data = {
            "beta": np.array([0.6, 0.8, 1.0, 1.2, 1.4]),
            "mu": np.array([0.5, 0.8, 1.0, 1.2, 1.5]),
            "eta": np.array([0.0, 0.1, 0.2, 0.3]),
        }

        # Test global sensitivity analysis
        results = analyzer.analyze_global_sensitivity(test_data)

        assert results is not None
        assert "sobol_indices" in results
        assert "total_indices" in results
        assert "interaction_indices" in results

        # Check Sobol indices
        sobol_indices = results["sobol_indices"]
        assert len(sobol_indices) == 3
        assert all(0 <= index <= 1 for index in sobol_indices.values())

        # Check total indices
        total_indices = results["total_indices"]
        assert len(total_indices) == 3
        assert all(0 <= index <= 1 for index in total_indices.values())

        # Check interaction indices
        interaction_indices = results["interaction_indices"]
        assert len(interaction_indices) == 3
        assert all(0 <= index <= 1 for index in interaction_indices.values())

    def test_sensitivity_ranking(self):
        """Test sensitivity ranking."""
        parameter_ranges = {"beta": (0.6, 1.4), "mu": (0.5, 1.5), "eta": (0.0, 0.3)}

        analyzer = SensitivityAnalyzer(parameter_ranges)

        # Mock test data
        test_data = {
            "beta": np.array([0.6, 0.8, 1.0, 1.2, 1.4]),
            "mu": np.array([0.5, 0.8, 1.0, 1.2, 1.5]),
            "eta": np.array([0.0, 0.1, 0.2, 0.3]),
        }

        # Test sensitivity ranking
        ranking = analyzer.rank_parameter_sensitivity(test_data)

        assert ranking is not None
        assert len(ranking) == 3
        assert all(param in ranking for param in ["beta", "mu", "eta"])

        # Check ranking order (should be sorted by sensitivity)
        assert ranking[0] in ["beta", "mu", "eta"]
        assert ranking[1] in ["beta", "mu", "eta"]
        assert ranking[2] in ["beta", "mu", "eta"]

    def test_sensitivity_visualization(self):
        """Test sensitivity visualization."""
        parameter_ranges = {"beta": (0.6, 1.4), "mu": (0.5, 1.5), "eta": (0.0, 0.3)}

        analyzer = SensitivityAnalyzer(parameter_ranges)

        # Mock test data
        test_data = {
            "beta": np.array([0.6, 0.8, 1.0, 1.2, 1.4]),
            "mu": np.array([0.5, 0.8, 1.0, 1.2, 1.5]),
            "eta": np.array([0.0, 0.1, 0.2, 0.3]),
        }

        # Test sensitivity visualization
        visualization = analyzer.visualize_sensitivity(test_data)

        assert visualization is not None
        assert "sensitivity_plot" in visualization
        assert "correlation_plot" in visualization
        assert "importance_plot" in visualization

        # Check plot data
        sensitivity_plot = visualization["sensitivity_plot"]
        assert sensitivity_plot is not None

        correlation_plot = visualization["correlation_plot"]
        assert correlation_plot is not None

        importance_plot = visualization["importance_plot"]
        assert importance_plot is not None

    def test_sensitivity_report(self):
        """Test sensitivity report generation."""
        parameter_ranges = {"beta": (0.6, 1.4), "mu": (0.5, 1.5), "eta": (0.0, 0.3)}

        analyzer = SensitivityAnalyzer(parameter_ranges)

        # Mock test data
        test_data = {
            "beta": np.array([0.6, 0.8, 1.0, 1.2, 1.4]),
            "mu": np.array([0.5, 0.8, 1.0, 1.2, 1.5]),
            "eta": np.array([0.0, 0.1, 0.2, 0.3]),
        }

        # Test sensitivity report
        report = analyzer.generate_sensitivity_report(test_data)

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
