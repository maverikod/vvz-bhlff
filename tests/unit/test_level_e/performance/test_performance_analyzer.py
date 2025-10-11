"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for performance analysis functionality.

This module tests the performance analysis functionality
for Level E experiments in 7D phase field theory.

Physical Meaning:
    Tests ensure that performance analysis correctly
    analyzes system performance for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/performance/test_performance_analyzer.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    """Test performance analysis functionality."""

    def test_initialization(self):
        """Test PerformanceAnalyzer initialization."""
        performance_metrics = {
            "execution_time": True,
            "memory_usage": True,
            "cpu_usage": True,
            "gpu_usage": True,
        }

        analyzer = PerformanceAnalyzer(performance_metrics)

        assert analyzer.performance_metrics == performance_metrics
        assert analyzer.performance_data is None
        assert analyzer.performance_statistics is None

    def test_execution_time_analysis(self):
        """Test execution time analysis."""
        performance_metrics = {
            "execution_time": True,
            "memory_usage": True,
            "cpu_usage": True,
            "gpu_usage": True,
        }

        analyzer = PerformanceAnalyzer(performance_metrics)

        # Mock test data
        test_data = {
            "execution_times": [1.0, 1.1, 1.2, 1.3, 1.4],
            "iteration_times": [0.1, 0.11, 0.12, 0.13, 0.14],
            "total_time": 5.0,
        }

        # Test execution time analysis
        results = analyzer.analyze_execution_time(test_data)

        assert results is not None
        assert "execution_statistics" in results
        assert "performance_trends" in results
        assert "bottleneck_analysis" in results

        # Check execution statistics
        execution_statistics = results["execution_statistics"]
        assert isinstance(execution_statistics, dict)
        assert "mean_time" in execution_statistics
        assert "std_time" in execution_statistics
        assert "max_time" in execution_statistics

        # Check performance trends
        performance_trends = results["performance_trends"]
        assert isinstance(performance_trends, dict)
        assert "trend_direction" in performance_trends
        assert "trend_magnitude" in performance_trends

        # Check bottleneck analysis
        bottleneck_analysis = results["bottleneck_analysis"]
        assert isinstance(bottleneck_analysis, dict)
        assert "bottleneck_operations" in bottleneck_analysis
        assert "optimization_recommendations" in bottleneck_analysis

    def test_memory_usage_analysis(self):
        """Test memory usage analysis."""
        performance_metrics = {
            "execution_time": True,
            "memory_usage": True,
            "cpu_usage": True,
            "gpu_usage": True,
        }

        analyzer = PerformanceAnalyzer(performance_metrics)

        # Mock test data
        test_data = {
            "memory_usage": [100, 110, 120, 130, 140],
            "peak_memory": 150,
            "memory_leaks": [0, 1, 2, 3, 4],
        }

        # Test memory usage analysis
        results = analyzer.analyze_memory_usage(test_data)

        assert results is not None
        assert "memory_statistics" in results
        assert "memory_trends" in results
        assert "leak_analysis" in results

        # Check memory statistics
        memory_statistics = results["memory_statistics"]
        assert isinstance(memory_statistics, dict)
        assert "mean_usage" in memory_statistics
        assert "peak_usage" in memory_statistics
        assert "usage_variance" in memory_statistics

        # Check memory trends
        memory_trends = results["memory_trends"]
        assert isinstance(memory_trends, dict)
        assert "trend_direction" in memory_trends
        assert "trend_magnitude" in memory_trends

        # Check leak analysis
        leak_analysis = results["leak_analysis"]
        assert isinstance(leak_analysis, dict)
        assert "leak_count" in leak_analysis
        assert "leak_severity" in leak_analysis

    def test_cpu_usage_analysis(self):
        """Test CPU usage analysis."""
        performance_metrics = {
            "execution_time": True,
            "memory_usage": True,
            "cpu_usage": True,
            "gpu_usage": True,
        }

        analyzer = PerformanceAnalyzer(performance_metrics)

        # Mock test data
        test_data = {
            "cpu_usage": [50, 55, 60, 65, 70],
            "cpu_cores": 4,
            "cpu_frequency": 2.4,
        }

        # Test CPU usage analysis
        results = analyzer.analyze_cpu_usage(test_data)

        assert results is not None
        assert "cpu_statistics" in results
        assert "cpu_trends" in results
        assert "efficiency_analysis" in results

        # Check CPU statistics
        cpu_statistics = results["cpu_statistics"]
        assert isinstance(cpu_statistics, dict)
        assert "mean_usage" in cpu_statistics
        assert "peak_usage" in cpu_statistics
        assert "usage_variance" in cpu_statistics

        # Check CPU trends
        cpu_trends = results["cpu_trends"]
        assert isinstance(cpu_trends, dict)
        assert "trend_direction" in cpu_trends
        assert "trend_magnitude" in cpu_trends

        # Check efficiency analysis
        efficiency_analysis = results["efficiency_analysis"]
        assert isinstance(efficiency_analysis, dict)
        assert "efficiency_metrics" in efficiency_analysis
        assert "optimization_recommendations" in efficiency_analysis

    def test_gpu_usage_analysis(self):
        """Test GPU usage analysis."""
        performance_metrics = {
            "execution_time": True,
            "memory_usage": True,
            "cpu_usage": True,
            "gpu_usage": True,
        }

        analyzer = PerformanceAnalyzer(performance_metrics)

        # Mock test data
        test_data = {
            "gpu_usage": [30, 35, 40, 45, 50],
            "gpu_memory": [2000, 2100, 2200, 2300, 2400],
            "gpu_temperature": [60, 65, 70, 75, 80],
        }

        # Test GPU usage analysis
        results = analyzer.analyze_gpu_usage(test_data)

        assert results is not None
        assert "gpu_statistics" in results
        assert "gpu_trends" in results
        assert "thermal_analysis" in results

        # Check GPU statistics
        gpu_statistics = results["gpu_statistics"]
        assert isinstance(gpu_statistics, dict)
        assert "mean_usage" in gpu_statistics
        assert "peak_usage" in gpu_statistics
        assert "usage_variance" in gpu_statistics

        # Check GPU trends
        gpu_trends = results["gpu_trends"]
        assert isinstance(gpu_trends, dict)
        assert "trend_direction" in gpu_trends
        assert "trend_magnitude" in gpu_trends

        # Check thermal analysis
        thermal_analysis = results["thermal_analysis"]
        assert isinstance(thermal_analysis, dict)
        assert "temperature_trends" in thermal_analysis
        assert "thermal_recommendations" in thermal_analysis

    def test_performance_optimization(self):
        """Test performance optimization analysis."""
        performance_metrics = {
            "execution_time": True,
            "memory_usage": True,
            "cpu_usage": True,
            "gpu_usage": True,
        }

        analyzer = PerformanceAnalyzer(performance_metrics)

        # Mock test data
        test_data = {
            "execution_times": [1.0, 1.1, 1.2, 1.3, 1.4],
            "memory_usage": [100, 110, 120, 130, 140],
            "cpu_usage": [50, 55, 60, 65, 70],
            "gpu_usage": [30, 35, 40, 45, 50],
        }

        # Test performance optimization
        results = analyzer.analyze_performance_optimization(test_data)

        assert results is not None
        assert "optimization_opportunities" in results
        assert "optimization_impact" in results
        assert "optimization_recommendations" in results

        # Check optimization opportunities
        optimization_opportunities = results["optimization_opportunities"]
        assert isinstance(optimization_opportunities, list)
        assert len(optimization_opportunities) >= 0

        # Check optimization impact
        optimization_impact = results["optimization_impact"]
        assert isinstance(optimization_impact, dict)
        assert "performance_gain" in optimization_impact
        assert "resource_savings" in optimization_impact

        # Check optimization recommendations
        optimization_recommendations = results["optimization_recommendations"]
        assert isinstance(optimization_recommendations, list)
        assert len(optimization_recommendations) >= 0

    def test_performance_report(self):
        """Test performance report generation."""
        performance_metrics = {
            "execution_time": True,
            "memory_usage": True,
            "cpu_usage": True,
            "gpu_usage": True,
        }

        analyzer = PerformanceAnalyzer(performance_metrics)

        # Mock test data
        test_data = {
            "execution_times": [1.0, 1.1, 1.2, 1.3, 1.4],
            "memory_usage": [100, 110, 120, 130, 140],
            "cpu_usage": [50, 55, 60, 65, 70],
            "gpu_usage": [30, 35, 40, 45, 50],
        }

        # Test performance report
        report = analyzer.generate_performance_report(test_data)

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
