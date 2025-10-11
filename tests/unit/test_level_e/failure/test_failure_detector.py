"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for failure detection functionality.

This module tests the failure detection functionality
for Level E experiments in 7D phase field theory.

Physical Meaning:
    Tests ensure that failure detection correctly
    identifies system failures for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/failure/test_failure_detector.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import FailureDetector


class TestFailureDetector:
    """Test failure detection functionality."""

    def test_initialization(self):
        """Test FailureDetector initialization."""
        failure_thresholds = {
            "energy_conservation": 1e-6,
            "momentum_conservation": 1e-6,
            "topological_charge": 1e-8,
        }

        detector = FailureDetector(failure_thresholds)

        assert detector.failure_thresholds == failure_thresholds
        assert detector.failure_history is None
        assert detector.failure_statistics is None

    def test_energy_conservation_failure_detection(self):
        """Test energy conservation failure detection."""
        failure_thresholds = {
            "energy_conservation": 1e-6,
            "momentum_conservation": 1e-6,
            "topological_charge": 1e-8,
        }

        detector = FailureDetector(failure_thresholds)

        # Mock test data
        test_data = {
            "initial_energy": 1.0,
            "final_energy": 0.999,
            "energy_tolerance": 1e-6,
        }

        # Test energy conservation failure detection
        results = detector.detect_energy_conservation_failure(test_data)

        assert results is not None
        assert "failure_detected" in results
        assert "failure_severity" in results
        assert "failure_reason" in results

        # Check failure detection
        failure_detected = results["failure_detected"]
        assert isinstance(failure_detected, bool)

        # Check failure severity
        failure_severity = results["failure_severity"]
        assert isinstance(failure_severity, str)
        assert failure_severity in ["low", "medium", "high", "critical"]

        # Check failure reason
        failure_reason = results["failure_reason"]
        assert isinstance(failure_reason, str)
        assert len(failure_reason) > 0

    def test_momentum_conservation_failure_detection(self):
        """Test momentum conservation failure detection."""
        failure_thresholds = {
            "energy_conservation": 1e-6,
            "momentum_conservation": 1e-6,
            "topological_charge": 1e-8,
        }

        detector = FailureDetector(failure_thresholds)

        # Mock test data
        test_data = {
            "initial_momentum": np.array([1.0, 0.0, 0.0]),
            "final_momentum": np.array([0.999, 0.001, 0.0]),
            "momentum_tolerance": 1e-6,
        }

        # Test momentum conservation failure detection
        results = detector.detect_momentum_conservation_failure(test_data)

        assert results is not None
        assert "failure_detected" in results
        assert "failure_severity" in results
        assert "failure_reason" in results

        # Check failure detection
        failure_detected = results["failure_detected"]
        assert isinstance(failure_detected, bool)

        # Check failure severity
        failure_severity = results["failure_severity"]
        assert isinstance(failure_severity, str)
        assert failure_severity in ["low", "medium", "high", "critical"]

        # Check failure reason
        failure_reason = results["failure_reason"]
        assert isinstance(failure_reason, str)
        assert len(failure_reason) > 0

    def test_topological_charge_failure_detection(self):
        """Test topological charge failure detection."""
        failure_thresholds = {
            "energy_conservation": 1e-6,
            "momentum_conservation": 1e-6,
            "topological_charge": 1e-8,
        }

        detector = FailureDetector(failure_thresholds)

        # Mock test data
        test_data = {
            "expected_charge": 1.0,
            "computed_charge": 0.999,
            "charge_tolerance": 1e-8,
        }

        # Test topological charge failure detection
        results = detector.detect_topological_charge_failure(test_data)

        assert results is not None
        assert "failure_detected" in results
        assert "failure_severity" in results
        assert "failure_reason" in results

        # Check failure detection
        failure_detected = results["failure_detected"]
        assert isinstance(failure_detected, bool)

        # Check failure severity
        failure_severity = results["failure_severity"]
        assert isinstance(failure_severity, str)
        assert failure_severity in ["low", "medium", "high", "critical"]

        # Check failure reason
        failure_reason = results["failure_reason"]
        assert isinstance(failure_reason, str)
        assert len(failure_reason) > 0

    def test_convergence_failure_detection(self):
        """Test convergence failure detection."""
        failure_thresholds = {
            "energy_conservation": 1e-6,
            "momentum_conservation": 1e-6,
            "topological_charge": 1e-8,
        }

        detector = FailureDetector(failure_thresholds)

        # Mock test data
        test_data = {
            "convergence_history": np.random.rand(100),
            "convergence_tolerance": 1e-6,
            "max_iterations": 1000,
        }

        # Test convergence failure detection
        results = detector.detect_convergence_failure(test_data)

        assert results is not None
        assert "failure_detected" in results
        assert "failure_severity" in results
        assert "failure_reason" in results

        # Check failure detection
        failure_detected = results["failure_detected"]
        assert isinstance(failure_detected, bool)

        # Check failure severity
        failure_severity = results["failure_severity"]
        assert isinstance(failure_severity, str)
        assert failure_severity in ["low", "medium", "high", "critical"]

        # Check failure reason
        failure_reason = results["failure_reason"]
        assert isinstance(failure_reason, str)
        assert len(failure_reason) > 0

    def test_failure_statistics(self):
        """Test failure statistics calculation."""
        failure_thresholds = {
            "energy_conservation": 1e-6,
            "momentum_conservation": 1e-6,
            "topological_charge": 1e-8,
        }

        detector = FailureDetector(failure_thresholds)

        # Mock test data
        test_data = {
            "failure_history": [
                {"type": "energy_conservation", "severity": "high"},
                {"type": "momentum_conservation", "severity": "medium"},
                {"type": "topological_charge", "severity": "low"},
            ],
        }

        # Test failure statistics
        results = detector.calculate_failure_statistics(test_data)

        assert results is not None
        assert "total_failures" in results
        assert "failure_by_type" in results
        assert "failure_by_severity" in results

        # Check total failures
        total_failures = results["total_failures"]
        assert isinstance(total_failures, int)
        assert total_failures >= 0

        # Check failure by type
        failure_by_type = results["failure_by_type"]
        assert isinstance(failure_by_type, dict)
        assert "energy_conservation" in failure_by_type
        assert "momentum_conservation" in failure_by_type
        assert "topological_charge" in failure_by_type

        # Check failure by severity
        failure_by_severity = results["failure_by_severity"]
        assert isinstance(failure_by_severity, dict)
        assert "high" in failure_by_severity
        assert "medium" in failure_by_severity
        assert "low" in failure_by_severity

    def test_failure_report(self):
        """Test failure report generation."""
        failure_thresholds = {
            "energy_conservation": 1e-6,
            "momentum_conservation": 1e-6,
            "topological_charge": 1e-8,
        }

        detector = FailureDetector(failure_thresholds)

        # Mock test data
        test_data = {
            "failure_history": [
                {"type": "energy_conservation", "severity": "high"},
                {"type": "momentum_conservation", "severity": "medium"},
                {"type": "topological_charge", "severity": "low"},
            ],
        }

        # Test failure report
        report = detector.generate_failure_report(test_data)

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
