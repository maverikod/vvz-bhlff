"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for phase mapping functionality.

This module tests the phase mapping functionality
for Level E experiments in 7D phase field theory.

Physical Meaning:
    Tests ensure that phase mapping correctly
    maps phase structures for 7D theory validation.

Example:
    >>> pytest tests/unit/test_level_e/phase_mapping/test_phase_mapper.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import PhaseMapper


class TestPhaseMapper:
    """Test phase mapping functionality."""

    def test_initialization(self):
        """Test PhaseMapper initialization."""
        mapping_parameters = {
            "phase_resolution": 0.01,
            "mapping_tolerance": 1e-6,
            "max_iterations": 1000,
        }

        mapper = PhaseMapper(mapping_parameters)

        assert mapper.mapping_parameters == mapping_parameters
        assert mapper.phase_maps is None
        assert mapper.mapping_statistics is None

    def test_phase_structure_mapping(self):
        """Test phase structure mapping."""
        mapping_parameters = {
            "phase_resolution": 0.01,
            "mapping_tolerance": 1e-6,
            "max_iterations": 1000,
        }

        mapper = PhaseMapper(mapping_parameters)

        # Mock test data
        test_data = {
            "field_data": np.random.rand(64, 64, 64),
            "phase_reference": np.random.rand(64, 64, 64),
        }

        # Test phase structure mapping
        results = mapper.map_phase_structure(test_data)

        assert results is not None
        assert "phase_map" in results
        assert "mapping_accuracy" in results
        assert "phase_statistics" in results

        # Check phase map
        phase_map = results["phase_map"]
        assert isinstance(phase_map, np.ndarray)
        assert phase_map.shape == (64, 64, 64)

        # Check mapping accuracy
        mapping_accuracy = results["mapping_accuracy"]
        assert isinstance(mapping_accuracy, float)
        assert 0 <= mapping_accuracy <= 1

        # Check phase statistics
        phase_statistics = results["phase_statistics"]
        assert isinstance(phase_statistics, dict)
        assert "mean_phase" in phase_statistics
        assert "phase_variance" in phase_statistics

    def test_phase_transition_mapping(self):
        """Test phase transition mapping."""
        mapping_parameters = {
            "phase_resolution": 0.01,
            "mapping_tolerance": 1e-6,
            "max_iterations": 1000,
        }

        mapper = PhaseMapper(mapping_parameters)

        # Mock test data
        test_data = {
            "initial_phase": np.random.rand(64, 64, 64),
            "final_phase": np.random.rand(64, 64, 64),
            "transition_time": 1.0,
        }

        # Test phase transition mapping
        results = mapper.map_phase_transition(test_data)

        assert results is not None
        assert "transition_map" in results
        assert "transition_velocity" in results
        assert "transition_statistics" in results

        # Check transition map
        transition_map = results["transition_map"]
        assert isinstance(transition_map, np.ndarray)
        assert transition_map.shape == (64, 64, 64)

        # Check transition velocity
        transition_velocity = results["transition_velocity"]
        assert isinstance(transition_velocity, float)
        assert transition_velocity >= 0

        # Check transition statistics
        transition_statistics = results["transition_statistics"]
        assert isinstance(transition_statistics, dict)
        assert "transition_rate" in transition_statistics
        assert "transition_duration" in transition_statistics

    def test_phase_defect_mapping(self):
        """Test phase defect mapping."""
        mapping_parameters = {
            "phase_resolution": 0.01,
            "mapping_tolerance": 1e-6,
            "max_iterations": 1000,
        }

        mapper = PhaseMapper(mapping_parameters)

        # Mock test data
        test_data = {
            "field_data": np.random.rand(64, 64, 64),
            "defect_locations": [(32, 32, 32), (16, 16, 16)],
        }

        # Test phase defect mapping
        results = mapper.map_phase_defects(test_data)

        assert results is not None
        assert "defect_map" in results
        assert "defect_properties" in results
        assert "defect_statistics" in results

        # Check defect map
        defect_map = results["defect_map"]
        assert isinstance(defect_map, np.ndarray)
        assert defect_map.shape == (64, 64, 64)

        # Check defect properties
        defect_properties = results["defect_properties"]
        assert isinstance(defect_properties, list)
        assert len(defect_properties) >= 0

        # Check defect statistics
        defect_statistics = results["defect_statistics"]
        assert isinstance(defect_statistics, dict)
        assert "defect_count" in defect_statistics
        assert "defect_density" in defect_statistics

    def test_phase_coherence_mapping(self):
        """Test phase coherence mapping."""
        mapping_parameters = {
            "phase_resolution": 0.01,
            "mapping_tolerance": 1e-6,
            "max_iterations": 1000,
        }

        mapper = PhaseMapper(mapping_parameters)

        # Mock test data
        test_data = {
            "field_data": np.random.rand(64, 64, 64),
            "coherence_reference": np.random.rand(64, 64, 64),
        }

        # Test phase coherence mapping
        results = mapper.map_phase_coherence(test_data)

        assert results is not None
        assert "coherence_map" in results
        assert "coherence_statistics" in results
        assert "coherence_analysis" in results

        # Check coherence map
        coherence_map = results["coherence_map"]
        assert isinstance(coherence_map, np.ndarray)
        assert coherence_map.shape == (64, 64, 64)

        # Check coherence statistics
        coherence_statistics = results["coherence_statistics"]
        assert isinstance(coherence_statistics, dict)
        assert "mean_coherence" in coherence_statistics
        assert "coherence_variance" in coherence_statistics

        # Check coherence analysis
        coherence_analysis = results["coherence_analysis"]
        assert isinstance(coherence_analysis, dict)
        assert "coherence_regions" in coherence_analysis
        assert "coherence_transitions" in coherence_analysis

    def test_phase_mapping_validation(self):
        """Test phase mapping validation."""
        mapping_parameters = {
            "phase_resolution": 0.01,
            "mapping_tolerance": 1e-6,
            "max_iterations": 1000,
        }

        mapper = PhaseMapper(mapping_parameters)

        # Mock test data
        test_data = {
            "field_data": np.random.rand(64, 64, 64),
            "phase_reference": np.random.rand(64, 64, 64),
        }

        # Test phase mapping validation
        results = mapper.validate_phase_mapping(test_data)

        assert results is not None
        assert "validation_metrics" in results
        assert "validation_errors" in results
        assert "validation_status" in results

        # Check validation metrics
        validation_metrics = results["validation_metrics"]
        assert isinstance(validation_metrics, dict)
        assert "accuracy" in validation_metrics
        assert "precision" in validation_metrics
        assert "recall" in validation_metrics

        # Check validation errors
        validation_errors = results["validation_errors"]
        assert isinstance(validation_errors, list)
        assert len(validation_errors) >= 0

        # Check validation status
        validation_status = results["validation_status"]
        assert isinstance(validation_status, str)
        assert validation_status in ["valid", "invalid", "warning"]

    def test_phase_mapping_report(self):
        """Test phase mapping report generation."""
        mapping_parameters = {
            "phase_resolution": 0.01,
            "mapping_tolerance": 1e-6,
            "max_iterations": 1000,
        }

        mapper = PhaseMapper(mapping_parameters)

        # Mock test data
        test_data = {
            "field_data": np.random.rand(64, 64, 64),
            "phase_reference": np.random.rand(64, 64, 64),
        }

        # Test phase mapping report
        report = mapper.generate_phase_mapping_report(test_data)

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
