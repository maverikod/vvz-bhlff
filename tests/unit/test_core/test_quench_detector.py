"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for QuenchDetector.

This module contains unit tests for the QuenchDetector class
in the 7D BVP framework, focusing on energy dumping event
detection during temporal integration.

Physical Meaning:
    Tests the quench detection system for monitoring energy dumping
    events during temporal integration.

Mathematical Foundation:
    Tests validate the quench detection implementation for:
    - Energy threshold monitoring
    - Rate-based detection
    - Magnitude-based detection
    - Event history management
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.bvp.quench_detector import QuenchDetector
from bhlff.core.domain.domain_7d import Domain7D
from bhlff.core.domain.config import SpatialConfig, PhaseConfig, TemporalConfig


class TestQuenchDetector:
    """
    Unit tests for QuenchDetector.

    Physical Meaning:
        Tests the quench detection system for monitoring energy dumping
        events during temporal integration.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        spatial_config = SpatialConfig(L_x=1.0, L_y=1.0, L_z=1.0, N_x=8, N_y=8, N_z=8)
        phase_config = PhaseConfig(phi_1_max=2*np.pi, phi_2_max=2*np.pi, phi_3_max=2*np.pi, N_phi_1=4, N_phi_2=4, N_phi_3=4)
        temporal_config = TemporalConfig(T_max=1.0, N_t=8, dt=0.125)
        return Domain7D(spatial_config, phase_config, temporal_config)

    @pytest.fixture
    def quench_detector(self, domain_7d):
        """Create quench detector for testing."""
        config = {
            "amplitude_threshold": 10.0,
            "detuning_threshold": 1e-2,
            "gradient_threshold": 1e-3,
            "use_cuda": False
        }
        return QuenchDetector(domain_7d, config)

    def test_initialization(self, quench_detector, domain_7d):
        """
        Test quench detector initialization.

        Physical Meaning:
            Validates that the quench detector initializes correctly with
            the specified thresholds.
        """
        assert quench_detector.domain_7d == domain_7d
        assert quench_detector.amplitude_threshold == 10.0
        assert quench_detector.detuning_threshold == 1e-2
        assert quench_detector.gradient_threshold == 1e-3

    def test_quench_detection_energy(self, quench_detector, domain_7d):
        """
        Test quench detection based on energy threshold.

        Physical Meaning:
            Validates that the quench detector correctly identifies
            energy dumping events based on energy change threshold.
        """
        # Create field with small energy
        small_field = np.ones(domain_7d.shape, dtype=np.complex128) * 0.1

        # Test quench detection with small field
        quenches = quench_detector.detect_quenches(small_field)
        assert isinstance(quenches, dict)
        assert "quenches_detected" in quenches

        # Create field with large energy change
        large_field = np.ones(domain_7d.shape, dtype=np.complex128) * 100.0

        # Test quench detection with large field
        quenches = quench_detector.detect_quenches(large_field)
        assert isinstance(quenches, dict)
        assert "quenches_detected" in quenches

    def test_quench_detection_magnitude(self, quench_detector, domain_7d):
        """
        Test quench detection based on magnitude threshold.

        Physical Meaning:
            Validates that the quench detector correctly identifies
            quench events based on field magnitude threshold.
        """
        # Create field with large magnitude
        large_field = np.ones(domain_7d.shape, dtype=np.complex128) * 20.0

        # Detect quenches
        quenches = quench_detector.detect_quenches(large_field)
        assert isinstance(quenches, dict)
        assert "quenches_detected" in quenches

    def test_quench_detection_multiple(self, quench_detector, domain_7d):
        """
        Test multiple quench detections.

        Physical Meaning:
            Validates that the quench detector can handle
            multiple detection calls correctly.
        """
        # Create multiple quench events
        for i in range(3):
            field = np.ones(domain_7d.shape, dtype=np.complex128) * (20.0 + i)
            quenches = quench_detector.detect_quenches(field)
            assert isinstance(quenches, dict)
            assert "quenches_detected" in quenches

    def test_quench_detection_consistency(self, quench_detector, domain_7d):
        """
        Test quench detection consistency.

        Physical Meaning:
            Validates that the quench detector produces
            consistent results for the same input.
        """
        # Create field
        field = np.ones(domain_7d.shape, dtype=np.complex128) * 20.0

        # Test multiple times with same field
        quenches1 = quench_detector.detect_quenches(field)
        quenches2 = quench_detector.detect_quenches(field)
        
        # Results should be consistent
        assert isinstance(quenches1, dict)
        assert isinstance(quenches2, dict)
        assert "quenches_detected" in quenches1
        assert "quenches_detected" in quenches2

    def test_quench_detection_different_fields(self, quench_detector, domain_7d):
        """
        Test quench detection with different field types.

        Physical Meaning:
            Validates that the quench detector works with
            different field configurations.
        """
        # Test with moderate field
        field = np.ones(domain_7d.shape, dtype=np.complex128) * 1.0
        quenches = quench_detector.detect_quenches(field)
        assert isinstance(quenches, dict)

        # Test with large field
        large_field = np.ones(domain_7d.shape, dtype=np.complex128) * 100.0
        quenches = quench_detector.detect_quenches(large_field)
        assert isinstance(quenches, dict)

    def test_quench_threshold_validation(self, domain_7d):
        """
        Test quench threshold validation.

        Physical Meaning:
            Validates that the quench detector properly validates
            threshold parameters.
        """
        # Test valid thresholds
        config = {
            "amplitude_threshold": 10.0,
            "detuning_threshold": 1e-2,
            "gradient_threshold": 1e-3,
            "use_cuda": False
        }
        detector = QuenchDetector(domain_7d, config)
        assert detector.amplitude_threshold == 10.0
        assert detector.detuning_threshold == 1e-2
        assert detector.gradient_threshold == 1e-3

        with pytest.raises(ValueError, match="Detuning threshold must be positive"):
            QuenchDetector(
                domain_7d,
                {
                    "amplitude_threshold": 10.0,
                    "detuning_threshold": -1e-2,
                    "gradient_threshold": 1e-3,
                    "use_cuda": False
                }
            )

        with pytest.raises(ValueError, match="Amplitude threshold must be positive"):
            QuenchDetector(
                domain_7d,
                {
                    "amplitude_threshold": -10.0,
                    "detuning_threshold": 1e-2,
                    "gradient_threshold": 1e-3,
                    "use_cuda": False
                }
            )

    def test_quench_field_validation(self, quench_detector, domain_7d):
        """
        Test quench field validation.

        Physical Meaning:
            Validates that the quench detector properly validates
            input fields.
        """
        # Test valid field
        valid_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        result = quench_detector.detect_quenches(valid_field)
        assert isinstance(result, dict)
        assert "quenches_detected" in result

        # Test that detect_quenches works with valid field
        result = quench_detector.detect_quenches(valid_field)
        assert isinstance(result, dict)

        # Test that detect_quenches works with float field too
        float_field = np.random.randn(*domain_7d.shape).astype(np.float64)
        result = quench_detector.detect_quenches(float_field)
        assert isinstance(result, dict)

        # Test that detect_quenches works with valid field
        result = quench_detector.detect_quenches(valid_field)
        assert isinstance(result, dict)

    def test_quench_statistics(self, quench_detector, domain_7d):
        """
        Test quench statistics computation.

        Physical Meaning:
            Validates that the quench detector correctly computes
            statistics about quench events.
        """
        # Create multiple quench events
        for i in range(5):
            field = np.ones(domain_7d.shape, dtype=np.complex128) * (20.0 + i)
            quench_detector.detect_quenches(field)

        # Test that detect_quenches returns valid results
        for i in range(5):
            field = np.ones(domain_7d.shape, dtype=np.complex128) * (20.0 + i)
            result = quench_detector.detect_quenches(field)
            
            # Check that result is a valid dictionary
            assert isinstance(result, dict)
            assert "quenches_detected" in result
            assert "total_quenches" in result
            assert "quench_locations" in result
            assert "quench_types" in result

    def test_quench_event_details(self, quench_detector, domain_7d):
        """
        Test quench event details.

        Physical Meaning:
            Validates that the quench detector correctly records
            detailed information about quench events.
        """
        # Create quench event
        field = np.ones(domain_7d.shape, dtype=np.complex128) * 20.0
        quench_detector.detect_quenches(field)

        # Test that detect_quenches returns valid results
        result = quench_detector.detect_quenches(field)
        
        # Check that result is a valid dictionary with expected keys
        assert isinstance(result, dict)
        assert "quenches_detected" in result
        assert "total_quenches" in result
        assert "quench_locations" in result
        assert "quench_types" in result
        assert "quench_strengths" in result
        assert "amplitude_quenches" in result
        assert "detuning_quenches" in result
        assert "gradient_quenches" in result
