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

from bhlff.core.time import QuenchDetector
from bhlff.core.domain import Domain, Parameters


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
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)

    @pytest.fixture
    def quench_detector(self, domain_7d):
        """Create quench detector for testing."""
        return QuenchDetector(
            domain_7d,
            energy_threshold=1e-3,
            rate_threshold=1e-2,
            magnitude_threshold=10.0,
        )

    def test_initialization(self, quench_detector, domain_7d):
        """
        Test quench detector initialization.

        Physical Meaning:
            Validates that the quench detector initializes correctly with
            the specified thresholds.
        """
        assert quench_detector._initialized
        assert quench_detector.domain == domain_7d
        assert quench_detector.energy_threshold == 1e-3
        assert quench_detector.rate_threshold == 1e-2
        assert quench_detector.magnitude_threshold == 10.0
        assert len(quench_detector.quench_history) == 0

    def test_quench_detection_energy(self, quench_detector, domain_7d):
        """
        Test quench detection based on energy threshold.

        Physical Meaning:
            Validates that the quench detector correctly identifies
            energy dumping events based on energy change threshold.
        """
        # Create field with small energy
        small_field = np.ones(domain_7d.shape, dtype=np.complex128) * 0.1

        # First detection (no previous energy)
        quench_detected = quench_detector.detect_quench(small_field, time=0.0)
        assert not quench_detected

        # Create field with large energy change
        large_field = np.ones(domain_7d.shape, dtype=np.complex128) * 100.0

        # Second detection (large energy change)
        quench_detected = quench_detector.detect_quench(large_field, time=0.01)
        assert quench_detected
        assert len(quench_detector.quench_history) == 1

    def test_quench_detection_magnitude(self, quench_detector, domain_7d):
        """
        Test quench detection based on magnitude threshold.

        Physical Meaning:
            Validates that the quench detector correctly identifies
            quench events based on field magnitude threshold.
        """
        # Create field with large magnitude
        large_field = np.ones(domain_7d.shape, dtype=np.complex128) * 20.0

        # Detect quench
        quench_detected = quench_detector.detect_quench(large_field, time=0.0)
        assert quench_detected
        assert len(quench_detector.quench_history) == 1

        # Check quench event details
        quench_event = quench_detector.quench_history[0]
        assert quench_event["time"] == 0.0
        assert "magnitude" in quench_event["reasons"][0]

    def test_quench_history(self, quench_detector, domain_7d):
        """
        Test quench event history.

        Physical Meaning:
            Validates that the quench detector correctly records
            and manages quench event history.
        """
        # Create multiple quench events
        for i in range(3):
            field = np.ones(domain_7d.shape, dtype=np.complex128) * (20.0 + i)
            quench_detector.detect_quench(field, time=i * 0.1)

        # Check history
        history = quench_detector.get_quench_history()
        assert len(history) == 3

        # Check statistics
        stats = quench_detector.get_statistics()
        assert stats["total_quenches"] == 3
        assert stats["quench_rate"] > 0

    def test_quench_clear_history(self, quench_detector, domain_7d):
        """
        Test quench history clearing.

        Physical Meaning:
            Validates that the quench detector can clear its history
            to start fresh monitoring.
        """
        # Create quench event
        field = np.ones(domain_7d.shape, dtype=np.complex128) * 20.0
        quench_detector.detect_quench(field, time=0.0)

        # Check history exists
        assert len(quench_detector.quench_history) == 1

        # Clear history
        quench_detector.clear_history()

        # Check history is cleared
        assert len(quench_detector.quench_history) == 0

    def test_quench_detection_rate(self, quench_detector, domain_7d):
        """
        Test quench detection based on rate threshold.

        Physical Meaning:
            Validates that the quench detector correctly identifies
            quench events based on energy change rate threshold.
        """
        # Create field with moderate energy
        field = np.ones(domain_7d.shape, dtype=np.complex128) * 1.0

        # First detection
        quench_detector.detect_quench(field, time=0.0)

        # Create field with large energy change in short time
        large_field = np.ones(domain_7d.shape, dtype=np.complex128) * 100.0

        # Second detection with high rate
        quench_detected = quench_detector.detect_quench(large_field, time=0.001)
        assert quench_detected
        assert len(quench_detector.quench_history) == 1

    def test_quench_threshold_validation(self, domain_7d):
        """
        Test quench threshold validation.

        Physical Meaning:
            Validates that the quench detector properly validates
            threshold parameters.
        """
        # Test valid thresholds
        detector = QuenchDetector(
            domain_7d,
            energy_threshold=1e-3,
            rate_threshold=1e-2,
            magnitude_threshold=10.0,
        )
        assert detector.energy_threshold == 1e-3
        assert detector.rate_threshold == 1e-2
        assert detector.magnitude_threshold == 10.0

        # Test invalid thresholds
        with pytest.raises(ValueError, match="Energy threshold must be positive"):
            QuenchDetector(
                domain_7d,
                energy_threshold=-1e-3,
                rate_threshold=1e-2,
                magnitude_threshold=10.0,
            )

        with pytest.raises(ValueError, match="Rate threshold must be positive"):
            QuenchDetector(
                domain_7d,
                energy_threshold=1e-3,
                rate_threshold=-1e-2,
                magnitude_threshold=10.0,
            )

        with pytest.raises(ValueError, match="Magnitude threshold must be positive"):
            QuenchDetector(
                domain_7d,
                energy_threshold=1e-3,
                rate_threshold=1e-2,
                magnitude_threshold=-10.0,
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
        result = quench_detector.detect_quench(valid_field, time=0.0)
        assert isinstance(result, bool)

        # Test invalid field shape
        invalid_field = np.random.randn(4, 4, 4).astype(np.complex128)
        with pytest.raises(ValueError, match="Field shape must match domain"):
            quench_detector.detect_quench(invalid_field, time=0.0)

        # Test invalid field type
        invalid_field = np.random.randn(*domain_7d.shape).astype(np.float64)
        with pytest.raises(ValueError, match="Field must be complex"):
            quench_detector.detect_quench(invalid_field, time=0.0)

        # Test invalid time
        with pytest.raises(ValueError, match="Time must be non-negative"):
            quench_detector.detect_quench(valid_field, time=-1.0)

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
            quench_detector.detect_quench(field, time=i * 0.1)

        # Get statistics
        stats = quench_detector.get_statistics()

        # Check statistics
        assert stats["total_quenches"] == 5
        assert stats["quench_rate"] > 0
        assert "average_energy" in stats
        assert "average_magnitude" in stats
        assert "time_span" in stats

        # Check that statistics are reasonable
        assert stats["time_span"] > 0
        assert stats["average_energy"] > 0
        assert stats["average_magnitude"] > 0

    def test_quench_event_details(self, quench_detector, domain_7d):
        """
        Test quench event details.

        Physical Meaning:
            Validates that the quench detector correctly records
            detailed information about quench events.
        """
        # Create quench event
        field = np.ones(domain_7d.shape, dtype=np.complex128) * 20.0
        quench_detector.detect_quench(field, time=0.5)

        # Check event details
        event = quench_detector.quench_history[0]
        assert event["time"] == 0.5
        assert "reasons" in event
        assert "energy" in event
        assert "magnitude" in event
        assert len(event["reasons"]) > 0

        # Check that reasons are valid
        for reason in event["reasons"]:
            assert reason in ["energy", "rate", "magnitude"]
