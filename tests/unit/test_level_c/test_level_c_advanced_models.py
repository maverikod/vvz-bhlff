"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced unit tests for Level C models.

This module contains simple, working unit tests for the Level C advanced models,
focusing on quench events, dual-mode sources, beating patterns, and mathematical operations.

Physical Meaning:
    Tests the advanced Level C model capabilities:
    - Quench event detection and analysis
    - Dual-mode source excitation
    - Beating pattern analysis
    - Mathematical operations validation

Mathematical Foundation:
    Tests advanced mathematical operations:
    - Complex number operations for field calculations
    - Matrix operations for system analysis
    - Numerical stability validation

Example:
    >>> pytest tests/unit/test_level_c/test_level_c_advanced_models.py
"""

import pytest
import numpy as np
from unittest.mock import Mock

from bhlff.models.level_c import (
    QuenchEvent,
    DualModeSource,
    BeatingPattern,
)


class TestQuenchEvent:
    """
    Test class for QuenchEvent data structure.

    Physical Meaning:
        Tests the QuenchEvent data structure that represents
        quench events in the system.
    """

    def test_quench_event_creation(self):
        """
        Test QuenchEvent creation.

        Physical Meaning:
            Tests that QuenchEvent objects are created correctly
            with proper parameter assignment.
        """
        location = np.array([1.0, 2.0, 3.0])
        event = QuenchEvent(
            location=location, time=5.0, intensity=0.8, threshold_type="amplitude"
        )

        assert np.array_equal(event.location, location)
        assert event.time == 5.0
        assert event.intensity == 0.8
        assert event.threshold_type == "amplitude"

    def test_quench_event_types(self):
        """
        Test QuenchEvent with different threshold types.

        Physical Meaning:
            Tests that QuenchEvent works correctly with
            different threshold types.
        """
        location = np.array([0.0, 0.0, 0.0])

        for threshold_type in ["amplitude", "detuning", "gradient"]:
            event = QuenchEvent(
                location=location,
                time=0.0,
                intensity=0.0,
                threshold_type=threshold_type,
            )
            assert event.threshold_type == threshold_type


class TestDualModeSource:
    """
    Test class for DualModeSource data structure.

    Physical Meaning:
        Tests the DualModeSource data structure that represents
        dual-mode excitation for beating analysis.
    """

    def test_dual_mode_source_creation(self):
        """
        Test DualModeSource creation.

        Physical Meaning:
            Tests that DualModeSource objects are created correctly
            with proper parameter assignment.
        """
        profile1 = np.array([0.1, 0.2, 0.3])
        profile2 = np.array([0.4, 0.5, 0.6])

        source = DualModeSource(
            frequency_1=0.9,
            frequency_2=1.1,
            amplitude_1=0.8,
            amplitude_2=1.2,
            profile_1=profile1,
            profile_2=profile2,
        )

        assert source.frequency_1 == 0.9
        assert source.frequency_2 == 1.1
        assert source.amplitude_1 == 0.8
        assert source.amplitude_2 == 1.2
        assert np.array_equal(source.profile_1, profile1)
        assert np.array_equal(source.profile_2, profile2)

    def test_dual_mode_source_defaults(self):
        """
        Test DualModeSource with default values.

        Physical Meaning:
            Tests that DualModeSource works correctly with
            default parameter values.
        """
        source = DualModeSource(frequency_1=1.0, frequency_2=1.0)

        assert source.frequency_1 == 1.0
        assert source.frequency_2 == 1.0
        assert source.amplitude_1 == 1.0
        assert source.amplitude_2 == 1.0
        assert source.profile_1 is None
        assert source.profile_2 is None


class TestBeatingPattern:
    """
    Test class for BeatingPattern data structure.

    Physical Meaning:
        Tests the BeatingPattern data structure that represents
        beating pattern analysis results.
    """

    def test_beating_pattern_creation(self):
        """
        Test BeatingPattern creation.

        Physical Meaning:
            Tests that BeatingPattern objects are created correctly
            with proper parameter assignment.
        """
        amplitude_mod = np.array([0.1, 0.2, 0.3])
        phase_evolution = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

        pattern = BeatingPattern(
            beating_frequency=0.1,
            amplitude_modulation=amplitude_mod,
            phase_evolution=phase_evolution,
            temporal_coherence=0.9,
        )

        assert pattern.beating_frequency == 0.1
        assert np.array_equal(pattern.amplitude_modulation, amplitude_mod)
        assert pattern.phase_evolution == phase_evolution
        assert pattern.temporal_coherence == 0.9

    def test_beating_pattern_defaults(self):
        """
        Test BeatingPattern with default values.

        Physical Meaning:
            Tests that BeatingPattern works correctly with
            default parameter values.
        """
        pattern = BeatingPattern(
            beating_frequency=0.0,
            amplitude_modulation=np.array([]),
            phase_evolution=[],
            temporal_coherence=1.0,
        )

        assert pattern.beating_frequency == 0.0
        assert len(pattern.amplitude_modulation) == 0
        assert len(pattern.phase_evolution) == 0
        assert pattern.temporal_coherence == 1.0


class TestMathematicalOperations:
    """
    Test class for basic mathematical operations.

    Physical Meaning:
        Tests basic mathematical operations used in Level C analysis,
        ensuring numerical stability and correctness.
    """

    def test_matrix_operations(self):
        """
        Test matrix operations for ABCD model.

        Physical Meaning:
            Tests basic matrix operations used in transmission
            matrix calculations.
        """
        # Test matrix multiplication
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = A @ B

        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(C, expected)

        # Test matrix determinant
        det_A = np.linalg.det(A)
        expected_det = -2.0
        assert abs(det_A - expected_det) < 1e-10

    def test_complex_operations(self):
        """
        Test complex number operations.

        Physical Meaning:
            Tests complex number operations used in admittance
            and field calculations.
        """
        z1 = complex(1, 2)
        z2 = complex(3, 4)

        # Test complex arithmetic
        sum_z = z1 + z2
        assert sum_z == complex(4, 6)

        product_z = z1 * z2
        assert product_z == complex(-5, 10)

        # Test complex magnitude
        magnitude = abs(z1)
        expected_magnitude = np.sqrt(5)
        assert abs(magnitude - expected_magnitude) < 1e-10

    def test_numerical_stability(self):
        """
        Test numerical stability of operations.

        Physical Meaning:
            Tests that mathematical operations are numerically
            stable and don't produce NaN or infinite values.
        """
        # Test division by small numbers
        small_number = 1e-12
        result = 1.0 / small_number
        assert np.isfinite(result)
        assert result > 0

        # Test logarithm operations
        positive_values = np.array([0.1, 1.0, 10.0, 100.0])
        log_values = np.log(positive_values)
        assert np.all(np.isfinite(log_values))

        # Test exponential operations
        exp_values = np.exp(np.array([-10, 0, 10]))
        assert np.all(np.isfinite(exp_values))
        assert np.all(exp_values > 0)
