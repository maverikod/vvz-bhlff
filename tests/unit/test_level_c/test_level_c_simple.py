"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple unit tests for Level C modules.

This module contains simple, working unit tests for the Level C functionality,
focusing on basic initialization and core functionality without complex mocking.

Physical Meaning:
    Tests the basic Level C analysis capabilities:
    - Module initialization and basic functionality
    - Data structure creation and validation
    - Core mathematical operations

Mathematical Foundation:
    Tests basic mathematical operations:
    - Matrix operations for ABCD model
    - Basic field operations for boundary analysis
    - Simple memory kernel operations

Example:
    >>> pytest tests/unit/test_level_c/test_level_c_simple.py
"""

import pytest
import numpy as np
from unittest.mock import Mock

from bhlff.models.level_c import (
    ABCDModel,
    ResonatorLayer,
    SystemMode,
    MemoryParameters,
    QuenchEvent,
    DualModeSource,
    BeatingPattern,
)


class TestABCDModel:
    """
    Test class for ABCD model functionality.

    Physical Meaning:
        Tests the ABCD model implementation for resonator
        chain analysis and system mode detection.
    """

    def test_abcd_model_initialization(self):
        """
        Test ABCD model initialization.

        Physical Meaning:
            Tests that the ABCD model initializes correctly
            with resonator layers.
        """
        resonators = [
            ResonatorLayer(radius=1.0, thickness=0.1, contrast=0.3),
            ResonatorLayer(radius=2.0, thickness=0.1, contrast=0.5),
        ]

        model = ABCDModel(resonators)

        assert len(model.resonators) == 2
        assert model.resonators[0].radius == 1.0
        assert model.resonators[1].radius == 2.0
        assert model.resonators[0].contrast == 0.3
        assert model.resonators[1].contrast == 0.5

    def test_compute_transmission_matrix(self):
        """
        Test transmission matrix computation.

        Physical Meaning:
            Tests the computation of 2x2 transmission matrices
            for the resonator chain at given frequencies.
        """
        resonators = [
            ResonatorLayer(radius=1.0, thickness=0.1, contrast=0.3),
            ResonatorLayer(radius=2.0, thickness=0.1, contrast=0.5),
        ]

        model = ABCDModel(resonators)
        T = model.compute_transmission_matrix(1.0)

        assert T.shape == (2, 2)
        assert isinstance(T, np.ndarray)
        # Check that matrix is properly formed
        assert np.all(np.isfinite(T))

    def test_compute_system_admittance(self):
        """
        Test system admittance computation.

        Physical Meaning:
            Tests the computation of complex admittance Y(ω)
            for the resonator chain.
        """
        resonators = [ResonatorLayer(radius=1.0, thickness=0.1, contrast=0.3)]

        model = ABCDModel(resonators)
        admittance = model.compute_system_admittance(1.0)

        # Admittance can be real or complex
        assert isinstance(admittance, (complex, float))
        if isinstance(admittance, complex):
            assert np.isfinite(admittance.real)
            assert np.isfinite(admittance.imag)
        else:
            assert np.isfinite(admittance)

    def test_find_resonance_conditions(self):
        """
        Test resonance condition finding.

        Physical Meaning:
            Tests the identification of frequencies where
            det(T_total - I) = 0, corresponding to system resonances.
        """
        resonators = [
            ResonatorLayer(radius=1.0, thickness=0.1, contrast=0.3),
            ResonatorLayer(radius=2.0, thickness=0.1, contrast=0.5),
        ]

        model = ABCDModel(resonators)
        resonances = model.find_resonance_conditions((0.1, 3.0))

        assert isinstance(resonances, list)
        # Check that all resonances are within the specified range
        for resonance in resonances:
            assert 0.1 <= resonance <= 3.0
            assert np.isfinite(resonance)

    def test_find_system_modes(self):
        """
        Test system mode finding.

        Physical Meaning:
            Tests the identification of system resonance modes
            with their frequencies and quality factors.
        """
        resonators = [ResonatorLayer(radius=1.0, thickness=0.1, contrast=0.3)]

        model = ABCDModel(resonators)
        modes = model.find_system_modes((0.1, 2.0))

        assert isinstance(modes, list)
        for mode in modes:
            assert isinstance(mode, SystemMode)
            assert 0.1 <= mode.frequency <= 2.0
            assert mode.quality_factor > 0
            assert np.isfinite(mode.amplitude)
            assert np.isfinite(mode.phase)


class TestResonatorLayer:
    """
    Test class for ResonatorLayer data structure.

    Physical Meaning:
        Tests the ResonatorLayer data structure that represents
        individual resonator layers in the chain.
    """

    def test_resonator_layer_creation(self):
        """
        Test ResonatorLayer creation.

        Physical Meaning:
            Tests that ResonatorLayer objects are created correctly
            with proper parameter assignment.
        """
        layer = ResonatorLayer(
            radius=2.5, thickness=0.3, contrast=0.7, memory_gamma=0.4, memory_tau=1.5
        )

        assert layer.radius == 2.5
        assert layer.thickness == 0.3
        assert layer.contrast == 0.7
        assert layer.memory_gamma == 0.4
        assert layer.memory_tau == 1.5
        assert layer.material_params is None

    def test_resonator_layer_defaults(self):
        """
        Test ResonatorLayer with default values.

        Physical Meaning:
            Tests that ResonatorLayer works correctly with
            default parameter values.
        """
        layer = ResonatorLayer(radius=1.0, thickness=0.1, contrast=0.5)

        assert layer.radius == 1.0
        assert layer.thickness == 0.1
        assert layer.contrast == 0.5
        assert layer.memory_gamma == 0.0
        assert layer.memory_tau == 1.0


class TestSystemMode:
    """
    Test class for SystemMode data structure.

    Physical Meaning:
        Tests the SystemMode data structure that represents
        system resonance modes with their properties.
    """

    def test_system_mode_creation(self):
        """
        Test SystemMode creation.

        Physical Meaning:
            Tests that SystemMode objects are created correctly
            with proper parameter assignment.
        """
        mode = SystemMode(
            frequency=1.5,
            quality_factor=25.0,
            amplitude=0.8,
            phase=0.3,
            mode_index=2,
            coupling_strength=0.15,
        )

        assert mode.frequency == 1.5
        assert mode.quality_factor == 25.0
        assert mode.amplitude == 0.8
        assert mode.phase == 0.3
        assert mode.mode_index == 2
        assert mode.coupling_strength == 0.15

    def test_system_mode_defaults(self):
        """
        Test SystemMode with default values.

        Physical Meaning:
            Tests that SystemMode works correctly with
            default parameter values.
        """
        mode = SystemMode(
            frequency=1.0, quality_factor=10.0, amplitude=1.0, phase=0.0, mode_index=0
        )

        assert mode.frequency == 1.0
        assert mode.quality_factor == 10.0
        assert mode.amplitude == 1.0
        assert mode.phase == 0.0
        assert mode.mode_index == 0
        assert mode.coupling_strength == 0.0


class TestMemoryParameters:
    """
    Test class for MemoryParameters data structure.

    Physical Meaning:
        Tests the MemoryParameters data structure that represents
        memory parameters for quench analysis.
    """

    def test_memory_parameters_creation(self):
        """
        Test MemoryParameters creation.

        Physical Meaning:
            Tests that MemoryParameters objects are created correctly
            with proper parameter assignment.
        """
        spatial_dist = np.array([0.1, 0.2, 0.3])
        memory = MemoryParameters(gamma=0.6, tau=2.0, spatial_distribution=spatial_dist)

        assert memory.gamma == 0.6
        assert memory.tau == 2.0
        assert np.array_equal(memory.spatial_distribution, spatial_dist)

    def test_memory_parameters_defaults(self):
        """
        Test MemoryParameters with default values.

        Physical Meaning:
            Tests that MemoryParameters works correctly with
            default parameter values.
        """
        memory = MemoryParameters(gamma=0.5, tau=1.0)

        assert memory.gamma == 0.5
        assert memory.tau == 1.0
        assert memory.spatial_distribution is None


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


if __name__ == "__main__":
    pytest.main([__file__])
