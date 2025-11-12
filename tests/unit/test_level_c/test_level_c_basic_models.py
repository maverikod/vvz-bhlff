"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic unit tests for Level C models.

This module contains simple, working unit tests for the Level C basic models,
focusing on ABCD model, resonator layers, system modes, and memory parameters.

Physical Meaning:
    Tests the basic Level C model capabilities:
    - ABCD model initialization and functionality
    - Resonator layer data structures
    - System mode detection and properties
    - Memory parameter handling

Mathematical Foundation:
    Tests basic mathematical operations:
    - Matrix operations for ABCD model
    - Basic field operations for boundary analysis
    - Simple memory kernel operations

Example:
    >>> pytest tests/unit/test_level_c/test_level_c_basic_models.py
"""

import pytest
import numpy as np
from unittest.mock import Mock

from bhlff.models.level_c import (
    ABCDModel,
    ResonatorLayer,
    SystemMode,
    MemoryParameters,
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
        # spatial_distribution may have a default value, check if it exists
        if hasattr(memory, 'spatial_distribution'):
            # If it exists, just verify it's a valid value (None or array)
            assert memory.spatial_distribution is None or isinstance(
                memory.spatial_distribution, (np.ndarray, type(None))
            )
