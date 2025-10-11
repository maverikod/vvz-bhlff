"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for NonlinearEffects initialization and basic functionality.

This module contains tests for the initialization and basic
functionality of the NonlinearEffects class in Level F models.

Physical Meaning:
    Tests verify that nonlinear effects are correctly
    initialized and basic functionality works properly.

Example:
    >>> pytest tests/unit/test_level_f/nonlinear/test_nonlinear_initialization.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.nonlinear import NonlinearEffects
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestNonlinearInitialization:
    """
    Test cases for NonlinearEffects initialization and basic functionality.

    Physical Meaning:
        Tests verify the correct initialization of nonlinear
        effects and basic functionality.
    """

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=20.0, N=16, N_phi=8, N_t=16, T=10.0, dimensions=7)

    @pytest.fixture
    def particles(self):
        """Create test particles."""
        return [
            Particle(position=np.array([5.0, 10.0, 10.0]), charge=1, phase=0.0),
            Particle(position=np.array([15.0, 10.0, 10.0]), charge=-1, phase=np.pi),
        ]

    @pytest.fixture
    def system(self, domain, particles):
        """Create test system."""
        return MultiParticleSystem(domain, particles)

    @pytest.fixture
    def nonlinear_params(self):
        """Create test nonlinear parameters."""
        return {
            "cubic_coefficient": 0.1,
            "quartic_coefficient": 0.01,
            "sine_gordon_amplitude": 1.0,
            "nonlinear_threshold": 0.5,
        }

    def test_initialization(self, system, nonlinear_params):
        """
        Test NonlinearEffects initialization.

        Physical Meaning:
            Tests that NonlinearEffects is correctly initialized
            with proper parameters and system configuration.
        """
        nonlinear = NonlinearEffects(system, nonlinear_params)

        assert nonlinear.system == system
        assert nonlinear.cubic_coefficient == 0.1
        assert nonlinear.quartic_coefficient == 0.01
        assert nonlinear.sine_gordon_amplitude == 1.0
        assert nonlinear.nonlinear_threshold == 0.5
        assert nonlinear.nonlinear_interactions is None
        assert nonlinear.nonlinear_modes is None
        assert nonlinear.soliton_solutions is None

    def test_cubic_nonlinearity_setup(self, system):
        """
        Test cubic nonlinearity setup.

        Physical Meaning:
            Tests that cubic nonlinearity is correctly set up
            for the nonlinear effects system.
        """
        nonlinear_params = {
            "cubic_coefficient": 0.1,
            "quartic_coefficient": 0.0,
            "sine_gordon_amplitude": 0.0,
            "nonlinear_threshold": 0.5,
        }

        nonlinear = NonlinearEffects(system, nonlinear_params)

        # Test cubic nonlinearity setup
        nonlinear.setup_cubic_nonlinearity()

        assert nonlinear.cubic_coefficient == 0.1
        assert nonlinear.quartic_coefficient == 0.0
        assert nonlinear.sine_gordon_amplitude == 0.0

    def test_quartic_nonlinearity_setup(self, system):
        """
        Test quartic nonlinearity setup.

        Physical Meaning:
            Tests that quartic nonlinearity is correctly set up
            for the nonlinear effects system.
        """
        nonlinear_params = {
            "cubic_coefficient": 0.0,
            "quartic_coefficient": 0.01,
            "sine_gordon_amplitude": 0.0,
            "nonlinear_threshold": 0.5,
        }

        nonlinear = NonlinearEffects(system, nonlinear_params)

        # Test quartic nonlinearity setup
        nonlinear.setup_quartic_nonlinearity()

        assert nonlinear.cubic_coefficient == 0.0
        assert nonlinear.quartic_coefficient == 0.01
        assert nonlinear.sine_gordon_amplitude == 0.0

    def test_sine_gordon_nonlinearity_setup(self, system):
        """
        Test sine-Gordon nonlinearity setup.

        Physical Meaning:
            Tests that sine-Gordon nonlinearity is correctly set up
            for the nonlinear effects system.
        """
        nonlinear_params = {
            "cubic_coefficient": 0.0,
            "quartic_coefficient": 0.0,
            "sine_gordon_amplitude": 1.0,
            "nonlinear_threshold": 0.5,
        }

        nonlinear = NonlinearEffects(system, nonlinear_params)

        # Test sine-Gordon nonlinearity setup
        nonlinear.setup_sine_gordon_nonlinearity()

        assert nonlinear.cubic_coefficient == 0.0
        assert nonlinear.quartic_coefficient == 0.0
        assert nonlinear.sine_gordon_amplitude == 1.0

    def test_add_nonlinear_interactions(self, nonlinear):
        """
        Test adding nonlinear interactions.

        Physical Meaning:
            Tests that nonlinear interactions are correctly added
            to the system.
        """
        # Test adding nonlinear interactions
        nonlinear.add_nonlinear_interactions()

        assert nonlinear.nonlinear_interactions is not None
        assert len(nonlinear.nonlinear_interactions) > 0

        # Check interaction properties
        for interaction in nonlinear.nonlinear_interactions:
            assert "type" in interaction
            assert "strength" in interaction
            assert "range" in interaction

    def test_find_nonlinear_modes(self, nonlinear):
        """
        Test finding nonlinear modes.

        Physical Meaning:
            Tests that nonlinear modes are correctly identified
            in the system.
        """
        # Test finding nonlinear modes
        nonlinear.find_nonlinear_modes()

        assert nonlinear.nonlinear_modes is not None
        assert len(nonlinear.nonlinear_modes) > 0

        # Check mode properties
        for mode in nonlinear.nonlinear_modes:
            assert "frequency" in mode
            assert "amplitude" in mode
            assert "phase" in mode
            assert "stability" in mode

    def test_find_soliton_solutions(self, nonlinear):
        """
        Test finding soliton solutions.

        Physical Meaning:
            Tests that soliton solutions are correctly found
            in the nonlinear system.
        """
        # Test finding soliton solutions
        nonlinear.find_soliton_solutions()

        assert nonlinear.soliton_solutions is not None
        assert len(nonlinear.soliton_solutions) > 0

        # Check soliton properties
        for soliton in nonlinear.soliton_solutions:
            assert "position" in soliton
            assert "velocity" in soliton
            assert "amplitude" in soliton
            assert "width" in soliton
            assert "stability" in soliton

    def test_sine_gordon_solitons(self, system):
        """
        Test sine-Gordon solitons.

        Physical Meaning:
            Tests that sine-Gordon solitons are correctly
            implemented in the nonlinear system.
        """
        nonlinear_params = {
            "cubic_coefficient": 0.0,
            "quartic_coefficient": 0.0,
            "sine_gordon_amplitude": 1.0,
            "nonlinear_threshold": 0.5,
        }

        nonlinear = NonlinearEffects(system, nonlinear_params)

        # Test sine-Gordon solitons
        solitons = nonlinear.find_sine_gordon_solitons()

        assert solitons is not None
        assert len(solitons) > 0

        # Check soliton properties
        for soliton in solitons:
            assert "position" in soliton
            assert "velocity" in soliton
            assert "amplitude" in soliton
            assert "width" in soliton

    def test_cubic_solitons(self, system):
        """
        Test cubic solitons.

        Physical Meaning:
            Tests that cubic solitons are correctly
            implemented in the nonlinear system.
        """
        nonlinear_params = {
            "cubic_coefficient": 0.1,
            "quartic_coefficient": 0.0,
            "sine_gordon_amplitude": 0.0,
            "nonlinear_threshold": 0.5,
        }

        nonlinear = NonlinearEffects(system, nonlinear_params)

        # Test cubic solitons
        solitons = nonlinear.find_cubic_solitons()

        assert solitons is not None
        assert len(solitons) > 0

        # Check soliton properties
        for soliton in solitons:
            assert "position" in soliton
            assert "velocity" in soliton
            assert "amplitude" in soliton
            assert "width" in soliton

    def test_quartic_solitons(self, system):
        """
        Test quartic solitons.

        Physical Meaning:
            Tests that quartic solitons are correctly
            implemented in the nonlinear system.
        """
        nonlinear_params = {
            "cubic_coefficient": 0.0,
            "quartic_coefficient": 0.01,
            "sine_gordon_amplitude": 0.0,
            "nonlinear_threshold": 0.5,
        }

        nonlinear = NonlinearEffects(system, nonlinear_params)

        # Test quartic solitons
        solitons = nonlinear.find_quartic_solitons()

        assert solitons is not None
        assert len(solitons) > 0

        # Check soliton properties
        for soliton in solitons:
            assert "position" in soliton
            assert "velocity" in soliton
            assert "amplitude" in soliton
            assert "width" in soliton
