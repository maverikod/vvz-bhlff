"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic tests for PhaseTransitions class in Level F models.

This module contains basic tests for the PhaseTransitions
class, including initialization, parameter sweeps, and
order parameter calculations.

Physical Meaning:
    Tests verify that phase transitions are correctly
    identified and analyzed in multi-particle systems,
    including basic functionality and order parameters.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.transitions import PhaseTransitions
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestPhaseTransitionsBasic:
    """
    Basic test cases for PhaseTransitions class.

    Physical Meaning:
        Tests verify the correct implementation of basic
        phase transition analysis including parameter sweeps,
        order parameters, and critical point identification.
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
    def transitions(self, system):
        """Create PhaseTransitions instance."""
        return PhaseTransitions(system)

    def test_initialization(self, system):
        """
        Test PhaseTransitions initialization.

        Physical Meaning:
            Verifies that the PhaseTransitions class is
            correctly initialized with proper parameters.
        """
        transitions = PhaseTransitions(system)
        
        assert transitions.system == system
        assert transitions.order_parameters == {}
        assert transitions.critical_points == []
        assert transitions.phase_diagram == {}

    def test_parameter_sweep(self, transitions):
        """
        Test parameter sweep functionality.

        Physical Meaning:
            Verifies that parameter sweeps are correctly
            performed to identify phase transitions and
            critical points.
        """
        parameter = "temperature"
        values = np.linspace(0.1, 2.0, 10)
        
        phase_diagram = transitions.parameter_sweep(parameter, values)
        
        assert phase_diagram is not None
        assert "parameter" in phase_diagram
        assert "values" in phase_diagram
        assert "order_parameters" in phase_diagram
        assert "phases" in phase_diagram
        assert "critical_points" in phase_diagram
        
        assert phase_diagram["parameter"] == parameter
        assert len(phase_diagram["values"]) == len(values)
        assert len(phase_diagram["phases"]) == len(values)

    def test_order_parameters_computation(self, transitions):
        """
        Test order parameters computation.

        Physical Meaning:
            Verifies that order parameters are correctly
            computed for the current system state.
        """
        order_params = transitions.compute_order_parameters()
        
        assert order_params is not None
        assert "topological_order" in order_params
        assert "phase_coherence" in order_params
        assert "spatial_order" in order_params
        assert "energy_density" in order_params
        
        # Check that order parameters are reasonable
        assert order_params["topological_order"] >= 0
        assert 0 <= order_params["phase_coherence"] <= 1
        assert order_params["spatial_order"] >= 0
        assert order_params["energy_density"] >= 0

    def test_critical_points_identification(self, transitions):
        """
        Test critical points identification.

        Physical Meaning:
            Verifies that critical points are correctly
            identified in the phase diagram.
        """
        # Create mock phase diagram
        phase_diagram = {
            "parameter": "temperature",
            "values": np.linspace(0.1, 2.0, 10),
            "order_parameters": {
                str(v): {
                    "topological_order": v,
                    "phase_coherence": 1.0 - v/2.0,
                    "spatial_order": v/2.0,
                    "energy_density": v**2
                } for v in np.linspace(0.1, 2.0, 10)
            },
            "phases": [{"parameter_value": v, "phase": "test", "stability": True} for v in np.linspace(0.1, 2.0, 10)]
        }
        
        critical_points = transitions.identify_critical_points(phase_diagram)
        
        assert critical_points is not None
        assert isinstance(critical_points, list)
        
        # Check that critical points have required fields
        for point in critical_points:
            assert "parameter_value" in point
            assert "transition_type" in point
            assert "order_parameter" in point
            assert "discontinuity_magnitude" in point

    def test_topological_order_computation(self, transitions):
        """
        Test topological order computation.

        Physical Meaning:
            Verifies that topological order is correctly
            computed as the total topological charge.
        """
        topological_order = transitions._compute_topological_order()
        
        assert topological_order is not None
        assert topological_order >= 0
        
        # For our test system with charges +1 and -1
        expected_order = 2.0  # |+1| + |-1| = 2
        assert abs(topological_order - expected_order) < 1e-10

    def test_phase_coherence_computation(self, transitions):
        """
        Test phase coherence computation.

        Physical Meaning:
            Verifies that phase coherence is correctly
            computed as a measure of phase synchronization.
        """
        phase_coherence = transitions._compute_phase_coherence()
        
        assert phase_coherence is not None
        assert 0 <= phase_coherence <= 1
        
        # For our test system with phases 0 and π
        expected_coherence = 0.0  # |e^(i0) + e^(iπ)|/2 = |1 - 1|/2 = 0
        assert abs(phase_coherence - expected_coherence) < 1e-10

    def test_spatial_order_computation(self, transitions):
        """
        Test spatial order computation.

        Physical Meaning:
            Verifies that spatial order is correctly
            computed as a measure of spatial organization.
        """
        spatial_order = transitions._compute_spatial_order()
        
        assert spatial_order is not None
        assert spatial_order >= 0

    def test_energy_density_computation(self, transitions):
        """
        Test energy density computation.

        Physical Meaning:
            Verifies that energy density is correctly
            computed as the system's energy per unit volume.
        """
        energy_density = transitions._compute_energy_density()
        
        assert energy_density is not None
        assert energy_density >= 0

    def test_system_state_analysis(self, transitions):
        """
        Test system state analysis.

        Physical Meaning:
            Verifies that system state is correctly
            analyzed to determine phase and stability.
        """
        state = transitions._analyze_system_state()
        
        assert state is not None
        assert "order_parameters" in state
        assert "phase" in state
        assert "stability" in state
        
        # Check that state analysis is reasonable
        assert state["order_parameters"] is not None
        assert state["phase"] is not None
        assert state["stability"] is not None

    def test_discontinuity_detection(self, transitions):
        """
        Test discontinuity detection.

        Physical Meaning:
            Verifies that discontinuities in order
            parameters are correctly detected.
        """
        # Create mock phase diagram
        phase_diagram = {
            "values": np.linspace(0.1, 2.0, 10),
            "order_parameters": {
                str(v): {
                    "topological_order": v,
                    "phase_coherence": 1.0 - v/2.0,
                    "spatial_order": v/2.0,
                    "energy_density": v**2
                } for v in np.linspace(0.1, 2.0, 10)
            }
        }
        
        discontinuities = transitions._find_discontinuities(phase_diagram)
        
        assert discontinuities is not None
        assert isinstance(discontinuities, list)
        
        # Check that discontinuities have required fields
        for discontinuity in discontinuities:
            assert "parameter" in discontinuity
            assert "value" in discontinuity
            assert "discontinuity" in discontinuity

    def test_critical_point_detection(self, transitions):
        """
        Test critical point detection.

        Physical Meaning:
            Verifies that critical points are correctly
            detected from discontinuities.
        """
        # Create mock phase diagram and discontinuities
        phase_diagram = {
            "values": np.linspace(0.1, 2.0, 10),
            "order_parameters": {
                str(v): {
                    "topological_order": v,
                    "phase_coherence": 1.0 - v/2.0,
                    "spatial_order": v/2.0,
                    "energy_density": v**2
                } for v in np.linspace(0.1, 2.0, 10)
            }
        }
        
        discontinuities = [
            {
                "parameter": "topological_order",
                "value": 1.0,
                "discontinuity": 0.5
            }
        ]
        
        critical_points = transitions._find_critical_points(phase_diagram, discontinuities)
        
        assert critical_points is not None
        assert isinstance(critical_points, list)
        assert len(critical_points) == len(discontinuities)
        
        # Check that critical points have required fields
        for point in critical_points:
            assert "parameter_value" in point
            assert "transition_type" in point
            assert "order_parameter" in point
            assert "discontinuity_magnitude" in point

    def test_critical_exponents_computation(self, transitions):
        """
        Test critical exponents computation.

        Physical Meaning:
            Verifies that critical exponents are correctly
            computed near critical points.
        """
        # Create mock phase diagram and critical point
        phase_diagram = {
            "values": np.linspace(0.1, 2.0, 10),
            "order_parameters": {
                str(v): {
                    "topological_order": v,
                    "phase_coherence": 1.0 - v/2.0,
                    "spatial_order": v/2.0,
                    "energy_density": v**2
                } for v in np.linspace(0.1, 2.0, 10)
            }
        }
        
        critical_point = {
            "parameter_value": 1.0,
            "transition_type": "first_order",
            "order_parameter": "topological_order",
            "discontinuity_magnitude": 0.5
        }
        
        exponents = transitions._compute_critical_exponents(phase_diagram, critical_point)
        
        assert exponents is not None
        assert "beta" in exponents
        assert "gamma" in exponents
        assert "delta" in exponents
        assert "nu" in exponents
        
        # Check that exponents are reasonable
        assert exponents["beta"] > 0
        assert exponents["gamma"] > 0
        assert exponents["delta"] > 0
        assert exponents["nu"] > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
