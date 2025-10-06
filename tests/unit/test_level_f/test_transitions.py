"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for PhaseTransitions class in Level F models.

This module contains comprehensive tests for the PhaseTransitions
class, including tests for parameter sweeps, order parameter
calculations, and critical point identification.

Physical Meaning:
    Tests verify that phase transitions are correctly
    identified and analyzed in multi-particle systems,
    including critical points and order parameters.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.transitions import PhaseTransitions
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestPhaseTransitions:
    """
    Test cases for PhaseTransitions class.

    Physical Meaning:
        Tests verify the correct implementation of phase
        transition analysis including parameter sweeps,
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
        return MultiParticleSystem(domain, particles, interaction_range=5.0)

    @pytest.fixture
    def transitions(self, system):
        """Create test transitions."""
        return PhaseTransitions(system)

    def test_initialization(self, system):
        """
        Test transitions initialization.

        Physical Meaning:
            Verifies that the phase transitions model is
            correctly initialized with the system.
        """
        transitions = PhaseTransitions(system)

        assert transitions.system == system
        assert transitions.order_parameters == {}
        assert transitions.critical_points == []

    def test_parameter_sweep(self, transitions):
        """
        Test parameter sweep.

        Physical Meaning:
            Verifies that parameter sweeps are correctly
            performed to study phase transitions.
        """
        parameter = "interaction_strength"
        values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

        result = transitions.parameter_sweep(parameter, values)

        # Check that result is returned
        assert "parameter_values" in result
        assert "phase_diagram" in result
        assert "critical_points" in result

        # Check that parameter values are returned
        assert np.array_equal(result["parameter_values"], values)

        # Check that phase diagram is created
        assert len(result["phase_diagram"]) == len(values)

        # Check that each point in phase diagram has required fields
        for point in result["phase_diagram"]:
            assert "parameter_value" in point
            assert "state" in point
            assert "order_parameters" in point

    def test_order_parameters_computation(self, transitions):
        """
        Test order parameters computation.

        Physical Meaning:
            Verifies that order parameters are correctly
            computed for the system.
        """
        order_params = transitions.compute_order_parameters()

        # Check that order parameters are returned
        assert "topological_order" in order_params
        assert "phase_coherence" in order_params
        assert "spatial_order" in order_params
        assert "energy_density" in order_params

        # Check that order parameters are finite
        for key, value in order_params.items():
            assert np.isfinite(value)

        # Check that order parameters are stored
        assert transitions.order_parameters == order_params

    def test_critical_points_identification(self, transitions):
        """
        Test critical points identification.

        Physical Meaning:
            Verifies that critical points are correctly
            identified in the phase diagram.
        """
        # Create mock phase diagram with critical point
        phase_diagram = [
            {
                "parameter_value": 0.1,
                "order_parameters": {
                    "topological_order": 2.0,
                    "phase_coherence": 0.8,
                    "spatial_order": 0.6,
                },
            },
            {
                "parameter_value": 0.5,
                "order_parameters": {
                    "topological_order": 1.8,
                    "phase_coherence": 0.7,
                    "spatial_order": 0.5,
                },
            },
            {
                "parameter_value": 1.0,
                "order_parameters": {
                    "topological_order": 1.5,
                    "phase_coherence": 0.6,
                    "spatial_order": 0.4,
                },
            },
            {
                "parameter_value": 1.5,
                "order_parameters": {
                    "topological_order": 1.2,
                    "phase_coherence": 0.5,
                    "spatial_order": 0.3,
                },
            },
            {
                "parameter_value": 2.0,
                "order_parameters": {
                    "topological_order": 0.8,
                    "phase_coherence": 0.4,
                    "spatial_order": 0.2,
                },
            },
        ]

        critical_points = transitions.identify_critical_points(phase_diagram)

        # Check that critical points are returned
        assert isinstance(critical_points, list)

        # Check that critical points are stored
        assert transitions.critical_points == critical_points

    def test_topological_order_computation(self, transitions):
        """
        Test topological order parameter computation.

        Physical Meaning:
            Verifies that the topological order parameter
            is correctly computed.
        """
        topological_order = transitions._compute_topological_order()

        # Check that topological order is returned
        assert isinstance(topological_order, (int, float, np.integer, np.floating))
        assert topological_order >= 0  # Should be non-negative
        assert np.isfinite(topological_order)

        # Check that it matches expected value
        expected_order = sum(abs(p.charge) for p in transitions.system.particles)
        assert topological_order == expected_order

    def test_phase_coherence_computation(self, transitions):
        """
        Test phase coherence computation.

        Physical Meaning:
            Verifies that the phase coherence order
            parameter is correctly computed.
        """
        phase_coherence = transitions._compute_phase_coherence()

        # Check that phase coherence is returned
        assert isinstance(phase_coherence, (int, float))
        assert 0 <= phase_coherence <= 1  # Should be between 0 and 1
        assert np.isfinite(phase_coherence)

    def test_spatial_order_computation(self, transitions):
        """
        Test spatial order computation.

        Physical Meaning:
            Verifies that the spatial order parameter
            is correctly computed.
        """
        spatial_order = transitions._compute_spatial_order()

        # Check that spatial order is returned
        assert isinstance(spatial_order, (int, float))
        assert spatial_order >= 0  # Should be non-negative
        assert np.isfinite(spatial_order)

    def test_energy_density_computation(self, transitions):
        """
        Test energy density computation.

        Physical Meaning:
            Verifies that the energy density order
            parameter is correctly computed.
        """
        energy_density = transitions._compute_energy_density()

        # Check that energy density is returned
        assert isinstance(energy_density, (int, float))
        assert np.isfinite(energy_density)

    def test_system_state_analysis(self, transitions):
        """
        Test system state analysis.

        Physical Meaning:
            Verifies that the system state is correctly
            analyzed.
        """
        state = transitions._analyze_system_state()

        # Check that state analysis is returned
        assert "total_charge" in state
        assert "charge_dispersion" in state
        assert "phase_dispersion" in state
        assert "mean_distance" in state
        assert "distance_dispersion" in state
        assert "n_particles" in state

        # Check that values are finite
        for key, value in state.items():
            if key != "n_particles":
                assert np.isfinite(value)

        # Check that n_particles is correct
        assert state["n_particles"] == len(transitions.system.particles)

    def test_discontinuity_detection(self, transitions):
        """
        Test discontinuity detection.

        Physical Meaning:
            Verifies that discontinuities are correctly
            detected in order parameters.
        """
        param_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        order_values = [2.0, 1.8, 1.5, 0.8, 0.5]  # Large jump at 1.5

        discontinuities = transitions._find_discontinuities(param_values, order_values)

        # Check that discontinuities are returned
        assert isinstance(discontinuities, list)

        # Check that discontinuity is detected
        assert len(discontinuities) > 0
        assert 1.5 in discontinuities

    def test_critical_point_detection(self, transitions):
        """
        Test critical point detection.

        Physical Meaning:
            Verifies that critical points are correctly
            detected in order parameters.
        """
        param_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        order_values = [2.0, 1.8, 1.5, 1.2, 0.8]  # Smooth transition

        critical_points = transitions._find_critical_points(param_values, order_values)

        # Check that critical points are returned
        assert isinstance(critical_points, list)

    def test_critical_exponents_computation(self, transitions):
        """
        Test critical exponents computation.

        Physical Meaning:
            Verifies that critical exponents are correctly
            computed for phase transitions.
        """
        param_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        order_values = [2.0, 1.8, 1.5, 1.2, 0.8]
        critical_point = 1.0

        exponents = transitions._compute_critical_exponents(
            param_values, order_values, critical_point
        )

        # Check that critical exponents are returned
        assert "alpha" in exponents
        assert "beta" in exponents
        assert "gamma" in exponents
        assert "delta" in exponents

        # Check that exponents are finite
        for key, value in exponents.items():
            assert np.isfinite(value)

    def test_phase_stability_analysis(self, transitions):
        """
        Test phase stability analysis.

        Physical Meaning:
            Verifies that phase stability is correctly
            analyzed.
        """
        stability = transitions.analyze_phase_stability()

        # Check that stability analysis is returned
        assert "current_stability" in stability
        assert "phase_boundaries" in stability
        assert "stability_regions" in stability

        # Check that current stability is returned
        current_stability = stability["current_stability"]
        assert "is_stable" in current_stability
        assert "stability_margin" in current_stability
        assert "phase_stability" in current_stability

    def test_parameter_update(self, transitions):
        """
        Test parameter update.

        Physical Meaning:
            Verifies that system parameters are correctly
            updated.
        """
        # Test temperature update
        transitions._update_system_parameter("temperature", 1.0)
        assert hasattr(transitions.system, "temperature")
        assert transitions.system.temperature == 1.0

        # Test interaction strength update
        transitions._update_system_parameter("interaction_strength", 2.0)
        assert transitions.system.interaction_strength == 2.0

        # Test interaction range update
        transitions._update_system_parameter("interaction_range", 8.0)
        assert transitions.system.interaction_range == 8.0

    def test_invalid_parameter_update(self, transitions):
        """
        Test invalid parameter update.

        Physical Meaning:
            Verifies that invalid parameters are handled
            correctly.
        """
        with pytest.raises(ValueError):
            transitions._update_system_parameter("invalid_parameter", 1.0)

    def test_equilibration(self, transitions):
        """
        Test system equilibration.

        Physical Meaning:
            Verifies that the system is correctly
            equilibrated.
        """
        # This is a placeholder test since equilibration
        # is not fully implemented
        transitions._equilibrate_system()

        # Check that no errors are raised
        assert True

    def test_phase_boundary_analysis(self, transitions):
        """
        Test phase boundary analysis.

        Physical Meaning:
            Verifies that phase boundaries are correctly
            analyzed.
        """
        boundaries = transitions._analyze_phase_boundaries()

        # Check that boundaries are returned
        assert "boundaries" in boundaries
        assert "metastable_regions" in boundaries
        assert "coexistence_regions" in boundaries

    def test_stability_region_identification(self, transitions):
        """
        Test stability region identification.

        Physical Meaning:
            Verifies that stability regions are correctly
            identified.
        """
        regions = transitions._identify_stability_regions()

        # Check that regions are returned
        assert "stable_regions" in regions
        assert "unstable_regions" in regions
        assert "metastable_regions" in regions

    def test_phase_stability_check(self, transitions):
        """
        Test phase stability check.

        Physical Meaning:
            Verifies that phase stability is correctly
            checked.
        """
        state = {"total_charge": 0, "charge_dispersion": 0.1}

        stability = transitions._check_phase_stability(state)

        # Check that stability is returned
        assert "is_stable" in stability
        assert "stability_margin" in stability
        assert "phase_stability" in stability

    def test_different_parameter_types(self, transitions):
        """
        Test different parameter types.

        Physical Meaning:
            Verifies that different parameter types
            work correctly.
        """
        parameters = ["temperature", "interaction_strength", "interaction_range"]

        for param in parameters:
            values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

            result = transitions.parameter_sweep(param, values)

            # Check that result is returned
            assert "parameter_values" in result
            assert "phase_diagram" in result
            assert "critical_points" in result

    def test_order_parameter_consistency(self, transitions):
        """
        Test order parameter consistency.

        Physical Meaning:
            Verifies that order parameters are consistent
            across multiple computations.
        """
        # Compute order parameters multiple times
        order_params_1 = transitions.compute_order_parameters()
        order_params_2 = transitions.compute_order_parameters()

        # Check that results are consistent
        for key in order_params_1:
            assert key in order_params_2
            assert np.isclose(order_params_1[key], order_params_2[key], rtol=1e-10)

    def test_critical_point_properties(self, transitions):
        """
        Test critical point properties.

        Physical Meaning:
            Verifies that critical points have the
            expected properties.
        """
        # Create mock phase diagram
        phase_diagram = [
            {
                "parameter_value": 0.1,
                "order_parameters": {
                    "topological_order": 2.0,
                    "phase_coherence": 0.8,
                    "spatial_order": 0.6,
                },
            },
            {
                "parameter_value": 0.5,
                "order_parameters": {
                    "topological_order": 1.8,
                    "phase_coherence": 0.7,
                    "spatial_order": 0.5,
                },
            },
            {
                "parameter_value": 1.0,
                "order_parameters": {
                    "topological_order": 1.5,
                    "phase_coherence": 0.6,
                    "spatial_order": 0.4,
                },
            },
            {
                "parameter_value": 1.5,
                "order_parameters": {
                    "topological_order": 1.2,
                    "phase_coherence": 0.5,
                    "spatial_order": 0.3,
                },
            },
            {
                "parameter_value": 2.0,
                "order_parameters": {
                    "topological_order": 0.8,
                    "phase_coherence": 0.4,
                    "spatial_order": 0.2,
                },
            },
        ]

        critical_points = transitions.identify_critical_points(phase_diagram)

        # Check that critical points have required properties
        for point in critical_points:
            assert "parameter_value" in point
            assert "transition_type" in point
            assert "order_parameter" in point
            assert "critical_exponents" in point

            # Check that critical exponents are returned
            exponents = point["critical_exponents"]
            assert "alpha" in exponents
            assert "beta" in exponents
            assert "gamma" in exponents
            assert "delta" in exponents
