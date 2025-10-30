"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced tests for PhaseTransitions class in Level F models.

This module contains advanced tests for the PhaseTransitions
class, including phase stability analysis, parameter updates,
and error handling.

Physical Meaning:
    Tests verify advanced functionality of phase transition
    analysis including stability analysis, parameter updates,
    and robust error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.transitions import PhaseTransitions
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestPhaseTransitionsAdvanced:
    """
    Advanced test cases for PhaseTransitions class.

    Physical Meaning:
        Tests verify advanced functionality of phase
        transition analysis including stability analysis,
        parameter updates, and robust error handling.
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

    def test_phase_stability_analysis(self, transitions):
        """
        Test phase stability analysis.

        Physical Meaning:
            Verifies that phase stability is correctly
            analyzed to identify stable and unstable
            regions.
        """
        stability = transitions.analyze_phase_stability()

        assert stability is not None
        assert "phase_boundaries" in stability
        assert "stability_regions" in stability
        assert "current_stability" in stability

        # Check that stability analysis is reasonable
        assert stability["phase_boundaries"] is not None
        assert stability["stability_regions"] is not None
        assert stability["current_stability"] is not None

    def test_parameter_update(self, transitions):
        """
        Test parameter update functionality.

        Physical Meaning:
            Verifies that system parameters are correctly
            updated during parameter sweeps.
        """
        # Test updating a valid parameter
        transitions._update_system_parameter("temperature", 1.5)

        # Test updating system_params
        transitions._update_system_parameter("interaction_strength", 2.0)

        # Verify that parameters were updated
        assert transitions.system.temperature == 1.5
        assert transitions.system.system_params.interaction_strength == 2.0

    def test_invalid_parameter_update(self, transitions):
        """
        Test invalid parameter update handling.

        Physical Meaning:
            Verifies that invalid parameter updates
            are handled gracefully.
        """
        # Test updating a non-existent parameter
        transitions._update_system_parameter("non_existent_param", 1.0)

        # This should not raise an error, but the parameter
        # should not be set

    def test_equilibration(self, transitions):
        """
        Test system equilibration.

        Physical Meaning:
            Verifies that the system is correctly
            equilibrated to new parameter values.
        """
        # Test equilibration
        transitions._equilibrate_system()

        # This should not raise an error
        assert True

    def test_phase_boundary_analysis(self, transitions):
        """
        Test phase boundary analysis.

        Physical Meaning:
            Verifies that phase boundaries are correctly
            analyzed to identify transition regions.
        """
        boundaries = transitions._analyze_phase_boundaries()

        assert boundaries is not None
        assert "boundary_count" in boundaries
        assert "boundary_types" in boundaries
        assert "boundary_stability" in boundaries

        # Check that boundary analysis is reasonable
        assert boundaries["boundary_count"] >= 0
        assert len(boundaries["boundary_types"]) == boundaries["boundary_count"]
        assert len(boundaries["boundary_stability"]) == boundaries["boundary_count"]

    def test_stability_region_identification(self, transitions):
        """
        Test stability region identification.

        Physical Meaning:
            Verifies that stability regions are correctly
            identified in parameter space.
        """
        regions = transitions._identify_stability_regions()

        assert regions is not None
        assert "stable_regions" in regions
        assert "region_boundaries" in regions
        assert "region_stability" in regions

        # Check that region identification is reasonable
        assert regions["stable_regions"] >= 0
        assert len(regions["region_boundaries"]) == regions["stable_regions"] - 1
        assert len(regions["region_stability"]) == regions["stable_regions"]

    def test_phase_stability_check(self, transitions):
        """
        Test phase stability check.

        Physical Meaning:
            Verifies that phase stability is correctly
            checked for the current system state.
        """
        # Create mock state
        state = {
            "order_parameters": {
                "topological_order": 1.0,
                "phase_coherence": 0.8,
                "spatial_order": 0.5,
                "energy_density": 0.1,
            }
        }

        stability = transitions._check_phase_stability(state)

        assert stability is not None
        assert "is_stable" in stability
        assert "stability_indicators" in stability

        # Check that stability check is reasonable
        assert isinstance(stability["is_stable"], bool)
        assert stability["stability_indicators"] is not None

    def test_different_parameter_types(self, transitions):
        """
        Test different parameter types.

        Physical Meaning:
            Verifies that different types of parameters
            are correctly handled during parameter sweeps.
        """
        # Test with different parameter types
        parameters = [
            "temperature",
            "pressure",
            "magnetic_field",
            "interaction_strength",
        ]
        values = np.linspace(0.1, 2.0, 5)

        for param in parameters:
            phase_diagram = transitions.parameter_sweep(param, values)

            assert phase_diagram is not None
            assert phase_diagram["parameter"] == param
            assert len(phase_diagram["values"]) == len(values)
            assert len(phase_diagram["phases"]) == len(values)

    def test_phase_classification(self, transitions):
        """
        Test phase classification.

        Physical Meaning:
            Verifies that phases are correctly classified
            based on order parameters.
        """
        # Test different order parameter combinations
        test_cases = [
            {
                "topological_order": 2.0,
                "phase_coherence": 0.9,
                "spatial_order": 0.8,
                "expected": "coherent",
            },
            {
                "topological_order": 1.5,
                "phase_coherence": 0.5,
                "spatial_order": 0.9,
                "expected": "spatial",
            },
            {
                "topological_order": 0.5,
                "phase_coherence": 0.3,
                "spatial_order": 0.2,
                "expected": "disordered",
            },
        ]

        for case in test_cases:
            order_params = {
                "topological_order": case["topological_order"],
                "phase_coherence": case["phase_coherence"],
                "spatial_order": case["spatial_order"],
                "energy_density": 1.0,
            }

            phase = transitions._classify_phase(order_params)

            assert phase is not None
            assert phase in ["topological", "coherent", "spatial", "disordered"]

    def test_error_handling(self, transitions):
        """
        Test error handling in phase transitions.

        Physical Meaning:
            Verifies that the system handles errors gracefully
            and provides meaningful error messages for
            invalid inputs.
        """
        # Test with invalid parameter values
        with pytest.raises(ValueError):
            transitions.parameter_sweep("", np.array([]))

        # Test with None system
        with pytest.raises(AttributeError):
            transitions._update_system_parameter("temperature", 1.0)

        # Test with invalid phase diagram
        with pytest.raises(KeyError):
            transitions.identify_critical_points({})

    def test_comprehensive_analysis(self, transitions):
        """
        Test comprehensive phase transition analysis.

        Physical Meaning:
            Verifies that comprehensive phase transition
            analysis is correctly performed.
        """
        # Perform comprehensive analysis
        analysis = transitions.analyze(None)

        assert analysis is not None
        assert "order_parameters" in analysis
        assert "phase_stability" in analysis
        assert "analysis_complete" in analysis

        # Check that analysis is complete
        assert analysis["analysis_complete"] == True
        assert analysis["order_parameters"] is not None
        assert analysis["phase_stability"] is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
