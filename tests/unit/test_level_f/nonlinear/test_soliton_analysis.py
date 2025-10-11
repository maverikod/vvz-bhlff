"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for soliton analysis functionality.

This module contains tests for soliton analysis functionality
of the NonlinearEffects class in Level F models.

Physical Meaning:
    Tests verify that soliton analysis is correctly
    implemented in the nonlinear system.

Example:
    >>> pytest tests/unit/test_level_f/nonlinear/test_soliton_analysis.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.nonlinear import NonlinearEffects
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestSolitonAnalysis:
    """
    Test cases for soliton analysis functionality.

    Physical Meaning:
        Tests verify the correct implementation of soliton
        analysis including profile computation, corrections,
        and stability analysis.
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

    @pytest.fixture
    def nonlinear(self, system, nonlinear_params):
        """Create test nonlinear effects."""
        return NonlinearEffects(system, nonlinear_params)

    def test_soliton_profile_computation(self, nonlinear):
        """
        Test soliton profile computation.

        Physical Meaning:
            Tests that soliton profiles are correctly computed
            for the nonlinear system.
        """
        # Test soliton profile computation
        profiles = nonlinear.compute_soliton_profiles()

        assert profiles is not None
        assert len(profiles) > 0

        # Check profile properties
        for profile in profiles:
            assert "position" in profile
            assert "amplitude" in profile
            assert "width" in profile
            assert "phase" in profile
            assert "field_data" in profile

            # Check field data
            field_data = profile["field_data"]
            assert isinstance(field_data, np.ndarray)
            assert field_data.shape == (16, 16, 16)

    def test_nonlinear_corrections(self, nonlinear):
        """
        Test nonlinear corrections computation.

        Physical Meaning:
            Tests that nonlinear corrections are correctly
            computed for the soliton solutions.
        """
        # Test nonlinear corrections
        corrections = nonlinear.compute_nonlinear_corrections()

        assert corrections is not None
        assert len(corrections) > 0

        # Check correction properties
        for correction in corrections:
            assert "type" in correction
            assert "magnitude" in correction
            assert "position" in correction
            assert "effect" in correction

    def test_bifurcation_points(self, nonlinear):
        """
        Test bifurcation points analysis.

        Physical Meaning:
            Tests that bifurcation points are correctly
            identified in the nonlinear system.
        """
        # Test bifurcation points
        bifurcations = nonlinear.find_bifurcation_points()

        assert bifurcations is not None
        assert len(bifurcations) >= 0

        # Check bifurcation properties
        for bifurcation in bifurcations:
            assert "parameter_value" in bifurcation
            assert "bifurcation_type" in bifurcation
            assert "stability_change" in bifurcation
            assert "critical_mode" in bifurcation

    def test_nonlinear_stability_analysis(self, nonlinear):
        """
        Test nonlinear stability analysis.

        Physical Meaning:
            Tests that nonlinear stability analysis is correctly
            performed for the system.
        """
        # Test nonlinear stability analysis
        stability = nonlinear.analyze_nonlinear_stability()

        assert stability is not None
        assert "stability_matrix" in stability
        assert "eigenvalues" in stability
        assert "stability_regions" in stability

        # Check stability matrix
        stability_matrix = stability["stability_matrix"]
        assert isinstance(stability_matrix, np.ndarray)
        assert stability_matrix.shape[0] == stability_matrix.shape[1]

        # Check eigenvalues
        eigenvalues = stability["eigenvalues"]
        assert isinstance(eigenvalues, np.ndarray)
        assert len(eigenvalues) > 0

        # Check stability regions
        stability_regions = stability["stability_regions"]
        assert isinstance(stability_regions, list)
        assert len(stability_regions) >= 0

    def test_stability_check(self, nonlinear):
        """
        Test stability check.

        Physical Meaning:
            Tests that stability check is correctly performed
            for the nonlinear system.
        """
        # Test stability check
        is_stable = nonlinear.check_stability()

        assert isinstance(is_stable, bool)

    def test_growth_rates_computation(self, nonlinear):
        """
        Test growth rates computation.

        Physical Meaning:
            Tests that growth rates are correctly computed
            for the nonlinear system.
        """
        # Test growth rates computation
        growth_rates = nonlinear.compute_growth_rates()

        assert growth_rates is not None
        assert len(growth_rates) > 0

        # Check growth rate properties
        for rate in growth_rates:
            assert "mode_index" in rate
            assert "growth_rate" in rate
            assert "frequency" in rate
            assert "stability" in rate

    def test_stability_region_identification(self, nonlinear):
        """
        Test stability region identification.

        Physical Meaning:
            Tests that stability regions are correctly
            identified in the nonlinear system.
        """
        # Test stability region identification
        regions = nonlinear.identify_stability_regions()

        assert regions is not None
        assert len(regions) >= 0

        # Check region properties
        for region in regions:
            assert "parameter_range" in region
            assert "stability_type" in region
            assert "boundary_conditions" in region
            assert "critical_points" in region

    def test_soliton_analysis(self, nonlinear):
        """
        Test comprehensive soliton analysis.

        Physical Meaning:
            Tests that comprehensive soliton analysis is correctly
            performed for the nonlinear system.
        """
        # Test soliton analysis
        analysis = nonlinear.analyze_solitons()

        assert analysis is not None
        assert "soliton_count" in analysis
        assert "soliton_properties" in analysis
        assert "interaction_analysis" in analysis
        assert "stability_analysis" in analysis

        # Check soliton count
        soliton_count = analysis["soliton_count"]
        assert isinstance(soliton_count, int)
        assert soliton_count >= 0

        # Check soliton properties
        soliton_properties = analysis["soliton_properties"]
        assert isinstance(soliton_properties, list)
        assert len(soliton_properties) == soliton_count

        # Check interaction analysis
        interaction_analysis = analysis["interaction_analysis"]
        assert isinstance(interaction_analysis, dict)
        assert "interaction_strength" in interaction_analysis
        assert "interaction_range" in interaction_analysis

        # Check stability analysis
        stability_analysis = analysis["stability_analysis"]
        assert isinstance(stability_analysis, dict)
        assert "overall_stability" in stability_analysis
        assert "stability_metrics" in stability_analysis
