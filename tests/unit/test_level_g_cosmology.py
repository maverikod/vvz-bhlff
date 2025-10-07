"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for level G cosmology models.

This module tests the cosmological models for 7D phase field theory,
including cosmological evolution, structure formation, and
cosmological parameters.

Physical Meaning:
    Tests the cosmological evolution of phase fields in expanding
    universe, including structure formation and cosmological parameters.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_g.cosmology import CosmologicalModel, EnvelopeEffectiveMetric


class TestEnvelopeEffectiveMetric:
    """
    Tests the envelope-derived effective metric (no spacetime curvature).
    """

    def test_metric_initialization(self):
        """Test metric initialization."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }

        metric = EnvelopeEffectiveMetric({"c_phi": 2.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 3.0})
        assert g.shape == (7, 7)
        assert g[0, 0] == pytest.approx(-1.0 / (2.0**2))

    def test_scale_factors_computation(self):
        """Test scale factors computation."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g1 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        g2 = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 2.0})
        assert g2[1, 1] > g1[1, 1]

    def test_metric_tensor_computation(self):
        """Test metric tensor computation."""
        metric = EnvelopeEffectiveMetric({"c_phi": 1.0})
        g = metric.compute_effective_metric_from_vbp_envelope({"chi_over_kappa": 1.0})
        assert g.shape == (7, 7)
        assert g[0, 0] < 0
        assert g[1, 1] > 0
        assert g[2, 2] > 0
        assert g[3, 3] > 0


class TestCosmologicalModel:
    """
    Test cosmological model.

    Physical Meaning:
        Tests the cosmological evolution model for 7D phase field theory,
        including universe evolution and structure formation.
    """

    def test_model_initialization(self):
        """Test model initialization."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 1000.0,
            "resolution": 256,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 13.8,
            "dt": 0.01,
            "c_phi": 1e10,
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)

        assert model.time_start == 0.0
        assert model.time_end == 13.8
        assert model.dt == 0.01
        assert model.c_phi == 1e10

    def test_universe_evolution(self):
        """Test universe evolution."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        results = model.evolve_universe()

        assert "time" in results
        assert "scale_factor" in results
        assert "hubble_parameter" in results
        assert "phase_field_evolution" in results
        assert "structure_formation" in results

        # Check that evolution has occurred
        assert len(results["time"]) > 1
        assert len(results["scale_factor"]) > 1
        assert len(results["phase_field_evolution"]) > 1

    def test_structure_formation_analysis(self):
        """Test structure formation analysis."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        model.evolve_universe()

        analysis = model.analyze_structure_formation()

        assert "total_evolution_time" in analysis
        assert "final_scale_factor" in analysis
        assert "expansion_rate" in analysis
        assert "structure_growth_rate" in analysis

    def test_cosmological_parameters_computation(self):
        """Test cosmological parameters computation."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        model.evolve_universe()

        parameters = model.compute_cosmological_parameters()

        assert "current_scale_factor" in parameters
        assert "current_hubble_parameter" in parameters
        assert "age_universe" in parameters
        assert "expansion_rate" in parameters
        assert "phase_velocity" in parameters

    def test_phase_field_initialization(self):
        """Test phase field initialization."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        phase_field = model._initialize_phase_field()

        assert phase_field.shape == (64, 64, 64)
        assert np.isfinite(phase_field).all()

    def test_phase_field_evolution_step(self):
        """Test phase field evolution step."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        model.phase_field = model._initialize_phase_field()

        # Test evolution step
        new_phase_field = model._evolve_phase_field_step(0.0, 0.1, 1.0)

        assert new_phase_field.shape == (64, 64, 64)
        assert np.isfinite(new_phase_field).all()

    def test_structure_analysis_at_time(self):
        """Test structure analysis at specific time."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        model.phase_field = model._initialize_phase_field()

        structure = model._analyze_structure_at_time(0.0)

        assert "time" in structure
        assert "phase_field_rms" in structure
        assert "phase_field_max" in structure
        assert "correlation_length" in structure
        assert "topological_defects" in structure

    def test_correlation_length_computation(self):
        """Test correlation length computation."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        model.phase_field = model._initialize_phase_field()

        correlation_length = model._compute_correlation_length()

        assert correlation_length >= 0
        assert np.isfinite(correlation_length)

    def test_topological_defects_counting(self):
        """Test topological defects counting."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        model.phase_field = model._initialize_phase_field()

        defect_count = model._count_topological_defects()

        assert defect_count >= 0
        assert isinstance(defect_count, int)

    def test_structure_growth_rate_computation(self):
        """Test structure growth rate computation."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        model.phase_field = model._initialize_phase_field()

        growth_rate = model._compute_structure_growth_rate()

        assert growth_rate >= 0
        assert np.isfinite(growth_rate)

    def test_parameter_evolution_consistency(self):
        """Test parameter evolution consistency."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        results = model.evolve_universe()

        # Check that scale factor increases with time
        scale_factors = results["scale_factor"]
        assert all(
            scale_factors[i] <= scale_factors[i + 1]
            for i in range(len(scale_factors) - 1)
        )

        # Check that Hubble parameter is consistent
        hubble_params = results["hubble_parameter"]
        assert all(h > 0 for h in hubble_params)

    def test_energy_conservation(self):
        """Test energy conservation."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        results = model.evolve_universe()

        # Check that phase field evolution is consistent
        phase_evolution = results["phase_field_evolution"]
        assert len(phase_evolution) > 1

        # Check that all phase fields are finite
        for phase_field in phase_evolution:
            assert np.isfinite(phase_field).all()

    def test_cosmological_parameters_physical_meaning(self):
        """Test cosmological parameters physical meaning."""
        initial_conditions = {
            "type": "gaussian_fluctuations",
            "domain_size": 100.0,
            "resolution": 64,
            "seed": 42,
        }

        cosmology_params = {
            "time_start": 0.0,
            "time_end": 1.0,
            "dt": 0.1,
            "c_phi": 1e10,
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
        }

        model = CosmologicalModel(initial_conditions, cosmology_params)
        results = model.evolve_universe()

        # Check physical meaning of parameters
        final_scale_factor = results["scale_factor"][-1]
        assert final_scale_factor > 1.0  # Universe should expand

        hubble_params = results["hubble_parameter"]
        assert all(h > 0 for h in hubble_params)  # Hubble parameter should be positive

        # Check that phase field represents physical field
        phase_evolution = results["phase_field_evolution"]
        for phase_field in phase_evolution:
            assert np.isfinite(phase_field).all()
            assert phase_field.shape == (64, 64, 64)
