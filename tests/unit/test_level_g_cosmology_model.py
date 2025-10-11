"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for level G cosmology model.

This module tests the cosmological model for 7D phase field theory,
including cosmological evolution, structure formation, and cosmological parameters.

Physical Meaning:
    Tests the cosmological evolution of phase fields in expanding
    universe, including structure formation and cosmological parameters.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_g.cosmology import CosmologicalModel


class TestCosmologicalModel:
    """
    Tests the cosmological model for 7D phase field theory.
    """

    def test_cosmological_model_initialization(self):
        """Test cosmological model initialization."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        assert model.H0 == 70.0
        assert model.omega_m == 0.3
        assert model.omega_lambda == 0.7
        assert model.omega_k == 0.0

    def test_scale_factor_computation(self):
        """Test scale factor computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0, 5.0]
        for z in z_values:
            a = model.scale_factor(z)
            assert a > 0
            assert a <= 1.0  # Scale factor at z >= 0 should be <= 1

    def test_hubble_parameter_computation(self):
        """Test Hubble parameter computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            H = model.hubble_parameter(z)
            assert H > 0
            assert H >= model.H0  # Hubble parameter should increase with redshift

    def test_density_parameter_evolution(self):
        """Test density parameter evolution."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at z=0
        omega_m_z0 = model.density_parameter_matter(0.0)
        omega_lambda_z0 = model.density_parameter_lambda(0.0)
        
        assert abs(omega_m_z0 - 0.3) < 1e-6
        assert abs(omega_lambda_z0 - 0.7) < 1e-6

    def test_age_of_universe(self):
        """Test age of universe computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        age = model.age_of_universe()
        assert age > 0
        assert age < 20.0  # Age should be reasonable in Gyr

    def test_lookback_time(self):
        """Test lookback time computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            t_lookback = model.lookback_time(z)
            assert t_lookback >= 0
            assert t_lookback <= model.age_of_universe()

    def test_comoving_distance(self):
        """Test comoving distance computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            d_comoving = model.comoving_distance(z)
            assert d_comoving >= 0

    def test_luminosity_distance(self):
        """Test luminosity distance computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            d_lum = model.luminosity_distance(z)
            assert d_lum >= 0
            assert d_lum >= model.comoving_distance(z)

    def test_angular_diameter_distance(self):
        """Test angular diameter distance computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            d_ang = model.angular_diameter_distance(z)
            assert d_ang >= 0
            assert d_ang <= model.comoving_distance(z)

    def test_volume_element(self):
        """Test volume element computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            dV = model.volume_element(z)
            assert dV >= 0

    def test_growth_factor(self):
        """Test growth factor computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            D = model.growth_factor(z)
            assert D > 0
            assert D <= 1.0  # Growth factor should be <= 1 at z >= 0

    def test_growth_rate(self):
        """Test growth rate computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            f = model.growth_rate(z)
            assert f > 0
            assert f <= 1.0  # Growth rate should be <= 1

    def test_critical_density(self):
        """Test critical density computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        rho_crit = model.critical_density()
        assert rho_crit > 0

    def test_matter_density(self):
        """Test matter density computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            rho_m = model.matter_density(z)
            assert rho_m > 0

    def test_dark_energy_density(self):
        """Test dark energy density computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            rho_lambda = model.dark_energy_density(z)
            assert rho_lambda > 0

    def test_curvature_density(self):
        """Test curvature density computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            rho_k = model.curvature_density(z)
            assert rho_k == 0.0  # Flat universe

    def test_equation_of_state(self):
        """Test equation of state computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test at different redshifts
        z_values = [0.0, 1.0, 2.0]
        for z in z_values:
            w = model.equation_of_state(z)
            assert isinstance(w, (int, float))

    def test_sound_horizon(self):
        """Test sound horizon computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        r_s = model.sound_horizon()
        assert r_s > 0

    def test_photon_horizon(self):
        """Test photon horizon computation."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        r_photon = model.photon_horizon()
        assert r_photon > 0

    def test_consistency_checks(self):
        """Test cosmological model consistency checks."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Check that omega_m + omega_lambda + omega_k = 1
        total_omega = model.omega_m + model.omega_lambda + model.omega_k
        assert abs(total_omega - 1.0) < 1e-10

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with invalid parameters
        invalid_params = {
            "H0": -70.0,  # Negative Hubble constant
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        with pytest.raises(ValueError):
            CosmologicalModel(invalid_params)

    def test_redshift_limits(self):
        """Test redshift limits."""
        cosmology_params = {
            "H0": 70.0,
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            "omega_k": 0.0,
        }
        
        model = CosmologicalModel(cosmology_params)
        
        # Test with extreme redshifts
        z_extreme = [0.0, 1000.0]
        for z in z_extreme:
            a = model.scale_factor(z)
            assert a > 0
            assert a <= 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
