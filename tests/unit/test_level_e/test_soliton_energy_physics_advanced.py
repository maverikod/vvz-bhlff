"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced physical tests for soliton energy calculations.

This module contains advanced tests for energy calculations in soliton models,
including energy gradient and Hessian properties.

Theoretical Background:
    Tests verify advanced aspects of energy calculations
    including gradient and Hessian properties for optimization.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_energy_physics_advanced.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from bhlff.models.level_e.soliton_models import BaryonSoliton
from bhlff.core.domain.domain import Domain


class TestSolitonEnergyPhysicsAdvanced:
    """
    Advanced physical tests for soliton energy calculations.

    Tests verify advanced aspects of energy calculations
    including gradient and Hessian properties for optimization.
    """

    @pytest.fixture
    def domain_3d(self):
        """Create 7D domain for testing."""
        return Domain(
            L=4.0,  # Larger domain for better resolution
            N=64,  # Higher resolution
            dimensions=7,
        )

    @pytest.fixture
    def physics_params(self):
        """Create realistic physics parameters."""
        return {
            "mu": 1.0,
            "beta": 1.5,
            "lambda_param": 0.1,
            "nu": 1.0,
            "energy_tolerance": 1e-6,
            "kinetic_coefficient": 1.0,
            "skyrme_coefficient": 1.0,
            "wzw_coefficient": 1.0,
        }

    def test_energy_gradient_properties(
        self, domain_3d, physics_params
    ):
        """
        Test energy gradient properties.

        Physical Meaning:
            Verifies that energy gradient calculations
            are physically meaningful and consistent.

        Mathematical Foundation:
            Energy gradient should point in the direction
            of steepest increase of energy.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Calculate energy gradient
        gradient = soliton.compute_energy_gradient(field)
        
        # Verify gradient has correct shape
        assert gradient.shape == field.shape
        
        # Verify gradient is finite
        assert np.all(np.isfinite(gradient))
        
        # Verify gradient is not all zero
        assert np.any(gradient != 0)
        
        # Test gradient consistency with finite differences
        h = 1e-6
        field_plus = field + h * gradient
        field_minus = field - h * gradient
        
        energy_plus = soliton.compute_total_energy(field_plus)
        energy_minus = soliton.compute_total_energy(field_minus)
        energy_center = soliton.compute_total_energy(field)
        
        # Verify gradient points in direction of energy increase
        assert energy_plus > energy_center
        assert energy_minus < energy_center

    def test_energy_hessian_properties(
        self, domain_3d, physics_params
    ):
        """
        Test energy Hessian properties.

        Physical Meaning:
            Verifies that energy Hessian calculations
            are physically meaningful and consistent.

        Mathematical Foundation:
            Energy Hessian should be symmetric and
            positive definite at energy minima.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Calculate energy Hessian
        hessian = soliton.compute_energy_hessian(field)
        
        # Verify Hessian has correct shape
        expected_shape = (field.size, field.size)
        assert hessian.shape == expected_shape
        
        # Verify Hessian is finite
        assert np.all(np.isfinite(hessian))
        
        # Verify Hessian is symmetric
        assert np.allclose(hessian, hessian.T, atol=1e-10)
        
        # Verify Hessian is positive definite (eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(hessian)
        assert np.all(eigenvalues > -1e-10)  # Allow small numerical errors

    def test_energy_optimization_properties(
        self, domain_3d, physics_params
    ):
        """
        Test energy optimization properties.

        Physical Meaning:
            Verifies that energy optimization
            produces physically reasonable results.

        Mathematical Foundation:
            Energy optimization should converge
            to stable configurations.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create initial field
        field_initial = soliton.create_b1_configuration()
        
        # Perform energy optimization
        field_optimized = soliton.optimize_energy(field_initial)
        
        # Calculate energies
        energy_initial = soliton.compute_total_energy(field_initial)
        energy_optimized = soliton.compute_total_energy(field_optimized)
        
        # Verify optimization reduces energy
        assert energy_optimized <= energy_initial
        
        # Verify optimized field is finite
        assert np.all(np.isfinite(field_optimized))

    def test_energy_stability_analysis(
        self, domain_3d, physics_params
    ):
        """
        Test energy stability analysis.

        Physical Meaning:
            Verifies that energy stability analysis
            correctly identifies stable configurations.

        Mathematical Foundation:
            Stable configurations should have
            positive definite Hessian matrices.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Perform stability analysis
        is_stable = soliton.is_energy_stable(field)
        
        # Verify stability analysis returns boolean
        assert isinstance(is_stable, bool)
        
        # For B=1 configuration, should be stable
        assert is_stable

    def test_energy_perturbation_response(
        self, domain_3d, physics_params
    ):
        """
        Test energy response to perturbations.

        Physical Meaning:
            Verifies that energy calculations
            respond correctly to field perturbations.

        Mathematical Foundation:
            Energy should change smoothly with
            field perturbations.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Calculate initial energy
        energy_initial = soliton.compute_total_energy(field)
        
        # Add small perturbation
        perturbation = 0.01 * np.random.random(field.shape) + 1j * 0.01 * np.random.random(field.shape)
        field_perturbed = field + perturbation
        
        # Calculate perturbed energy
        energy_perturbed = soliton.compute_total_energy(field_perturbed)
        
        # Verify energy change is reasonable
        energy_change = abs(energy_perturbed - energy_initial)
        assert energy_change < 1.0  # Should be small for small perturbation

    def test_energy_convergence_properties(
        self, physics_params
    ):
        """
        Test energy convergence properties.

        Physical Meaning:
            Verifies that energy calculations
            converge with increasing resolution.

        Mathematical Foundation:
            Energy should converge to the
            correct value as resolution increases.
        """
        # Test different grid resolutions
        grid_sizes = [16, 32, 64, 128]
        energies = []
        
        for N in grid_sizes:
            domain = Domain(L=4.0, N=N, dimensions=7)
            soliton = BaryonSoliton(domain, physics_params)
            field = soliton.create_b1_configuration()
            energy = soliton.compute_total_energy(field)
            energies.append(energy)
        
        # Verify energy convergence
        for i in range(1, len(energies)):
            # Energy should become more stable with higher resolution
            assert abs(energies[i] - energies[i-1]) < abs(energies[i-1] - energies[i-2]) if i > 1 else True

    def test_energy_boundary_conditions(
        self, domain_3d, physics_params
    ):
        """
        Test energy calculation with boundary conditions.

        Physical Meaning:
            Verifies that energy calculations
            handle boundary conditions correctly.

        Mathematical Foundation:
            Energy should be calculated correctly
            even with complex boundary conditions.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field with boundary conditions
        field = soliton.create_field_with_boundary_conditions()
        
        # Calculate energy
        energy = soliton.compute_total_energy(field)
        
        # Verify energy is finite
        assert np.isfinite(energy)
        
        # Verify energy is positive
        assert energy > 0

    def test_energy_parameter_dependence(
        self, domain_3d
    ):
        """
        Test energy dependence on parameters.

        Physical Meaning:
            Verifies that energy calculations
            depend correctly on physical parameters.

        Mathematical Foundation:
            Energy should scale appropriately
            with physical parameters.
        """
        # Test different parameter sets
        param_sets = [
            {"mu": 1.0, "beta": 1.5, "lambda_param": 0.1, "nu": 1.0},
            {"mu": 2.0, "beta": 1.5, "lambda_param": 0.1, "nu": 1.0},
            {"mu": 1.0, "beta": 2.0, "lambda_param": 0.1, "nu": 1.0},
        ]
        
        energies = []
        for params in param_sets:
            soliton = BaryonSoliton(domain_3d, params)
            field = soliton.create_b1_configuration()
            energy = soliton.compute_total_energy(field)
            energies.append(energy)
        
        # Verify energy changes with parameters
        assert not np.allclose(energies, energies[0])

    def test_energy_spectral_properties(
        self, domain_3d, physics_params
    ):
        """
        Test energy spectral properties.

        Physical Meaning:
            Verifies that energy calculations
            have correct spectral properties.

        Mathematical Foundation:
            Energy should have appropriate
            spectral characteristics.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Calculate energy spectrum
        energy_spectrum = soliton.compute_energy_spectrum(field)
        
        # Verify spectrum has correct shape
        assert energy_spectrum.shape == field.shape
        
        # Verify spectrum is finite
        assert np.all(np.isfinite(energy_spectrum))
        
        # Verify spectrum is positive
        assert np.all(energy_spectrum >= 0)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
