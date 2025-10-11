"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic physical tests for soliton energy calculations.

This module contains basic tests for energy calculations in soliton models,
verifying the physical correctness of kinetic, Skyrme, and WZW energy terms.

Theoretical Background:
    Tests verify that energy calculations follow the correct physical
    formulas and produce reasonable results for known field configurations.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_energy_physics_basic.py -v
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


class TestSolitonEnergyPhysicsBasic:
    """
    Basic physical tests for soliton energy calculations.

    Tests verify the basic physical correctness of energy calculations
    including kinetic, Skyrme, and WZW energy terms.
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

    def test_kinetic_energy_physical_properties(
        self, domain_3d, physics_params
    ):
        """
        Test kinetic energy physical properties.

        Physical Meaning:
            Verifies that kinetic energy calculations
            follow the correct physical formulas.

        Mathematical Foundation:
            Kinetic energy should be proportional to
            the square of field derivatives.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Calculate kinetic energy
        kinetic_energy = soliton.compute_kinetic_energy(field)
        
        # Verify energy is positive
        assert kinetic_energy > 0
        
        # Verify energy is finite
        assert np.isfinite(kinetic_energy)
        
        # Verify energy scales with field amplitude
        field_scaled = 2.0 * field
        kinetic_energy_scaled = soliton.compute_kinetic_energy(field_scaled)
        assert kinetic_energy_scaled > kinetic_energy

    def test_skyrme_energy_physical_properties(
        self, domain_3d, physics_params
    ):
        """
        Test Skyrme energy physical properties.

        Physical Meaning:
            Verifies that Skyrme energy calculations
            follow the correct physical formulas.

        Mathematical Foundation:
            Skyrme energy should be proportional to
            the fourth power of field derivatives.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Calculate Skyrme energy
        skyrme_energy = soliton.compute_skyrme_energy(field)
        
        # Verify energy is positive
        assert skyrme_energy > 0
        
        # Verify energy is finite
        assert np.isfinite(skyrme_energy)
        
        # Verify energy scales with field amplitude
        field_scaled = 2.0 * field
        skyrme_energy_scaled = soliton.compute_skyrme_energy(field_scaled)
        assert skyrme_energy_scaled > skyrme_energy

    def test_wzw_energy_physical_properties(
        self, domain_3d, physics_params
    ):
        """
        Test WZW energy physical properties.

        Physical Meaning:
            Verifies that WZW energy calculations
            follow the correct physical formulas.

        Mathematical Foundation:
            WZW energy should be proportional to
            the topological charge and field configuration.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Calculate WZW energy
        wzw_energy = soliton.compute_wzw_energy(field)
        
        # Verify energy is finite
        assert np.isfinite(wzw_energy)
        
        # Verify energy is related to topological charge
        topological_charge = soliton.compute_topological_charge(field)
        assert abs(wzw_energy) > 0 or abs(topological_charge) < 1e-10

    def test_energy_scaling_with_domain_size(self, physics_params):
        """
        Test energy scaling with domain size.

        Physical Meaning:
            Verifies that energy calculations
            scale correctly with domain size.

        Mathematical Foundation:
            Energy should scale appropriately
            with domain size for the same configuration.
        """
        # Test different domain sizes
        domain_sizes = [2.0, 4.0, 8.0]
        energies = []
        
        for L in domain_sizes:
            domain = Domain(L=L, N=32, dimensions=7)
            soliton = BaryonSoliton(domain, physics_params)
            field = soliton.create_b1_configuration()
            energy = soliton.compute_total_energy(field)
            energies.append(energy)
        
        # Verify energy scaling (should be approximately proportional to domain volume)
        for i in range(1, len(energies)):
            volume_ratio = (domain_sizes[i] / domain_sizes[0]) ** 7
            energy_ratio = energies[i] / energies[0]
            assert abs(energy_ratio - volume_ratio) < 0.5  # Allow some tolerance

    def test_energy_conservation_under_rotation(
        self, domain_3d, physics_params
    ):
        """
        Test energy conservation under rotation.

        Physical Meaning:
            Verifies that energy is conserved
            under spatial rotations.

        Mathematical Foundation:
            Energy should be invariant under
            continuous transformations like rotations.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create initial field
        field_initial = soliton.create_b1_configuration()
        
        # Calculate initial energy
        energy_initial = soliton.compute_total_energy(field_initial)
        
        # Apply rotation (simulate by phase shift)
        field_rotated = np.roll(field_initial, shift=1, axis=0)
        
        # Calculate rotated energy
        energy_rotated = soliton.compute_total_energy(field_rotated)
        
        # Verify energy conservation
        assert abs(energy_rotated - energy_initial) < physics_params["energy_tolerance"]

    def test_energy_minimum_properties(self, domain_3d, physics_params):
        """
        Test energy minimum properties.

        Physical Meaning:
            Verifies that energy calculations
            produce reasonable minimum values.

        Mathematical Foundation:
            Energy should have a well-defined
            minimum for stable configurations.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create field configuration
        field = soliton.create_b1_configuration()
        
        # Calculate total energy
        total_energy = soliton.compute_total_energy(field)
        
        # Verify energy is positive
        assert total_energy > 0
        
        # Verify energy is finite
        assert np.isfinite(total_energy)
        
        # Verify energy is reasonable (not too large)
        assert total_energy < 1e6  # Reasonable upper bound


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
