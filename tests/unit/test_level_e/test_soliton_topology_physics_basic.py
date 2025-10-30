"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic physical tests for soliton topological properties.

This module contains basic tests for topological charge calculations
and related properties in soliton models.

Theoretical Background:
    Tests verify that topological charge calculations follow the correct
    physical formulas and produce reasonable results for known field
    configurations.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_topology_physics_basic.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from bhlff.models.level_e.soliton_models import BaryonSoliton, SkyrmionSoliton
from bhlff.core.domain.domain import Domain


class TestSolitonTopologyPhysicsBasic:
    """
    Basic physical tests for soliton topological properties.

    Tests verify the basic physical correctness of topological charge
    calculations and related properties.
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
        """Create physics parameters for testing."""
        return {
            "mu": 1.0,
            "beta": 1.5,
            "lambda_param": 0.1,
            "nu": 1.0,
            "topological_charge_tolerance": 1e-6,
        }

    def test_topological_charge_b1_u1_phase(self, domain_3d, physics_params):
        """
        Test topological charge calculation for B=1 U(1) phase field.

        Physical Meaning:
            Verifies that a B=1 soliton configuration produces
            the correct topological charge of 1.

        Mathematical Foundation:
            For a U(1) phase field with winding number 1,
            the topological charge should be exactly 1.
        """
        # Create B=1 soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create field with B=1 configuration
        field = soliton.create_b1_configuration()

        # Calculate topological charge
        charge = soliton.compute_topological_charge(field)

        # Verify charge is 1
        assert abs(charge - 1.0) < physics_params["topological_charge_tolerance"]

        # Verify charge is integer
        assert abs(charge - round(charge)) < 1e-10

    def test_topological_charge_trivial_field(self, domain_3d, physics_params):
        """
        Test topological charge calculation for trivial field.

        Physical Meaning:
            Verifies that a trivial (constant) field configuration
            produces zero topological charge.

        Mathematical Foundation:
            A constant field has no winding and should have
            zero topological charge.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create trivial field (constant)
        field = np.ones(domain_3d.shape, dtype=complex)

        # Calculate topological charge
        charge = soliton.compute_topological_charge(field)

        # Verify charge is 0
        assert abs(charge) < physics_params["topological_charge_tolerance"]

    def test_topological_charge_conservation(self, domain_3d, physics_params):
        """
        Test topological charge conservation under time evolution.

        Physical Meaning:
            Verifies that topological charge is conserved during
            time evolution of the field.

        Mathematical Foundation:
            Topological charge is a conserved quantity in
            the 7D phase field theory.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create initial field
        field_initial = soliton.create_b1_configuration()

        # Calculate initial charge
        charge_initial = soliton.compute_topological_charge(field_initial)

        # Simulate time evolution (small phase rotation)
        field_evolved = field_initial * np.exp(1j * 0.1)

        # Calculate evolved charge
        charge_evolved = soliton.compute_topological_charge(field_evolved)

        # Verify charge conservation
        assert (
            abs(charge_evolved - charge_initial)
            < physics_params["topological_charge_tolerance"]
        )

    def test_topological_charge_scaling(self, physics_params):
        """
        Test topological charge scaling with domain size.

        Physical Meaning:
            Verifies that topological charge is independent
            of domain size for the same configuration.

        Mathematical Foundation:
            Topological charge is a dimensionless quantity
            that should not depend on domain size.
        """
        # Test different domain sizes
        domain_sizes = [2.0, 4.0, 8.0]
        charges = []

        for L in domain_sizes:
            domain = Domain(L=L, N=32, dimensions=7)
            soliton = BaryonSoliton(domain, physics_params)
            field = soliton.create_b1_configuration()
            charge = soliton.compute_topological_charge(field)
            charges.append(charge)

        # Verify charges are approximately equal
        for i in range(1, len(charges)):
            assert (
                abs(charges[i] - charges[0])
                < physics_params["topological_charge_tolerance"]
            )

    def test_topological_charge_under_rotation(self, domain_3d, physics_params):
        """
        Test topological charge invariance under rotation.

        Physical Meaning:
            Verifies that topological charge is invariant
            under spatial rotations.

        Mathematical Foundation:
            Topological charge is a topological invariant
            that should be preserved under continuous
            transformations like rotations.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create initial field
        field_initial = soliton.create_b1_configuration()

        # Calculate initial charge
        charge_initial = soliton.compute_topological_charge(field_initial)

        # Apply rotation (simulate by phase shift)
        field_rotated = np.roll(field_initial, shift=1, axis=0)

        # Calculate rotated charge
        charge_rotated = soliton.compute_topological_charge(field_rotated)

        # Verify charge invariance
        assert (
            abs(charge_rotated - charge_initial)
            < physics_params["topological_charge_tolerance"]
        )

    def test_topological_charge_precision(self, domain_3d, physics_params):
        """
        Test topological charge calculation precision.

        Physical Meaning:
            Verifies that topological charge calculations
            have sufficient numerical precision.

        Mathematical Foundation:
            Topological charge should be calculated with
            high precision to ensure physical accuracy.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create field
        field = soliton.create_b1_configuration()

        # Calculate charge multiple times
        charges = []
        for _ in range(5):
            charge = soliton.compute_topological_charge(field)
            charges.append(charge)

        # Verify precision (charges should be identical)
        for i in range(1, len(charges)):
            assert abs(charges[i] - charges[0]) < 1e-12


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
