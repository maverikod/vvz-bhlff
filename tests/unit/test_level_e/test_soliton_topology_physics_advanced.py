"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced physical tests for soliton topological properties.

This module contains advanced tests for topological charge calculations
and related properties in soliton models, including boundary effects
and integration radius.

Theoretical Background:
    Tests verify advanced aspects of topological charge calculations
    including boundary effects, integration radius dependence,
    and charge-specific terms.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_topology_physics_advanced.py -v
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


class TestSolitonTopologyPhysicsAdvanced:
    """
    Advanced physical tests for soliton topological properties.

    Tests verify advanced aspects of topological charge calculations
    including boundary effects, integration radius, and charge-specific terms.
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

    def test_topological_charge_boundary_effects(self, physics_params):
        """
        Test topological charge calculation with boundary effects.

        Physical Meaning:
            Verifies that topological charge calculations
            are robust against boundary effects.

        Mathematical Foundation:
            Topological charge should be calculated correctly
            even when the soliton is near domain boundaries.
        """
        # Test different domain sizes and positions
        domain_sizes = [2.0, 4.0, 8.0]

        for L in domain_sizes:
            domain = Domain(L=L, N=32, dimensions=7)
            soliton = BaryonSoliton(domain, physics_params)

            # Create field with soliton at different positions
            field = soliton.create_b1_configuration()

            # Calculate charge
            charge = soliton.compute_topological_charge(field)

            # Verify charge is approximately 1
            assert abs(charge - 1.0) < physics_params["topological_charge_tolerance"]

    def test_topological_charge_integration_radius(self, domain_3d, physics_params):
        """
        Test topological charge calculation with different integration radii.

        Physical Meaning:
            Verifies that topological charge calculations
            are consistent across different integration radii.

        Mathematical Foundation:
            Topological charge should be independent of
            the integration radius for sufficiently large radii.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create field
        field = soliton.create_b1_configuration()

        # Test different integration radii
        integration_radii = [0.5, 1.0, 2.0, 4.0]
        charges = []

        for radius in integration_radii:
            charge = soliton.compute_topological_charge_with_radius(field, radius)
            charges.append(charge)

        # Verify charges are approximately equal for large radii
        for i in range(2, len(charges)):  # Skip small radii
            assert (
                abs(charges[i] - charges[-1])
                < physics_params["topological_charge_tolerance"]
            )

    def test_topological_charge_charge_specific_terms(self, domain_3d, physics_params):
        """
        Test topological charge calculation with charge-specific terms.

        Physical Meaning:
            Verifies that topological charge calculations
            correctly handle charge-specific terms in the
            field configuration.

        Mathematical Foundation:
            Different charge configurations should produce
            the correct topological charge values.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Test different charge configurations
        charge_configs = [1, 2, 3, -1, -2]

        for target_charge in charge_configs:
            # Create field with specific charge
            field = soliton.create_charge_configuration(target_charge)

            # Calculate charge
            computed_charge = soliton.compute_topological_charge(field)

            # Verify charge matches target
            assert (
                abs(computed_charge - target_charge)
                < physics_params["topological_charge_tolerance"]
            )

    def test_topological_charge_skyrmion_model(self, domain_3d, physics_params):
        """
        Test topological charge calculation for Skyrmion model.

        Physical Meaning:
            Verifies that Skyrmion solitons produce
            the correct topological charge.

        Mathematical Foundation:
            Skyrmion solitons should have integer
            topological charge values.
        """
        # Create Skyrmion soliton
        skyrmion = SkyrmionSoliton(domain_3d, physics_params)

        # Create Skyrmion field
        field = skyrmion.create_skyrmion_configuration()

        # Calculate charge
        charge = skyrmion.compute_topological_charge(field)

        # Verify charge is integer
        assert abs(charge - round(charge)) < 1e-10

        # Verify charge is positive
        assert charge > 0

    def test_topological_charge_multiple_solitons(self, domain_3d, physics_params):
        """
        Test topological charge calculation for multiple solitons.

        Physical Meaning:
            Verifies that multiple solitons produce
            the correct total topological charge.

        Mathematical Foundation:
            Total topological charge should be the sum
            of individual soliton charges.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create field with multiple solitons
        field = soliton.create_multi_soliton_configuration([1, 1, -1])

        # Calculate total charge
        total_charge = soliton.compute_topological_charge(field)

        # Verify total charge is 1 (1 + 1 - 1 = 1)
        assert abs(total_charge - 1.0) < physics_params["topological_charge_tolerance"]

    def test_topological_charge_energy_relation(self, domain_3d, physics_params):
        """
        Test relationship between topological charge and energy.

        Physical Meaning:
            Verifies that topological charge is related
            to the energy of the field configuration.

        Mathematical Foundation:
            Higher topological charge should generally
            correspond to higher energy configurations.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Test different charge configurations
        charges = [1, 2, 3]
        energies = []

        for charge in charges:
            field = soliton.create_charge_configuration(charge)
            energy = soliton.compute_energy(field)
            energies.append(energy)

        # Verify energy increases with charge
        for i in range(1, len(energies)):
            assert energies[i] > energies[i - 1]

    def test_topological_charge_stability(self, domain_3d, physics_params):
        """
        Test topological charge stability under perturbations.

        Physical Meaning:
            Verifies that topological charge is stable
            under small perturbations.

        Mathematical Foundation:
            Topological charge should be robust against
            small perturbations of the field.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create field
        field = soliton.create_b1_configuration()

        # Calculate initial charge
        charge_initial = soliton.compute_topological_charge(field)

        # Add small perturbation
        perturbation = 0.01 * np.random.random(
            field.shape
        ) + 1j * 0.01 * np.random.random(field.shape)
        field_perturbed = field + perturbation

        # Calculate perturbed charge
        charge_perturbed = soliton.compute_topological_charge(field_perturbed)

        # Verify charge stability
        assert (
            abs(charge_perturbed - charge_initial) < 0.1
        )  # Allow some tolerance for perturbations

    def test_topological_charge_convergence(self, physics_params):
        """
        Test topological charge convergence with grid resolution.

        Physical Meaning:
            Verifies that topological charge calculations
            converge with increasing grid resolution.

        Mathematical Foundation:
            Topological charge should converge to the
            correct value as grid resolution increases.
        """
        # Test different grid resolutions
        grid_sizes = [16, 32, 64, 128]
        charges = []

        for N in grid_sizes:
            domain = Domain(L=4.0, N=N, dimensions=7)
            soliton = BaryonSoliton(domain, physics_params)
            field = soliton.create_b1_configuration()
            charge = soliton.compute_topological_charge(field)
            charges.append(charge)

        # Verify convergence (charges should approach 1.0)
        for i in range(1, len(charges)):
            assert abs(charges[i] - 1.0) < abs(
                charges[i - 1] - 1.0
            )  # Should get closer to 1.0

    def test_topological_charge_phase_continuity(self, domain_3d, physics_params):
        """
        Test topological charge calculation with phase continuity.

        Physical Meaning:
            Verifies that topological charge calculations
            handle phase continuity correctly.

        Mathematical Foundation:
            Phase field should be continuous and
            topological charge should be well-defined.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create field with phase continuity
        field = soliton.create_continuous_phase_field()

        # Calculate charge
        charge = soliton.compute_topological_charge(field)

        # Verify charge is finite and reasonable
        assert np.isfinite(charge)
        assert abs(charge) < 10.0  # Reasonable upper bound

    def test_topological_charge_symmetry(self, domain_3d, physics_params):
        """
        Test topological charge calculation with symmetry.

        Physical Meaning:
            Verifies that topological charge calculations
            respect field symmetries.

        Mathematical Foundation:
            Symmetric field configurations should
            produce symmetric topological charge distributions.
        """
        # Create soliton
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Create symmetric field
        field = soliton.create_symmetric_field()

        # Calculate charge
        charge = soliton.compute_topological_charge(field)

        # Verify charge is integer (symmetric fields should have integer charge)
        assert abs(charge - round(charge)) < 1e-10


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
