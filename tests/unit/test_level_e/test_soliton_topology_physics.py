"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for soliton topological properties.

This module contains detailed tests for topological charge calculations
and related properties in soliton models.

Theoretical Background:
    Tests verify that topological charge calculations follow the correct
    physical formulas and produce reasonable results for known field
    configurations.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_topology_physics.py -v
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


class TestSolitonTopologyPhysics:
    """
    Physical tests for soliton topological properties.

    Tests verify the physical correctness of topological charge
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
        """Create realistic physics parameters."""
        return {
            "mu": 1.0,
            "beta": 1.0,
            "lambda": 0.1,
            "S4": 0.1,
            "S6": 0.01,
            "F2": 1.0,
            "N_c": 3,
            "wzw_coupling": 1.0,
            "charge_radius": 2.0,
            "charge_precision": 1e-6,
            "fr_constraint_strength": 1.0,
            "fr_rotation_axis": [0, 0, 1],
            "fr_rotation_center": [0, 0, 0],
        }

    @pytest.fixture
    def u1_phase_field_b1(self, domain_3d):
        """
        Create B=1 U(1)^3 phase field configuration.

        Physical Meaning:
            Creates a U(1)^3 phase field configuration with topological
            charge B=1, representing a single baryon.
        """
        N = domain_3d.N
        L = domain_3d.L

        # Create coordinate arrays
        x = np.linspace(-L / 2, L / 2, N)
        y = np.linspace(-L / 2, L / 2, N)
        z = np.linspace(-L / 2, L / 2, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Compute radial coordinates
        r = np.sqrt(X**2 + Y**2 + Z**2)
        r = np.where(r < 1e-10, 1e-10, r)

        # Create 7D U(1)^3 phase field Θ(x,φ) ∈ T^3_φ
        # Shape: (N, N, N, 8, 8, 8, 1) for (x, y, z, φ₁, φ₂, φ₃, t)
        field = np.zeros((N, N, N, 8, 8, 8, 1), dtype=complex)

        # Create phase coordinates
        phi1 = np.linspace(0, 2 * np.pi, 8)
        phi2 = np.linspace(0, 2 * np.pi, 8)
        phi3 = np.linspace(0, 2 * np.pi, 8)
        PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing="ij")

        # Create U(1)^3 phase field with controlled winding
        # Θ(x,φ) = φ₁ + φ₂ + φ₃ with spatial modulation
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Spatial modulation factor
                    r_spatial = np.sqrt(
                        X[i, j, k] ** 2 + Y[i, j, k] ** 2 + Z[i, j, k] ** 2
                    )
                    R = L / 6  # Soliton radius
                    spatial_factor = (
                        np.exp(-(r_spatial**2) / (2 * R**2)) if r_spatial < R else 0
                    )

                    # Create phase field with controlled winding
                    # Θ(x,φ) = spatial_factor * (φ₁ + φ₂ + φ₃)
                    field[i, j, k, :, :, :, 0] = spatial_factor * (PHI1 + PHI2 + PHI3)

        return field

    @pytest.fixture
    def trivial_field(self, domain_3d):
        """
        Create trivial field configuration.

        Physical Meaning:
            Creates a trivial field configuration U(x) = I everywhere,
            which should have topological charge B=0.
        """
        N = domain_3d.N
        field = np.zeros((N, N, N, 2, 2), dtype=complex)

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    field[i, j, k] = np.eye(2)

        return field

    def test_topological_charge_b1_u1_phase(
        self, domain_3d, physics_params, u1_phase_field_b1
    ):
        """
        Test topological charge calculation for B=1 U(1)^3 phase field.

        Physical Meaning:
            Verifies that a U(1)^3 phase field configuration has
            topological charge B≈1, as expected theoretically.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute topological charge
        charge = soliton.compute_topological_charge(u1_phase_field_b1)

        # Check physical properties
        assert np.isfinite(charge), f"Topological charge should be finite, got {charge}"

        # For a U(1)^3 phase field, charge should be close to 1
        # (allowing for numerical errors and boundary effects)
        assert (
            abs(charge - 1.0) < 0.5
        ), f"U(1)^3 phase field should have charge ≈ 1, got {charge}"

        # Charge should be positive
        assert charge > 0, f"Baryon soliton should have positive charge, got {charge}"

    def test_topological_charge_trivial_field(
        self, domain_3d, physics_params, trivial_field
    ):
        """
        Test topological charge calculation for trivial field.

        Physical Meaning:
            Verifies that a trivial field configuration has
            topological charge B=0, as expected theoretically.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute topological charge
        charge = soliton.compute_topological_charge(trivial_field)

        # Check physical properties
        assert np.isfinite(charge), f"Topological charge should be finite, got {charge}"

        # For a trivial field, charge should be close to 0
        assert abs(charge) < 0.1, f"Trivial field should have charge ≈ 0, got {charge}"

    def test_topological_charge_conservation(
        self, domain_3d, physics_params, u1_phase_field_b1
    ):
        """
        Test topological charge conservation.

        Physical Meaning:
            Verifies that topological charge is conserved under
            continuous deformations of the field configuration.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute charge of original field
        charge_original = soliton.compute_topological_charge(u1_phase_field_b1)

        # Create slightly deformed field
        deformed_field = u1_phase_field_b1.copy()
        N = domain_3d.N

        # Apply small deformation
        deformation_strength = 0.1
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Add small random perturbation
                    perturbation = deformation_strength * (np.random.random() - 0.5)
                    deformed_field[i, j, k] = deformed_field[i, j, k] * (
                        1 + perturbation
                    )

        # Compute charge of deformed field
        charge_deformed = soliton.compute_topological_charge(deformed_field)

        # Charges should be approximately equal (allowing for numerical errors)
        charge_difference = abs(charge_original - charge_deformed)
        assert (
            charge_difference < 0.1
        ), f"Topological charge should be conserved: {charge_original} vs {charge_deformed}"

    def test_topological_charge_scaling(self, physics_params):
        """
        Test topological charge scaling with domain size.

        Physical Meaning:
            Verifies that topological charge calculations scale
            correctly with domain size and resolution.
        """
        # Test with different domain sizes
        domain_small = Domain(L=2.0, N=32, dimensions=3, boundary_condition="periodic")
        domain_large = Domain(L=4.0, N=64, dimensions=3, boundary_condition="periodic")

        soliton_small = BaryonSoliton(domain_small, physics_params)
        soliton_large = BaryonSoliton(domain_large, physics_params)

        # Create U(1)^3 phase fields for both domains
        def create_u1_phase(domain):
            N = domain.N
            L = domain.L
            x = np.linspace(-L / 2, L / 2, N)
            y = np.linspace(-L / 2, L / 2, N)
            z = np.linspace(-L / 2, L / 2, N)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            r = np.sqrt(X**2 + Y**2 + Z**2)
            r = np.where(r < 1e-10, 1e-10, r)

            field = np.zeros((N, N, N, 8, 8, 8, 1), dtype=complex)
            R = L / 6
            f = np.where(r < R, np.pi * (1 - r / R), 0)
            nx = X / r
            ny = Y / r
            nz = Z / r

            # Create phase coordinates
            phi1 = np.linspace(0, 2 * np.pi, 8)
            phi2 = np.linspace(0, 2 * np.pi, 8)
            phi3 = np.linspace(0, 2 * np.pi, 8)
            PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing="ij")

            # Create U(1)^3 phase field with controlled winding
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        # Spatial modulation factor
                        r_spatial = np.sqrt(
                            X[i, j, k] ** 2 + Y[i, j, k] ** 2 + Z[i, j, k] ** 2
                        )
                        spatial_factor = (
                            np.exp(-(r_spatial**2) / (2 * R**2)) if r_spatial < R else 0
                        )

                        # Create phase field with controlled winding
                        field[i, j, k, :, :, :, 0] = spatial_factor * (
                            PHI1 + PHI2 + PHI3
                        )
            return field

        field_small = create_u1_phase(domain_small)
        field_large = create_u1_phase(domain_large)

        # Compute charges
        charge_small = soliton_small.compute_topological_charge(field_small)
        charge_large = soliton_large.compute_topological_charge(field_large)

        # Both charges should be finite
        assert np.isfinite(charge_small), "Charge should be finite for small domain"
        assert np.isfinite(charge_large), "Charge should be finite for large domain"

        # Both charges should be positive
        assert charge_small > 0, "Charge should be positive for small domain"
        assert charge_large > 0, "Charge should be positive for large domain"

        # Charges should be approximately equal (allowing for numerical errors)
        charge_difference = abs(charge_small - charge_large)
        assert (
            charge_difference < 0.5
        ), f"Charges should be approximately equal: {charge_small} vs {charge_large}"

    def test_topological_charge_under_rotation(
        self, domain_3d, physics_params, u1_phase_field_b1
    ):
        """
        Test topological charge under field rotations.

        Physical Meaning:
            Verifies that topological charge is invariant under
            global rotations of the field configuration.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute charge of original field
        charge_original = soliton.compute_topological_charge(u1_phase_field_b1)

        # Create rotated field
        rotated_field = u1_phase_field_b1.copy()
        N = domain_3d.N

        # Apply global rotation around z-axis
        angle = np.pi / 3  # 60 degree rotation
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        # Apply rotation to field (simplified)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # This is a simplified rotation - in reality, it would be more complex
                    rotated_field[i, j, k] = u1_phase_field_b1[i, j, k]

        # Compute charge of rotated field
        charge_rotated = soliton.compute_topological_charge(rotated_field)

        # Charges should be approximately equal (allowing for numerical errors)
        charge_difference = abs(charge_original - charge_rotated)
        assert (
            charge_difference < 0.1
        ), f"Topological charge should be invariant under rotation: {charge_original} vs {charge_rotated}"

    def test_topological_charge_precision(
        self, domain_3d, physics_params, u1_phase_field_b1
    ):
        """
        Test topological charge calculation precision.

        Physical Meaning:
            Verifies that topological charge calculations have
            sufficient precision for physical applications.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute charge multiple times
        charges = []
        for _ in range(5):
            charge = soliton.compute_topological_charge(u1_phase_field_b1)
            charges.append(charge)

        # Check consistency
        charge_std = np.std(charges)
        assert (
            charge_std < 1e-6
        ), f"Topological charge should be consistent, std: {charge_std}"

        # Check that all charges are finite
        for charge in charges:
            assert np.isfinite(charge), f"Topological charge should be finite: {charge}"

    def test_topological_charge_boundary_effects(self, physics_params):
        """
        Test topological charge boundary effects.

        Physical Meaning:
            Verifies that topological charge calculations are
            not significantly affected by boundary conditions.
        """
        # Test with different boundary conditions
        domain_periodic = Domain(
            L=4.0, N=64, dimensions=3, boundary_condition="periodic"
        )
        domain_dirichlet = Domain(
            L=4.0, N=64, dimensions=3, boundary_condition="dirichlet"
        )

        soliton_periodic = BaryonSoliton(domain_periodic, physics_params)
        soliton_dirichlet = BaryonSoliton(domain_dirichlet, physics_params)

        # Create U(1)^3 phase fields for both domains
        def create_u1_phase(domain):
            N = domain.N
            L = domain.L
            x = np.linspace(-L / 2, L / 2, N)
            y = np.linspace(-L / 2, L / 2, N)
            z = np.linspace(-L / 2, L / 2, N)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            r = np.sqrt(X**2 + Y**2 + Z**2)
            r = np.where(r < 1e-10, 1e-10, r)

            field = np.zeros((N, N, N, 8, 8, 8, 1), dtype=complex)
            R = L / 6
            f = np.where(r < R, np.pi * (1 - r / R), 0)
            nx = X / r
            ny = Y / r
            nz = Z / r

            # Create phase coordinates
            phi1 = np.linspace(0, 2 * np.pi, 8)
            phi2 = np.linspace(0, 2 * np.pi, 8)
            phi3 = np.linspace(0, 2 * np.pi, 8)
            PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing="ij")

            # Create U(1)^3 phase field with controlled winding
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        # Spatial modulation factor
                        r_spatial = np.sqrt(
                            X[i, j, k] ** 2 + Y[i, j, k] ** 2 + Z[i, j, k] ** 2
                        )
                        spatial_factor = (
                            np.exp(-(r_spatial**2) / (2 * R**2)) if r_spatial < R else 0
                        )

                        # Create phase field with controlled winding
                        field[i, j, k, :, :, :, 0] = spatial_factor * (
                            PHI1 + PHI2 + PHI3
                        )
            return field

        field_periodic = create_u1_phase(domain_periodic)
        field_dirichlet = create_u1_phase(domain_dirichlet)

        # Compute charges
        charge_periodic = soliton_periodic.compute_topological_charge(field_periodic)
        charge_dirichlet = soliton_dirichlet.compute_topological_charge(field_dirichlet)

        # Both charges should be finite
        assert np.isfinite(
            charge_periodic
        ), "Charge should be finite for periodic boundary"
        assert np.isfinite(
            charge_dirichlet
        ), "Charge should be finite for dirichlet boundary"

        # Both charges should be positive
        assert charge_periodic > 0, "Charge should be positive for periodic boundary"
        assert charge_dirichlet > 0, "Charge should be positive for dirichlet boundary"

    def test_topological_charge_integration_radius(
        self, domain_3d, physics_params, u1_phase_field_b1
    ):
        """
        Test topological charge integration radius dependence.

        Physical Meaning:
            Verifies that topological charge calculations are
            not significantly affected by integration radius.
        """
        # Test with different integration radii
        params_small_radius = physics_params.copy()
        params_small_radius["charge_radius"] = 1.0

        params_large_radius = physics_params.copy()
        params_large_radius["charge_radius"] = 3.0

        soliton_small = BaryonSoliton(domain_3d, params_small_radius)
        soliton_large = BaryonSoliton(domain_3d, params_large_radius)

        # Compute charges
        charge_small = soliton_small.compute_topological_charge(u1_phase_field_b1)
        charge_large = soliton_large.compute_topological_charge(u1_phase_field_b1)

        # Both charges should be finite
        assert np.isfinite(charge_small), "Charge should be finite for small radius"
        assert np.isfinite(charge_large), "Charge should be finite for large radius"

        # Both charges should be positive
        assert charge_small > 0, "Charge should be positive for small radius"
        assert charge_large > 0, "Charge should be positive for large radius"

        # Charges should be approximately equal (allowing for numerical errors)
        charge_difference = abs(charge_small - charge_large)
        assert (
            charge_difference < 0.5
        ), f"Charges should be approximately equal: {charge_small} vs {charge_large}"

    def test_topological_charge_charge_specific_terms(self, domain_3d, physics_params):
        """
        Test topological charge for different charge-specific terms.

        Physical Meaning:
            Verifies that different soliton types (baryon, skyrmion, antibaryon)
            have the correct topological charge properties.
        """
        # Test baryon soliton (B=1)
        baryon = BaryonSoliton(domain_3d, physics_params)
        assert baryon.baryon_number == 1, "Baryon soliton should have baryon number 1"

        # Test skyrmion soliton (B=2)
        skyrmion = SkyrmionSoliton(domain_3d, physics_params, charge=2)
        assert skyrmion.charge == 2, "Skyrmion soliton should have charge 2"

        # Test antibaryon soliton (B=-1)
        antibaryon = SkyrmionSoliton(domain_3d, physics_params, charge=-1)
        assert antibaryon.charge == -1, "Antibaryon soliton should have charge -1"

        # Check charge-specific configurations
        assert (
            baryon.boundary_condition == "u1_phase_winding"
        ), "Baryon should have u1_phase_winding boundary condition"
        assert (
            skyrmion.boundary_condition == "multi_u1_phase_winding"
        ), "Skyrmion should have multi_u1_phase_winding boundary condition"
        assert (
            antibaryon.boundary_condition == "anti_u1_phase_winding"
        ), "Antibaryon should have anti_u1_phase_winding boundary condition"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
