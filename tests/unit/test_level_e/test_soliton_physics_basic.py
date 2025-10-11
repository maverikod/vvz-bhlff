"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic physical tests for soliton models in Level E experiments.

This module contains basic physical tests for soliton models,
testing the setup and basic energy calculations.

Theoretical Background:
    Tests verify the physical correctness of soliton implementations
    including WZW terms, topological charge conservation, FR constraints,
    and basic energy calculations.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_physics_basic.py -v
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


class TestSolitonPhysics:
    """
    Basic physical tests for soliton models.

    Tests verify the physical correctness of soliton implementations
    against known theoretical results and physical constraints.
    """

    @pytest.fixture
    def domain_3d(self):
        """Create 7D domain for testing."""
        return Domain(L=2.0, N=32, dimensions=7)  # Domain size  # Grid points

    @pytest.fixture
    def physics_params(self):
        """Create realistic physics parameters."""
        return {
            "mu": 1.0,  # Diffusion coefficient
            "beta": 1.0,  # Fractional order
            "lambda": 0.1,  # Damping parameter
            "S4": 0.1,  # Skyrme quartic coupling
            "S6": 0.01,  # Skyrme sextic coupling
            "F2": 1.0,  # Kinetic coupling
            "N_c": 3,  # Number of colors
            "wzw_coupling": 1.0,  # WZW coupling
            "charge_radius": 2.0,  # Integration radius for charge
            "charge_precision": 1e-6,
            "fr_constraint_strength": 1.0,
            "fr_rotation_axis": [0, 0, 1],
            "fr_rotation_center": [0, 0, 0],
        }

    @pytest.fixture
    def u1_phase_field(self, domain_3d):
        """
        Create U(1)^3 phase field configuration.

        Physical Meaning:
            Creates a 7D phase field configuration Θ(x,φ) ∈ T^3_φ with
            controlled winding over φ-coordinates. This represents a B=1
            soliton with U(1)^3 phase winding boundary conditions.

        Mathematical Foundation:
            Θ(x,φ) = φ₁ + φ₂ + φ₃ with controlled winding over T^3_φ.
            The U(1)^3 phase configuration has topological charge B=1
            via winding integrals over φ-coordinates.
        """
        N = domain_3d.N
        L = domain_3d.L

        # Create 7D phase field Θ(x,φ) ∈ T^3_φ
        # Shape: (N, N, N, 8, 8, 8, 1) for (x, y, z, φ₁, φ₂, φ₃, t)
        field = np.zeros((N, N, N, 8, 8, 8, 1), dtype=complex)

        # Create coordinate arrays
        x = np.linspace(-L / 2, L / 2, N)
        y = np.linspace(-L / 2, L / 2, N)
        z = np.linspace(-L / 2, L / 2, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

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
                    R = L / 4  # Soliton radius
                    spatial_factor = (
                        np.exp(-(r_spatial**2) / (2 * R**2)) if r_spatial < R else 0
                    )

                    # Create phase field with controlled winding
                    # Θ(x,φ) = spatial_factor * (φ₁ + φ₂ + φ₃)
                    field[i, j, k, :, :, :, 0] = spatial_factor * (PHI1 + PHI2 + PHI3)

        return field

    def test_wzw_term_setup(self, domain_3d, physics_params):
        """
        Test WZW term setup with physical parameters.

        Physical Meaning:
            Verifies that the WZW term is correctly initialized with
            the proper coefficient (N_c/240π²) and integration parameters.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Check WZW coefficient
        expected_coefficient = physics_params["N_c"] / (240 * np.pi**2)
        assert abs(soliton.wzw_coefficient - expected_coefficient) < 1e-10

        # Check integration parameters
        assert soliton.wzw_integration_order == 5
        assert soliton.wzw_precision == 1e-6
        assert soliton.N_c == physics_params["N_c"]
        assert soliton.wzw_coupling == physics_params["wzw_coupling"]

    def test_topological_charge_setup(self, domain_3d, physics_params):
        """
        Test topological charge setup with physical parameters.

        Physical Meaning:
            Verifies that the topological charge calculation is correctly
            initialized with proper integration parameters.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Check integration parameters
        assert soliton.charge_integration_radius == physics_params["charge_radius"]
        assert soliton.charge_precision == physics_params["charge_precision"]
        assert soliton.charge_integration_points == 64

    def test_fr_constraints_setup(self, domain_3d, physics_params):
        """
        Test FR constraints setup with physical parameters.

        Physical Meaning:
            Verifies that Finkelstein-Rubinstein constraints are correctly
            initialized to ensure fermionic statistics.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Check FR parameters
        assert soliton.fr_rotation_angle == 2 * np.pi
        assert soliton.fr_sign_change == True
        assert (
            soliton.fr_constraint_strength == physics_params["fr_constraint_strength"]
        )

        # Check rotation axis and center
        expected_axis = np.array(physics_params["fr_rotation_axis"])
        expected_axis = expected_axis / np.linalg.norm(expected_axis)
        np.testing.assert_array_almost_equal(soliton.fr_rotation_axis, expected_axis)

        expected_center = np.array(physics_params["fr_rotation_center"])
        np.testing.assert_array_almost_equal(
            soliton.fr_rotation_center, expected_center
        )

    def test_charge_specific_terms_baryon(self, domain_3d, physics_params):
        """
        Test charge-specific terms for B=1 baryon.

        Physical Meaning:
            Verifies that B=1 soliton is correctly configured with
            U(1)^3 phase boundary condition and single baryon constraints.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Check baryon-specific configuration
        assert soliton.boundary_condition == "u1_phase_winding"
        assert soliton.constraint_type == "single_baryon"
        assert soliton.charge_specific_coupling == physics_params.get(
            "baryon_coupling", 1.0
        )

    def test_charge_specific_terms_skyrmion(self, domain_3d, physics_params):
        """
        Test charge-specific terms for B>1 skyrmion.

        Physical Meaning:
            Verifies that B>1 soliton is correctly configured with
            multi-U(1)^3 phase boundary condition and multi-baryon constraints.
        """
        physics_params["multi_baryon_coupling"] = 1.5
        physics_params["baryon_separation"] = 1.0
        physics_params["interaction_strength"] = 0.1

        soliton = SkyrmionSoliton(domain_3d, physics_params, charge=2)

        # Check multi-baryon configuration
        assert soliton.boundary_condition == "multi_u1_phase_winding"
        assert soliton.constraint_type == "multi_baryon"
        assert (
            soliton.charge_specific_coupling == physics_params["multi_baryon_coupling"]
        )
        assert soliton.baryon_separation == physics_params["baryon_separation"]
        assert soliton.interaction_strength == physics_params["interaction_strength"]

    def test_charge_specific_terms_antibaryon(self, domain_3d, physics_params):
        """
        Test charge-specific terms for B<0 antibaryon.

        Physical Meaning:
            Verifies that B<0 soliton is correctly configured with
            anti-U(1)^3 phase boundary condition and antibaryon constraints.
        """
        physics_params["antibaryon_coupling"] = -1.0

        soliton = SkyrmionSoliton(domain_3d, physics_params, charge=-1)

        # Check antibaryon configuration
        assert soliton.boundary_condition == "anti_u1_phase_winding"
        assert soliton.constraint_type == "antibaryon"
        assert soliton.charge_specific_coupling == physics_params["antibaryon_coupling"]
        assert soliton.antibaryon_coupling == physics_params["antibaryon_coupling"]

    def test_kinetic_energy_calculation(
        self, domain_3d, physics_params, u1_phase_field
    ):
        """
        Test kinetic energy calculation with U(1)^3 phase field.

        Physical Meaning:
            Verifies that kinetic energy is correctly computed for
            a U(1)^3 phase field configuration. The kinetic energy should
            be positive and finite.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute kinetic energy
        kinetic_energy = soliton._energy_calculator.compute_kinetic_energy(
            u1_phase_field
        )

        # Check physical properties
        assert kinetic_energy >= 0, "Kinetic energy should be non-negative"
        assert np.isfinite(kinetic_energy), "Kinetic energy should be finite"

        # For a static U(1)^3 phase field, kinetic energy should be zero
        # (no time dependence in the test field)
        assert (
            abs(kinetic_energy) < 1e-10
        ), "Static field should have zero kinetic energy"

    def test_skyrme_energy_calculation(self, domain_3d, physics_params, u1_phase_field):
        """
        Test Skyrme energy calculation with U(1)^3 phase field.

        Physical Meaning:
            Verifies that Skyrme energy is correctly computed for
            a U(1)^3 phase field configuration. The Skyrme energy should
            be positive and provide stability against collapse.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute Skyrme energy
        skyrme_energy = soliton._compute_skyrme_energy(u1_phase_field)

        # Check physical properties
        assert skyrme_energy >= 0, "Skyrme energy should be non-negative"
        assert np.isfinite(skyrme_energy), "Skyrme energy should be finite"

        # Skyrme energy should be significant for U(1)^3 phase configuration
        assert (
            skyrme_energy > 0
        ), "U(1)^3 phase field should have non-zero Skyrme energy"

    def test_wzw_energy_calculation(self, domain_3d, physics_params, u1_phase_field):
        """
        Test WZW energy calculation with U(1)^3 phase field.

        Physical Meaning:
            Verifies that WZW energy is correctly computed for
            a U(1)^3 phase field configuration. The WZW energy should
            be finite and contribute to baryon number conservation.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute WZW energy
        wzw_energy = soliton._compute_wzw_energy(u1_phase_field)

        # Check physical properties
        assert np.isfinite(wzw_energy), "WZW energy should be finite"

        # WZW energy can be positive or negative depending on configuration
        # but should be finite

    def test_total_energy_calculation(self, domain_3d, physics_params, u1_phase_field):
        """
        Test total energy calculation with U(1)^3 phase field.

        Physical Meaning:
            Verifies that total energy is correctly computed as the sum
            of kinetic, Skyrme, and WZW contributions.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute total energy
        total_energy = soliton.compute_soliton_energy(u1_phase_field)

        # Compute individual contributions
        kinetic_energy = soliton._compute_kinetic_energy(u1_phase_field)
        skyrme_energy = soliton._compute_skyrme_energy(u1_phase_field)
        wzw_energy = soliton._compute_wzw_energy(u1_phase_field)

        # Check that total energy equals sum of contributions
        expected_total = kinetic_energy + skyrme_energy + wzw_energy
        assert (
            abs(total_energy - expected_total) < 1e-10
        ), "Total energy should equal sum of contributions"

        # Check physical properties
        assert np.isfinite(total_energy), "Total energy should be finite"
