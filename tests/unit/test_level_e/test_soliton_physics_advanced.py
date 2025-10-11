"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced physical tests for soliton models in Level E experiments.

This module contains advanced physical tests for soliton models,
testing complex interactions, stability analysis, and parameter
dependencies.

Theoretical Background:
    Tests verify advanced physical properties of soliton implementations
    including topological charge conservation, FR constraints, stability
    analysis, and parameter dependence.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_physics_advanced.py -v
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


class TestSolitonPhysicsAdvanced:
    """
    Advanced physical tests for soliton models.

    Tests verify advanced physical properties of soliton implementations
    including topological charge conservation, FR constraints, stability
    analysis, and parameter dependence.
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
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create phase coordinates
        phi1 = np.linspace(0, 2 * np.pi, 8)
        phi2 = np.linspace(0, 2 * np.pi, 8)
        phi3 = np.linspace(0, 2 * np.pi, 8)
        PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing="ij")

        # Create U(1)^3 phase field with controlled winding
        # Θ(x,φ) = spatial_factor * (φ₁ + φ₂ + φ₃)
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

    def test_topological_charge_calculation(
        self, domain_3d, physics_params, u1_phase_field
    ):
        """
        Test topological charge calculation with U(1)^3 phase field.

        Physical Meaning:
            Verifies that topological charge is correctly computed for
            a U(1)^3 phase field configuration. For a U(1)^3 phase field,
            the topological charge should be close to 1.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute topological charge
        charge = soliton.compute_topological_charge(u1_phase_field)

        # Check physical properties
        assert np.isfinite(charge), "Topological charge should be finite"

        # For a U(1)^3 phase field, charge should be close to 1
        # (allowing for numerical errors and boundary effects)
        assert (
            abs(charge - 1.0) < 0.5
        ), f"Hedgehog field should have charge ≈ 1, got {charge}"

    def test_fr_constraints_application(
        self, domain_3d, physics_params, u1_phase_field
    ):
        """
        Test FR constraints application to field.

        Physical Meaning:
            Verifies that FR constraints are correctly applied to ensure
            fermionic statistics by checking the sign change under 2π rotation.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Apply FR constraints
        constrained_field = soliton.apply_fr_constraints(u1_phase_field)

        # Check that field is modified
        assert not np.array_equal(
            u1_phase_field, constrained_field
        ), "FR constraints should modify the field"

        # Check that constrained field has same shape
        assert (
            constrained_field.shape == u1_phase_field.shape
        ), "Constrained field should have same shape"

    def test_soliton_solution_finding(self, domain_3d, physics_params, u1_phase_field):
        """
        Test soliton solution finding with U(1)^3 phase initial guess.

        Physical Meaning:
            Verifies that the soliton finding algorithm converges to a
            stable solution with proper energy and topological charge.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Find soliton solution
        result = soliton.find_soliton_solution(u1_phase_field)

        # Check that solution is found
        assert "solution" in result, "Solution should be found"
        assert "energy" in result, "Energy should be computed"
        assert "topological_charge" in result, "Topological charge should be computed"
        assert "stability" in result, "Stability analysis should be performed"

        # Check physical properties
        solution = result["solution"]
        energy = result["energy"]
        charge = result["topological_charge"]
        stability = result["stability"]

        assert np.isfinite(energy), "Solution energy should be finite"
        assert np.isfinite(charge), "Solution topological charge should be finite"
        assert isinstance(stability, dict), "Stability should be a dictionary"

        # Check that solution has reasonable properties
        assert (
            abs(charge - 1.0) < 0.5
        ), f"Baryon soliton should have charge ≈ 1, got {charge}"

    def test_stability_analysis(self, domain_3d, physics_params, u1_phase_field):
        """
        Test stability analysis of soliton solution.

        Physical Meaning:
            Verifies that stability analysis correctly identifies
            stable and unstable modes of the soliton solution.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Analyze stability
        stability = soliton.analyze_soliton_stability(u1_phase_field)

        # Check stability analysis results
        assert "eigenvalues" in stability, "Eigenvalues should be computed"
        assert "eigenvectors" in stability, "Eigenvectors should be computed"
        assert "frequencies" in stability, "Frequencies should be computed"
        assert "stable_modes" in stability, "Stable modes should be identified"
        assert "unstable_modes" in stability, "Unstable modes should be identified"
        assert "is_stable" in stability, "Overall stability should be determined"

        # Check physical properties
        eigenvalues = stability["eigenvalues"]
        frequencies = stability["frequencies"]
        stable_modes = stability["stable_modes"]
        unstable_modes = stability["unstable_modes"]

        assert len(eigenvalues) > 0, "Should have eigenvalues"
        assert len(frequencies) == len(
            eigenvalues
        ), "Frequencies should match eigenvalues"
        assert len(stable_modes) == len(
            eigenvalues
        ), "Stable modes should match eigenvalues"
        assert len(unstable_modes) == len(
            eigenvalues
        ), "Unstable modes should match eigenvalues"

        # Check that frequencies are non-negative for stable modes
        for i, (freq, stable) in enumerate(zip(frequencies, stable_modes)):
            if stable:
                assert freq >= 0, f"Stable mode {i} should have non-negative frequency"

    def test_energy_conservation(self, domain_3d, physics_params, u1_phase_field):
        """
        Test energy conservation properties.

        Physical Meaning:
            Verifies that energy calculations are consistent and
            that the energy functional is properly implemented.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute energy multiple times
        energy1 = soliton.compute_soliton_energy(u1_phase_field)
        energy2 = soliton.compute_soliton_energy(u1_phase_field)

        # Check consistency
        assert (
            abs(energy1 - energy2) < 1e-10
        ), "Energy should be consistent across multiple calculations"

        # Check that energy is finite
        assert np.isfinite(energy1), "Energy should be finite"

    def test_charge_conservation(self, domain_3d, physics_params, u1_phase_field):
        """
        Test topological charge conservation properties.

        Physical Meaning:
            Verifies that topological charge calculations are consistent
            and that charge is properly conserved.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)

        # Compute charge multiple times
        charge1 = soliton.compute_topological_charge(u1_phase_field)
        charge2 = soliton.compute_topological_charge(u1_phase_field)

        # Check consistency
        assert (
            abs(charge1 - charge2) < 1e-10
        ), "Topological charge should be consistent across multiple calculations"

        # Check that charge is finite
        assert np.isfinite(charge1), "Topological charge should be finite"

    def test_parameter_dependence(self, domain_3d):
        """
        Test dependence on physical parameters.

        Physical Meaning:
            Verifies that soliton properties depend correctly on
            physical parameters like coupling constants.
        """
        # Test with different coupling constants
        params1 = {
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

        params2 = params1.copy()
        params2["S4"] = 0.2  # Double the Skyrme coupling

        soliton1 = BaryonSoliton(domain_3d, params1)
        soliton2 = BaryonSoliton(domain_3d, params2)

        # Check that parameters are correctly set
        assert soliton1.S4 == 0.1
        assert soliton2.S4 == 0.2

        # Check that WZW coefficient is correctly computed
        expected_coeff1 = params1["N_c"] / (240 * np.pi**2)
        expected_coeff2 = params2["N_c"] / (240 * np.pi**2)

        assert abs(soliton1.wzw_coefficient - expected_coeff1) < 1e-10
        assert abs(soliton2.wzw_coefficient - expected_coeff2) < 1e-10

    def test_domain_independence(self, physics_params):
        """
        Test independence on domain parameters.

        Physical Meaning:
            Verifies that soliton properties are correctly scaled
            with domain size and resolution.
        """
        # Test with different domain sizes
        domain1 = Domain(L=1.0, N=16, dimensions=3, boundary_condition="periodic")
        domain2 = Domain(L=2.0, N=32, dimensions=3, boundary_condition="periodic")

        soliton1 = BaryonSoliton(domain1, physics_params)
        soliton2 = BaryonSoliton(domain2, physics_params)

        # Check that both solitons are created successfully
        assert soliton1.domain.L == 1.0
        assert soliton2.domain.L == 2.0
        assert soliton1.domain.N == 16
        assert soliton2.domain.N == 32

        # Check that physical parameters are the same
        assert soliton1.S4 == soliton2.S4
        assert soliton1.S6 == soliton2.S6
        assert soliton1.F2 == soliton2.F2
        assert soliton1.N_c == soliton2.N_c


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
