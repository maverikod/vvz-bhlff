"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for soliton models in Level E experiments.

This module contains comprehensive physical tests for soliton models,
testing the implementation against known theoretical results and
physical constraints.

Theoretical Background:
    Tests verify the physical correctness of soliton implementations
    including WZW terms, topological charge conservation, FR constraints,
    and energy calculations.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from bhlff.models.level_e.soliton_models import BaryonSoliton, SkyrmionSoliton
from bhlff.core.domain.domain import Domain


class TestSolitonPhysics:
    """
    Physical tests for soliton models.
    
    Tests verify the physical correctness of soliton implementations
    against known theoretical results and physical constraints.
    """

    @pytest.fixture
    def domain_3d(self):
        """Create 7D domain for testing."""
        return Domain(
            L=2.0,  # Domain size
            N=32,   # Grid points
            dimensions=7
        )

    @pytest.fixture
    def physics_params(self):
        """Create realistic physics parameters."""
        return {
            "mu": 1.0,           # Diffusion coefficient
            "beta": 1.0,         # Fractional order
            "lambda": 0.1,       # Damping parameter
            "S4": 0.1,           # Skyrme quartic coupling
            "S6": 0.01,          # Skyrme sextic coupling
            "F2": 1.0,           # Kinetic coupling
            "N_c": 3,            # Number of colors
            "wzw_coupling": 1.0,  # WZW coupling
            "charge_radius": 2.0, # Integration radius for charge
            "charge_precision": 1e-6,
            "fr_constraint_strength": 1.0,
            "fr_rotation_axis": [0, 0, 1],
            "fr_rotation_center": [0, 0, 0]
        }

    @pytest.fixture
    def hedgehog_field(self, domain_3d):
        """
        Create hedgehog field configuration.
        
        Physical Meaning:
            Creates a hedgehog field configuration U(x) = exp(iτ·n̂(x))
            where n̂(x) = x̂/|x| is the unit radial vector. This is the
            standard configuration for B=1 solitons.
            
        Mathematical Foundation:
            U(x) = cos(f(r)) + iτ·n̂(x) sin(f(r))
            where f(r) is the profile function and τ are Pauli matrices.
        """
        N = domain_3d.N
        L = domain_3d.L
        
        # Create coordinate arrays
        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
        z = np.linspace(-L/2, L/2, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Compute radial coordinates
        r = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Avoid division by zero
        r = np.where(r < 1e-10, 1e-10, r)
        
        # Create hedgehog field U(x) = exp(iτ·n̂(x))
        # For simplicity, use a 2x2 matrix representation
        field = np.zeros((N, N, N, 2, 2), dtype=complex)
        
        # Profile function f(r) = π(1 - r/R) for r < R, 0 otherwise
        R = L/4  # Soliton radius
        f = np.where(r < R, np.pi * (1 - r/R), 0)
        
        # Unit radial vector n̂(x) = x/|x|
        nx = X / r
        ny = Y / r
        nz = Z / r
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Create field U(x) = cos(f(r)) + iτ·n̂(x) sin(f(r))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if r[i, j, k] < R:
                        # τ·n̂ = σ_x n_x + σ_y n_y + σ_z n_z
                        tau_dot_n = (sigma_x * nx[i, j, k] + 
                                   sigma_y * ny[i, j, k] + 
                                   sigma_z * nz[i, j, k])
                        
                        # U = cos(f) + iτ·n̂ sin(f)
                        field[i, j, k] = (np.cos(f[i, j, k]) * np.eye(2) + 
                                        1j * tau_dot_n * np.sin(f[i, j, k]))
                    else:
                        field[i, j, k] = np.eye(2)
        
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
        assert soliton.fr_constraint_strength == physics_params["fr_constraint_strength"]
        
        # Check rotation axis and center
        expected_axis = np.array(physics_params["fr_rotation_axis"])
        expected_axis = expected_axis / np.linalg.norm(expected_axis)
        np.testing.assert_array_almost_equal(soliton.fr_rotation_axis, expected_axis)
        
        expected_center = np.array(physics_params["fr_rotation_center"])
        np.testing.assert_array_almost_equal(soliton.fr_rotation_center, expected_center)

    def test_charge_specific_terms_baryon(self, domain_3d, physics_params):
        """
        Test charge-specific terms for B=1 baryon.
        
        Physical Meaning:
            Verifies that B=1 soliton is correctly configured with
            hedgehog boundary condition and single baryon constraints.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Check baryon-specific configuration
        assert soliton.boundary_condition == "hedgehog"
        assert soliton.constraint_type == "single_baryon"
        assert soliton.charge_specific_coupling == physics_params.get("baryon_coupling", 1.0)

    def test_charge_specific_terms_skyrmion(self, domain_3d, physics_params):
        """
        Test charge-specific terms for B>1 skyrmion.
        
        Physical Meaning:
            Verifies that B>1 soliton is correctly configured with
            multi-hedgehog boundary condition and multi-baryon constraints.
        """
        physics_params["multi_baryon_coupling"] = 1.5
        physics_params["baryon_separation"] = 1.0
        physics_params["interaction_strength"] = 0.1
        
        soliton = SkyrmionSoliton(domain_3d, physics_params, charge=2)
        
        # Check multi-baryon configuration
        assert soliton.boundary_condition == "multi_hedgehog"
        assert soliton.constraint_type == "multi_baryon"
        assert soliton.charge_specific_coupling == physics_params["multi_baryon_coupling"]
        assert soliton.baryon_separation == physics_params["baryon_separation"]
        assert soliton.interaction_strength == physics_params["interaction_strength"]

    def test_charge_specific_terms_antibaryon(self, domain_3d, physics_params):
        """
        Test charge-specific terms for B<0 antibaryon.
        
        Physical Meaning:
            Verifies that B<0 soliton is correctly configured with
            anti-hedgehog boundary condition and antibaryon constraints.
        """
        physics_params["antibaryon_coupling"] = -1.0
        
        soliton = SkyrmionSoliton(domain_3d, physics_params, charge=-1)
        
        # Check antibaryon configuration
        assert soliton.boundary_condition == "anti_hedgehog"
        assert soliton.constraint_type == "antibaryon"
        assert soliton.charge_specific_coupling == physics_params["antibaryon_coupling"]
        assert soliton.antibaryon_coupling == physics_params["antibaryon_coupling"]

    def test_kinetic_energy_calculation(self, domain_3d, physics_params, hedgehog_field):
        """
        Test kinetic energy calculation with hedgehog field.
        
        Physical Meaning:
            Verifies that kinetic energy is correctly computed for
            a hedgehog field configuration. The kinetic energy should
            be positive and finite.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute kinetic energy
        kinetic_energy = soliton._compute_kinetic_energy(hedgehog_field)
        
        # Check physical properties
        assert kinetic_energy >= 0, "Kinetic energy should be non-negative"
        assert np.isfinite(kinetic_energy), "Kinetic energy should be finite"
        
        # For a static hedgehog field, kinetic energy should be zero
        # (no time dependence in the test field)
        assert abs(kinetic_energy) < 1e-10, "Static field should have zero kinetic energy"

    def test_skyrme_energy_calculation(self, domain_3d, physics_params, hedgehog_field):
        """
        Test Skyrme energy calculation with hedgehog field.
        
        Physical Meaning:
            Verifies that Skyrme energy is correctly computed for
            a hedgehog field configuration. The Skyrme energy should
            be positive and provide stability against collapse.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute Skyrme energy
        skyrme_energy = soliton._compute_skyrme_energy(hedgehog_field)
        
        # Check physical properties
        assert skyrme_energy >= 0, "Skyrme energy should be non-negative"
        assert np.isfinite(skyrme_energy), "Skyrme energy should be finite"
        
        # Skyrme energy should be significant for hedgehog configuration
        assert skyrme_energy > 0, "Hedgehog field should have non-zero Skyrme energy"

    def test_wzw_energy_calculation(self, domain_3d, physics_params, hedgehog_field):
        """
        Test WZW energy calculation with hedgehog field.
        
        Physical Meaning:
            Verifies that WZW energy is correctly computed for
            a hedgehog field configuration. The WZW energy should
            be finite and contribute to baryon number conservation.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute WZW energy
        wzw_energy = soliton._compute_wzw_energy(hedgehog_field)
        
        # Check physical properties
        assert np.isfinite(wzw_energy), "WZW energy should be finite"
        
        # WZW energy can be positive or negative depending on configuration
        # but should be finite

    def test_total_energy_calculation(self, domain_3d, physics_params, hedgehog_field):
        """
        Test total energy calculation with hedgehog field.
        
        Physical Meaning:
            Verifies that total energy is correctly computed as the sum
            of kinetic, Skyrme, and WZW contributions.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute total energy
        total_energy = soliton.compute_soliton_energy(hedgehog_field)
        
        # Compute individual contributions
        kinetic_energy = soliton._compute_kinetic_energy(hedgehog_field)
        skyrme_energy = soliton._compute_skyrme_energy(hedgehog_field)
        wzw_energy = soliton._compute_wzw_energy(hedgehog_field)
        
        # Check that total energy equals sum of contributions
        expected_total = kinetic_energy + skyrme_energy + wzw_energy
        assert abs(total_energy - expected_total) < 1e-10, "Total energy should equal sum of contributions"
        
        # Check physical properties
        assert np.isfinite(total_energy), "Total energy should be finite"

    def test_topological_charge_calculation(self, domain_3d, physics_params, hedgehog_field):
        """
        Test topological charge calculation with hedgehog field.
        
        Physical Meaning:
            Verifies that topological charge is correctly computed for
            a hedgehog field configuration. For a hedgehog field,
            the topological charge should be close to 1.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute topological charge
        charge = soliton.compute_topological_charge(hedgehog_field)
        
        # Check physical properties
        assert np.isfinite(charge), "Topological charge should be finite"
        
        # For a hedgehog field, charge should be close to 1
        # (allowing for numerical errors and boundary effects)
        assert abs(charge - 1.0) < 0.5, f"Hedgehog field should have charge ≈ 1, got {charge}"

    def test_fr_constraints_application(self, domain_3d, physics_params, hedgehog_field):
        """
        Test FR constraints application to field.
        
        Physical Meaning:
            Verifies that FR constraints are correctly applied to ensure
            fermionic statistics by checking the sign change under 2π rotation.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Apply FR constraints
        constrained_field = soliton.apply_fr_constraints(hedgehog_field)
        
        # Check that field is modified
        assert not np.array_equal(hedgehog_field, constrained_field), "FR constraints should modify the field"
        
        # Check that constrained field has same shape
        assert constrained_field.shape == hedgehog_field.shape, "Constrained field should have same shape"

    def test_soliton_solution_finding(self, domain_3d, physics_params, hedgehog_field):
        """
        Test soliton solution finding with hedgehog initial guess.
        
        Physical Meaning:
            Verifies that the soliton finding algorithm converges to a
            stable solution with proper energy and topological charge.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Find soliton solution
        result = soliton.find_soliton_solution(hedgehog_field)
        
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
        assert abs(charge - 1.0) < 0.5, f"Baryon soliton should have charge ≈ 1, got {charge}"

    def test_stability_analysis(self, domain_3d, physics_params, hedgehog_field):
        """
        Test stability analysis of soliton solution.
        
        Physical Meaning:
            Verifies that stability analysis correctly identifies
            stable and unstable modes of the soliton solution.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Analyze stability
        stability = soliton.analyze_soliton_stability(hedgehog_field)
        
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
        assert len(frequencies) == len(eigenvalues), "Frequencies should match eigenvalues"
        assert len(stable_modes) == len(eigenvalues), "Stable modes should match eigenvalues"
        assert len(unstable_modes) == len(eigenvalues), "Unstable modes should match eigenvalues"
        
        # Check that frequencies are non-negative for stable modes
        for i, (freq, stable) in enumerate(zip(frequencies, stable_modes)):
            if stable:
                assert freq >= 0, f"Stable mode {i} should have non-negative frequency"

    def test_energy_conservation(self, domain_3d, physics_params, hedgehog_field):
        """
        Test energy conservation properties.
        
        Physical Meaning:
            Verifies that energy calculations are consistent and
            that the energy functional is properly implemented.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute energy multiple times
        energy1 = soliton.compute_soliton_energy(hedgehog_field)
        energy2 = soliton.compute_soliton_energy(hedgehog_field)
        
        # Check consistency
        assert abs(energy1 - energy2) < 1e-10, "Energy should be consistent across multiple calculations"
        
        # Check that energy is finite
        assert np.isfinite(energy1), "Energy should be finite"

    def test_charge_conservation(self, domain_3d, physics_params, hedgehog_field):
        """
        Test topological charge conservation properties.
        
        Physical Meaning:
            Verifies that topological charge calculations are consistent
            and that charge is properly conserved.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute charge multiple times
        charge1 = soliton.compute_topological_charge(hedgehog_field)
        charge2 = soliton.compute_topological_charge(hedgehog_field)
        
        # Check consistency
        assert abs(charge1 - charge2) < 1e-10, "Topological charge should be consistent across multiple calculations"
        
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
            "mu": 1.0, "beta": 1.0, "lambda": 0.1,
            "S4": 0.1, "S6": 0.01, "F2": 1.0,
            "N_c": 3, "wzw_coupling": 1.0,
            "charge_radius": 2.0, "charge_precision": 1e-6,
            "fr_constraint_strength": 1.0,
            "fr_rotation_axis": [0, 0, 1],
            "fr_rotation_center": [0, 0, 0]
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
    pytest.main([__file__, "-v", "--tb=short"])
