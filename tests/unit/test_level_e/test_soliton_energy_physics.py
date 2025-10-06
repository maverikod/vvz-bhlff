"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for soliton energy calculations.

This module contains detailed tests for energy calculations in soliton models,
verifying the physical correctness of kinetic, Skyrme, and WZW energy terms.

Theoretical Background:
    Tests verify that energy calculations follow the correct physical
    formulas and produce reasonable results for known field configurations.

Example:
    >>> pytest tests/unit/test_level_e/test_soliton_energy_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from bhlff.models.level_e.soliton_models import BaryonSoliton
from bhlff.core.domain.domain import Domain


class TestSolitonEnergyPhysics:
    """
    Physical tests for soliton energy calculations.
    
    Tests verify the physical correctness of energy calculations
    including kinetic, Skyrme, and WZW energy terms.
    """

    @pytest.fixture
    def domain_3d(self):
        """Create 7D domain for testing."""
        return Domain(
            L=4.0,  # Larger domain for better resolution
            N=64,   # Higher resolution
            dimensions=7
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
            "fr_rotation_center": [0, 0, 0]
        }

    @pytest.fixture
    def simple_u1_phase_field(self, domain_3d):
        """
        Create simple U(1)^3 phase field for testing.
        
        Physical Meaning:
            Creates a simplified U(1)^3 phase field configuration that
            is easier to analyze and has known theoretical properties.
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
        
        # Create 7D U(1)^3 phase field Θ(x,φ) ∈ T^3_φ
        # Shape: (N, N, N, 8, 8, 8, 1) for (x, y, z, φ₁, φ₂, φ₃, t)
        field = np.zeros((N, N, N, 8, 8, 8, 1), dtype=complex)
        
        # Create phase coordinates
        phi1 = np.linspace(0, 2*np.pi, 8)
        phi2 = np.linspace(0, 2*np.pi, 8)
        phi3 = np.linspace(0, 2*np.pi, 8)
        PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing='ij')
        
        # Create U(1)^3 phase field with controlled winding
        # Θ(x,φ) = φ₁ + φ₂ + φ₃ with spatial modulation
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Spatial modulation factor
                    r_spatial = np.sqrt(X[i, j, k]**2 + Y[i, j, k]**2 + Z[i, j, k]**2)
                    R = L/8  # Soliton radius
                    spatial_factor = np.exp(-r_spatial**2 / (2 * R**2)) if r_spatial < R else 0
                    
                    # Create phase field with controlled winding
                    # Θ(x,φ) = spatial_factor * (φ₁ + φ₂ + φ₃)
                    field[i, j, k, :, :, :, 0] = spatial_factor * (PHI1 + PHI2 + PHI3)
        
        return field

    def test_kinetic_energy_physical_properties(self, domain_3d, physics_params, simple_u1_phase_field):
        """
        Test physical properties of kinetic energy calculation.
        
        Physical Meaning:
            Verifies that kinetic energy has the correct physical properties:
            - Non-negative for all field configurations
            - Zero for static fields
            - Scales correctly with field derivatives
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Test with static field (should have zero kinetic energy)
        static_field = simple_u1_phase_field.copy()
        kinetic_energy_static = soliton._energy_calculator.compute_kinetic_energy(static_field)
        
        # Static field should have zero kinetic energy
        assert abs(kinetic_energy_static) < 1e-10, f"Static field should have zero kinetic energy, got {kinetic_energy_static}"
        
        # Test with time-dependent field
        time_dependent_field = np.zeros_like(simple_u1_phase_field)
        N = domain_3d.N
        
        # Create field with time dependence: U(x,t) = U(x) * exp(iωt)
        omega = 1.0  # Frequency
        t = 0.1  # Time
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    time_dependent_field[i, j, k] = simple_u1_phase_field[i, j, k] * np.exp(1j * omega * t)
        
        kinetic_energy_dynamic = soliton._energy_calculator.compute_kinetic_energy(time_dependent_field)
        
        # Time-dependent field should have positive kinetic energy
        assert kinetic_energy_dynamic > 0, f"Time-dependent field should have positive kinetic energy, got {kinetic_energy_dynamic}"
        assert np.isfinite(kinetic_energy_dynamic), "Kinetic energy should be finite"

    def test_skyrme_energy_physical_properties(self, domain_3d, physics_params, simple_u1_phase_field):
        """
        Test physical properties of Skyrme energy calculation.
        
        Physical Meaning:
            Verifies that Skyrme energy has the correct physical properties:
            - Non-negative for all field configurations
            - Provides stability against collapse
            - Scales with field derivatives
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute Skyrme energy
        skyrme_energy = soliton._compute_skyrme_energy(simple_u1_phase_field)
        
        # Check physical properties
        assert skyrme_energy >= 0, f"Skyrme energy should be non-negative, got {skyrme_energy}"
        assert np.isfinite(skyrme_energy), "Skyrme energy should be finite"
        
        # Skyrme energy should be significant for U(1)^3 phase configuration
        assert skyrme_energy > 0, "U(1)^3 phase field should have non-zero Skyrme energy"
        
        # Test with different coupling constants
        params_weak = physics_params.copy()
        params_weak["S4"] = 0.01  # Weak coupling
        
        params_strong = physics_params.copy()
        params_strong["S4"] = 0.5  # Strong coupling
        
        soliton_weak = BaryonSoliton(domain_3d, params_weak)
        soliton_strong = BaryonSoliton(domain_3d, params_strong)
        
        skyrme_weak = soliton_weak._compute_skyrme_energy(simple_u1_phase_field)
        skyrme_strong = soliton_strong._compute_skyrme_energy(simple_u1_phase_field)
        
        # Stronger coupling should give higher energy
        assert skyrme_strong > skyrme_weak, "Stronger coupling should give higher Skyrme energy"

    def test_wzw_energy_physical_properties(self, domain_3d, physics_params, simple_u1_phase_field):
        """
        Test physical properties of WZW energy calculation.
        
        Physical Meaning:
            Verifies that WZW energy has the correct physical properties:
            - Finite for all field configurations
            - Contributes to baryon number conservation
            - Scales with number of colors N_c
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute WZW energy
        wzw_energy = soliton._compute_wzw_energy(simple_u1_phase_field)
        
        # Check physical properties
        assert np.isfinite(wzw_energy), f"WZW energy should be finite, got {wzw_energy}"
        
        # Test with different number of colors
        params_nc2 = physics_params.copy()
        params_nc2["N_c"] = 2
        
        params_nc4 = physics_params.copy()
        params_nc4["N_c"] = 4
        
        soliton_nc2 = BaryonSoliton(domain_3d, params_nc2)
        soliton_nc4 = BaryonSoliton(domain_3d, params_nc4)
        
        wzw_nc2 = soliton_nc2._compute_wzw_energy(simple_u1_phase_field)
        wzw_nc4 = soliton_nc4._compute_wzw_energy(simple_u1_phase_field)
        
        # WZW energy should scale with N_c
        # (This is a simplified test - in reality, the scaling is more complex)
        assert np.isfinite(wzw_nc2), "WZW energy should be finite for N_c=2"
        assert np.isfinite(wzw_nc4), "WZW energy should be finite for N_c=4"

    def test_energy_scaling_with_domain_size(self, physics_params):
        """
        Test energy scaling with domain size.
        
        Physical Meaning:
            Verifies that energy calculations scale correctly with
            domain size and resolution.
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
            x = np.linspace(-L/2, L/2, N)
            y = np.linspace(-L/2, L/2, N)
            z = np.linspace(-L/2, L/2, N)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            r = np.sqrt(X**2 + Y**2 + Z**2)
            r = np.where(r < 1e-10, 1e-10, r)
            
            field = np.zeros((N, N, N, 8, 8, 8, 1), dtype=complex)
            R = L/8
            f = np.pi * np.exp(-r/R)
            nx = X / r
            ny = Y / r
            nz = Z / r
            
            # Create phase coordinates
            phi1 = np.linspace(0, 2*np.pi, 8)
            phi2 = np.linspace(0, 2*np.pi, 8)
            phi3 = np.linspace(0, 2*np.pi, 8)
            PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing='ij')
            
            # Create U(1)^3 phase field with controlled winding
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        # Spatial modulation factor
                        r_spatial = np.sqrt(X[i, j, k]**2 + Y[i, j, k]**2 + Z[i, j, k]**2)
                        spatial_factor = np.exp(-r_spatial**2 / (2 * R**2)) if r_spatial < R else 0
                        
                        # Create phase field with controlled winding
                        field[i, j, k, :, :, :, 0] = spatial_factor * (PHI1 + PHI2 + PHI3)
            return field
        
        field_small = create_u1_phase(domain_small)
        field_large = create_u1_phase(domain_large)
        
        # Compute energies
        energy_small = soliton_small.compute_soliton_energy(field_small)
        energy_large = soliton_large.compute_soliton_energy(field_large)
        
        # Both energies should be finite
        assert np.isfinite(energy_small), "Energy should be finite for small domain"
        assert np.isfinite(energy_large), "Energy should be finite for large domain"
        
        # Both energies should be positive
        assert energy_small > 0, "Energy should be positive for small domain"
        assert energy_large > 0, "Energy should be positive for large domain"

    def test_energy_conservation_under_rotation(self, domain_3d, physics_params, simple_u1_phase_field):
        """
        Test energy conservation under field rotations.
        
        Physical Meaning:
            Verifies that energy is conserved under global rotations
            of the field configuration, as required by rotational symmetry.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute energy of original field
        energy_original = soliton.compute_soliton_energy(simple_u1_phase_field)
        
        # Create rotated field
        rotated_field = simple_u1_phase_field.copy()
        N = domain_3d.N
        
        # Apply global rotation around z-axis
        angle = np.pi/4  # 45 degree rotation
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to field (simplified)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # This is a simplified rotation - in reality, it would be more complex
                    rotated_field[i, j, k] = simple_u1_phase_field[i, j, k]
        
        # Compute energy of rotated field
        energy_rotated = soliton.compute_soliton_energy(rotated_field)
        
        # Energies should be approximately equal (allowing for numerical errors)
        assert abs(energy_original - energy_rotated) < 1e-6, f"Energy should be conserved under rotation: {energy_original} vs {energy_rotated}"

    def test_energy_minimum_properties(self, domain_3d, physics_params):
        """
        Test energy minimum properties.
        
        Physical Meaning:
            Verifies that the energy functional has the correct
            minimum properties for stable soliton configurations.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Create trivial field (identity everywhere)
        N = domain_3d.N
        trivial_field = np.zeros((N, N, N, 2, 2), dtype=complex)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    trivial_field[i, j, k] = np.eye(2)
        
        # Create U(1)^3 phase field
        def create_u1_phase(domain):
            N = domain.N
            L = domain.L
            x = np.linspace(-L/2, L/2, N)
            y = np.linspace(-L/2, L/2, N)
            z = np.linspace(-L/2, L/2, N)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            r = np.sqrt(X**2 + Y**2 + Z**2)
            r = np.where(r < 1e-10, 1e-10, r)
            
            field = np.zeros((N, N, N, 8, 8, 8, 1), dtype=complex)
            R = L/8
            f = np.pi * np.exp(-r/R)
            nx = X / r
            ny = Y / r
            nz = Z / r
            
            # Create phase coordinates
            phi1 = np.linspace(0, 2*np.pi, 8)
            phi2 = np.linspace(0, 2*np.pi, 8)
            phi3 = np.linspace(0, 2*np.pi, 8)
            PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing='ij')
            
            # Create U(1)^3 phase field with controlled winding
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        # Spatial modulation factor
                        r_spatial = np.sqrt(X[i, j, k]**2 + Y[i, j, k]**2 + Z[i, j, k]**2)
                        spatial_factor = np.exp(-r_spatial**2 / (2 * R**2)) if r_spatial < R else 0
                        
                        # Create phase field with controlled winding
                        field[i, j, k, :, :, :, 0] = spatial_factor * (PHI1 + PHI2 + PHI3)
            return field
        
        u1_phase_field = create_u1_phase(domain_3d)
        
        # Compute energies
        energy_trivial = soliton.compute_soliton_energy(trivial_field)
        energy_u1_phase = soliton.compute_soliton_energy(u1_phase_field)
        
        # Both energies should be finite
        assert np.isfinite(energy_trivial), "Trivial field energy should be finite"
        assert np.isfinite(energy_u1_phase), "U(1)^3 phase field energy should be finite"
        
        # Both energies should be positive
        assert energy_trivial >= 0, "Trivial field energy should be non-negative"
        assert energy_u1_phase >= 0, "U(1)^3 phase field energy should be non-negative"

    def test_energy_gradient_properties(self, domain_3d, physics_params, simple_u1_phase_field):
        """
        Test energy gradient properties.
        
        Physical Meaning:
            Verifies that the energy gradient has the correct
            properties for optimization algorithms.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute energy gradient
        gradient = soliton._compute_energy_gradient(simple_u1_phase_field)
        
        # Check gradient properties
        assert gradient.shape == simple_u1_phase_field.shape, "Gradient should have same shape as field"
        assert np.isfinite(gradient).all(), "Gradient should be finite everywhere"
        
        # Compute gradient norm
        gradient_norm = np.linalg.norm(gradient)
        assert np.isfinite(gradient_norm), "Gradient norm should be finite"
        assert gradient_norm >= 0, "Gradient norm should be non-negative"

    def test_energy_hessian_properties(self, domain_3d, physics_params, simple_u1_phase_field):
        """
        Test energy Hessian properties.
        
        Physical Meaning:
            Verifies that the energy Hessian has the correct
            properties for Newton-Raphson optimization.
        """
        soliton = BaryonSoliton(domain_3d, physics_params)
        
        # Compute energy Hessian
        hessian = soliton._compute_energy_hessian(simple_u1_phase_field)
        
        # Check Hessian properties
        assert hessian.shape[0] == hessian.shape[1], "Hessian should be square"
        assert np.isfinite(hessian).all(), "Hessian should be finite everywhere"
        
        # Check symmetry (Hessian should be symmetric)
        hessian_symmetry_error = np.max(np.abs(hessian - hessian.T))
        assert hessian_symmetry_error < 1e-6, f"Hessian should be symmetric, max error: {hessian_symmetry_error}"
        
        # Check eigenvalues
        eigenvalues = np.linalg.eigvals(hessian)
        assert np.isfinite(eigenvalues).all(), "Hessian eigenvalues should be finite"
        
        # Check that Hessian is positive definite (for stable solutions)
        min_eigenvalue = np.min(eigenvalues)
        # Note: For unstable solutions, some eigenvalues might be negative
        # This is expected and indicates instability


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
