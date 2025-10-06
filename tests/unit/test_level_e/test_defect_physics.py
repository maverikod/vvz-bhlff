"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for defect models in Level E experiments.

This module contains comprehensive physical tests for topological
defect models, verifying their behavior against theoretical predictions
and physical constraints.

Theoretical Background:
    Tests verify the physical correctness of defect implementations
    including topological charge conservation, interaction forces,
    and annihilation processes.

Example:
    >>> pytest tests/unit/test_level_e/test_defect_physics.py -v
"""

import pytest
import numpy as np
from bhlff.core.domain.domain import Domain
from bhlff.models.level_e.defect_models import (
    DefectModel, VortexDefect, MultiDefectSystem,
    DefectDynamics, DefectInteractions
)


class TestDefectPhysics:
    """
    Physical tests for defect models.
    
    Tests verify the physical correctness of defect implementations
    against known theoretical results and physical constraints.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(
            L=4.0,  # Domain size
            N=64,   # Grid points
            dimensions=7
        )

    @pytest.fixture
    def physics_params(self):
        """Create realistic physics parameters."""
        return {
            "mu": 1.0,
            "beta": 1.0,
            "lambda": 0.0,
            "interaction_strength": 1.0,
            "interaction_range": 1.0,
            "screening_length": 0.5,
            "cutoff_radius": 0.1,
            "coherence_length": 0.5,
            "core_radius": 0.1,
            "circulation": 1,
            "annihilation_radius": 0.2,
            "defect_mass": 1.0,
            "gyroscopic_coefficient": 1.0,
            "dissipation_coefficient": 0.1,
            "time_step": 0.01
        }

    def test_vortex_defect_creation(self, domain_7d, physics_params):
        """
        Test vortex defect creation with physical properties.
        
        Physical Meaning:
            Verifies that vortex defects are created with correct
            topological charge and phase winding properties.
        """
        vortex = VortexDefect(domain_7d, physics_params)
        
        # Test position
        position = np.array([2.0, 2.0, 2.0])
        charge = 1
        
        # Create vortex field
        field = vortex.create_defect(position, charge)
        
        # Check field properties
        assert field.shape == (64, 64, 64), "Field should have correct shape"
        assert np.iscomplexobj(field), "Field should be complex"
        
        # Check amplitude profile
        amplitude = np.abs(field)
        assert np.all(amplitude >= 0), "Amplitude should be non-negative"
        assert np.all(amplitude <= 1), "Amplitude should be bounded"
        
        # Check phase winding
        phase = np.angle(field)
        assert np.all(np.isfinite(phase)), "Phase should be finite"

    def test_topological_charge_calculation(self, domain_7d, physics_params):
        """
        Test topological charge calculation.
        
        Physical Meaning:
            Verifies that topological charge is correctly calculated
            for different defect configurations.
        """
        vortex = VortexDefect(domain_7d, physics_params)
        
        # Create vortex with charge +1
        position = np.array([2.0, 2.0, 2.0])
        field = vortex.create_defect(position, 1)
        
        # Calculate topological charge
        calculated_charge = vortex.compute_defect_charge(field, position)
        
        # Check charge calculation
        assert abs(calculated_charge - 1) < 0.1, f"Charge should be 1, got {calculated_charge}"

    def test_defect_interaction_forces(self, domain_7d, physics_params):
        """
        Test interaction forces between defects.
        
        Physical Meaning:
            Verifies that interaction forces follow expected
            physical behavior for different charge configurations.
        """
        interactions = DefectInteractions(domain_7d, physics_params)
        
        # Test two defects with opposite charges (should attract)
        positions = [
            np.array([1.0, 2.0, 2.0]),
            np.array([3.0, 2.0, 2.0])
        ]
        charges = [1, -1]
        
        forces = interactions.compute_interaction_forces(positions, charges)
        
        # Check force properties
        assert len(forces) == 2, "Should have forces for both defects"
        assert all(isinstance(f, np.ndarray) for f in forces), "Forces should be arrays"
        assert all(f.shape == (3,) for f in forces), "Forces should be 3D vectors"
        
        # Forces should be opposite (Newton's third law)
        force_diff = np.linalg.norm(forces[0] + forces[1])
        assert force_diff < 1e-10, "Forces should sum to zero (Newton's third law)"

    def test_defect_annihilation_physics(self, domain_7d, physics_params):
        """
        Test defect annihilation process.
        
        Physical Meaning:
            Verifies that defect-antidefect annihilation
            follows correct physical behavior.
        """
        interactions = DefectInteractions(domain_7d, physics_params)
        
        # Test annihilation of opposite charges
        positions = [
            np.array([1.0, 2.0, 2.0]),
            np.array([1.1, 2.0, 2.0])  # Close together
        ]
        charges = [1, -1]
        
        result = interactions.simulate_defect_annihilation([0, 1], positions, charges)
        
        # Check annihilation results
        assert "annihilated" in result, "Result should contain annihilation status"
        assert "annihilation_energy" in result, "Result should contain energy"
        assert "energy_release_rate" in result, "Result should contain release rate"
        
        # Should annihilate for opposite charges close together
        if result["annihilated"]:
            assert result["annihilation_energy"] > 0, "Annihilation should release energy"
            assert result["energy_release_rate"] > 0, "Energy release rate should be positive"

    def test_multi_defect_system_physics(self, domain_7d, physics_params):
        """
        Test multi-defect system behavior.
        
        Physical Meaning:
            Verifies that multi-defect systems exhibit correct
            collective behavior and energy conservation.
        """
        # Create multi-defect system
        initial_defects = [
            {"position": np.array([1.0, 2.0, 2.0]), "charge": 1},
            {"position": np.array([3.0, 2.0, 2.0]), "charge": -1}
        ]
        
        system = MultiDefectSystem(domain_7d, physics_params, initial_defects)
        
        # Test system properties
        assert len(system.defects) == 2, "Should have 2 defects"
        
        # Test interaction forces
        forces = system.compute_interaction_forces()
        assert len(forces) == 2, "Should have forces for both defects"
        
        # Test system energy
        energy = system.get_system_energy()
        assert isinstance(energy, (int, float)), "Energy should be numeric"
        assert np.isfinite(energy), "Energy should be finite"

    def test_defect_dynamics_physics(self, domain_7d, physics_params):
        """
        Test defect dynamics and motion.
        
        Physical Meaning:
            Verifies that defect motion follows the Thiele equation
            with correct force calculations and dynamics.
        """
        dynamics = DefectDynamics(domain_7d, physics_params)
        
        # Test initial position
        initial_position = np.array([2.0, 2.0, 2.0])
        time_steps = 10
        
        # Create simple field for testing
        field = np.ones((64, 64, 64), dtype=complex)
        
        # Simulate motion
        result = dynamics.simulate_defect_motion(initial_position, time_steps, field)
        
        # Check results
        assert "positions" in result, "Should contain positions"
        assert "velocities" in result, "Should contain velocities"
        assert "forces" in result, "Should contain forces"
        
        positions = result["positions"]
        velocities = result["velocities"]
        forces = result["forces"]
        
        # Check array shapes
        assert positions.shape == (time_steps, 3), "Positions should have correct shape"
        assert velocities.shape == (time_steps, 3), "Velocities should have correct shape"
        assert forces.shape == (time_steps, 3), "Forces should have correct shape"
        
        # Check physical properties
        assert np.all(np.isfinite(positions)), "Positions should be finite"
        assert np.all(np.isfinite(velocities)), "Velocities should be finite"
        assert np.all(np.isfinite(forces)), "Forces should be finite"

    def test_interaction_potential_setup(self, domain_7d, physics_params):
        """
        Test interaction potential setup.
        
        Physical Meaning:
            Verifies that interaction potentials are correctly
            initialized with proper physical parameters.
        """
        interactions = DefectInteractions(domain_7d, physics_params)
        
        # Check that parameters are set
        assert hasattr(interactions, 'interaction_strength'), "Should have interaction strength"
        assert hasattr(interactions, 'interaction_range'), "Should have interaction range"
        assert hasattr(interactions, 'screening_length'), "Should have screening length"
        assert hasattr(interactions, 'cutoff_radius'), "Should have cutoff radius"
        
        # Check parameter values
        assert interactions.interaction_strength > 0, "Interaction strength should be positive"
        assert interactions.interaction_range > 0, "Interaction range should be positive"
        assert interactions.screening_length > 0, "Screening length should be positive"
        assert interactions.cutoff_radius > 0, "Cutoff radius should be positive"

    def test_green_function_physics(self, domain_7d, physics_params):
        """
        Test Green function calculations.
        
        Physical Meaning:
            Verifies that Green functions follow correct
            physical behavior for defect interactions.
        """
        interactions = DefectInteractions(domain_7d, physics_params)
        
        # Test Green function at different distances
        distances = [0.1, 0.5, 1.0, 2.0]
        
        for r in distances:
            green_value, green_gradient = interactions._compute_green_function(r)
            
            # Check Green function properties
            assert np.isfinite(green_value), f"Green function should be finite at r={r}"
            assert np.isfinite(green_gradient), f"Green gradient should be finite at r={r}"
            
            # Green function should decrease with distance
            if r > interactions.cutoff_radius:
                assert green_value > 0, f"Green function should be positive at r={r}"
                assert green_gradient < 0, f"Green gradient should be negative at r={r}"

    def test_defect_charge_conservation(self, domain_7d, physics_params):
        """
        Test topological charge conservation.
        
        Physical Meaning:
            Verifies that topological charge is conserved
            during defect interactions and dynamics.
        """
        vortex = VortexDefect(domain_7d, physics_params)
        
        # Create vortex with specific charge
        position = np.array([2.0, 2.0, 2.0])
        charge = 2
        field = vortex.create_defect(position, charge)
        
        # Calculate charge
        calculated_charge = vortex.compute_defect_charge(field, position)
        
        # Charge should be conserved
        assert abs(calculated_charge - charge) < 0.1, f"Charge should be conserved: {charge} vs {calculated_charge}"

    def test_defect_energy_scaling(self, domain_7d, physics_params):
        """
        Test energy scaling with defect parameters.
        
        Physical Meaning:
            Verifies that defect energies scale correctly
            with physical parameters.
        """
        # Test with different interaction strengths
        strengths = [0.5, 1.0, 2.0]
        energies = []
        
        for strength in strengths:
            params = physics_params.copy()
            params["interaction_strength"] = strength
            
            interactions = DefectInteractions(domain_7d, params)
            
            # Create two defects
            positions = [
                np.array([1.0, 2.0, 2.0]),
                np.array([3.0, 2.0, 2.0])
            ]
            charges = [1, -1]
            
            energy = interactions.compute_interaction_potential(positions, charges)
            energies.append(energy)
        
        # Energy should scale with interaction strength
        for i in range(1, len(energies)):
            ratio = energies[i] / energies[0]
            expected_ratio = strengths[i] / strengths[0]
            assert abs(ratio - expected_ratio) < 0.1, f"Energy should scale with strength: {ratio} vs {expected_ratio}"
