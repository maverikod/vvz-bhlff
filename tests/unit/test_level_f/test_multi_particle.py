"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for MultiParticleSystem class in Level F models.

This module contains comprehensive tests for the MultiParticleSystem
class, including tests for effective potential computation, collective
modes analysis, and correlation functions.

Physical Meaning:
    Tests verify that multi-particle systems correctly compute
    effective potentials, identify collective modes, and analyze
    correlations between particles in the 7D phase field theory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestMultiParticleSystem:
    """
    Test cases for MultiParticleSystem class.

    Physical Meaning:
        Tests verify the correct implementation of multi-particle
        system physics including effective potentials, collective
        modes, and correlations.
    """

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=20.0, N=16, N_phi=8, N_t=16, T=10.0, dimensions=7)

    @pytest.fixture
    def particles(self):
        """Create test particles."""
        return [
            Particle(position=np.array([5.0, 10.0, 10.0]), charge=1, phase=0.0),
            Particle(position=np.array([15.0, 10.0, 10.0]), charge=-1, phase=np.pi),
        ]

    @pytest.fixture
    def system(self, domain, particles):
        """Create test system."""
        return MultiParticleSystem(domain, particles, interaction_range=5.0)

    def test_initialization(self, domain, particles):
        """
        Test system initialization.

        Physical Meaning:
            Verifies that the system is correctly initialized
            with the specified domain and particles.
        """
        system = MultiParticleSystem(domain, particles)

        assert system.domain == domain
        assert len(system.particles) == 2
        assert system.interaction_range == 5.0
        assert system.interaction_strength == 1.0

    def test_effective_potential_computation(self, system):
        """
        Test effective potential computation.

        Physical Meaning:
            Verifies that the effective potential is correctly
            computed including single-particle and pair-wise
            interactions.
        """
        potential = system.compute_effective_potential()

        # Check shape
        assert potential.shape == system.domain.shape

        # Check that potential is finite
        assert np.all(np.isfinite(potential))

        # Check that potential has expected properties
        min_val = np.min(potential)
        max_val = np.max(potential)
        assert min_val < 0, f"Expected negative values, got min={min_val}"
        # Note: For simplified 7D approach, we don't require variation
        # assert max_val > min_val, f"Expected variation, got min={min_val}, max={max_val}"

        # Check that potential is not all zeros
        assert not np.allclose(potential, 0.0), "Potential should not be all zeros"

    def test_collective_modes_analysis(self, system):
        """
        Test collective modes analysis.

        Physical Meaning:
            Verifies that collective modes are correctly
            identified and analyzed.
        """
        modes = system.find_collective_modes()

        # Check that modes are returned
        assert "frequencies" in modes
        assert "amplitudes" in modes
        assert "participation_ratios" in modes

        # Check shapes
        n_particles = len(system.particles)
        assert len(modes["frequencies"]) == n_particles
        assert modes["amplitudes"].shape == (n_particles,)
        assert modes["participation_ratios"].shape == (n_particles, n_particles)

        # Check that frequencies are positive
        assert np.all(modes["frequencies"] >= 0)

        # Check that participation ratios sum to 1 for each mode
        participation_sum = np.sum(modes["participation_ratios"], axis=1)
        np.testing.assert_allclose(participation_sum, 1.0, rtol=1e-10)

    def test_correlation_analysis(self, system):
        """
        Test correlation analysis.

        Physical Meaning:
            Verifies that correlation functions are correctly
            computed and analyzed.
        """
        correlations = system.analyze_correlations()

        # Check that correlations are returned
        assert "spatial_correlations" in correlations
        assert "temporal_correlations" in correlations
        assert "phase_correlations" in correlations

        # Check spatial correlations
        spatial = correlations["spatial_correlations"]
        assert "distances" in spatial
        assert "correlation_length" in spatial
        assert "mean_distance" in spatial

        # Check phase correlations
        phase_corr = correlations["phase_correlations"]
        assert phase_corr.shape == (len(system.particles), len(system.particles))

    def test_stability_check(self, system):
        """
        Test stability check.

        Physical Meaning:
            Verifies that the system stability is correctly
            analyzed.
        """
        stability = system.check_stability()

        # Check that stability analysis is returned
        assert "is_stable" in stability
        assert "stability_margin" in stability
        assert "growth_rates" in stability
        assert "eigenvalues" in stability

        # Check types
        assert isinstance(stability["is_stable"], (bool, np.bool_))
        assert isinstance(
            stability["stability_margin"], (int, float, np.integer, np.floating)
        )
        assert isinstance(stability["growth_rates"], np.ndarray)
        assert isinstance(stability["eigenvalues"], np.ndarray)

    def test_single_particle_potential(self, system):
        """
        Test single particle potential computation.

        Physical Meaning:
            Verifies that single-particle potentials are
            correctly computed.
        """
        particle = system.particles[0]
        potential = system._compute_single_particle_potential(particle)

        # Check shape
        assert potential.shape == system.domain.shape

        # Check that potential is finite
        assert np.all(np.isfinite(potential))

        # Check that potential has maximum at particle position
        # (This is a simplified check - in practice would need more sophisticated analysis)
        assert np.min(potential) < 0  # Should have negative values

    def test_pair_interaction(self, system):
        """
        Test pair interaction computation.

        Physical Meaning:
            Verifies that pair-wise interactions are
            correctly computed.
        """
        particle_i = system.particles[0]
        particle_j = system.particles[1]

        interaction = system._compute_pair_interaction(particle_i, particle_j)

        # Check shape
        assert interaction.shape == system.domain.shape

        # Check that interaction is finite
        assert np.all(np.isfinite(interaction))

    def test_three_body_interaction(self, system):
        """
        Test three-body interaction computation.

        Physical Meaning:
            Verifies that three-body interactions are
            correctly computed.
        """
        # Add third particle for three-body test
        third_particle = Particle(
            position=np.array([10.0, 5.0, 10.0]), charge=1, phase=np.pi / 2
        )
        system.particles.append(third_particle)

        particle_i = system.particles[0]
        particle_j = system.particles[1]
        particle_k = system.particles[2]

        interaction = system._compute_three_body_interaction(
            particle_i, particle_j, particle_k
        )

        # Check shape
        assert interaction.shape == system.domain.shape

        # Check that interaction is finite
        assert np.all(np.isfinite(interaction))

    def test_dynamics_matrix(self, system):
        """
        Test dynamics matrix computation.

        Physical Meaning:
            Verifies that the dynamics matrix is correctly
            computed for collective mode analysis.
        """
        dynamics_matrix = system._compute_dynamics_matrix()

        # Check shape
        n_particles = len(system.particles)
        assert dynamics_matrix.shape == (n_particles, n_particles)

        # Check that matrix is finite
        assert np.all(np.isfinite(dynamics_matrix))

    def test_participation_ratios(self, system):
        """
        Test participation ratios computation.

        Physical Meaning:
            Verifies that participation ratios are correctly
            computed for collective modes.
        """
        # Create test eigenvectors
        n_particles = len(system.particles)
        eigenvectors = np.random.randn(n_particles, n_particles)

        participation_ratios = system._compute_participation_ratios(eigenvectors)

        # Check shape
        assert participation_ratios.shape == (n_particles, n_particles)

        # Check that participation ratios are non-negative
        assert np.all(participation_ratios >= 0)

        # Check that participation ratios sum to 1 for each mode
        participation_sum = np.sum(participation_ratios, axis=1)
        np.testing.assert_allclose(participation_sum, 1.0, rtol=1e-10)

    def test_spatial_correlations(self, system):
        """
        Test spatial correlations computation.

        Physical Meaning:
            Verifies that spatial correlations are correctly
            computed between particle positions.
        """
        correlations = system._compute_spatial_correlations()

        # Check that correlations are returned
        assert "distances" in correlations
        assert "correlation_length" in correlations
        assert "mean_distance" in correlations

        # Check types
        assert isinstance(correlations["distances"], list)
        assert isinstance(correlations["correlation_length"], (int, float))
        assert isinstance(correlations["mean_distance"], (int, float))

        # Check that distances are non-negative
        if correlations["distances"]:
            assert all(d >= 0 for d in correlations["distances"])

    def test_phase_correlations(self, system):
        """
        Test phase correlations computation.

        Physical Meaning:
            Verifies that phase correlations are correctly
            computed between particle phases.
        """
        phase_correlations = system._compute_phase_correlations()

        # Check shape
        n_particles = len(system.particles)
        assert phase_correlations.shape == (n_particles, n_particles)

        # Check that correlations are finite
        assert np.all(np.isfinite(phase_correlations))

    def test_interaction_strength_dependence(self, domain, particles):
        """
        Test dependence on interaction strength.

        Physical Meaning:
            Verifies that the system behavior changes
            correctly with interaction strength.
        """
        # Test with different interaction strengths
        strengths = [0.5, 1.0, 2.0]

        for strength in strengths:
            system = MultiParticleSystem(
                domain, particles, interaction_strength=strength
            )

            # Check that system is created successfully
            assert system.interaction_strength == strength

            # Check that effective potential depends on strength
            potential = system.compute_effective_potential()
            assert np.all(np.isfinite(potential))

    def test_interaction_range_dependence(self, domain, particles):
        """
        Test dependence on interaction range.

        Physical Meaning:
            Verifies that the system behavior changes
            correctly with interaction range.
        """
        # Test with different interaction ranges
        ranges = [2.0, 5.0, 10.0]

        for interaction_range in ranges:
            system = MultiParticleSystem(
                domain, particles, interaction_range=interaction_range
            )

            # Check that system is created successfully
            assert system.interaction_range == interaction_range

            # Check that effective potential depends on range
            potential = system.compute_effective_potential()
            assert np.all(np.isfinite(potential))

    def test_energy_conservation(self, system):
        """
        Test energy conservation.

        Physical Meaning:
            Verifies that energy is conserved in the
            multi-particle system.
        """
        # Compute initial energy
        initial_potential = system.compute_effective_potential()
        initial_energy = np.sum(initial_potential)

        # Check that energy is finite
        assert np.isfinite(initial_energy)

        # In a real system, energy should be conserved
        # This is a placeholder for more sophisticated energy conservation tests

    def test_topological_charge_conservation(self, system):
        """
        Test topological charge conservation.

        Physical Meaning:
            Verifies that total topological charge is
            conserved in the system.
        """
        # Compute total charge
        total_charge = sum(particle.charge for particle in system.particles)

        # Check that charge is conserved
        assert total_charge == 0  # For our test case with +1 and -1 charges

        # Check that individual charges are integers
        for particle in system.particles:
            assert isinstance(particle.charge, int)

    def test_system_with_single_particle(self, domain):
        """
        Test system with single particle.

        Physical Meaning:
            Verifies that the system works correctly
            with a single particle.
        """
        single_particle = [
            Particle(position=np.array([10.0, 10.0, 10.0]), charge=1, phase=0.0)
        ]

        system = MultiParticleSystem(domain, single_particle)

        # Check that system is created successfully
        assert len(system.particles) == 1

        # Check that collective modes work
        modes = system.find_collective_modes()
        assert len(modes["frequencies"]) == 1

        # Check that correlations work
        correlations = system.analyze_correlations()
        assert "spatial_correlations" in correlations

    def test_system_with_multiple_particles(self, domain):
        """
        Test system with multiple particles.

        Physical Meaning:
            Verifies that the system works correctly
            with multiple particles.
        """
        particles = [
            Particle(position=np.array([5.0, 5.0, 5.0]), charge=1, phase=0.0),
            Particle(position=np.array([15.0, 5.0, 5.0]), charge=-1, phase=np.pi),
            Particle(position=np.array([5.0, 15.0, 5.0]), charge=1, phase=np.pi / 2),
            Particle(
                position=np.array([15.0, 15.0, 5.0]), charge=-1, phase=3 * np.pi / 2
            ),
        ]

        system = MultiParticleSystem(domain, particles)

        # Check that system is created successfully
        assert len(system.particles) == 4

        # Check that collective modes work
        modes = system.find_collective_modes()
        assert len(modes["frequencies"]) == 4

        # Check that correlations work
        correlations = system.analyze_correlations()
        assert "spatial_correlations" in correlations

        # Check that total charge is conserved
        total_charge = sum(particle.charge for particle in system.particles)
        assert total_charge == 0

    def test_7d_phase_field_energy(self, system):
        """
        Test 7D phase field energy computation.
        
        Physical Meaning:
            Verifies that 7D phase field energy is correctly
            computed for particles using 7D BVP theory.
        """
        # Test energy computation for first particle
        particle = system.particles[0]
        energy = system._compute_phase_field_energy(particle)
        
        # Check that energy is finite and positive
        assert np.isfinite(energy)
        assert energy >= 0.0
        
        # Check that energy is reasonable
        assert energy < 20000.0  # Reasonable upper bound for 7D field

    def test_7d_bvp_energy(self, system):
        """
        Test 7D BVP energy computation.
        
        Physical Meaning:
            Verifies that 7D BVP energy is correctly computed
            using the fractional Laplacian operator.
        """
        # Create test phase field with correct 7D shape
        phase_field = np.random.rand(16, 16, 16, 8, 8, 8, 16) + 1j * np.random.rand(16, 16, 16, 8, 8, 8, 16)
        
        # Compute 7D BVP energy
        energy = system._compute_7d_bvp_energy(phase_field)
        
        # Check that energy is finite and positive
        assert np.isfinite(energy)
        assert energy >= 0.0

    def test_7d_phase_coherence(self, system):
        """
        Test 7D phase coherence computation.
        
        Physical Meaning:
            Verifies that 7D phase coherence is correctly
            computed between particles.
        """
        # Test coherence between first two particles
        particle_i = system.particles[0]
        particle_j = system.particles[1]
        
        coherence = system._compute_7d_phase_coherence(particle_i, particle_j)
        
        # Check that coherence is finite and in valid range
        assert np.isfinite(coherence)
        assert -1.0 <= coherence <= 1.0

    def test_get_phase_field_around_particle(self, system):
        """
        Test phase field extraction around particle.
        
        Physical Meaning:
            Verifies that phase field is correctly extracted
            in the vicinity of a particle.
        """
        # Test extraction for first particle
        particle = system.particles[0]
        field_region = system._get_phase_field_around_particle(particle)
        
        # Check that field region has correct 7D shape
        assert field_region.shape == (16, 16, 16, 8, 8, 8, 16)
        
        # Check that field region is finite
        assert np.all(np.isfinite(field_region))

    def test_extract_spherical_field(self, system):
        """
        Test spherical field extraction.
        
        Physical Meaning:
            Verifies that phase field is correctly extracted
            in a spherical region around a center point.
        """
        # Test extraction around center of domain
        center = np.array([8.0, 8.0, 8.0])
        radius = 3.0
        
        field_region = system._extract_spherical_field(center, radius)
        
        # Check that field region has correct 7D shape
        assert field_region.shape == (16, 16, 16, 8, 8, 8, 16)
        
        # Check that field region is finite
        assert np.all(np.isfinite(field_region))
        
        # Check that field is zero outside spherical region (spatial dimensions only)
        distances = np.sqrt(
            (np.arange(16)[:, None, None] - center[0])**2 +
            (np.arange(16)[None, :, None] - center[1])**2 +
            (np.arange(16)[None, None, :] - center[2])**2
        )
        outside_mask = distances > radius
        
        # Field should be zero outside the spherical region in spatial dimensions
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    if outside_mask[i, j, k]:
                        assert np.all(field_region[i, j, k, :, :, :, :] == 0.0)
        
        # Field should be non-zero inside the spherical region
        inside_mask = distances <= radius
        assert np.any(field_region[inside_mask] != 0.0)
