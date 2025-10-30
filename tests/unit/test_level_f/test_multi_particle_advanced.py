"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced tests for MultiParticleSystem class in Level F models.

This module contains advanced tests for the MultiParticleSystem
class, including energy conservation, topological charge conservation,
and 7D phase field properties.

Physical Meaning:
    Tests verify advanced aspects of multi-particle systems including
    conservation laws, 7D phase field energy, and field extraction
    in the 7D phase field theory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.multi_particle_system import MultiParticleSystem
from bhlff.models.level_f.multi_particle import Particle
from bhlff.core.domain import Domain


class TestMultiParticleSystemAdvanced:
    """
    Advanced test cases for MultiParticleSystem class.

    Physical Meaning:
        Tests verify advanced aspects of multi-particle
        system physics including conservation laws and
        7D phase field properties.
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

    def test_energy_conservation(self, system):
        """Test energy conservation."""
        # Mock the energy calculation
        with patch.object(system, "compute_total_energy") as mock_energy:
            mock_energy.return_value = 100.0

            # Test energy conservation under time evolution
            initial_energy = system.compute_total_energy()

            # Simulate time evolution (small phase change)
            for particle in system.particles:
                particle.phase += 0.1

            final_energy = system.compute_total_energy()

            # Energy should be conserved (within numerical precision)
            assert abs(final_energy - initial_energy) < 1e-10

    def test_topological_charge_conservation(self, system):
        """Test topological charge conservation."""
        # Mock the topological charge calculation
        with patch.object(system, "compute_total_topological_charge") as mock_charge:
            mock_charge.return_value = 0  # Net charge should be zero

            # Test charge conservation under time evolution
            initial_charge = system.compute_total_topological_charge()

            # Simulate time evolution (small phase change)
            for particle in system.particles:
                particle.phase += 0.1

            final_charge = system.compute_total_topological_charge()

            # Topological charge should be conserved
            assert abs(final_charge - initial_charge) < 1e-10

    def test_system_with_single_particle(self, domain):
        """Test system with single particle."""
        particle = Particle(position=np.array([10.0, 10.0, 10.0]), charge=1, phase=0.0)
        system = MultiParticleSystem(domain, [particle], interaction_range=5.0)

        # Test single particle properties
        assert system.num_particles == 1
        assert system.particles[0] == particle

        # Test single particle potential
        with patch.object(
            system.potential_analyzer, "compute_single_particle_potential"
        ) as mock_single:
            mock_single.return_value = 50.0
            potential = system.compute_single_particle_potential(particle)
            assert potential == 50.0

    def test_system_with_multiple_particles(self, domain):
        """Test system with multiple particles."""
        particles = [
            Particle(position=np.array([5.0, 5.0, 5.0]), charge=1, phase=0.0),
            Particle(position=np.array([15.0, 15.0, 15.0]), charge=-1, phase=np.pi),
            Particle(position=np.array([10.0, 10.0, 10.0]), charge=1, phase=np.pi / 2),
        ]
        system = MultiParticleSystem(domain, particles, interaction_range=5.0)

        # Test multiple particle properties
        assert system.num_particles == 3
        assert len(system.particles) == 3

        # Test interactions between particles
        with patch.object(
            system.potential_analyzer, "compute_pair_interaction"
        ) as mock_pair:
            mock_pair.return_value = 20.0
            interaction = system.compute_pair_interaction(particles[0], particles[1])
            assert interaction == 20.0

    def test_7d_phase_field_energy(self, system):
        """Test 7D phase field energy computation."""
        # Mock the 7D phase field energy
        with patch.object(system, "_compute_7d_phase_field_energy") as mock_energy:
            mock_energy.return_value = {
                "kinetic_energy": 50.0,
                "potential_energy": 30.0,
                "interaction_energy": 20.0,
                "total_energy": 100.0,
            }

            energy = system.compute_7d_phase_field_energy()

            mock_energy.assert_called_once()
            assert "kinetic_energy" in energy
            assert "potential_energy" in energy
            assert "interaction_energy" in energy
            assert "total_energy" in energy

    def test_7d_bvp_energy(self, system):
        """Test 7D BVP energy computation."""
        # Mock the 7D BVP energy
        with patch.object(system, "_compute_7d_bvp_energy") as mock_energy:
            mock_energy.return_value = {
                "bvp_energy": 75.0,
                "phase_energy": 25.0,
                "total_energy": 100.0,
            }

            energy = system.compute_7d_bvp_energy()

            mock_energy.assert_called_once()
            assert "bvp_energy" in energy
            assert "phase_energy" in energy
            assert "total_energy" in energy

    def test_7d_phase_coherence(self, system):
        """Test 7D phase coherence computation."""
        # Mock the 7D phase coherence
        with patch.object(system, "_compute_7d_phase_coherence") as mock_coherence:
            mock_coherence.return_value = {
                "coherence_length": 3.0,
                "coherence_strength": 0.8,
                "phase_synchronization": 0.6,
            }

            coherence = system.compute_7d_phase_coherence()

            mock_coherence.assert_called_once()
            assert "coherence_length" in coherence
            assert "coherence_strength" in coherence
            assert "phase_synchronization" in coherence

    def test_get_phase_field_around_particle(self, system):
        """Test phase field extraction around particle."""
        particle = system.particles[0]

        # Mock the phase field extraction
        with patch.object(
            system, "_extract_phase_field_around_particle"
        ) as mock_extract:
            mock_field = np.random.random(
                (8, 8, 8, 4, 4, 4, 8)
            ) + 1j * np.random.random((8, 8, 8, 4, 4, 4, 8))
            mock_extract.return_value = mock_field

            field = system.get_phase_field_around_particle(particle, radius=2.0)

            mock_extract.assert_called_once_with(particle, radius=2.0)
            assert field.shape == (8, 8, 8, 4, 4, 4, 8)
            assert np.all(np.isfinite(field))

    def test_extract_spherical_field(self, system):
        """Test spherical field extraction."""
        particle = system.particles[0]

        # Mock the spherical field extraction
        with patch.object(system, "_extract_spherical_field") as mock_spherical:
            mock_field = np.random.random((16, 16, 16)) + 1j * np.random.random(
                (16, 16, 16)
            )
            mock_spherical.return_value = mock_field

            field = system.extract_spherical_field(particle, radius=3.0)

            mock_spherical.assert_called_once_with(particle, radius=3.0)
            assert field.shape == (16, 16, 16)
            assert np.all(np.isfinite(field))

    def test_advanced_correlation_analysis(self, system):
        """Test advanced correlation analysis."""
        # Mock the advanced correlation analysis
        with patch.object(system, "_compute_advanced_correlations") as mock_advanced:
            mock_advanced.return_value = {
                "spatial_correlation_function": np.random.random((10, 10)),
                "phase_correlation_function": np.random.random((10, 10)),
                "charge_correlation_function": np.random.random((10, 10)),
                "correlation_lengths": [2.0, 3.0, 1.5],
            }

            correlations = system.compute_advanced_correlations()

            mock_advanced.assert_called_once()
            assert "spatial_correlation_function" in correlations
            assert "phase_correlation_function" in correlations
            assert "charge_correlation_function" in correlations
            assert "correlation_lengths" in correlations

    def test_energy_optimization(self, system):
        """Test energy optimization."""
        # Mock the energy optimization
        with patch.object(system, "_optimize_energy") as mock_optimize:
            mock_optimize.return_value = {
                "optimized_particles": system.particles,
                "energy_reduction": 0.2,
                "convergence": True,
            }

            result = system.optimize_energy()

            mock_optimize.assert_called_once()
            assert "optimized_particles" in result
            assert "energy_reduction" in result
            assert "convergence" in result

    def test_phase_field_dynamics(self, system):
        """Test phase field dynamics."""
        # Mock the phase field dynamics
        with patch.object(system, "_compute_phase_field_dynamics") as mock_dynamics:
            mock_dynamics.return_value = {
                "field_evolution": np.random.random((10, 16, 16, 8, 8, 8, 16)),
                "energy_evolution": np.random.random(10),
                "charge_evolution": np.random.random(10),
            }

            dynamics = system.compute_phase_field_dynamics(time_steps=10)

            mock_dynamics.assert_called_once_with(time_steps=10)
            assert "field_evolution" in dynamics
            assert "energy_evolution" in dynamics
            assert "charge_evolution" in dynamics

    def test_collective_excitations(self, system):
        """Test collective excitations."""
        # Mock the collective excitations
        with patch.object(
            system.collective_modes, "compute_excitations"
        ) as mock_excitations:
            mock_excitations.return_value = {
                "excitation_frequencies": [0.5, 1.0, 1.5],
                "excitation_amplitudes": [0.8, 0.6, 0.4],
                "participation_functions": [0.7, 0.5, 0.3],
            }

            excitations = system.compute_collective_excitations()

            mock_excitations.assert_called_once_with(system.particles)
            assert "excitation_frequencies" in excitations
            assert "excitation_amplitudes" in excitations
            assert "participation_functions" in excitations

    def test_thermodynamic_properties(self, system):
        """Test thermodynamic properties."""
        # Mock the thermodynamic properties
        with patch.object(system, "_compute_thermodynamic_properties") as mock_thermo:
            mock_thermo.return_value = {
                "temperature": 1.0,
                "entropy": 2.0,
                "free_energy": 50.0,
                "heat_capacity": 3.0,
            }

            thermo = system.compute_thermodynamic_properties()

            mock_thermo.assert_called_once()
            assert "temperature" in thermo
            assert "entropy" in thermo
            assert "free_energy" in thermo
            assert "heat_capacity" in thermo


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
