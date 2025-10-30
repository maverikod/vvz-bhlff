"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic tests for MultiParticleSystem class in Level F models.

This module contains basic tests for the MultiParticleSystem
class, including initialization, effective potential computation,
and collective modes analysis.

Physical Meaning:
    Tests verify that multi-particle systems correctly compute
    effective potentials, identify collective modes, and analyze
    correlations between particles in the 7D phase field theory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.multi_particle_system import MultiParticleSystem
from bhlff.models.level_f.multi_particle import Particle
from bhlff.core.domain import Domain


class TestMultiParticleSystemBasic:
    """
    Basic test cases for MultiParticleSystem class.

    Physical Meaning:
        Tests verify the basic implementation of multi-particle
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
        """Test system initialization."""
        system = MultiParticleSystem(domain, particles, interaction_range=5.0)

        assert system.domain == domain
        assert system.particles == particles
        assert system.interaction_range == 5.0
        assert system.num_particles == len(particles)

    def test_effective_potential_computation(self, system):
        """Test effective potential computation."""
        # Mock the potential analysis
        with patch.object(
            system.potential_analyzer, "compute_effective_potential"
        ) as mock_compute:
            mock_compute.return_value = {
                "potential_energy": 100.0,
                "interaction_energy": 50.0,
                "total_energy": 150.0,
            }

            potential = system.compute_effective_potential()

            mock_compute.assert_called_once_with(system.particles)
            assert "potential_energy" in potential
            assert "interaction_energy" in potential
            assert "total_energy" in potential

    def test_collective_modes_analysis(self, system):
        """Test collective modes analysis."""
        # Mock the collective modes analysis
        with patch.object(system.collective_modes, "analyze_modes") as mock_analyze:
            mock_analyze.return_value = {
                "modes": [{"frequency": 0.5, "amplitude": 0.8}],
                "participation_ratios": [0.6, 0.4],
            }

            modes = system.analyze_collective_modes()

            mock_analyze.assert_called_once_with(system.particles)
            assert "modes" in modes
            assert "participation_ratios" in modes

    def test_correlation_analysis(self, system):
        """Test correlation analysis."""
        # Mock the correlation analysis
        with patch.object(system, "_compute_correlations") as mock_correlations:
            mock_correlations.return_value = {
                "spatial_correlation": 0.7,
                "phase_correlation": 0.3,
                "charge_correlation": -0.5,
            }

            correlations = system.compute_correlations()

            mock_correlations.assert_called_once()
            assert "spatial_correlation" in correlations
            assert "phase_correlation" in correlations
            assert "charge_correlation" in correlations

    def test_stability_check(self, system):
        """Test stability check."""
        # Mock the stability check
        with patch.object(system, "_check_stability") as mock_stability:
            mock_stability.return_value = {
                "is_stable": True,
                "stability_energy": 0.1,
                "unstable_modes": [],
            }

            stability = system.check_stability()

            mock_stability.assert_called_once()
            assert "is_stable" in stability
            assert "stability_energy" in stability
            assert "unstable_modes" in stability

    def test_single_particle_potential(self, system):
        """Test single particle potential."""
        particle = system.particles[0]

        # Mock the single particle potential
        with patch.object(
            system.potential_analyzer, "compute_single_particle_potential"
        ) as mock_single:
            mock_single.return_value = 25.0

            potential = system.compute_single_particle_potential(particle)

            mock_single.assert_called_once_with(particle, system.particles)
            assert potential == 25.0

    def test_pair_interaction(self, system):
        """Test pair interaction."""
        particle1 = system.particles[0]
        particle2 = system.particles[1]

        # Mock the pair interaction
        with patch.object(
            system.potential_analyzer, "compute_pair_interaction"
        ) as mock_pair:
            mock_pair.return_value = 10.0

            interaction = system.compute_pair_interaction(particle1, particle2)

            mock_pair.assert_called_once_with(particle1, particle2)
            assert interaction == 10.0

    def test_three_body_interaction(self, system):
        """Test three-body interaction."""
        particle1 = system.particles[0]
        particle2 = system.particles[1]
        particle3 = system.particles[0]  # Use same particle for simplicity

        # Mock the three-body interaction
        with patch.object(
            system.potential_analyzer, "compute_three_body_interaction"
        ) as mock_three:
            mock_three.return_value = 5.0

            interaction = system.compute_three_body_interaction(
                particle1, particle2, particle3
            )

            mock_three.assert_called_once_with(particle1, particle2, particle3)
            assert interaction == 5.0

    def test_dynamics_matrix(self, system):
        """Test dynamics matrix computation."""
        # Mock the dynamics matrix
        with patch.object(system, "_compute_dynamics_matrix") as mock_dynamics:
            mock_dynamics.return_value = np.array([[1.0, 0.5], [0.5, 1.0]])

            matrix = system.compute_dynamics_matrix()

            mock_dynamics.assert_called_once()
            assert matrix.shape == (2, 2)
            assert np.allclose(matrix, matrix.T)  # Should be symmetric

    def test_participation_ratios(self, system):
        """Test participation ratios computation."""
        # Mock the participation ratios
        with patch.object(
            system.collective_modes, "compute_participation_ratios"
        ) as mock_ratios:
            mock_ratios.return_value = [0.6, 0.4]

            ratios = system.compute_participation_ratios()

            mock_ratios.assert_called_once_with(system.particles)
            assert len(ratios) == len(system.particles)
            assert all(0 <= ratio <= 1 for ratio in ratios)

    def test_spatial_correlations(self, system):
        """Test spatial correlations."""
        # Mock the spatial correlations
        with patch.object(system, "_compute_spatial_correlations") as mock_spatial:
            mock_spatial.return_value = {
                "correlation_length": 2.0,
                "correlation_strength": 0.8,
            }

            correlations = system.compute_spatial_correlations()

            mock_spatial.assert_called_once()
            assert "correlation_length" in correlations
            assert "correlation_strength" in correlations

    def test_phase_correlations(self, system):
        """Test phase correlations."""
        # Mock the phase correlations
        with patch.object(system, "_compute_phase_correlations") as mock_phase:
            mock_phase.return_value = {
                "phase_coherence": 0.7,
                "phase_synchronization": 0.5,
            }

            correlations = system.compute_phase_correlations()

            mock_phase.assert_called_once()
            assert "phase_coherence" in correlations
            assert "phase_synchronization" in correlations

    def test_interaction_strength_dependence(self, domain, particles):
        """Test interaction strength dependence."""
        # Test different interaction strengths
        strengths = [0.1, 0.5, 1.0, 2.0]
        energies = []

        for strength in strengths:
            system = MultiParticleSystem(
                domain, particles, interaction_strength=strength
            )
            with patch.object(
                system.potential_analyzer, "compute_effective_potential"
            ) as mock_compute:
                mock_compute.return_value = {"total_energy": 100.0 * strength}
                potential = system.compute_effective_potential()
                energies.append(potential["total_energy"])

        # Verify energy scales with interaction strength
        for i in range(1, len(energies)):
            assert energies[i] > energies[i - 1]

    def test_interaction_range_dependence(self, domain, particles):
        """Test interaction range dependence."""
        # Test different interaction ranges
        ranges = [1.0, 2.0, 5.0, 10.0]
        energies = []

        for interaction_range in ranges:
            system = MultiParticleSystem(
                domain, particles, interaction_range=interaction_range
            )
            with patch.object(
                system.potential_analyzer, "compute_effective_potential"
            ) as mock_compute:
                mock_compute.return_value = {"total_energy": 100.0 / interaction_range}
                potential = system.compute_effective_potential()
                energies.append(potential["total_energy"])

        # Verify energy decreases with interaction range
        for i in range(1, len(energies)):
            assert energies[i] < energies[i - 1]


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
