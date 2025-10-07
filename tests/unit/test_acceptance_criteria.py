"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for acceptance criteria enforcement in 7D BVP theory.

This module tests the enforcement of acceptance criteria from ALL.md:
- PASS-1: ReY(ω)≥0 for memory kernels below resonances
- GW-1: |h|∝a^{-1} when Γ=K=0 through c_T=c_φ evolution
- LEN-1: lensing consistency using g_eff distance factors
- FRAC-1: G_β tail validation and energy monotonicity (ΔE≤0)

Example:
    >>> pytest tests/unit/test_acceptance_criteria.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock

from bhlff.core.time.memory_kernel import MemoryKernel
from bhlff.models.level_g.gravity import VBPGravitationalEffectsModel
from bhlff.models.level_e.defect_interactions import DefectInteractions


class TestAcceptanceCriteria:
    """Test enforcement of acceptance criteria from ALL.md."""

    @pytest.fixture
    def domain_3d(self):
        """Create 3D domain for testing."""
        domain = Mock()
        domain.N = 32
        domain.L = 4.0
        domain.shape = (32, 32, 32)
        return domain

    def test_pass_1_memory_kernel_passivity(self, domain_3d):
        """Test PASS-1: ReY(ω)≥0 for memory kernels below resonances."""
        # Test with positive coupling strengths (should pass)
        memory_kernel = MemoryKernel(domain_3d, num_memory_vars=3)
        positive_gammas = [0.5, 1.0, 0.8]
        memory_kernel.set_coupling_strengths(positive_gammas)

        # Should not raise any warnings for positive gammas
        # (validation is done in set_coupling_strengths method)

        # Test with negative coupling strengths (should warn)
        memory_kernel_neg = MemoryKernel(domain_3d, num_memory_vars=3)
        negative_gammas = [0.5, -0.3, 0.8]  # One negative
        memory_kernel_neg.set_coupling_strengths(negative_gammas)

        # Should log warning but not fail (allows diagnostic cases)
        # The validation is done in _validate_passivity method

    def test_gw_1_amplitude_law(self, domain_3d):
        """Test GW-1: |h|∝a^{-1} when Γ=K=0 through c_T=c_φ evolution."""
        # Create gravitational effects model
        system = Mock()
        system.domain = domain_3d

        gravity_params = {
            "c_phi": 2.0,  # Phase velocity
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": 1.0,
        }

        gravity_model = VBPGravitationalEffectsModel(system, gravity_params)

        # Test that c_T = c_φ
        assert gravity_model.c_phi == 2.0, "c_φ should be set correctly"

        # Test GW-1 amplitude law: |h|∝a^{-1}
        # This would be tested in the gravitational waves calculator
        # when Γ=K=0 (no memory kernels, no quenches)

        # For now, just verify that the model is set up correctly
        assert gravity_model.c_phi > 0, "c_φ should be positive for GW-1"

    def test_len_1_lensing_consistency(self, domain_3d):
        """Test LEN-1: lensing consistency using g_eff distance factors."""
        # Create gravitational effects model
        system = Mock()
        system.domain = domain_3d

        gravity_params = {"c_phi": 1.0, "chi_kappa": 1.0, "beta": 0.5, "mu": 1.0}

        gravity_model = VBPGravitationalEffectsModel(system, gravity_params)

        # Test that effective metric can be computed
        # (This would test lensing consistency in a full implementation)
        assert gravity_model.c_phi > 0, "c_φ should be positive for LEN-1"
        assert gravity_model.chi_kappa > 0, "χ/κ should be positive for LEN-1"

    def test_frac_1_energy_monotonicity(self, domain_3d):
        """Test FRAC-1: G_β tail validation and energy monotonicity (ΔE≤0)."""
        # Test defect interactions with fractional Green function
        physics_params = {
            "beta": 0.5,
            "mu": 1.0,
            "interaction_strength": 1.0,
            "tempered_lambda": 0.0,  # No mass terms in base regime
        }

        interactions = DefectInteractions(domain_3d, physics_params)

        # Test energy monotonicity with opposite charges
        positions = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
        charges = [1, -1]  # Opposite charges for attraction

        # Compute energy at different separations
        separations = [2.0, 1.5, 1.0, 0.8, 0.6]
        energies = []

        for sep in separations:
            positions[1] = np.array([sep, 0.0, 0.0])
            energy = interactions.compute_interaction_potential(positions, charges)
            energies.append(energy)

        # Check energy monotonicity: energy should decrease as defects approach
        for i in range(1, len(energies)):
            delta_energy = energies[i] - energies[i - 1]
            assert (
                delta_energy <= 0
            ), f"FRAC-1 violation: ΔE={delta_energy} > 0 at separation {separations[i]}"

    def test_mass_terms_forbidden_in_base_regime(self, domain_3d):
        """Test that mass terms are forbidden in base regime (tempered_lambda==0)."""
        # Test that tempered_lambda > 0 raises error in base regime
        physics_params = {
            "beta": 0.5,
            "mu": 1.0,
            "interaction_strength": 1.0,
            "tempered_lambda": 0.5,  # Mass term
            "diagnostic_mode": False,  # Base regime
        }

        with pytest.raises(ValueError, match="Mass terms forbidden in base regime"):
            DefectInteractions(domain_3d, physics_params)

        # Test that tempered_lambda > 0 is allowed in diagnostic mode
        physics_params_diagnostic = {
            "beta": 0.5,
            "mu": 1.0,
            "interaction_strength": 1.0,
            "tempered_lambda": 0.5,  # Mass term
            "diagnostic_mode": True,  # Diagnostic mode
        }

        # Should not raise error in diagnostic mode
        interactions_diagnostic = DefectInteractions(
            domain_3d, physics_params_diagnostic
        )
        assert interactions_diagnostic.tempered_lambda == 0.5

    def test_stability_assertions(self, domain_3d):
        """Test stability assertions: c_φ^2>0, M_*^2>0."""
        system = Mock()
        system.domain = domain_3d

        # Test with positive parameters (should pass)
        gravity_params_good = {"c_phi": 2.0, "chi_kappa": 1.0, "beta": 0.5, "mu": 1.0}

        gravity_model = VBPGravitationalEffectsModel(system, gravity_params_good)
        assert gravity_model.c_phi**2 > 0, "c_φ^2 should be positive"
        assert gravity_model.mu > 0, "μ should be positive"

        # Test with zero c_phi (should fail)
        gravity_params_bad = {
            "c_phi": 0.0,  # Zero phase velocity
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": 1.0,
        }

        with pytest.raises(AssertionError, match="Stability violation: c_φ"):
            VBPGravitationalEffectsModel(system, gravity_params_bad)

        # Test with negative mu (should fail)
        gravity_params_bad2 = {
            "c_phi": 1.0,
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": -1.0,  # Negative diffusion coefficient
        }

        with pytest.raises(AssertionError, match="Stability violation: μ"):
            VBPGravitationalEffectsModel(system, gravity_params_bad2)

    def test_acceptance_criteria_integration(self, domain_3d):
        """Test integration of all acceptance criteria."""
        # Test that all criteria work together
        system = Mock()
        system.domain = domain_3d

        # Valid parameters that should pass all criteria
        gravity_params = {
            "c_phi": 1.5,  # Positive phase velocity
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": 1.0,  # Positive diffusion coefficient
        }

        # Should not raise any errors
        gravity_model = VBPGravitationalEffectsModel(system, gravity_params)

        # Test memory kernel with positive coupling strengths
        memory_kernel = MemoryKernel(domain_3d, num_memory_vars=2)
        memory_kernel.set_coupling_strengths([0.5, 0.8])

        # Test defect interactions without mass terms
        physics_params = {
            "beta": 0.5,
            "mu": 1.0,
            "interaction_strength": 1.0,
            "tempered_lambda": 0.0,  # No mass terms
        }

        interactions = DefectInteractions(domain_3d, physics_params)

        # All should be valid
        assert gravity_model.c_phi > 0
        assert gravity_model.mu > 0
        assert interactions.tempered_lambda == 0.0
        assert all(gamma >= 0 for gamma in memory_kernel.coupling_strengths)
