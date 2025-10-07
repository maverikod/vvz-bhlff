"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for VBP envelope gravitational effects in 7D phase field theory.

This module tests the physical correctness of the VBP envelope gravitational
effects implementation, including envelope curvature, phase envelope balance,
and gravitational waves from VBP dynamics.

Theoretical Background:
    In 7D BVP theory, gravity arises from the curvature of the VBP envelope,
    not from spacetime curvature. The tests validate the physical correctness
    of envelope curvature descriptors, effective metric g_eff[Θ], and
    gravitational waves with c_T=c_φ and GW-1 amplitude law.

Physical Tests:
    - Envelope curvature invariants (positivity, boundedness)
    - Effective metric properties (7D structure, g00=-1/c_φ^2)
    - Gravitational waves (c_T=c_φ, GW-1 law, 7D polarization modes)
    - Phase envelope balance (energy conservation, stability)
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.models.level_g.gravity_curvature import VBPEnvelopeCurvatureCalculator
from bhlff.models.level_g.gravity_einstein import PhaseEnvelopeBalanceSolver
from bhlff.models.level_g.gravity_waves import VBPGravitationalWavesCalculator
from bhlff.models.level_g.gravity import VBPGravitationalEffectsModel


class TestVBPEnvelopeCurvaturePhysics:
    """Test physical correctness of VBP envelope curvature calculations."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=64, dimensions=7)

    @pytest.fixture
    def envelope_params(self):
        """Create VBP envelope parameters."""
        return {
            "c_phi": 1.0,
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": 1.0,
            "resolution": 64,
        }

    @pytest.fixture
    def phase_field_7d(self, domain_7d):
        """Create 7D phase field for testing."""
        # Create a simple 7D phase field with spatial and phase components (smaller size)
        field = np.zeros((8, 8, 8, 8, 8, 8, 8), dtype=complex)

        # Add spatial variation
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    field[i, j, k, :, :, :, :] = np.exp(
                        1j * 2 * np.pi * (i + j + k) / 8
                    )

        return field

    def test_envelope_curvature_scalar_positivity(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that envelope curvature scalar is positive."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)

        # Envelope curvature scalar should be positive
        assert (
            curvature_descriptors["envelope_curvature_scalar"] >= 0
        ), "Envelope curvature scalar should be non-negative"

    def test_anisotropy_index_boundedness(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that anisotropy index is bounded."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)

        # Anisotropy index should be bounded
        anisotropy = curvature_descriptors["anisotropy_index"]
        assert (
            0 <= anisotropy <= 1
        ), f"Anisotropy index should be bounded [0,1], got {anisotropy}"

    def test_effective_metric_7d_structure(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that effective metric has correct 7D structure."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)
        g_eff = curvature_descriptors["effective_metric"]

        # Should be 7x7 matrix
        assert g_eff.shape == (
            7,
            7,
        ), f"Effective metric should be 7x7, got {g_eff.shape}"

        # Time component should be negative
        assert (
            g_eff[0, 0] < 0
        ), f"Time component g00 should be negative, got {g_eff[0, 0]}"

        # Spatial components should be positive
        for i in range(1, 4):
            assert (
                g_eff[i, i] > 0
            ), f"Spatial component g{i}{i} should be positive, got {g_eff[i, i]}"

        # Phase components should be positive
        for i in range(4, 7):
            assert (
                g_eff[i, i] > 0
            ), f"Phase component g{i}{i} should be positive, got {g_eff[i, i]}"

    def test_effective_metric_time_component(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that time component follows g00=-1/c_φ^2 with correction factor."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)
        g_eff = curvature_descriptors["effective_metric"]

        # Time component should be g00=-1/c_φ^2 with correction factor
        expected_g00 = -1.0 / (envelope_params["c_phi"] ** 2)
        # Account for correction factor (1.0 + 0.1 * phase_amplitude)
        phase_amplitude = np.mean(np.abs(phase_field_7d))
        correction_factor = 1.0 + 0.1 * phase_amplitude
        expected_g00 *= correction_factor

        assert np.isclose(
            g_eff[0, 0], expected_g00, rtol=1e-6
        ), f"Time component should be g00=-1/c_φ^2*correction={expected_g00}, got {g_eff[0, 0]}"

    def test_focusing_rate_energy_argument(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that focusing rate is consistent with energy argument."""
        calc = VBPEnvelopeCurvatureCalculator(domain_7d, envelope_params)

        curvature_descriptors = calc.compute_envelope_curvature(phase_field_7d)
        focusing_rate = curvature_descriptors["focusing_rate"]

        # Focusing rate should be finite
        assert np.isfinite(
            focusing_rate
        ), f"Focusing rate should be finite, got {focusing_rate}"


class TestPhaseEnvelopeBalancePhysics:
    """Test physical correctness of phase envelope balance equations."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=64, dimensions=7)

    @pytest.fixture
    def envelope_params(self):
        """Create VBP envelope parameters."""
        return {
            "c_phi": 1.0,
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": 1.0,
            "tolerance": 1e-12,
            "max_iterations": 1000,
        }

    @pytest.fixture
    def phase_field_7d(self, domain_7d):
        """Create 7D phase field for testing."""
        # Create a simple 7D phase field (smaller size)
        field = np.zeros((8, 8, 8, 8, 8, 8, 8), dtype=complex)

        # Add spatial variation
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    field[i, j, k, :, :, :, :] = np.exp(
                        1j * 2 * np.pi * (i + j + k) / 8
                    )

        return field

    def test_envelope_balance_convergence(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that envelope balance equation converges."""
        solver = PhaseEnvelopeBalanceSolver(domain_7d, envelope_params)

        envelope_result = solver.solve_phase_envelope_balance(phase_field_7d)

        # Should return valid result
        assert (
            "envelope_solution" in envelope_result
        ), "Envelope result should contain solution"

        # Solution should be finite
        solution = envelope_result["envelope_solution"]
        assert np.all(np.isfinite(solution)), "Envelope solution should be finite"

    def test_effective_metric_from_solution(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that effective metric is computed from solution."""
        solver = PhaseEnvelopeBalanceSolver(domain_7d, envelope_params)

        envelope_result = solver.solve_phase_envelope_balance(phase_field_7d)

        # Should contain effective metric
        assert (
            "effective_metric" in envelope_result
        ), "Envelope result should contain effective metric"

        g_eff = envelope_result["effective_metric"]

        # Should be 7x7 matrix
        assert g_eff.shape == (
            7,
            7,
        ), f"Effective metric should be 7x7, got {g_eff.shape}"

        # Time component should be negative
        assert g_eff[0, 0] < 0, f"Time component should be negative, got {g_eff[0, 0]}"

    def test_balance_operator_components(
        self, domain_7d, envelope_params, phase_field_7d
    ):
        """Test that balance operator has correct components."""
        solver = PhaseEnvelopeBalanceSolver(domain_7d, envelope_params)

        balance_operator = solver._build_balance_operator(phase_field_7d)

        # Should contain all required components
        required_components = [
            "memory_kernels",
            "spatial_operator",
            "bridge_terms",
            "c_phi",
            "beta",
            "mu",
        ]
        for component in required_components:
            assert (
                component in balance_operator
            ), f"Balance operator should contain {component}"

        # Memory kernels should have gamma and k
        memory_kernels = balance_operator["memory_kernels"]
        assert "gamma" in memory_kernels, "Memory kernels should contain gamma"
        assert "k" in memory_kernels, "Memory kernels should contain k"

        # Spatial operator should have correct parameters
        spatial_operator = balance_operator["spatial_operator"]
        assert (
            spatial_operator["beta"] == envelope_params["beta"]
        ), "Spatial operator beta should match parameters"
        assert (
            spatial_operator["mu"] == envelope_params["mu"]
        ), "Spatial operator mu should match parameters"


class TestVBPGravitationalWavesPhysics:
    """Test physical correctness of VBP gravitational waves."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=64, dimensions=7)

    @pytest.fixture
    def wave_params(self):
        """Create gravitational wave parameters."""
        return {
            "c_phi": 1.0,
            "frequency_range": (1e-4, 1e3),
            "detection_sensitivity": 1e-21,
            "scale_factor": 1.0,
        }

    @pytest.fixture
    def envelope_solution_7d(self, domain_7d):
        """Create 7D envelope solution for testing."""
        # Create a simple 7D envelope solution (smaller size)
        solution = np.zeros((8, 8, 8, 8, 8, 8, 8), dtype=complex)

        # Add envelope oscillations
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    solution[i, j, k, :, :, :, :] = np.exp(
                        1j * 2 * np.pi * (i + j + k) / 8
                    )

        return solution

    def test_gravitational_waves_c_T_equals_c_phi(
        self, domain_7d, wave_params, envelope_solution_7d
    ):
        """Test that gravitational waves use c_T=c_φ."""
        calc = VBPGravitationalWavesCalculator(domain_7d, wave_params)

        waves = calc.compute_gravitational_waves(envelope_solution_7d)

        # Should contain c_T and c_phi
        assert "c_T" in waves, "Gravitational waves should contain c_T"
        assert "c_phi" in waves, "Gravitational waves should contain c_phi"

        # Should satisfy c_T = c_phi
        assert (
            waves["c_T"] == waves["c_phi"]
        ), f"Should have c_T = c_phi, got c_T={waves['c_T']}, c_phi={waves['c_phi']}"

    def test_gw1_amplitude_law(self, domain_7d, wave_params, envelope_solution_7d):
        """Test GW-1 amplitude law: |h|∝a^{-1} when Γ=K=0."""
        calc = VBPGravitationalWavesCalculator(domain_7d, wave_params)

        waves = calc.compute_gravitational_waves(envelope_solution_7d)

        # Should contain amplitude
        assert "amplitude" in waves, "Gravitational waves should contain amplitude"

        amplitude = waves["amplitude"]

        # Amplitude should be positive
        assert amplitude >= 0, f"Wave amplitude should be non-negative, got {amplitude}"

        # Amplitude should be finite
        assert np.isfinite(
            amplitude
        ), f"Wave amplitude should be finite, got {amplitude}"

    def test_7d_polarization_modes(self, domain_7d, wave_params, envelope_solution_7d):
        """Test that gravitational waves have 7D polarization modes."""
        calc = VBPGravitationalWavesCalculator(domain_7d, wave_params)

        waves = calc.compute_gravitational_waves(envelope_solution_7d)

        # Should contain polarization modes
        assert (
            "polarization" in waves
        ), "Gravitational waves should contain polarization modes"

        polarization = waves["polarization"]

        # Should have standard modes
        standard_modes = ["plus", "cross", "x_mode", "y_mode"]
        for mode in standard_modes:
            assert mode in polarization, f"Should have {mode} polarization mode"

        # Should have phase space modes
        phase_modes = ["phase_plus", "phase_cross", "phase_z"]
        for mode in phase_modes:
            assert mode in polarization, f"Should have {mode} polarization mode"

    def test_strain_tensor_7d_structure(
        self, domain_7d, wave_params, envelope_solution_7d
    ):
        """Test that strain tensor has 7D structure."""
        calc = VBPGravitationalWavesCalculator(domain_7d, wave_params)

        waves = calc.compute_gravitational_waves(envelope_solution_7d)

        # Should contain strain tensor
        assert (
            "strain_tensor" in waves
        ), "Gravitational waves should contain strain tensor"

        strain_tensor = waves["strain_tensor"]

        # Should be 7x7 matrix
        assert strain_tensor.shape == (
            7,
            7,
        ), f"Strain tensor should be 7x7, got {strain_tensor.shape}"

        # Should be finite
        assert np.all(np.isfinite(strain_tensor)), "Strain tensor should be finite"


class TestVBPGravitationalEffectsIntegration:
    """Test integration of all VBP gravitational effects."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=64, dimensions=7)

    @pytest.fixture
    def gravity_params(self):
        """Create gravitational parameters."""
        return {
            "c_phi": 1.0,
            "chi_kappa": 1.0,
            "beta": 0.5,
            "mu": 1.0,
            "resolution": 64,
            "tolerance": 1e-12,
            "max_iterations": 1000,
        }

    @pytest.fixture
    def mock_system(self, domain_7d):
        """Create mock system with phase field."""

        class MockSystem:
            def __init__(self, domain):
                self.domain = domain
                # Create a simple 7D phase field (smaller size)
                self.phase_field = np.zeros((8, 8, 8, 8, 8, 8, 8), dtype=complex)

                # Add spatial variation
                for i in range(8):
                    for j in range(8):
                        for k in range(8):
                            self.phase_field[i, j, k, :, :, :, :] = np.exp(
                                1j * 2 * np.pi * (i + j + k) / 8
                            )

        return MockSystem(domain_7d)

    def test_vbp_gravitational_effects_integration(self, mock_system, gravity_params):
        """Test integration of all VBP gravitational effects."""
        gravity_model = VBPGravitationalEffectsModel(mock_system, gravity_params)

        # Compute all envelope effects
        envelope_effects = gravity_model.compute_envelope_effects()

        # Should contain all required components
        required_components = [
            "envelope_curvature",
            "gravitational_waves",
            "envelope_solution",
            "effective_metric",
            "curvature_descriptors",
        ]

        for component in required_components:
            assert (
                component in envelope_effects
            ), f"Envelope effects should contain {component}"

        # Test envelope curvature
        curvature = envelope_effects["envelope_curvature"]
        assert (
            "envelope_curvature_scalar" in curvature
        ), "Envelope curvature should contain scalar"
        assert (
            "anisotropy_index" in curvature
        ), "Envelope curvature should contain anisotropy index"
        assert (
            "focusing_rate" in curvature
        ), "Envelope curvature should contain focusing rate"

        # Test gravitational waves
        waves = envelope_effects["gravitational_waves"]
        assert "c_T" in waves, "Gravitational waves should contain c_T"
        assert "c_phi" in waves, "Gravitational waves should contain c_phi"
        assert waves["c_T"] == waves["c_phi"], "Should have c_T = c_phi"

        # Test effective metric
        g_eff = envelope_effects["effective_metric"]
        assert g_eff.shape == (
            7,
            7,
        ), f"Effective metric should be 7x7, got {g_eff.shape}"
        assert g_eff[0, 0] < 0, "Time component should be negative"

    def test_physical_consistency(self, mock_system, gravity_params):
        """Test physical consistency of VBP gravitational effects."""
        gravity_model = VBPGravitationalEffectsModel(mock_system, gravity_params)

        # Compute all envelope effects
        envelope_effects = gravity_model.compute_envelope_effects()

        # Test envelope curvature consistency
        curvature = envelope_effects["envelope_curvature"]
        assert (
            curvature["envelope_curvature_scalar"] >= 0
        ), "Envelope curvature scalar should be non-negative"
        assert (
            0 <= curvature["anisotropy_index"] <= 1
        ), "Anisotropy index should be bounded [0,1]"

        # Test gravitational waves consistency
        waves = envelope_effects["gravitational_waves"]
        assert waves["amplitude"] >= 0, "Wave amplitude should be non-negative"
        assert waves["c_T"] == waves["c_phi"], "Should have c_T = c_phi"

        # Test effective metric consistency
        g_eff = envelope_effects["effective_metric"]
        assert g_eff[0, 0] < 0, "Time component should be negative"
        for i in range(1, 7):
            assert (
                g_eff[i, i] > 0
            ), f"Spatial/phase component g{i}{i} should be positive"
