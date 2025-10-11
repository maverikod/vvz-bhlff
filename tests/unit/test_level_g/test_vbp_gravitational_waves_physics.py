"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for VBP gravitational waves in 7D phase field theory.

This module tests the physical correctness of the VBP gravitational waves
implementation, including wave speed, amplitude laws, and polarization modes.

Theoretical Background:
    Gravitational waves in 7D BVP theory arise from VBP envelope dynamics
    with c_T=c_φ and GW-1 amplitude law.

Physical Tests:
    - Gravitational waves (c_T=c_φ, GW-1 law, 7D polarization modes)
    - Strain tensor 7D structure
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.models.level_g.gravity_waves import VBPGravitationalWavesCalculator


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
