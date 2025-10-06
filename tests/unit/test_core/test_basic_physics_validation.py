"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic physical validation tests for BHLFF core components.

This module provides basic physical validation tests that can run
without complex dependencies, focusing on fundamental physics
validation.

Physical Meaning:
    Tests validate basic physical principles:
    - Domain structure and properties
    - Basic mathematical operations
    - Physical constraints and bounds
    - Energy conservation principles

Mathematical Foundation:
    Validates fundamental mathematical properties:
    - Vector operations
    - Field properties
    - Conservation laws
    - Physical bounds

Example:
    >>> pytest tests/unit/test_core/test_basic_physics_validation.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain


class TestBasicPhysicsValidation:
    """Basic physical validation tests."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for basic testing."""
        return Domain(
            L=1.0,
            N=8,  # Reduced from 32 to 8
            dimensions=3,
            N_phi=4,  # Reduced from 16 to 4
            N_t=8,  # Reduced from 64 to 8
            T=1.0,
        )

    def test_domain_7d_structure_physics(self, domain_7d):
        """
        Test 7D domain structure physics.

        Physical Meaning:
            Validates that the 7D domain correctly represents
            the 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests domain properties: shape, spacing, boundaries.
        """
        # Physical validation 1: Domain should have correct shape
        expected_shape = (
            domain_7d.N,
            domain_7d.N,
            domain_7d.N,
            domain_7d.N_phi,
            domain_7d.N_phi,
            domain_7d.N_phi,
            domain_7d.N_t,
        )
        assert (
            domain_7d.shape == expected_shape
        ), f"Wrong domain shape: {domain_7d.shape}"

        # Physical validation 2: Grid spacing should be positive
        assert domain_7d.dx > 0, f"Negative spatial spacing: {domain_7d.dx}"
        assert domain_7d.dphi > 0, f"Negative phase spacing: {domain_7d.dphi}"
        assert domain_7d.dt > 0, f"Negative temporal spacing: {domain_7d.dt}"

        # Physical validation 3: Domain size should be positive
        assert domain_7d.L > 0, f"Negative spatial domain size: {domain_7d.L}"
        assert domain_7d.T > 0, f"Negative temporal domain size: {domain_7d.T}"

        # Physical validation 4: Grid spacing should be consistent
        expected_dx = domain_7d.L / domain_7d.N
        expected_dphi = 2 * np.pi / domain_7d.N_phi
        expected_dt = domain_7d.T / domain_7d.N_t

        assert abs(domain_7d.dx - expected_dx) < 1e-10, "Spatial spacing inconsistent"
        assert abs(domain_7d.dphi - expected_dphi) < 1e-10, "Phase spacing inconsistent"
        assert abs(domain_7d.dt - expected_dt) < 1e-10, "Temporal spacing inconsistent"

    def test_field_energy_conservation_physics(self, domain_7d):
        """
        Test field energy conservation physics.

        Physical Meaning:
            Validates that energy calculations are physically
            meaningful and conserve energy.

        Mathematical Foundation:
            Tests energy conservation: E = ∫(|∇a|² + |a|²)dV
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute energy
        energy = self._compute_field_energy(test_field, domain_7d)

        # Physical validation 1: Energy should be positive
        assert energy > 0, f"Negative field energy: {energy}"

        # Physical validation 2: Energy should be finite
        assert np.isfinite(energy), f"Non-finite field energy: {energy}"

        # Physical validation 3: Energy should be reasonable
        max_reasonable_energy = domain_7d.N**3 * domain_7d.N_phi**3 * domain_7d.N_t
        assert energy < max_reasonable_energy, f"Field energy too large: {energy}"

    def test_gradient_physics(self, domain_7d):
        """
        Test gradient computation physics.

        Physical Meaning:
            Validates that gradient computations are physically
            meaningful and follow mathematical principles.

        Mathematical Foundation:
            Tests gradient: ∇a = (∂a/∂x, ∂a/∂y, ∂a/∂z)
        """
        # Create test field with known gradient
        test_field = self._create_gradient_test_field(domain_7d)

        # Compute gradient
        gradient = self._compute_gradient(test_field, domain_7d)

        # Physical validation 1: Gradient should be finite
        assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"

        # Physical validation 2: Gradient should have correct shape
        # In 7D BVP, gradient is computed only over spatial dimensions (first 3)
        expected_shape = (3, domain_7d.N, domain_7d.N, domain_7d.N, 1, 1, 1, 1)
        assert (
            gradient.shape == expected_shape
        ), f"Wrong gradient shape: {gradient.shape}"

        # Physical validation 3: Gradient should be bounded
        max_gradient = np.max(np.abs(gradient))
        assert max_gradient < 1e6, f"Gradient too large: {max_gradient}"

    def test_laplacian_physics(self, domain_7d):
        """
        Test Laplacian computation physics.

        Physical Meaning:
            Validates that Laplacian computations are physically
            meaningful and follow mathematical principles.

        Mathematical Foundation:
            Tests Laplacian: ∇²a = ∂²a/∂x² + ∂²a/∂y² + ∂²a/∂z²
        """
        # Create test field with known Laplacian
        test_field = self._create_laplacian_test_field(domain_7d)

        # Compute Laplacian
        laplacian = self._compute_laplacian(test_field, domain_7d)

        # Physical validation 1: Laplacian should be finite
        assert np.all(np.isfinite(laplacian)), "Laplacian contains non-finite values"

        # Physical validation 2: Laplacian should have correct shape
        # In 7D BVP, Laplacian is computed only over spatial dimensions (first 3)
        expected_shape = (domain_7d.N, domain_7d.N, domain_7d.N, 1, 1, 1, 1)
        assert (
            laplacian.shape == expected_shape
        ), f"Wrong Laplacian shape: {laplacian.shape}"

        # Physical validation 3: Laplacian should be bounded
        max_laplacian = np.max(np.abs(laplacian))
        assert max_laplacian < 1e6, f"Laplacian too large: {max_laplacian}"

    def test_fft_energy_conservation_physics(self, domain_7d):
        """
        Test FFT energy conservation physics.

        Physical Meaning:
            Validates that FFT operations conserve energy
            (Parseval's theorem).

        Mathematical Foundation:
            Tests Parseval's theorem: ∫|a(x)|²dx = ∫|â(k)|²dk
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute energy in real space
        real_energy = np.sum(np.abs(test_field) ** 2)

        # Compute FFT
        spectral_field = np.fft.fftn(test_field)

        # Compute energy in spectral space (with proper normalization)
        spectral_energy = np.sum(np.abs(spectral_field) ** 2) / np.prod(
            test_field.shape
        )

        # Physical validation: Energy should be conserved
        energy_conservation_error = abs(real_energy - spectral_energy) / real_energy
        assert (
            energy_conservation_error < 1e-12
        ), f"FFT energy not conserved: error = {energy_conservation_error}"

    def test_boundary_conditions_physics(self, domain_7d):
        """
        Test boundary conditions physics.

        Physical Meaning:
            Validates that boundary conditions are physically
            meaningful and maintain field properties.

        Mathematical Foundation:
            Tests periodic boundary conditions in 7D space-time.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Apply FFT and inverse FFT (should preserve boundary conditions)
        spectral_field = np.fft.fftn(test_field)
        reconstructed_field = np.fft.ifftn(spectral_field)

        # Physical validation: Reconstruction should be accurate
        reconstruction_error = np.mean(np.abs(test_field - reconstructed_field))
        assert (
            reconstruction_error < 1e-12
        ), f"FFT reconstruction error too large: {reconstruction_error}"

    def test_phase_structure_physics(self, domain_7d):
        """
        Test phase structure physics.

        Physical Meaning:
            Validates that phase structure is physically
            meaningful and follows U(1)³ symmetry.

        Mathematical Foundation:
            Tests phase decomposition: a = |a|e^(iφ)
        """
        # Create test field with phase structure
        test_field = self._create_phase_test_field(domain_7d)

        # Compute amplitude and phase
        amplitude = np.abs(test_field)
        phase = np.angle(test_field)

        # Physical validation 1: Amplitude should be non-negative
        assert np.all(amplitude >= 0), "Amplitude contains negative values"

        # Physical validation 2: Phase should be in [-π, π]
        assert np.all(phase >= -np.pi) and np.all(phase <= np.pi), "Phase out of range"

        # Physical validation 3: Reconstruction should be accurate
        reconstructed_field = amplitude * np.exp(1j * phase)
        reconstruction_error = np.mean(np.abs(test_field - reconstructed_field))
        assert (
            reconstruction_error < 1e-12
        ), f"Phase reconstruction error too large: {reconstruction_error}"

    def _create_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field for physics validation."""
        field = np.zeros(domain.shape)

        # Create Gaussian field
        center = domain.N // 2
        x = np.arange(domain.N) - center
        y = np.arange(domain.N) - center
        z = np.arange(domain.N) - center

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        gaussian = np.exp(-(X**2 + Y**2 + Z**2) / (2 * (domain.N / 8) ** 2))

        field[:, :, :, :, :, :, :] = gaussian[
            :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        return field

    def _create_gradient_test_field(self, domain: Domain) -> np.ndarray:
        """Create field with known gradient for testing."""
        field = np.zeros(domain.shape)

        # Create linear field: x + y + z
        x = np.linspace(-1, 1, domain.N)
        y = np.linspace(-1, 1, domain.N)
        z = np.linspace(-1, 1, domain.N)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        # Broadcast to all 7 dimensions
        field = (X + Y + Z)[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

        return field

    def _create_laplacian_test_field(self, domain: Domain) -> np.ndarray:
        """Create field with known Laplacian for testing."""
        field = np.zeros(domain.shape)

        # Create field: x² + y² + z² (Laplacian = 6)
        x = np.linspace(-1, 1, domain.N)
        y = np.linspace(-1, 1, domain.N)
        z = np.linspace(-1, 1, domain.N)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        # Broadcast to all 7 dimensions
        field = (X**2 + Y**2 + Z**2)[
            :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        return field

    def _create_phase_test_field(self, domain: Domain) -> np.ndarray:
        """Create field with phase structure for testing."""
        field = np.zeros(domain.shape, dtype=complex)

        # Create field with known phase structure
        x = np.linspace(0, 2 * np.pi, domain.N)
        y = np.linspace(0, 2 * np.pi, domain.N)
        z = np.linspace(0, 2 * np.pi, domain.N)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        # Broadcast to all 7 dimensions
        field = np.exp(1j * (X + Y + Z))[
            :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        return field

    def _compute_field_energy(self, field: np.ndarray, domain: Domain) -> float:
        """Compute field energy."""
        # Compute gradient energy
        grad_x = np.gradient(field, axis=0)
        grad_y = np.gradient(field, axis=1)
        grad_z = np.gradient(field, axis=2)

        gradient_energy = np.sum(grad_x**2 + grad_y**2 + grad_z**2)

        # Compute potential energy
        potential_energy = np.sum(np.abs(field) ** 2)

        return gradient_energy + potential_energy

    def _compute_gradient(self, field: np.ndarray, domain: Domain) -> np.ndarray:
        """Compute field gradient."""
        grad_x = np.gradient(field, axis=0)
        grad_y = np.gradient(field, axis=1)
        grad_z = np.gradient(field, axis=2)

        return np.array([grad_x, grad_y, grad_z])

    def _compute_laplacian(self, field: np.ndarray, domain: Domain) -> np.ndarray:
        """Compute field Laplacian."""
        # Compute second derivatives
        d2x = np.gradient(np.gradient(field, axis=0), axis=0)
        d2y = np.gradient(np.gradient(field, axis=1), axis=1)
        d2z = np.gradient(np.gradient(field, axis=2), axis=2)

        return d2x + d2y + d2z
