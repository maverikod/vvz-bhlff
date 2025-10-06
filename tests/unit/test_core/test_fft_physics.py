"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for FFT operations in 7D BVP theory.

This module provides physical validation tests for FFT operations,
ensuring mathematical correctness and physical consistency.
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from bhlff.core.domain import Domain
from bhlff.core.fft.spectral_operations import SpectralOperations


class TestFFTPhysics:
    """Physical validation tests for FFT operations."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for spectral testing."""
        return Domain(
            L=1.0,  # Smaller domain for memory efficiency
            N=8,  # Lower resolution
            dimensions=7,
            N_phi=4,
            N_t=8,
            T=1.0,
        )

    @pytest.fixture
    def fft_backend(self, domain_7d):
        """Create FFT backend for testing."""
        from bhlff.core.fft.fft_backend_core import FFTBackend

        return FFTBackend(domain_7d)

    @pytest.fixture
    def spectral_ops(self, fft_backend):
        """Create spectral operations for testing."""
        return SpectralOperations(fft_backend)

    def test_fft_energy_conservation_physics(self, domain_7d, spectral_ops):
        """
        Test FFT energy conservation (Parseval's theorem).

        Physical Meaning:
            Validates that FFT operations conserve energy, ensuring
            Parseval's theorem is satisfied: ∫|a(x)|²dx = ∫|â(k)|²dk

        Mathematical Foundation:
            Tests Parseval's theorem for 7D FFT operations.
        """
        # Create test field with known energy
        test_field = self._create_test_field(domain_7d)

        # Compute energy in real space
        real_energy = np.sum(np.abs(test_field) ** 2)

        # Compute FFT
        spectral_field = spectral_ops.fft_backend.forward_transform(test_field)

        # Compute energy in spectral space
        spectral_energy = np.sum(np.abs(spectral_field) ** 2)

        # Physical validation: Energy should be conserved
        energy_ratio = spectral_energy / real_energy
        assert np.isclose(
            energy_ratio, 1.0, atol=1e-10
        ), f"Energy not conserved: ratio = {energy_ratio}"

    def test_fft_round_trip_physics(self, domain_7d, spectral_ops):
        """
        Test FFT round-trip accuracy.

        Physical Meaning:
            Validates that forward and inverse FFT operations
            preserve field information accurately.

        Mathematical Foundation:
            Tests FFT round-trip: a(x) → â(k) → a(x)
        """
        # Create test field
        original_field = self._create_test_field(domain_7d)

        # Forward FFT
        spectral_field = spectral_ops.fft_backend.forward_transform(original_field)

        # Inverse FFT
        reconstructed_field = spectral_ops.fft_backend.inverse_transform(spectral_field)

        # Physical validation: Should reconstruct original
        reconstruction_error = np.max(np.abs(original_field - reconstructed_field))
        assert (
            reconstruction_error < 1e-12
        ), f"FFT round-trip error too large: {reconstruction_error}"

    def test_fft_7d_structure_physics(self, domain_7d, spectral_ops):
        """
        Test FFT 7D structure preservation.

        Physical Meaning:
            Validates that FFT operations preserve the 7D structure
            of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests 7D FFT structure preservation.
        """
        # Create 7D test field with specific structure
        test_field = np.zeros(domain_7d.shape)
        test_field[0, 0, 0, 0, 0, 0, 0] = 1.0  # Origin
        test_field[-1, -1, -1, -1, -1, -1, -1] = 1.0  # Corner

        # Forward FFT
        spectral_field = spectral_ops.fft_backend.forward_transform(test_field)

        # Physical validation: Should preserve 7D structure
        assert (
            spectral_field.shape == domain_7d.shape
        ), "FFT does not preserve 7D structure"

        # Inverse FFT
        reconstructed_field = spectral_ops.fft_backend.inverse_transform(spectral_field)

        # Should reconstruct original structure
        reconstruction_error = np.max(np.abs(test_field - reconstructed_field))
        assert (
            reconstruction_error < 1e-12
        ), f"7D structure reconstruction error: {reconstruction_error}"

    def test_fft_numerical_stability_physics(self, domain_7d, spectral_ops):
        """
        Test FFT numerical stability.

        Physical Meaning:
            Validates that FFT operations are numerically stable
            for extreme field values.

        Mathematical Foundation:
            Tests numerical stability of 7D FFT operations.
        """
        # Test with extreme values
        extreme_field = np.array([1e10, -1e10, 1e-10, -1e-10])
        extreme_field = np.broadcast_to(
            extreme_field.reshape(-1, 1, 1, 1, 1, 1, 1), domain_7d.shape
        )

        # Forward FFT
        spectral_field = spectral_ops.fft_backend.forward_transform(extreme_field)

        # Physical validation: Should be stable
        assert np.isfinite(
            spectral_field
        ).all(), "FFT not numerically stable for extreme values"

        # Inverse FFT
        reconstructed_field = spectral_ops.fft_backend.inverse_transform(spectral_field)

        # Should be stable
        assert np.isfinite(
            reconstructed_field
        ).all(), "Inverse FFT not numerically stable for extreme values"

    def test_fft_precision_physics(self, domain_7d, spectral_ops):
        """
        Test FFT precision with known functions.

        Physical Meaning:
            Validates that FFT operations maintain high precision
            for known analytical functions.

        Mathematical Foundation:
            Tests FFT precision with sinusoidal functions.
        """
        # Test with sinusoidal function
        x = np.linspace(0, 2 * np.pi, domain_7d.shape[0], endpoint=False)
        test_field = np.sin(x)
        test_field = np.broadcast_to(
            test_field.reshape(-1, 1, 1, 1, 1, 1, 1), domain_7d.shape
        )

        # Forward FFT
        spectral_field = spectral_ops.fft_backend.forward_transform(test_field)

        # Inverse FFT
        reconstructed_field = spectral_ops.fft_backend.inverse_transform(spectral_field)

        # Physical validation: Should be very precise
        precision_error = np.max(np.abs(test_field - reconstructed_field))
        assert (
            precision_error < 1e-12
        ), f"FFT precision error too large: {precision_error}"

    def test_fft_boundary_conditions_physics(self, domain_7d, spectral_ops):
        """
        Test FFT boundary condition handling.

        Physical Meaning:
            Validates that FFT operations handle boundary conditions
            correctly in 7D space-time.

        Mathematical Foundation:
            Tests boundary condition handling in spectral space.
        """
        # Create field with specific boundary conditions
        test_field = np.zeros(domain_7d.shape)

        # Set boundary values
        test_field[0, :, :, :, :, :, :] = 1.0  # x=0 boundary
        test_field[-1, :, :, :, :, :, :] = 1.0  # x=L boundary

        # Forward FFT
        spectral_field = spectral_ops.fft_backend.forward_transform(test_field)

        # Physical validation: Should handle boundaries correctly
        assert np.isfinite(
            spectral_field
        ).all(), "FFT does not handle boundary conditions correctly"

        # Inverse FFT
        reconstructed_field = spectral_ops.fft_backend.inverse_transform(spectral_field)

        # Should reconstruct boundaries
        boundary_error = np.max(np.abs(test_field - reconstructed_field))
        assert (
            boundary_error < 1e-12
        ), f"Boundary condition reconstruction error: {boundary_error}"

    def test_fft_phase_structure_physics(self, domain_7d, spectral_ops):
        """
        Test FFT phase structure preservation.

        Physical Meaning:
            Validates that FFT operations preserve phase structure
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests phase structure preservation in spectral space.
        """
        # Create field with complex phase structure
        test_field = np.zeros(domain_7d.shape, dtype=complex)

        # Set phase structure
        for i in range(domain_7d.shape[0]):
            for j in range(domain_7d.shape[1]):
                for k in range(domain_7d.shape[2]):
                    phase = 2 * np.pi * (i + j + k) / 8
                    test_field[i, j, k, 0, 0, 0, 0] = np.exp(1j * phase)

        # Forward FFT
        spectral_field = spectral_ops.fft_backend.forward_transform(test_field)

        # Physical validation: Should preserve phase structure
        assert np.iscomplexobj(
            spectral_field
        ), "FFT does not preserve complex phase structure"

        # Inverse FFT
        reconstructed_field = spectral_ops.fft_backend.inverse_transform(spectral_field)

        # Should reconstruct phase structure
        phase_error = np.max(np.abs(test_field - reconstructed_field))
        assert (
            phase_error < 1e-12
        ), f"Phase structure reconstruction error: {phase_error}"

    def _create_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field for FFT testing."""
        # Create random test field
        test_field = np.random.random(domain.shape)

        # Normalize for numerical stability
        test_field = test_field / np.max(np.abs(test_field))

        return test_field
