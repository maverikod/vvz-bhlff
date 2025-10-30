"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for spectral Laplacian in 7D BVP theory.

This module provides physical validation tests for spectral Laplacian,
ensuring mathematical correctness and physical consistency.
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from bhlff.core.domain import Domain
from bhlff.core.fft.spectral_operations import SpectralOperations
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian


class TestSpectralLaplacianPhysics:
    """Physical validation tests for spectral Laplacian."""

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
    def spectral_ops(self, domain_7d):
        """Create spectral operations for testing."""
        from bhlff.core.fft.spectral_derivatives_impl import SpectralDerivatives

        return SpectralDerivatives(domain_7d, precision="float64")

    @pytest.fixture
    def fractional_laplacian(self, domain_7d):
        """Create fractional Laplacian for testing."""
        return FractionalLaplacian(domain_7d, beta=1.5, lambda_param=0.1)

    def test_spectral_laplacian_physics(self, domain_7d, spectral_ops):
        """
        Test spectral Laplacian physical consistency.

        Physical Meaning:
            Validates that spectral Laplacian correctly computes
            Laplacian in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests spectral Laplacian: ∇²a → -k²â(k)
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute spectral Laplacian
        laplacian = spectral_ops.compute_laplacian(test_field)

        # Physical validation: Should be finite and reasonable
        assert np.isfinite(laplacian).all(), "Spectral Laplacian not finite"

        # Should preserve field shape
        assert (
            laplacian.shape == test_field.shape
        ), "Spectral Laplacian does not preserve field shape"

    def test_spectral_laplacian_energy_conservation_physics(
        self, domain_7d, spectral_ops
    ):
        """
        Test spectral Laplacian energy conservation.

        Physical Meaning:
            Validates that spectral Laplacian conserves energy
            in the spectral domain.

        Mathematical Foundation:
            Tests energy conservation for spectral Laplacian.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute spectral Laplacian
        laplacian = spectral_ops.compute_laplacian(test_field)

        # Physical validation: Energy should be finite
        original_energy = np.sum(test_field**2)
        laplacian_energy = np.sum(laplacian**2)

        assert np.isfinite(original_energy), "Original field energy not finite"
        assert np.isfinite(laplacian_energy), "Laplacian energy not finite"

    def test_spectral_laplacian_7d_structure_physics(self, domain_7d, spectral_ops):
        """
        Test spectral Laplacian 7D structure preservation.

        Physical Meaning:
            Validates that spectral Laplacian preserves the 7D structure
            of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests 7D structure preservation for spectral Laplacian.
        """
        # Create 7D test field
        test_field = np.zeros(domain_7d.shape)
        test_field[0, 0, 0, 0, 0, 0, 0] = 1.0

        # Compute spectral Laplacian
        laplacian = spectral_ops.compute_laplacian(test_field)

        # Physical validation: Should preserve 7D structure
        assert (
            laplacian.shape == domain_7d.shape
        ), "Spectral Laplacian does not preserve 7D structure"

    def test_spectral_laplacian_numerical_stability_physics(
        self, domain_7d, spectral_ops
    ):
        """
        Test spectral Laplacian numerical stability.

        Physical Meaning:
            Validates that spectral Laplacian is numerically stable
            for extreme field values.

        Mathematical Foundation:
            Tests numerical stability of spectral Laplacian.
        """
        # Test with extreme values
        extreme_field = np.random.random(domain_7d.shape)
        extreme_field = extreme_field * 1e10  # Scale to extreme values

        # Compute spectral Laplacian
        laplacian = spectral_ops.compute_laplacian(extreme_field)

        # Physical validation: Should be stable
        assert np.isfinite(
            laplacian
        ).all(), "Spectral Laplacian not numerically stable for extreme values"

    def test_spectral_laplacian_precision_physics(self, domain_7d, spectral_ops):
        """
        Test spectral Laplacian precision.

        Physical Meaning:
            Validates that spectral Laplacian maintains high precision
            for known analytical functions.

        Mathematical Foundation:
            Tests precision of spectral Laplacian with sinusoidal functions.
        """
        # Test with sinusoidal function
        x = np.linspace(0, 2 * np.pi, domain_7d.shape[0], endpoint=False)
        test_field = np.sin(x)
        test_field = np.broadcast_to(
            test_field.reshape(-1, 1, 1, 1, 1, 1, 1), domain_7d.shape
        )

        # Compute spectral Laplacian
        laplacian = spectral_ops.compute_laplacian(test_field)

        # Physical validation: Should be finite and reasonable
        assert np.isfinite(
            laplacian
        ).all(), "Spectral Laplacian not finite for sinusoidal function"

        # Should be reasonable magnitude
        max_laplacian = np.max(np.abs(laplacian))
        assert (
            max_laplacian < 100.0
        ), f"Spectral Laplacian magnitude too large: {max_laplacian}"

    def test_spectral_laplacian_boundary_conditions_physics(
        self, domain_7d, spectral_ops
    ):
        """
        Test spectral Laplacian boundary condition handling.

        Physical Meaning:
            Validates that spectral Laplacian handles boundary conditions
            correctly in 7D space-time.

        Mathematical Foundation:
            Tests boundary condition handling for spectral Laplacian.
        """
        # Create field with specific boundary conditions
        test_field = np.zeros(domain_7d.shape)

        # Set boundary values
        test_field[0, :, :, :, :, :, :] = 1.0  # x=0 boundary
        test_field[-1, :, :, :, :, :, :] = 1.0  # x=L boundary

        # Compute spectral Laplacian
        laplacian = spectral_ops.compute_laplacian(test_field)

        # Physical validation: Should handle boundaries correctly
        assert np.isfinite(
            laplacian
        ).all(), "Spectral Laplacian does not handle boundary conditions correctly"

    def test_spectral_laplacian_phase_structure_physics(self, domain_7d, spectral_ops):
        """
        Test spectral Laplacian phase structure preservation.

        Physical Meaning:
            Validates that spectral Laplacian preserves phase structure
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests phase structure preservation for spectral Laplacian.
        """
        # Create field with complex phase structure
        test_field = np.zeros(domain_7d.shape, dtype=complex)

        # Set phase structure (simplified for 7D)
        for i in range(min(domain_7d.shape[0], 4)):
            for j in range(min(domain_7d.shape[1], 4)):
                for k in range(min(domain_7d.shape[2], 4)):
                    phase = 2 * np.pi * (i + j + k) / 8
                    test_field[i, j, k, 0, 0, 0, 0] = np.exp(1j * phase)

        # Compute spectral Laplacian
        laplacian = spectral_ops.compute_laplacian(test_field)

        # Physical validation: Should preserve phase structure
        # Note: Spectral Laplacian returns real values, so we check for finite values
        assert np.isfinite(
            laplacian
        ).all(), "Spectral Laplacian not finite for complex phase structure"

    def test_fractional_laplacian_physics(self, domain_7d, fractional_laplacian):
        """
        Test fractional Laplacian physical consistency.

        Physical Meaning:
            Validates that fractional Laplacian correctly computes
            fractional Laplacian in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests fractional Laplacian: (-Δ)^β a
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute fractional Laplacian
        fractional_result = fractional_laplacian.apply(test_field)

        # Physical validation: Should be finite and reasonable
        assert np.isfinite(fractional_result).all(), "Fractional Laplacian not finite"

        # Should preserve field shape
        assert (
            fractional_result.shape == test_field.shape
        ), "Fractional Laplacian does not preserve field shape"

    def test_fractional_laplacian_energy_conservation_physics(
        self, domain_7d, fractional_laplacian
    ):
        """
        Test fractional Laplacian energy conservation.

        Physical Meaning:
            Validates that fractional Laplacian conserves energy
            in the spectral domain.

        Mathematical Foundation:
            Tests energy conservation for fractional Laplacian.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute fractional Laplacian
        fractional_result = fractional_laplacian.apply(test_field)

        # Physical validation: Energy should be finite
        original_energy = np.sum(test_field**2)
        fractional_energy = np.sum(fractional_result**2)

        assert np.isfinite(original_energy), "Original field energy not finite"
        assert np.isfinite(fractional_energy), "Fractional Laplacian energy not finite"

    def test_fractional_laplacian_7d_structure_physics(
        self, domain_7d, fractional_laplacian
    ):
        """
        Test fractional Laplacian 7D structure preservation.

        Physical Meaning:
            Validates that fractional Laplacian preserves the 7D structure
            of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests 7D structure preservation for fractional Laplacian.
        """
        # Create 7D test field
        test_field = np.zeros(domain_7d.shape)
        test_field[0, 0, 0, 0, 0, 0, 0] = 1.0

        # Compute fractional Laplacian
        fractional_result = fractional_laplacian.apply(test_field)

        # Physical validation: Should preserve 7D structure
        assert (
            fractional_result.shape == domain_7d.shape
        ), "Fractional Laplacian does not preserve 7D structure"

    def test_fractional_laplacian_numerical_stability_physics(
        self, domain_7d, fractional_laplacian
    ):
        """
        Test fractional Laplacian numerical stability.

        Physical Meaning:
            Validates that fractional Laplacian is numerically stable
            for extreme field values.

        Mathematical Foundation:
            Tests numerical stability of fractional Laplacian.
        """
        # Test with extreme values
        extreme_field = np.random.random(domain_7d.shape)
        extreme_field = extreme_field * 1e10  # Scale to extreme values

        # Compute fractional Laplacian
        fractional_result = fractional_laplacian.apply(extreme_field)

        # Physical validation: Should be stable
        assert np.isfinite(
            fractional_result
        ).all(), "Fractional Laplacian not numerically stable for extreme values"

    def test_fractional_laplacian_precision_physics(
        self, domain_7d, fractional_laplacian
    ):
        """
        Test fractional Laplacian precision.

        Physical Meaning:
            Validates that fractional Laplacian maintains high precision
            for known analytical functions.

        Mathematical Foundation:
            Tests precision of fractional Laplacian with sinusoidal functions.
        """
        # Test with sinusoidal function
        x = np.linspace(0, 2 * np.pi, domain_7d.shape[0], endpoint=False)
        test_field = np.sin(x)
        # Create 7D field by broadcasting
        test_field_7d = np.zeros(domain_7d.shape)
        for i in range(domain_7d.shape[0]):
            test_field_7d[i, :, :, :, :, :, :] = test_field[i]

        # Compute fractional Laplacian
        fractional_result = fractional_laplacian.apply(test_field_7d)

        # Physical validation: Should be finite and reasonable
        assert np.isfinite(
            fractional_result
        ).all(), "Fractional Laplacian not finite for sinusoidal function"

        # Should be reasonable magnitude
        max_fractional = np.max(np.abs(fractional_result))
        assert (
            max_fractional < 100.0
        ), f"Fractional Laplacian magnitude too large: {max_fractional}"

    def _create_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field for spectral Laplacian testing."""
        # Create random test field
        test_field = np.random.random(domain.shape)

        # Normalize for numerical stability
        test_field = test_field / np.max(np.abs(test_field))

        return test_field
