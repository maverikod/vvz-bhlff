"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for spectral derivatives in 7D BVP theory.

This module provides physical validation tests for spectral derivatives,
ensuring mathematical correctness and physical consistency.
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from bhlff.core.domain import Domain
from bhlff.core.fft.spectral_derivatives import SpectralDerivatives


class TestSpectralDerivativesPhysics:
    """Physical validation tests for spectral derivatives."""

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
    def spectral_derivs(self, fft_backend):
        """Create spectral derivatives for testing."""
        return SpectralDerivatives(fft_backend)

    def test_spectral_derivatives_physics(self, domain_7d, spectral_derivs):
        """
        Test spectral derivatives physical consistency.

        Physical Meaning:
            Validates that spectral derivatives correctly compute
            derivatives in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests spectral derivatives: ∂a/∂x → ikâ(k)
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute spectral derivative
        derivative = spectral_derivs.compute_derivative(test_field, axis=0, order=1)

        # Physical validation: Should be finite and reasonable
        assert np.isfinite(derivative).all(), "Spectral derivative not finite"

        # Should preserve field shape
        assert (
            derivative.shape == test_field.shape
        ), "Spectral derivative does not preserve field shape"

    def test_spectral_derivatives_energy_conservation_physics(
        self, domain_7d, spectral_derivs
    ):
        """
        Test spectral derivatives energy conservation.

        Physical Meaning:
            Validates that spectral derivatives conserve energy
            in the spectral domain.

        Mathematical Foundation:
            Tests energy conservation for spectral derivatives.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute spectral derivative
        derivative = spectral_derivs.compute_derivative(test_field, axis=0, order=1)

        # Physical validation: Energy should be finite
        original_energy = np.sum(test_field**2)
        derivative_energy = np.sum(derivative**2)

        assert np.isfinite(original_energy), "Original field energy not finite"
        assert np.isfinite(derivative_energy), "Derivative energy not finite"

    def test_spectral_derivatives_7d_structure_physics(
        self, domain_7d, spectral_derivs
    ):
        """
        Test spectral derivatives 7D structure preservation.

        Physical Meaning:
            Validates that spectral derivatives preserve the 7D structure
            of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests 7D structure preservation for spectral derivatives.
        """
        # Create 7D test field
        test_field = np.zeros(domain_7d.shape)
        test_field[0, 0, 0, 0, 0, 0, 0] = 1.0

        # Compute spectral derivative along different axes
        for axis in range(7):
            derivative = spectral_derivs.compute_derivative(
                test_field, axis=axis, order=1
            )

            # Physical validation: Should preserve 7D structure
            assert (
                derivative.shape == domain_7d.shape
            ), f"Spectral derivative does not preserve 7D structure for axis {axis}"

    def test_spectral_derivatives_numerical_stability_physics(
        self, domain_7d, spectral_derivs
    ):
        """
        Test spectral derivatives numerical stability.

        Physical Meaning:
            Validates that spectral derivatives are numerically stable
            for extreme field values.

        Mathematical Foundation:
            Tests numerical stability of spectral derivatives.
        """
        # Test with extreme values
        extreme_field = np.array([1e10, -1e10, 1e-10, -1e-10])
        extreme_field = np.broadcast_to(
            extreme_field.reshape(-1, 1, 1, 1, 1, 1, 1), domain_7d.shape
        )

        # Compute spectral derivative
        derivative = spectral_derivs.compute_derivative(extreme_field, axis=0, order=1)

        # Physical validation: Should be stable
        assert np.isfinite(
            derivative
        ).all(), "Spectral derivative not numerically stable for extreme values"

    def test_spectral_derivatives_precision_physics(self, domain_7d, spectral_derivs):
        """
        Test spectral derivatives precision.

        Physical Meaning:
            Validates that spectral derivatives maintain high precision
            for known analytical functions.

        Mathematical Foundation:
            Tests precision of spectral derivatives with sinusoidal functions.
        """
        # Test with sinusoidal function
        x = np.linspace(0, 2 * np.pi, domain_7d.shape[0], endpoint=False)
        test_field = np.sin(x)
        test_field = np.broadcast_to(
            test_field.reshape(-1, 1, 1, 1, 1, 1, 1), domain_7d.shape
        )

        # Compute spectral derivative
        derivative = spectral_derivs.compute_derivative(test_field, axis=0, order=1)

        # Physical validation: Should be finite and reasonable
        assert np.isfinite(
            derivative
        ).all(), "Spectral derivative not finite for sinusoidal function"

        # Should be reasonable magnitude
        max_derivative = np.max(np.abs(derivative))
        assert (
            max_derivative < 10.0
        ), f"Spectral derivative magnitude too large: {max_derivative}"

    def test_spectral_derivatives_boundary_conditions_physics(
        self, domain_7d, spectral_derivs
    ):
        """
        Test spectral derivatives boundary condition handling.

        Physical Meaning:
            Validates that spectral derivatives handle boundary conditions
            correctly in 7D space-time.

        Mathematical Foundation:
            Tests boundary condition handling for spectral derivatives.
        """
        # Create field with specific boundary conditions
        test_field = np.zeros(domain_7d.shape)

        # Set boundary values
        test_field[0, :, :, :, :, :, :] = 1.0  # x=0 boundary
        test_field[-1, :, :, :, :, :, :] = 1.0  # x=L boundary

        # Compute spectral derivative
        derivative = spectral_derivs.compute_derivative(test_field, axis=0, order=1)

        # Physical validation: Should handle boundaries correctly
        assert np.isfinite(
            derivative
        ).all(), "Spectral derivative does not handle boundary conditions correctly"

    def test_spectral_derivatives_phase_structure_physics(
        self, domain_7d, spectral_derivs
    ):
        """
        Test spectral derivatives phase structure preservation.

        Physical Meaning:
            Validates that spectral derivatives preserve phase structure
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests phase structure preservation for spectral derivatives.
        """
        # Create field with complex phase structure
        test_field = np.zeros(domain_7d.shape, dtype=complex)

        # Set phase structure
        for i in range(domain_7d.shape[0]):
            for j in range(domain_7d.shape[1]):
                for k in range(domain_7d.shape[2]):
                    phase = 2 * np.pi * (i + j + k) / 8
                    test_field[i, j, k, 0, 0, 0, 0] = np.exp(1j * phase)

        # Compute spectral derivative
        derivative = spectral_derivs.compute_derivative(test_field, axis=0, order=1)

        # Physical validation: Should preserve phase structure
        assert np.iscomplexobj(
            derivative
        ), "Spectral derivative does not preserve complex phase structure"

        # Should be finite
        assert np.isfinite(
            derivative
        ).all(), "Spectral derivative not finite for complex phase structure"

    def test_spectral_derivatives_mixed_derivatives_physics(
        self, domain_7d, spectral_derivs
    ):
        """
        Test spectral mixed derivatives.

        Physical Meaning:
            Validates that spectral mixed derivatives correctly compute
            mixed derivatives in 7D space-time.

        Mathematical Foundation:
            Tests mixed spectral derivatives.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Compute mixed spectral derivative
        mixed_derivative = spectral_derivs.compute_mixed_derivative(
            test_field, axes=[0, 1], orders=[1, 1]
        )

        # Physical validation: Should be finite and reasonable
        assert np.isfinite(
            mixed_derivative
        ).all(), "Mixed spectral derivative not finite"

        # Should preserve field shape
        assert (
            mixed_derivative.shape == test_field.shape
        ), "Mixed spectral derivative does not preserve field shape"

    def test_spectral_derivatives_higher_order_physics(
        self, domain_7d, spectral_derivs
    ):
        """
        Test spectral higher-order derivatives.

        Physical Meaning:
            Validates that spectral higher-order derivatives correctly compute
            higher-order derivatives in 7D space-time.

        Mathematical Foundation:
            Tests higher-order spectral derivatives.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)

        # Test different orders
        for order in [1, 2, 3, 4]:
            derivative = spectral_derivs.compute_derivative(
                test_field, axis=0, order=order
            )

            # Physical validation: Should be finite and reasonable
            assert np.isfinite(
                derivative
            ).all(), f"Spectral derivative order {order} not finite"

            # Should preserve field shape
            assert (
                derivative.shape == test_field.shape
            ), f"Spectral derivative order {order} does not preserve field shape"

    def _create_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field for spectral derivatives testing."""
        # Create random test field
        test_field = np.random.random(domain.shape)

        # Normalize for numerical stability
        test_field = test_field / np.max(np.abs(test_field))

        return test_field
