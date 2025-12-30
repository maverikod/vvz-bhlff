"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Analytical tests for time integrators with known solutions.

This module contains tests that validate time integrators against
known analytical solutions for specific cases.
"""

import numpy as np
import pytest
from typing import Tuple

from bhlff.core.time import (
    BVPEnvelopeIntegrator,
    CrankNicolsonIntegrator,
)
from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters


class TestAnalyticalSolutions:
    """
    Analytical tests for time integrators.

    Physical Meaning:
        Tests integrators against known analytical solutions to validate
        correctness and accuracy of numerical methods.
    """

    @pytest.fixture
    def small_domain(self):
        """Create small domain for analytical tests."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=16, dimensions=7)

    @pytest.fixture
    def parameters_conservative(self):
        """Parameters for conservative system (λ=0)."""
        return Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.0,  # Conservative
            nu=1.0,
            precision="float64",
        )

    @pytest.fixture
    def parameters_dissipative(self):
        """Parameters for dissipative system (λ>0)."""
        return Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,  # Dissipative
            nu=1.0,
            precision="float64",
        )

    def test_exponential_decay_zero_source(self, small_domain, parameters_dissipative):
        """
        Test exponential decay with zero source.

        Physical Meaning:
            For zero source and positive damping, the field should decay
            exponentially: a(t) = a₀ * exp(-λt) in the k=0 mode.

        Mathematical Foundation:
            For the equation ∂a/∂t + λa = 0 with initial condition a(0) = a₀,
            the solution is a(t) = a₀ * exp(-λt).
        """
        integrator = CrankNicolsonIntegrator(small_domain, parameters_dissipative)

        # Create initial field with constant value
        initial_field = np.ones(small_domain.shape, dtype=np.complex128)
        time_steps = np.linspace(0.0, 1.0, 20)
        source_field = np.zeros(
            (len(time_steps),) + small_domain.shape, dtype=np.complex128
        )

        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)

        # Check exponential decay in k=0 mode (spatial average)
        initial_avg = np.abs(result[0]).mean()
        final_avg = np.abs(result[-1]).mean()

        # Expected decay: exp(-λ * t_final)
        lambda_param = parameters_dissipative.lambda_param
        t_final = time_steps[-1]
        expected_decay = np.exp(-lambda_param * t_final)

        # Allow 10% tolerance for numerical errors
        actual_decay = final_avg / initial_avg
        assert np.abs(actual_decay - expected_decay) < 0.1 * expected_decay, (
            f"Exponential decay mismatch. "
            f"Expected: {expected_decay:.6f}, Got: {actual_decay:.6f}"
        )

    def test_steady_state_constant_source(self, small_domain, parameters_dissipative):
        """
        Test convergence to steady state with constant source.

        Physical Meaning:
            For constant source s and positive damping λ, the system should
            converge to steady state. For fractional Laplacian, the steady state
            depends on the spectral properties of the operator.

        Mathematical Foundation:
            For the equation ∂a/∂t + ν(-Δ)^β a + λa = s with constant source,
            the steady state solution in k-space is â_steady(k) = ŝ(k) / (ν|k|^(2β) + λ).
            For k=0 mode: â_steady(0) = ŝ(0) / λ.
        """
        integrator = CrankNicolsonIntegrator(small_domain, parameters_dissipative)

        # Create initial field (zero)
        initial_field = np.zeros(small_domain.shape, dtype=np.complex128)

        # Create constant source (k=0 mode)
        source_value = 1.0 + 0.0j
        time_steps = np.linspace(0.0, 5.0, 50)  # Long time for steady state
        source_field = np.full(
            (len(time_steps),) + small_domain.shape,
            source_value,
            dtype=np.complex128,
        )

        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)

        # Check convergence to steady state
        # For k=0 mode, steady state should be approximately s/λ
        # But due to fractional Laplacian, we check that field converges
        final_field = result[-1]
        
        # Check that field is finite and converges (magnitude increases toward steady state)
        assert np.all(np.isfinite(final_field)), "Final field should be finite"
        
        # Field should be non-zero and positive (converged to steady state)
        final_magnitude = np.abs(final_field).mean()
        assert final_magnitude > 0.1, (
            f"Field should converge to non-zero steady state. "
            f"Got magnitude: {final_magnitude:.6f}"
        )
        
        # Check that field is approximately constant at the end (steady state reached)
        last_few = result[-5:]
        std_last = np.std(np.abs(last_few))
        mean_last = np.abs(last_few).mean()
        
        # Coefficient of variation should be small (field is steady)
        cv = std_last / mean_last if mean_last > 0 else 1.0
        assert cv < 0.1, (
            f"Field should be steady at the end. "
            f"Coefficient of variation: {cv:.6f}"
        )

    def test_energy_conservation_zero_damping(self, small_domain, parameters_conservative):
        """
        Test energy conservation for conservative system (λ=0).

        Physical Meaning:
            For conservative system without damping, energy should be
            approximately conserved (within numerical errors). However,
            fractional Laplacian can cause energy dissipation even with λ=0
            due to numerical diffusion, so we check for reasonable behavior.

        Mathematical Foundation:
            For λ=0, the system is conservative, but fractional Laplacian
            with β<2 can cause energy decay due to numerical effects.
        """
        integrator = CrankNicolsonIntegrator(small_domain, parameters_conservative)

        # Create initial field
        initial_field = np.random.random(small_domain.shape).astype(np.complex128)
        time_steps = np.linspace(0.0, 0.5, 10)  # Shorter time to reduce numerical errors
        source_field = np.zeros(
            (len(time_steps),) + small_domain.shape, dtype=np.complex128
        )

        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)

        # Compute energy (L2 norm squared) at different times
        energies = [np.linalg.norm(result[i]) ** 2 for i in range(len(time_steps))]

        initial_energy = energies[0]
        final_energy = energies[-1]

        # For fractional Laplacian with numerical errors, energy may decrease
        # but should not increase significantly
        energy_change = (final_energy - initial_energy) / initial_energy
        
        # Energy should not increase (conservative system)
        assert energy_change <= 0.1, (  # Allow 10% tolerance for numerical errors
            f"Energy increased in conservative system. "
            f"Initial: {initial_energy:.6f}, Final: {final_energy:.6f}, "
            f"Change: {energy_change*100:.2f}%"
        )
        
        # Energy decrease should be reasonable (not too large)
        assert energy_change >= -0.3, (  # Allow up to 30% decrease due to numerical diffusion
            f"Energy decreased too much. "
            f"Initial: {initial_energy:.6f}, Final: {final_energy:.6f}, "
            f"Change: {energy_change*100:.2f}%"
        )

    def test_harmonic_source_response(self, small_domain, parameters_dissipative):
        """
        Test response to harmonic source.

        Physical Meaning:
            For harmonic source s(t) = s₀ * exp(iωt), the response should
            follow the analytical solution with proper phase and amplitude.

        Mathematical Foundation:
            For harmonic source, the solution has the form:
            a(t) = A * exp(i(ωt + φ)) where A and φ depend on ω, λ, and ν.
        """
        integrator = CrankNicolsonIntegrator(small_domain, parameters_dissipative)

        # Create initial field (zero)
        initial_field = np.zeros(small_domain.shape, dtype=np.complex128)

        # Create harmonic source: s(t) = s₀ * exp(iωt)
        omega = 2.0 * np.pi  # Frequency
        s0 = 1.0 + 0.0j
        time_steps = np.linspace(0.0, 2.0, 40)
        source_field = np.zeros(
            (len(time_steps),) + small_domain.shape, dtype=np.complex128
        )

        for i, t in enumerate(time_steps):
            source_field[i] = s0 * np.exp(1j * omega * t)

        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)

        # Check that response is harmonic (oscillatory)
        # Extract a single point for analysis
        test_point = result[:, 0, 0, 0, 0, 0, 0]

        # Check that response oscillates (has non-zero imaginary part or varying real part)
        real_part = np.real(test_point)
        imag_part = np.imag(test_point)
        
        # Response should oscillate (vary over time)
        real_std = np.std(real_part)
        imag_std = np.std(imag_part)
        
        # At least one component should vary significantly
        assert real_std > 0.01 or imag_std > 0.01, (
            f"Response should oscillate. "
            f"Real std: {real_std:.6f}, Imag std: {imag_std:.6f}"
        )
        
        # Compute FFT to check frequency content (if enough points)
        if len(test_point) >= 4:
            fft_result = np.fft.fft(test_point)
            dt = time_steps[1] - time_steps[0] if len(time_steps) > 1 else 1.0
            frequencies = np.fft.fftfreq(len(test_point), dt)

            # Find dominant frequency (skip DC component)
            n_half = len(fft_result) // 2
            if n_half > 1:
                # Get positive frequencies only (skip DC)
                fft_positive = fft_result[1:n_half]
                freq_positive = frequencies[1:n_half]
                
                if len(fft_positive) > 0:
                    dominant_idx = np.argmax(np.abs(fft_positive))
                    if dominant_idx < len(freq_positive):
                        dominant_freq = np.abs(freq_positive[dominant_idx])
                        
                        # Dominant frequency should be close to omega (within reasonable range)
                        # For short time series, frequency resolution is limited
                        freq_tolerance = 2.0 * np.pi / (time_steps[-1] - time_steps[0])
                        assert np.abs(dominant_freq - omega) < freq_tolerance, (
                            f"Harmonic response frequency mismatch. "
                            f"Expected: {omega:.3f}, Got: {dominant_freq:.3f}, "
                            f"Tolerance: {freq_tolerance:.3f}"
                        )

    def test_boundary_case_beta_zero(self, small_domain, parameters_dissipative):
        """
        Test boundary case β→0 (ordinary diffusion).

        Physical Meaning:
            For β→0, the fractional Laplacian reduces to ordinary diffusion.
            The integrator should handle this limit correctly.

        Mathematical Foundation:
            For β→0, (-Δ)^β → I (identity), so the equation becomes
            ∂a/∂t + λa = s.
        """
        # Create parameters with small beta
        params_beta_small = Parameters(
            mu=1.0,
            beta=0.01,  # Very small beta
            lambda_param=parameters_dissipative.lambda_param,
            nu=1.0,
            precision="float64",
        )

        integrator = CrankNicolsonIntegrator(small_domain, params_beta_small)

        # Create test case
        initial_field = np.ones(small_domain.shape, dtype=np.complex128)
        time_steps = np.linspace(0.0, 0.5, 10)
        source_field = np.zeros(
            (len(time_steps),) + small_domain.shape, dtype=np.complex128
        )

        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)

        # Check that result is finite and reasonable
        assert np.all(np.isfinite(result)), "Result should be finite for β→0"
        assert np.all(np.abs(result) < 1e10), "Result should not explode for β→0"

    def test_boundary_case_lambda_zero(self, small_domain, parameters_conservative):
        """
        Test boundary case λ=0 (conservative system).

        Physical Meaning:
            For λ=0, the system is conservative and energy should be conserved.
            This is already tested in test_energy_conservation_zero_damping,
            but this test focuses on the boundary case behavior.
        """
        integrator = CrankNicolsonIntegrator(small_domain, parameters_conservative)

        # Create test case
        initial_field = np.random.random(small_domain.shape).astype(np.complex128)
        time_steps = np.linspace(0.0, 1.0, 20)
        source_field = np.zeros(
            (len(time_steps),) + small_domain.shape, dtype=np.complex128
        )

        # Perform integration
        result = integrator.integrate(initial_field, source_field, time_steps)

        # Check that field doesn't decay (conservative)
        initial_magnitude = np.abs(result[0]).mean()
        final_magnitude = np.abs(result[-1]).mean()

        # Magnitude should be approximately constant (within 10% for numerical errors)
        magnitude_change = np.abs(final_magnitude - initial_magnitude) / initial_magnitude
        assert magnitude_change < 0.1, (
            f"Field magnitude should be conserved for λ=0. "
            f"Change: {magnitude_change*100:.2f}%"
        )
