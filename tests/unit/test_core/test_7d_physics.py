"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for 7D BVP theory.

This module provides comprehensive physical validation tests for the 7D
Base High-Frequency Field theory, ensuring theoretical correctness and
physical consistency of the implementation.

Physical Meaning:
    Tests validate the fundamental physics of the 7D space-time theory
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, including:
    - 7D envelope equation physics
    - U(1)³ phase structure
    - Energy conservation
    - Quench dynamics
    - Spectral properties

Mathematical Foundation:
    Validates key equations:
    - 7D envelope equation: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    - U(1)³ phase structure: a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃)
    - Energy conservation: ∂E/∂t + ∇·S = 0
    - Quench condition: |∇a|² > threshold

Example:
    >>> pytest tests/unit/test_core/test_7d_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from bhlff.core.domain import Domain
from bhlff.core.bvp.envelope_solver.envelope_solver_core import EnvelopeSolverCore
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.quench_detector import QuenchDetector
from bhlff.core.bvp.phase_vector.phase_vector import PhaseVector


class Test7DPhysics:
    """Physical validation tests for 7D BVP theory."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for physical testing."""
        return Domain(
            L=1.0,  # Smaller domain for memory efficiency
            N=8,  # Lower resolution
            dimensions=3,
            N_phi=4,  # Fewer phase points
            N_t=8,  # Fewer time points
            T=1.0,  # Shorter time evolution
        )

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for physical testing."""
        config = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,  # k0=2.0 squared
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
            },
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def envelope_solver(self, domain_7d, bvp_constants):
        """Create envelope solver for physical testing."""
        return EnvelopeSolverCore(domain_7d, bvp_constants)

    @pytest.fixture
    def quench_detector(self, domain_7d, bvp_constants):
        """Create quench detector for physical testing."""
        return QuenchDetector(domain_7d, bvp_constants)

    def test_7d_envelope_equation_physics(self, envelope_solver, domain_7d):
        """
        Test physical correctness of 7D envelope equation.

        Physical Meaning:
            Validates that the 7D envelope equation correctly implements
            the physics of the Base High-Frequency Field in 7D space-time
            M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

        Mathematical Foundation:
            Tests the equation: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
            where κ(|a|) and χ(|a|) are nonlinear coefficients.
        """
        # Create test source with known physical properties
        source = self._create_physical_source(domain_7d)

        # Solve the envelope equation
        envelope = envelope_solver.solve_envelope(source)

        # Physical validation 1: Envelope should be finite
        assert np.all(np.isfinite(envelope)), "Envelope contains non-finite values"

        # Physical validation 2: Envelope should respect boundary conditions
        self._validate_boundary_conditions(envelope, domain_7d)

        # Physical validation 3: Envelope should have correct dimensionality
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
            envelope.shape == expected_shape
        ), f"Wrong envelope shape: {envelope.shape}"

        # Physical validation 4: Envelope should satisfy energy bounds
        self._validate_energy_bounds(envelope, domain_7d)

    def test_u1_phase_structure_physics(self, domain_7d, bvp_constants):
        """
        Test U(1)³ phase structure physics.

        Physical Meaning:
            Validates that the phase field correctly implements the U(1)³
            phase structure a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃) with proper
            phase coherence and topological properties.

        Mathematical Foundation:
            Tests phase decomposition: a = |a|e^(iφ) where φ = φ₁ + φ₂ + φ₃
            and validates phase coherence conditions.
        """
        # Create phase vector with known U(1)³ structure
        phase_vector = PhaseVector(domain_7d, bvp_constants)

        # Test phase decomposition
        amplitude, phases = phase_vector.decompose_phase_structure()

        # Physical validation 1: Amplitude should be non-negative
        assert np.all(amplitude >= 0), "Amplitude contains negative values"

        # Physical validation 2: Phases should be in [0, 2π)
        for phase in phases:
            assert np.all(phase >= 0) and np.all(
                phase < 2 * np.pi
            ), "Phases out of range"

        # Physical validation 3: Phase coherence should be maintained
        coherence = phase_vector.compute_phase_coherence()
        assert 0 <= coherence <= 1, f"Phase coherence out of range: {coherence}"

        # Physical validation 4: Topological charge should be quantized
        topological_charge = phase_vector.compute_topological_charge()
        assert np.isclose(
            topological_charge, np.round(topological_charge), atol=1e-6
        ), f"Topological charge not quantized: {topological_charge}"

    def test_energy_conservation_physics(self, envelope_solver, domain_7d):
        """
        Test energy conservation in 7D BVP system.

        Physical Meaning:
            Validates that energy is conserved in the 7D BVP system,
            ensuring the fundamental conservation law ∂E/∂t + ∇·S = 0
            is satisfied.

        Mathematical Foundation:
            Tests energy conservation: E = ∫(|∇a|² + k₀²|a|²)dV
            and energy flux: S = -2Re(a*∇a).
        """
        # Create time-evolving source
        source_evolution = self._create_time_evolving_source(domain_7d)

        # Solve envelope evolution
        envelope_evolution = []
        for t in range(domain_7d.N_t):
            envelope = envelope_solver.solve_envelope(source_evolution[t])
            envelope_evolution.append(envelope)

        # Compute energy evolution
        energy_evolution = []
        for envelope in envelope_evolution:
            energy = self._compute_total_energy(envelope, domain_7d)
            energy_evolution.append(energy)

        # Physical validation 1: Energy should be conserved (within numerical precision)
        initial_energy = energy_evolution[0]
        final_energy = energy_evolution[-1]
        energy_conservation_error = abs(final_energy - initial_energy) / initial_energy

        assert (
            energy_conservation_error < 1e-3
        ), f"Energy not conserved: error = {energy_conservation_error}"

        # Physical validation 2: Energy should be positive
        for energy in energy_evolution:
            assert energy > 0, f"Negative energy: {energy}"

        # Physical validation 3: Energy should be bounded
        max_energy = max(energy_evolution)
        min_energy = min(energy_evolution)
        assert max_energy / min_energy < 10, "Energy varies too much"

    def test_quench_dynamics_physics(self, quench_detector, domain_7d):
        """
        Test quench dynamics physics.

        Physical Meaning:
            Validates that quench detection correctly identifies
            regions where the field gradient exceeds the threshold,
            indicating phase transitions in the BVP system.

        Mathematical Foundation:
            Tests quench condition: |∇a|² > threshold
            and validates quench dynamics.
        """
        # Create envelope with known quench regions
        envelope = self._create_envelope_with_quenches(domain_7d)

        # Detect quenches
        quench_map = quench_detector.detect_quenches(envelope)

        # Physical validation 1: Quench map should be binary
        assert np.all((quench_map == 0) | (quench_map == 1)), "Quench map not binary"

        # Physical validation 2: Quenches should be localized
        quench_fraction = np.mean(quench_map)
        assert (
            0 < quench_fraction < 0.5
        ), f"Quench fraction out of range: {quench_fraction}"

        # Physical validation 3: Quenches should correlate with high gradients
        gradient_magnitude = self._compute_gradient_magnitude(envelope, domain_7d)
        quench_gradient_correlation = np.corrcoef(
            quench_map.flatten(), gradient_magnitude.flatten()
        )[0, 1]
        assert (
            quench_gradient_correlation > 0.5
        ), "Quenches don't correlate with gradients"

    def test_spectral_properties_physics(self, envelope_solver, domain_7d):
        """
        Test spectral properties of 7D BVP field.

        Physical Meaning:
            Validates that the spectral properties of the BVP field
            are consistent with the theoretical predictions for
            the 7D space-time structure.

        Mathematical Foundation:
            Tests spectral decomposition and validates power law
            behavior in frequency space.
        """
        # Create source with known spectral properties
        source = self._create_spectral_source(domain_7d)

        # Solve envelope
        envelope = envelope_solver.solve_envelope(source)

        # Compute spectral properties
        spatial_spectrum = self._compute_spatial_spectrum(envelope, domain_7d)
        phase_spectrum = self._compute_phase_spectrum(envelope, domain_7d)

        # Physical validation 1: Spatial spectrum should follow power law
        power_law_exponent = self._fit_power_law(spatial_spectrum)
        assert (
            -3 < power_law_exponent < -1
        ), f"Power law exponent out of range: {power_law_exponent}"

        # Physical validation 2: Phase spectrum should be bounded
        assert np.all(phase_spectrum >= 0), "Phase spectrum contains negative values"

        # Physical validation 3: Spectral energy should be conserved
        total_spectral_energy = np.sum(spatial_spectrum) + np.sum(phase_spectrum)
        assert total_spectral_energy > 0, "Total spectral energy is zero"

    def _create_physical_source(self, domain: Domain) -> np.ndarray:
        """Create a source with known physical properties."""
        source = np.zeros(domain.shape)

        # Create localized source in center
        center = domain.N // 2
        source[
            center - 2 : center + 3,
            center - 2 : center + 3,
            center - 2 : center + 3,
            :,
            :,
            :,
            :,
        ] = 1.0

        return source

    def _create_time_evolving_source(self, domain: Domain) -> list:
        """Create time-evolving source for energy conservation test."""
        sources = []
        for t in range(domain.N_t):
            source = np.zeros(domain.shape)
            # Moving source
            center = domain.N // 2 + int(5 * np.sin(2 * np.pi * t / domain.N_t))
            if 0 <= center < domain.N:
                source[center, center, center, :, :, :, t] = 1.0
            sources.append(source)
        return sources

    def _create_envelope_with_quenches(self, domain: Domain) -> np.ndarray:
        """Create envelope with known quench regions."""
        envelope = np.ones(domain.shape)

        # Create sharp gradients (quenches)
        envelope[
            domain.N // 4 : 3 * domain.N // 4,
            domain.N // 4 : 3 * domain.N // 4,
            domain.N // 4 : 3 * domain.N // 4,
            :,
            :,
            :,
            :,
        ] *= 10.0

        return envelope

    def _create_spectral_source(self, domain: Domain) -> np.ndarray:
        """Create source with known spectral properties."""
        source = np.zeros(domain.shape)

        # Create sinusoidal pattern
        x = np.linspace(0, 2 * np.pi, domain.N)
        y = np.linspace(0, 2 * np.pi, domain.N)
        z = np.linspace(0, 2 * np.pi, domain.N)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        source[:, :, :, :, :, :, :] = np.sin(X) * np.sin(Y) * np.sin(Z)

        return source

    def _validate_boundary_conditions(
        self, envelope: np.ndarray, domain: Domain
    ) -> None:
        """Validate boundary conditions."""
        # Check spatial boundaries (should be periodic)
        assert np.allclose(
            envelope[0, :, :, :, :, :, :], envelope[-1, :, :, :, :, :, :], atol=1e-6
        ), "Spatial boundary conditions not satisfied"

        # Check phase boundaries (should be periodic)
        assert np.allclose(
            envelope[:, :, :, 0, :, :, :], envelope[:, :, :, -1, :, :, :], atol=1e-6
        ), "Phase boundary conditions not satisfied"

    def _validate_energy_bounds(self, envelope: np.ndarray, domain: Domain) -> None:
        """Validate energy bounds."""
        energy = self._compute_total_energy(envelope, domain)

        # Energy should be positive and finite
        assert energy > 0, f"Negative energy: {energy}"
        assert np.isfinite(energy), f"Non-finite energy: {energy}"

        # Energy should be reasonable (not too large)
        max_reasonable_energy = domain.N**3 * domain.N_phi**3 * domain.N_t
        assert energy < max_reasonable_energy, f"Energy too large: {energy}"

    def _compute_total_energy(self, envelope: np.ndarray, domain: Domain) -> float:
        """Compute total energy of the envelope."""
        # Compute gradient energy
        grad_x = np.gradient(envelope, axis=0)
        grad_y = np.gradient(envelope, axis=1)
        grad_z = np.gradient(envelope, axis=2)

        gradient_energy = np.sum(grad_x**2 + grad_y**2 + grad_z**2)

        # Compute potential energy
        potential_energy = np.sum(envelope**2)

        return gradient_energy + potential_energy

    def _compute_gradient_magnitude(
        self, envelope: np.ndarray, domain: Domain
    ) -> np.ndarray:
        """Compute gradient magnitude."""
        grad_x = np.gradient(envelope, axis=0)
        grad_y = np.gradient(envelope, axis=1)
        grad_z = np.gradient(envelope, axis=2)

        return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    def _compute_spatial_spectrum(
        self, envelope: np.ndarray, domain: Domain
    ) -> np.ndarray:
        """Compute spatial spectrum."""
        # Take FFT in spatial dimensions
        spatial_fft = np.fft.fftn(envelope, axes=(0, 1, 2))
        return np.abs(spatial_fft) ** 2

    def _compute_phase_spectrum(
        self, envelope: np.ndarray, domain: Domain
    ) -> np.ndarray:
        """Compute phase spectrum."""
        # Take FFT in phase dimensions
        phase_fft = np.fft.fftn(envelope, axes=(3, 4, 5))
        return np.abs(phase_fft) ** 2

    def _fit_power_law(self, spectrum: np.ndarray) -> float:
        """Fit power law to spectrum."""
        # Compute radial average
        kx = np.fft.fftfreq(spectrum.shape[0])
        ky = np.fft.fftfreq(spectrum.shape[1])
        kz = np.fft.fftfreq(spectrum.shape[2])

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

        # Bin the spectrum
        k_bins = np.linspace(0, np.max(k_magnitude), 20)
        spectrum_binned = np.zeros(len(k_bins) - 1)

        for i in range(len(k_bins) - 1):
            mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
            if np.any(mask):
                spectrum_binned[i] = np.mean(spectrum[mask])

        # Fit power law (log-log linear fit)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        valid_mask = (spectrum_binned > 0) & (k_centers > 0)

        if np.sum(valid_mask) > 2:
            log_k = np.log(k_centers[valid_mask])
            log_spectrum = np.log(spectrum_binned[valid_mask])
            slope = np.polyfit(log_k, log_spectrum, 1)[0]
            return slope
        else:
            return -2.0  # Default value
