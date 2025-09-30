"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for spectral methods in 7D BVP theory.

This module provides comprehensive physical validation tests for spectral
methods used in the 7D BVP theory, ensuring mathematical correctness and
physical consistency of FFT-based computations.

Physical Meaning:
    Tests validate that spectral methods correctly implement:
    - FFT operations in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    - Spectral derivatives and Laplacian operators
    - Boundary condition handling in spectral space
    - Energy conservation in spectral domain
    - Convergence properties of spectral methods

Mathematical Foundation:
    Validates key spectral operations:
    - FFT: â(k) = ∫ a(x)e^(-ik·x)dx
    - Spectral derivatives: ∂a/∂x → ikâ(k)
    - Spectral Laplacian: ∇²a → -k²â(k)
    - Parseval's theorem: ∫|a(x)|²dx = ∫|â(k)|²dk

Example:
    >>> pytest tests/unit/test_core/test_spectral_methods_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from bhlff.core.domain import Domain
from bhlff.core.fft.spectral_operations import SpectralOperations
from bhlff.core.fft.spectral_derivatives import SpectralDerivatives
from bhlff.core.operators.fractional_laplacian import FractionalLaplacian
from bhlff.core.fft.spectral_operations import SpectralOperations as FFTSpectralOps


class TestSpectralMethodsPhysics:
    """Physical validation tests for spectral methods."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for spectral testing."""
        return Domain(
            L=1.0,  # Smaller domain for memory efficiency
            N=8,    # Lower resolution
            dimensions=3,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def fft_backend(self, domain_7d):
        """Create FFT backend for testing."""
        from bhlff.core.fft import FFTBackend
        return FFTBackend(domain_7d)
    
    @pytest.fixture
    def spectral_ops(self, domain_7d, fft_backend):
        """Create spectral operations for testing."""
        return SpectralOperations(domain_7d, fft_backend)

    @pytest.fixture
    def spectral_derivs(self, domain_7d, fft_backend):
        """Create spectral derivatives for testing."""
        return SpectralDerivatives(domain_7d, fft_backend)

    @pytest.fixture
    def fractional_laplacian(self, domain_7d):
        """Create fractional Laplacian for testing."""
        return FractionalLaplacian(domain_7d, beta=1.5)

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
        real_energy = np.sum(np.abs(test_field)**2)
        
        # Compute FFT
        spectral_field = spectral_ops.fft_forward(test_field)
        
        # Compute energy in spectral space
        spectral_energy = np.sum(np.abs(spectral_field)**2)
        
        # Physical validation: Energy should be conserved
        energy_conservation_error = abs(real_energy - spectral_energy) / real_energy
        assert energy_conservation_error < 1e-12, \
            f"FFT energy not conserved: error = {energy_conservation_error}"
        
        # Physical validation: Energies should be positive
        assert real_energy > 0, f"Negative real energy: {real_energy}"
        assert spectral_energy > 0, f"Negative spectral energy: {spectral_energy}"

    def test_spectral_derivatives_physics(self, domain_7d, spectral_derivs):
        """
        Test spectral derivatives physics.
        
        Physical Meaning:
            Validates that spectral derivatives correctly implement
            ∂a/∂x → ikâ(k) in 7D space-time.
            
        Mathematical Foundation:
            Tests spectral derivative formula: ∂a/∂x = FFT⁻¹(ik·FFT(a))
        """
        # Create test field with known derivatives
        test_field = self._create_analytical_field(domain_7d)
        
        # Compute analytical derivatives
        analytical_derivs = self._compute_analytical_derivatives(test_field, domain_7d)
        
        # Compute spectral derivatives
        spectral_derivs_x = spectral_derivs.compute_derivative_x(test_field)
        spectral_derivs_y = spectral_derivs.compute_derivative_y(test_field)
        spectral_derivs_z = spectral_derivs.compute_derivative_z(test_field)
        
        # Physical validation 1: Spectral derivatives should match analytical
        error_x = np.mean(np.abs(spectral_derivs_x - analytical_derivs['x']))
        error_y = np.mean(np.abs(spectral_derivs_y - analytical_derivs['y']))
        error_z = np.mean(np.abs(spectral_derivs_z - analytical_derivs['z']))
        
        assert error_x < 1e-10, f"X-derivative error too large: {error_x}"
        assert error_y < 1e-10, f"Y-derivative error too large: {error_y}"
        assert error_z < 1e-10, f"Z-derivative error too large: {error_z}"
        
        # Physical validation 2: Derivatives should be finite
        assert np.all(np.isfinite(spectral_derivs_x)), "X-derivative contains non-finite values"
        assert np.all(np.isfinite(spectral_derivs_y)), "Y-derivative contains non-finite values"
        assert np.all(np.isfinite(spectral_derivs_z)), "Z-derivative contains non-finite values"

    def test_spectral_laplacian_physics(self, domain_7d, spectral_derivs):
        """
        Test spectral Laplacian physics.
        
        Physical Meaning:
            Validates that spectral Laplacian correctly implements
            ∇²a → -k²â(k) in 7D space-time.
            
        Mathematical Foundation:
            Tests spectral Laplacian formula: ∇²a = FFT⁻¹(-k²·FFT(a))
        """
        # Create test field with known Laplacian
        test_field = self._create_laplacian_test_field(domain_7d)
        
        # Compute analytical Laplacian
        analytical_laplacian = self._compute_analytical_laplacian(test_field, domain_7d)
        
        # Compute spectral Laplacian
        spectral_laplacian = spectral_derivs.compute_laplacian(test_field)
        
        # Physical validation 1: Spectral Laplacian should match analytical
        error = np.mean(np.abs(spectral_laplacian - analytical_laplacian))
        assert error < 1e-10, f"Spectral Laplacian error too large: {error}"
        
        # Physical validation 2: Laplacian should be finite
        assert np.all(np.isfinite(spectral_laplacian)), "Laplacian contains non-finite values"
        
        # Physical validation 3: Laplacian should have correct sign for test field
        laplacian_sign = np.sign(np.mean(spectral_laplacian))
        expected_sign = np.sign(np.mean(analytical_laplacian))
        assert laplacian_sign == expected_sign, "Laplacian sign incorrect"

    def test_fractional_laplacian_physics(self, domain_7d, fractional_laplacian):
        """
        Test fractional Laplacian physics.
        
        Physical Meaning:
            Validates that fractional Laplacian correctly implements
            (-Δ)^β a → |k|^(2β)â(k) for β ∈ (0,2).
            
        Mathematical Foundation:
            Tests fractional Laplacian: (-Δ)^β a = FFT⁻¹(|k|^(2β)·FFT(a))
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Compute fractional Laplacian
        frac_laplacian = fractional_laplacian.apply(test_field)
        
        # Physical validation 1: Fractional Laplacian should be finite
        assert np.all(np.isfinite(frac_laplacian)), "Fractional Laplacian contains non-finite values"
        
        # Physical validation 2: For β = 1.5, should be between regular and squared Laplacian
        regular_laplacian = self._compute_regular_laplacian(test_field, domain_7d)
        squared_laplacian = self._compute_squared_laplacian(test_field, domain_7d)
        
        # Fractional Laplacian should be intermediate
        frac_mean = np.mean(np.abs(frac_laplacian))
        regular_mean = np.mean(np.abs(regular_laplacian))
        squared_mean = np.mean(np.abs(squared_laplacian))
        
        assert regular_mean < frac_mean < squared_mean, \
            f"Fractional Laplacian not intermediate: {regular_mean} < {frac_mean} < {squared_mean}"
        
        # Physical validation 3: Should preserve field structure
        structure_correlation = np.corrcoef(test_field.flatten(), frac_laplacian.flatten())[0, 1]
        assert abs(structure_correlation) > 0.1, "Fractional Laplacian doesn't preserve structure"

    def test_spectral_boundary_conditions_physics(self, domain_7d, spectral_ops):
        """
        Test spectral boundary conditions physics.
        
        Physical Meaning:
            Validates that spectral methods correctly handle boundary
            conditions in 7D space-time, ensuring periodicity.
            
        Mathematical Foundation:
            Tests that FFT operations respect periodic boundary conditions.
        """
        # Create field with known boundary values
        test_field = self._create_boundary_test_field(domain_7d)
        
        # Apply FFT and inverse FFT
        spectral_field = spectral_ops.fft_forward(test_field)
        reconstructed_field = spectral_ops.fft_inverse(spectral_field)
        
        # Physical validation 1: Reconstruction should be accurate
        reconstruction_error = np.mean(np.abs(test_field - reconstructed_field))
        assert reconstruction_error < 1e-12, \
            f"FFT reconstruction error too large: {reconstruction_error}"
        
        # Physical validation 2: Boundary conditions should be preserved
        self._validate_periodic_boundaries(reconstructed_field, domain_7d)
        
        # Physical validation 3: Spectral field should be finite
        assert np.all(np.isfinite(spectral_field)), "Spectral field contains non-finite values"

    def test_spectral_convergence_physics(self, domain_7d):
        """
        Test spectral method convergence physics.
        
        Physical Meaning:
            Validates that spectral methods converge to the correct
            solution as resolution increases.
            
        Mathematical Foundation:
            Tests convergence rate of spectral methods for smooth functions.
        """
        # Test convergence for different resolutions
        resolutions = [16, 32, 64]
        errors = []
        
        for N in resolutions:
            # Create domain with resolution N
            test_domain = Domain(L=1.0, N=N, dimensions=3, N_phi=16, N_t=32, T=1.0)
            test_spectral_ops = SpectralOperations(test_domain)
            
            # Create analytical field
            analytical_field = self._create_analytical_field(test_domain)
            
            # Compute spectral derivative
            spectral_deriv = test_spectral_ops.compute_derivative_x(analytical_field)
            
            # Compute analytical derivative
            analytical_deriv = self._compute_analytical_derivatives(analytical_field, test_domain)['x']
            
            # Compute error
            error = np.mean(np.abs(spectral_deriv - analytical_deriv))
            errors.append(error)
        
        # Physical validation: Error should decrease with resolution
        for i in range(1, len(errors)):
            assert errors[i] < errors[i-1], \
                f"Error doesn't decrease with resolution: {errors[i]} >= {errors[i-1]}"
        
        # Physical validation: Should achieve spectral accuracy
        final_error = errors[-1]
        assert final_error < 1e-10, f"Final error too large: {final_error}"

    def test_spectral_energy_spectrum_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum physics.
        
        Physical Meaning:
            Validates that the energy spectrum in spectral space
            has correct physical properties and scaling.
            
        Mathematical Foundation:
            Tests energy spectrum: E(k) = |â(k)|² and validates
            power law behavior for turbulent fields.
        """
        # Create field with known energy spectrum
        test_field = self._create_turbulent_field(domain_7d)
        
        # Compute spectral field
        spectral_field = spectral_ops.fft_forward(test_field)
        
        # Compute energy spectrum
        energy_spectrum = np.abs(spectral_field)**2
        
        # Physical validation 1: Energy spectrum should be positive
        assert np.all(energy_spectrum >= 0), "Energy spectrum contains negative values"
        
        # Physical validation 2: Total energy should be conserved
        total_energy = np.sum(energy_spectrum)
        real_energy = np.sum(np.abs(test_field)**2)
        energy_error = abs(total_energy - real_energy) / real_energy
        assert energy_error < 1e-12, f"Energy not conserved in spectrum: {energy_error}"
        
        # Physical validation 3: Energy spectrum should have correct scaling
        self._validate_energy_spectrum_scaling(energy_spectrum, domain_7d)

    def _create_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field for spectral operations."""
        field = np.zeros(domain.shape)
        
        # Create Gaussian field
        center = domain.N // 2
        x = np.arange(domain.N) - center
        y = np.arange(domain.N) - center
        z = np.arange(domain.N) - center
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        gaussian = np.exp(-(X**2 + Y**2 + Z**2) / (2 * (domain.N/8)**2))
        
        field[:, :, :, :, :, :, :] = gaussian[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        
        return field

    def _create_analytical_field(self, domain: Domain) -> np.ndarray:
        """Create field with known analytical derivatives."""
        field = np.zeros(domain.shape)
        
        # Create sinusoidal field: sin(x) * sin(y) * sin(z)
        x = np.linspace(0, 2*np.pi, domain.N)
        y = np.linspace(0, 2*np.pi, domain.N)
        z = np.linspace(0, 2*np.pi, domain.N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        field[:, :, :, :, :, :, :] = np.sin(X) * np.sin(Y) * np.sin(Z)
        
        return field

    def _create_laplacian_test_field(self, domain: Domain) -> np.ndarray:
        """Create field with known Laplacian."""
        field = np.zeros(domain.shape)
        
        # Create field: x² + y² + z² (Laplacian = 6)
        x = np.linspace(-1, 1, domain.N)
        y = np.linspace(-1, 1, domain.N)
        z = np.linspace(-1, 1, domain.N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        field[:, :, :, :, :, :, :] = X**2 + Y**2 + Z**2
        
        return field

    def _create_boundary_test_field(self, domain: Domain) -> np.ndarray:
        """Create field with known boundary values."""
        field = np.zeros(domain.shape)
        
        # Create field that's periodic
        x = np.linspace(0, 2*np.pi, domain.N)
        y = np.linspace(0, 2*np.pi, domain.N)
        z = np.linspace(0, 2*np.pi, domain.N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        field[:, :, :, :, :, :, :] = np.cos(X) * np.cos(Y) * np.cos(Z)
        
        return field

    def _create_turbulent_field(self, domain: Domain) -> np.ndarray:
        """Create field with turbulent-like energy spectrum."""
        field = np.zeros(domain.shape)
        
        # Create random field with power law spectrum
        np.random.seed(42)  # For reproducibility
        field = np.random.randn(*domain.shape)
        
        # Apply low-pass filter to create realistic spectrum
        kx = np.fft.fftfreq(domain.N)
        ky = np.fft.fftfreq(domain.N)
        kz = np.fft.fftfreq(domain.N)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Apply filter
        filter_mask = k_magnitude < 0.3
        field_fft = np.fft.fftn(field, axes=(0, 1, 2))
        field_fft[~filter_mask] = 0
        field = np.fft.ifftn(field_fft, axes=(0, 1, 2)).real
        
        return field

    def _compute_analytical_derivatives(self, field: np.ndarray, domain: Domain) -> Dict[str, np.ndarray]:
        """Compute analytical derivatives of the field."""
        # For sin(x) * sin(y) * sin(z)
        x = np.linspace(0, 2*np.pi, domain.N)
        y = np.linspace(0, 2*np.pi, domain.N)
        z = np.linspace(0, 2*np.pi, domain.N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        deriv_x = np.cos(X) * np.sin(Y) * np.sin(Z)
        deriv_y = np.sin(X) * np.cos(Y) * np.sin(Z)
        deriv_z = np.sin(X) * np.sin(Y) * np.cos(Z)
        
        return {
            'x': deriv_x,
            'y': deriv_y,
            'z': deriv_z
        }

    def _compute_analytical_laplacian(self, field: np.ndarray, domain: Domain) -> np.ndarray:
        """Compute analytical Laplacian of the field."""
        # For x² + y² + z², Laplacian = 6
        return np.full(domain.shape, 6.0)

    def _compute_regular_laplacian(self, field: np.ndarray, domain: Domain) -> np.ndarray:
        """Compute regular Laplacian using finite differences."""
        laplacian = np.zeros_like(field)
        
        # Compute second derivatives
        d2x = np.gradient(np.gradient(field, axis=0), axis=0)
        d2y = np.gradient(np.gradient(field, axis=1), axis=1)
        d2z = np.gradient(np.gradient(field, axis=2), axis=2)
        
        laplacian = d2x + d2y + d2z
        
        return laplacian

    def _compute_squared_laplacian(self, field: np.ndarray, domain: Domain) -> np.ndarray:
        """Compute squared Laplacian."""
        regular_laplacian = self._compute_regular_laplacian(field, domain)
        return regular_laplacian**2

    def _validate_periodic_boundaries(self, field: np.ndarray, domain: Domain) -> None:
        """Validate periodic boundary conditions."""
        # Check spatial boundaries
        assert np.allclose(field[0, :, :, :, :, :, :], 
                          field[-1, :, :, :, :, :, :], atol=1e-10), \
            "Spatial boundary conditions not periodic"
        
        # Check phase boundaries
        assert np.allclose(field[:, :, :, 0, :, :, :], 
                          field[:, :, :, -1, :, :, :], atol=1e-10), \
            "Phase boundary conditions not periodic"

    def _validate_energy_spectrum_scaling(self, energy_spectrum: np.ndarray, domain: Domain) -> None:
        """Validate energy spectrum scaling."""
        # Compute radial average
        kx = np.fft.fftfreq(domain.N)
        ky = np.fft.fftfreq(domain.N)
        kz = np.fft.fftfreq(domain.N)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Bin the spectrum
        k_bins = np.linspace(0, np.max(k_magnitude), 20)
        spectrum_binned = np.zeros(len(k_bins)-1)
        
        for i in range(len(k_bins)-1):
            mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i+1])
            if np.any(mask):
                spectrum_binned[i] = np.mean(energy_spectrum[mask])
        
        # Check that spectrum is decreasing (for turbulent field)
        valid_bins = spectrum_binned > 0
        if np.sum(valid_bins) > 3:
            decreasing_ratio = np.sum(np.diff(spectrum_binned[valid_bins]) < 0) / np.sum(valid_bins)
            assert decreasing_ratio > 0.5, "Energy spectrum not decreasing"
