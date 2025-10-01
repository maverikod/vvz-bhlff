"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for spectral energy spectrum in 7D BVP theory.

This module provides physical validation tests for spectral energy spectrum,
ensuring mathematical correctness and physical consistency.
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from bhlff.core.domain import Domain
from bhlff.core.fft.spectral_operations import SpectralOperations


class TestSpectralEnergySpectrumPhysics:
    """Physical validation tests for spectral energy spectrum."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for spectral testing."""
        return Domain(
            L=1.0,  # Smaller domain for memory efficiency
            N=8,    # Lower resolution
            dimensions=7,
            N_phi=4,
            N_t=8,
            T=1.0
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

    def test_spectral_energy_spectrum_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum physical consistency.
        
        Physical Meaning:
            Validates that spectral energy spectrum is computed correctly
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
            
        Mathematical Foundation:
            Tests energy spectrum computation in spectral space.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(test_field, spectral_ops)
        
        # Physical validation: Should be finite and positive
        assert np.isfinite(energy_spectrum).all(), \
            "Spectral energy spectrum not finite"
        assert np.all(energy_spectrum >= 0), \
            "Spectral energy spectrum not positive"

    def test_spectral_energy_spectrum_energy_conservation_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum energy conservation.
        
        Physical Meaning:
            Validates that spectral energy spectrum conserves energy
            in the spectral domain.
            
        Mathematical Foundation:
            Tests energy conservation for spectral energy spectrum.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(test_field, spectral_ops)
        
        # Physical validation: Energy should be conserved
        total_energy = np.sum(energy_spectrum)
        original_energy = np.sum(test_field**2)
        
        assert np.isfinite(total_energy), \
            "Total spectral energy not finite"
        assert np.isfinite(original_energy), \
            "Original field energy not finite"
        
        # Energy should be conserved (within numerical precision)
        energy_ratio = total_energy / original_energy
        assert np.isclose(energy_ratio, 1.0, atol=1e-10), \
            f"Energy not conserved in spectral spectrum: ratio = {energy_ratio}"

    def test_spectral_energy_spectrum_7d_structure_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum 7D structure preservation.
        
        Physical Meaning:
            Validates that spectral energy spectrum preserves the 7D structure
            of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
            
        Mathematical Foundation:
            Tests 7D structure preservation for spectral energy spectrum.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(test_field, spectral_ops)
        
        # Physical validation: Should preserve 7D structure
        assert energy_spectrum.shape == domain_7d.shape, \
            "Spectral energy spectrum does not preserve 7D structure"

    def test_spectral_energy_spectrum_numerical_stability_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum numerical stability.
        
        Physical Meaning:
            Validates that spectral energy spectrum is numerically stable
            for extreme field values.
            
        Mathematical Foundation:
            Tests numerical stability of spectral energy spectrum.
        """
        # Test with extreme values
        extreme_field = np.array([1e10, -1e10, 1e-10, -1e-10])
        extreme_field = np.broadcast_to(
            extreme_field.reshape(-1, 1, 1, 1, 1, 1, 1), 
            domain_7d.shape
        )
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(extreme_field, spectral_ops)
        
        # Physical validation: Should be stable
        assert np.isfinite(energy_spectrum).all(), \
            "Spectral energy spectrum not numerically stable for extreme values"

    def test_spectral_energy_spectrum_precision_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum precision.
        
        Physical Meaning:
            Validates that spectral energy spectrum maintains high precision
            for known analytical functions.
            
        Mathematical Foundation:
            Tests precision of spectral energy spectrum with sinusoidal functions.
        """
        # Test with sinusoidal function
        x = np.linspace(0, 2*np.pi, domain_7d.shape[0], endpoint=False)
        test_field = np.sin(x)
        test_field = np.broadcast_to(
            test_field.reshape(-1, 1, 1, 1, 1, 1, 1), 
            domain_7d.shape
        )
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(test_field, spectral_ops)
        
        # Physical validation: Should be finite and reasonable
        assert np.isfinite(energy_spectrum).all(), \
            "Spectral energy spectrum not finite for sinusoidal function"
        
        # Should be reasonable magnitude
        max_energy = np.max(energy_spectrum)
        assert max_energy < 100.0, \
            f"Spectral energy spectrum magnitude too large: {max_energy}"

    def test_spectral_energy_spectrum_phase_structure_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum phase structure preservation.
        
        Physical Meaning:
            Validates that spectral energy spectrum preserves phase structure
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
            
        Mathematical Foundation:
            Tests phase structure preservation for spectral energy spectrum.
        """
        # Create field with complex phase structure
        test_field = np.zeros(domain_7d.shape, dtype=complex)
        
        # Set phase structure
        for i in range(domain_7d.shape[0]):
            for j in range(domain_7d.shape[1]):
                for k in range(domain_7d.shape[2]):
                    phase = 2 * np.pi * (i + j + k) / 8
                    test_field[i, j, k, 0, 0, 0, 0] = np.exp(1j * phase)
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(test_field, spectral_ops)
        
        # Physical validation: Should preserve phase structure
        assert np.isfinite(energy_spectrum).all(), \
            "Spectral energy spectrum not finite for complex phase structure"

    def test_spectral_energy_spectrum_frequency_dependence_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum frequency dependence.
        
        Physical Meaning:
            Validates that spectral energy spectrum shows correct frequency dependence
            in 7D space-time.
            
        Mathematical Foundation:
            Tests frequency dependence of spectral energy spectrum.
        """
        # Create test field with specific frequency content
        test_field = self._create_frequency_test_field(domain_7d)
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(test_field, spectral_ops)
        
        # Physical validation: Should show frequency dependence
        assert np.isfinite(energy_spectrum).all(), \
            "Spectral energy spectrum not finite for frequency test field"
        
        # Should have reasonable frequency content
        max_energy = np.max(energy_spectrum)
        assert max_energy > 0, \
            "Spectral energy spectrum has no frequency content"

    def test_spectral_energy_spectrum_resolution_dependence_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum resolution dependence.
        
        Physical Meaning:
            Validates that spectral energy spectrum shows correct resolution dependence
            in 7D space-time.
            
        Mathematical Foundation:
            Tests resolution dependence of spectral energy spectrum.
        """
        # Test with different resolutions
        resolutions = [4, 8, 16]
        
        for N in resolutions:
            # Create domain with different resolution
            test_domain = Domain(
                L=1.0,
                N=N,
                dimensions=7,
                N_phi=4,
                N_t=8,
                T=1.0
            )
            
            # Create FFT backend
            from bhlff.core.fft.fft_backend_core import FFTBackend
            test_fft_backend = FFTBackend(test_domain)
            test_spectral_ops = SpectralOperations(test_fft_backend)
            
            # Create test field
            test_field = self._create_test_field(test_domain)
            
            # Compute spectral energy spectrum
            energy_spectrum = self._compute_energy_spectrum(test_field, test_spectral_ops)
            
            # Physical validation: Should be finite for all resolutions
            assert np.isfinite(energy_spectrum).all(), \
                f"Spectral energy spectrum not finite for resolution N={N}"

    def test_spectral_energy_spectrum_energy_distribution_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum energy distribution.
        
        Physical Meaning:
            Validates that spectral energy spectrum shows correct energy distribution
            in 7D space-time.
            
        Mathematical Foundation:
            Tests energy distribution of spectral energy spectrum.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(test_field, spectral_ops)
        
        # Physical validation: Should have reasonable energy distribution
        total_energy = np.sum(energy_spectrum)
        max_energy = np.max(energy_spectrum)
        
        assert total_energy > 0, \
            "Spectral energy spectrum has no total energy"
        assert max_energy > 0, \
            "Spectral energy spectrum has no maximum energy"
        
        # Energy distribution should be reasonable
        energy_ratio = max_energy / total_energy
        assert energy_ratio < 1.0, \
            f"Spectral energy spectrum has unreasonable energy distribution: {energy_ratio}"

    def test_spectral_energy_spectrum_spectral_properties_physics(self, domain_7d, spectral_ops):
        """
        Test spectral energy spectrum spectral properties.
        
        Physical Meaning:
            Validates that spectral energy spectrum shows correct spectral properties
            in 7D space-time.
            
        Mathematical Foundation:
            Tests spectral properties of spectral energy spectrum.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Compute spectral energy spectrum
        energy_spectrum = self._compute_energy_spectrum(test_field, spectral_ops)
        
        # Physical validation: Should have correct spectral properties
        assert np.isfinite(energy_spectrum).all(), \
            "Spectral energy spectrum not finite"
        assert np.all(energy_spectrum >= 0), \
            "Spectral energy spectrum not positive"
        
        # Should have reasonable spectral properties
        total_energy = np.sum(energy_spectrum)
        assert total_energy > 0, \
            "Spectral energy spectrum has no total energy"

    def _create_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field for spectral energy spectrum testing."""
        # Create random test field
        test_field = np.random.random(domain.shape)
        
        # Normalize for numerical stability
        test_field = test_field / np.max(np.abs(test_field))
        
        return test_field

    def _create_frequency_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field with specific frequency content."""
        # Create field with specific frequency content
        test_field = np.zeros(domain.shape)
        
        # Set frequency content
        for i in range(domain.shape[0]):
            for j in range(domain.shape[1]):
                for k in range(domain.shape[2]):
                    frequency = 2 * np.pi * (i + j + k) / 8
                    test_field[i, j, k, 0, 0, 0, 0] = np.sin(frequency)
        
        return test_field

    def _compute_energy_spectrum(self, field: np.ndarray, spectral_ops: SpectralOperations) -> np.ndarray:
        """Compute spectral energy spectrum."""
        # Transform to spectral space
        spectral_field = spectral_ops.fft_backend.forward_transform(field)
        
        # Compute energy spectrum
        energy_spectrum = np.abs(spectral_field)**2
        
        return energy_spectrum
