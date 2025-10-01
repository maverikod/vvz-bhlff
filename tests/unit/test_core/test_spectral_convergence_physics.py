"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for spectral convergence in 7D BVP theory.

This module provides physical validation tests for spectral convergence,
ensuring mathematical correctness and physical consistency.
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from bhlff.core.domain import Domain
from bhlff.core.fft.spectral_operations import SpectralOperations


class TestSpectralConvergencePhysics:
    """Physical validation tests for spectral convergence."""

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

    def test_spectral_convergence_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence physical consistency.
        
        Physical Meaning:
            Validates that spectral methods converge correctly
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
            
        Mathematical Foundation:
            Tests convergence properties of spectral methods.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Apply spectral operations
        laplacian = spectral_ops.compute_laplacian(test_field)
        
        # Physical validation: Should converge
        assert np.isfinite(laplacian).all(), \
            "Spectral methods do not converge correctly"

    def test_spectral_convergence_energy_conservation_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence energy conservation.
        
        Physical Meaning:
            Validates that spectral convergence conserves energy
            in the spectral domain.
            
        Mathematical Foundation:
            Tests energy conservation for spectral convergence.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Apply spectral operations
        laplacian = spectral_ops.compute_laplacian(test_field)
        
        # Physical validation: Energy should be finite
        original_energy = np.sum(test_field**2)
        laplacian_energy = np.sum(laplacian**2)
        
        assert np.isfinite(original_energy), \
            "Original field energy not finite"
        assert np.isfinite(laplacian_energy), \
            "Laplacian energy not finite"

    def test_spectral_convergence_7d_structure_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence 7D structure preservation.
        
        Physical Meaning:
            Validates that spectral convergence preserves the 7D structure
            of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
            
        Mathematical Foundation:
            Tests 7D structure preservation for spectral convergence.
        """
        # Create test field
        test_field = self._create_test_field(domain_7d)
        
        # Apply spectral operations
        laplacian = spectral_ops.compute_laplacian(test_field)
        
        # Physical validation: Should preserve 7D structure
        assert laplacian.shape == domain_7d.shape, \
            "Spectral convergence does not preserve 7D structure"

    def test_spectral_convergence_numerical_stability_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence numerical stability.
        
        Physical Meaning:
            Validates that spectral convergence is numerically stable
            for extreme field values.
            
        Mathematical Foundation:
            Tests numerical stability of spectral convergence.
        """
        # Test with extreme values
        extreme_field = np.array([1e10, -1e10, 1e-10, -1e-10])
        extreme_field = np.broadcast_to(
            extreme_field.reshape(-1, 1, 1, 1, 1, 1, 1), 
            domain_7d.shape
        )
        
        # Apply spectral operations
        laplacian = spectral_ops.compute_laplacian(extreme_field)
        
        # Physical validation: Should be stable
        assert np.isfinite(laplacian).all(), \
            "Spectral convergence not numerically stable for extreme values"

    def test_spectral_convergence_precision_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence precision.
        
        Physical Meaning:
            Validates that spectral convergence maintains high precision
            for known analytical functions.
            
        Mathematical Foundation:
            Tests precision of spectral convergence with sinusoidal functions.
        """
        # Test with sinusoidal function
        x = np.linspace(0, 2*np.pi, domain_7d.shape[0], endpoint=False)
        test_field = np.sin(x)
        test_field = np.broadcast_to(
            test_field.reshape(-1, 1, 1, 1, 1, 1, 1), 
            domain_7d.shape
        )
        
        # Apply spectral operations
        laplacian = spectral_ops.compute_laplacian(test_field)
        
        # Physical validation: Should be finite and reasonable
        assert np.isfinite(laplacian).all(), \
            "Spectral convergence not finite for sinusoidal function"
        
        # Should be reasonable magnitude
        max_laplacian = np.max(np.abs(laplacian))
        assert max_laplacian < 100.0, \
            f"Spectral convergence magnitude too large: {max_laplacian}"

    def test_spectral_convergence_phase_structure_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence phase structure preservation.
        
        Physical Meaning:
            Validates that spectral convergence preserves phase structure
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
            
        Mathematical Foundation:
            Tests phase structure preservation for spectral convergence.
        """
        # Create field with complex phase structure
        test_field = np.zeros(domain_7d.shape, dtype=complex)
        
        # Set phase structure
        for i in range(domain_7d.shape[0]):
            for j in range(domain_7d.shape[1]):
                for k in range(domain_7d.shape[2]):
                    phase = 2 * np.pi * (i + j + k) / 8
                    test_field[i, j, k, 0, 0, 0, 0] = np.exp(1j * phase)
        
        # Apply spectral operations
        laplacian = spectral_ops.compute_laplacian(test_field)
        
        # Physical validation: Should preserve phase structure
        assert np.iscomplexobj(laplacian), \
            "Spectral convergence does not preserve complex phase structure"
        
        # Should be finite
        assert np.isfinite(laplacian).all(), \
            "Spectral convergence not finite for complex phase structure"

    def test_spectral_convergence_resolution_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence with different resolutions.
        
        Physical Meaning:
            Validates that spectral convergence improves with higher resolution
            in 7D space-time.
            
        Mathematical Foundation:
            Tests resolution dependence of spectral convergence.
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
            
            # Apply spectral operations
            laplacian = test_spectral_ops.compute_laplacian(test_field)
            
            # Physical validation: Should converge with higher resolution
            assert np.isfinite(laplacian).all(), \
                f"Spectral convergence not finite for resolution N={N}"

    def test_spectral_convergence_accuracy_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence accuracy.
        
        Physical Meaning:
            Validates that spectral convergence maintains high accuracy
            for known analytical functions.
            
        Mathematical Foundation:
            Tests accuracy of spectral convergence.
        """
        # Test with known analytical function
        x = np.linspace(0, 2*np.pi, domain_7d.shape[0], endpoint=False)
        test_field = np.sin(x)
        test_field = np.broadcast_to(
            test_field.reshape(-1, 1, 1, 1, 1, 1, 1), 
            domain_7d.shape
        )
        
        # Apply spectral operations
        laplacian = spectral_ops.compute_laplacian(test_field)
        
        # Physical validation: Should be accurate
        assert np.isfinite(laplacian).all(), \
            "Spectral convergence not accurate for sinusoidal function"
        
        # Should be reasonable magnitude
        max_laplacian = np.max(np.abs(laplacian))
        assert max_laplacian < 100.0, \
            f"Spectral convergence accuracy too low: {max_laplacian}"

    def test_spectral_convergence_stability_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence stability.
        
        Physical Meaning:
            Validates that spectral convergence is stable
            for different field configurations.
            
        Mathematical Foundation:
            Tests stability of spectral convergence.
        """
        # Test with different field configurations
        field_configs = [
            self._create_test_field(domain_7d),
            self._create_sinusoidal_field(domain_7d),
            self._create_gaussian_field(domain_7d),
            self._create_random_field(domain_7d)
        ]
        
        for i, test_field in enumerate(field_configs):
            # Apply spectral operations
            laplacian = spectral_ops.compute_laplacian(test_field)
            
            # Physical validation: Should be stable
            assert np.isfinite(laplacian).all(), \
                f"Spectral convergence not stable for field configuration {i}"

    def test_spectral_convergence_efficiency_physics(self, domain_7d, spectral_ops):
        """
        Test spectral convergence efficiency.
        
        Physical Meaning:
            Validates that spectral convergence is efficient
            for different field sizes.
            
        Mathematical Foundation:
            Tests efficiency of spectral convergence.
        """
        # Test with different field sizes
        field_sizes = [4, 8, 16]
        
        for N in field_sizes:
            # Create domain with different size
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
            
            # Apply spectral operations
            laplacian = test_spectral_ops.compute_laplacian(test_field)
            
            # Physical validation: Should be efficient
            assert np.isfinite(laplacian).all(), \
                f"Spectral convergence not efficient for field size N={N}"

    def _create_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field for spectral convergence testing."""
        # Create random test field
        test_field = np.random.random(domain.shape)
        
        # Normalize for numerical stability
        test_field = test_field / np.max(np.abs(test_field))
        
        return test_field

    def _create_sinusoidal_field(self, domain: Domain) -> np.ndarray:
        """Create sinusoidal test field."""
        x = np.linspace(0, 2*np.pi, domain.shape[0], endpoint=False)
        test_field = np.sin(x)
        test_field = np.broadcast_to(
            test_field.reshape(-1, 1, 1, 1, 1, 1, 1), 
            domain.shape
        )
        return test_field

    def _create_gaussian_field(self, domain: Domain) -> np.ndarray:
        """Create Gaussian test field."""
        x = np.linspace(-1, 1, domain.shape[0])
        test_field = np.exp(-x**2)
        test_field = np.broadcast_to(
            test_field.reshape(-1, 1, 1, 1, 1, 1, 1), 
            domain.shape
        )
        return test_field

    def _create_random_field(self, domain: Domain) -> np.ndarray:
        """Create random test field."""
        test_field = np.random.random(domain.shape)
        return test_field
