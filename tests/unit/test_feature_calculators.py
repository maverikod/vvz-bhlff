"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for feature calculators.

This module contains comprehensive tests for feature calculators
in 7D phase field beating analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.models.level_c.beating.ml.core.feature_calculators import FeatureCalculator


class TestFeatureCalculator:
    """Test suite for feature calculators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = FeatureCalculator()
        self.test_envelope = np.random.random((32, 32, 32)) + 1j * np.random.random((32, 32, 32))
        self.test_envelope_2d = np.random.random((64, 64)) + 1j * np.random.random((64, 64))
    
    def test_initialization(self):
        """Test feature calculator initialization."""
        assert hasattr(self.calculator, 'feature_cache')
        assert hasattr(self.calculator, 'calculation_precision')
        assert hasattr(self.calculator, 'max_array_size')
        assert self.calculator.calculation_precision == 1e-12
        assert self.calculator.max_array_size == 1000000
    
    def test_spectral_entropy_calculation(self):
        """Test spectral entropy calculation."""
        entropy = self.calculator.calculate_spectral_entropy(self.test_envelope)
        
        assert isinstance(entropy, float)
        assert np.isfinite(entropy)
        assert entropy >= 0.0
        
        # Test with zero envelope
        zero_envelope = np.zeros((16, 16))
        entropy_zero = self.calculator.calculate_spectral_entropy(zero_envelope)
        assert entropy_zero == 0.0
    
    def test_frequency_spacing_calculation(self):
        """Test frequency spacing calculation."""
        spacing = self.calculator.calculate_frequency_spacing(self.test_envelope, (32, 32, 32))
        
        assert isinstance(spacing, float)
        assert np.isfinite(spacing)
        assert spacing > 0.0
    
    def test_frequency_bandwidth_calculation(self):
        """Test frequency bandwidth calculation."""
        bandwidth = self.calculator.calculate_frequency_bandwidth(self.test_envelope)
        
        assert isinstance(bandwidth, float)
        assert np.isfinite(bandwidth)
        assert 0.0 <= bandwidth <= 1.0
    
    def test_autocorrelation_calculation(self):
        """Test autocorrelation calculation."""
        autocorr = self.calculator.calculate_autocorrelation(self.test_envelope)
        
        assert isinstance(autocorr, float)
        assert np.isfinite(autocorr)
        assert 0.0 <= autocorr <= 1.0
        
        # Test with 2D envelope
        autocorr_2d = self.calculator.calculate_autocorrelation(self.test_envelope_2d)
        assert isinstance(autocorr_2d, float)
        assert np.isfinite(autocorr_2d)
    
    def test_frequency_coupling_strength_calculation(self):
        """Test frequency coupling strength calculation."""
        coupling = self.calculator.calculate_frequency_coupling_strength(self.test_envelope)
        
        assert isinstance(coupling, float)
        assert np.isfinite(coupling)
        assert coupling >= 0.0
    
    def test_mode_interaction_energy_calculation(self):
        """Test mode interaction energy calculation."""
        energy = self.calculator.calculate_mode_interaction_energy(self.test_envelope)
        
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        # Energy can be negative for complex fields, so we check it's finite
        assert np.isfinite(energy)
        
        # Test with 2D envelope
        energy_2d = self.calculator.calculate_mode_interaction_energy(self.test_envelope_2d)
        assert isinstance(energy_2d, float)
        assert np.isfinite(energy_2d)
    
    def test_coupling_symmetry_calculation(self):
        """Test coupling symmetry calculation."""
        symmetry = self.calculator.calculate_coupling_symmetry(self.test_envelope)
        
        assert isinstance(symmetry, float)
        assert np.isfinite(symmetry)
        assert symmetry >= 0.0
    
    def test_nonlinear_strength_calculation(self):
        """Test nonlinear strength calculation."""
        strength = self.calculator.calculate_nonlinear_strength(self.test_envelope)
        
        assert isinstance(strength, float)
        assert np.isfinite(strength)
        assert strength >= 0.0
    
    def test_mode_mixing_degree_calculation(self):
        """Test mode mixing degree calculation."""
        mixing = self.calculator.calculate_mode_mixing_degree(self.test_envelope)
        
        assert isinstance(mixing, float)
        assert np.isfinite(mixing)
        assert 0.0 <= mixing <= 1.0
    
    def test_coupling_efficiency_calculation(self):
        """Test coupling efficiency calculation."""
        efficiency = self.calculator.calculate_coupling_efficiency(self.test_envelope)
        
        assert isinstance(efficiency, float)
        assert np.isfinite(efficiency)
        assert 0.0 <= efficiency <= 1.0
    
    def test_large_array_handling(self):
        """Test handling of large arrays."""
        # Create large array
        large_envelope = np.random.random((100, 100, 100)) + 1j * np.random.random((100, 100, 100))
        
        # Test that calculations don't fail with large arrays
        entropy = self.calculator.calculate_spectral_entropy(large_envelope)
        assert isinstance(entropy, float)
        assert np.isfinite(entropy)
        
        autocorr = self.calculator.calculate_autocorrelation(large_envelope)
        assert isinstance(autocorr, float)
        assert np.isfinite(autocorr)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very small envelope
        small_envelope = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        entropy = self.calculator.calculate_spectral_entropy(small_envelope)
        assert isinstance(entropy, float)
        assert np.isfinite(entropy)
        
        # Test with constant envelope
        constant_envelope = np.ones((8, 8))
        
        entropy_const = self.calculator.calculate_spectral_entropy(constant_envelope)
        assert isinstance(entropy_const, float)
        assert np.isfinite(entropy_const)
    
    def test_feature_calculation_consistency(self):
        """Test consistency of feature calculations."""
        # Calculate all features for the same envelope
        features = {
            'spectral_entropy': self.calculator.calculate_spectral_entropy(self.test_envelope),
            'frequency_spacing': self.calculator.calculate_frequency_spacing(self.test_envelope, (32, 32, 32)),
            'frequency_bandwidth': self.calculator.calculate_frequency_bandwidth(self.test_envelope),
            'autocorrelation': self.calculator.calculate_autocorrelation(self.test_envelope),
            'coupling_strength': self.calculator.calculate_frequency_coupling_strength(self.test_envelope),
            'interaction_energy': self.calculator.calculate_mode_interaction_energy(self.test_envelope),
            'coupling_symmetry': self.calculator.calculate_coupling_symmetry(self.test_envelope),
            'nonlinear_strength': self.calculator.calculate_nonlinear_strength(self.test_envelope),
            'mixing_degree': self.calculator.calculate_mode_mixing_degree(self.test_envelope),
            'coupling_efficiency': self.calculator.calculate_coupling_efficiency(self.test_envelope)
        }
        
        # All features should be finite and of correct type
        for name, value in features.items():
            assert isinstance(value, float), f"{name} should be float"
            assert np.isfinite(value), f"{name} should be finite"
            if 'bandwidth' in name or 'autocorrelation' in name or 'mixing' in name or 'efficiency' in name:
                assert 0.0 <= value <= 1.0, f"{name} should be in [0,1]"
            elif 'entropy' in name:
                # Spectral entropy can be > 1.0 for complex fields
                assert value >= 0.0, f"{name} should be non-negative"
            elif 'interaction_energy' in name:
                # Interaction energy can be negative for complex fields
                assert np.isfinite(value), f"{name} should be finite"
            else:
                assert value >= 0.0, f"{name} should be non-negative"
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        # Test that large arrays are handled efficiently
        very_large_envelope = np.random.random((200, 200, 200)) + 1j * np.random.random((200, 200, 200))
        
        # This should not cause memory issues due to sampling in autocorrelation
        autocorr = self.calculator.calculate_autocorrelation(very_large_envelope)
        assert isinstance(autocorr, float)
        assert np.isfinite(autocorr)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        small_envelope = np.array([[1e-15, 1e-15], [1e-15, 1e-15]])
        
        entropy = self.calculator.calculate_spectral_entropy(small_envelope)
        assert isinstance(entropy, float)
        assert np.isfinite(entropy)
        
        # Test with very large values
        large_envelope = np.array([[1e15, 1e15], [1e15, 1e15]])
        
        entropy_large = self.calculator.calculate_spectral_entropy(large_envelope)
        assert isinstance(entropy_large, float)
        assert np.isfinite(entropy_large)
