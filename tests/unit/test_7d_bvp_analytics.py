"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for 7D BVP analytics.

This module contains comprehensive tests for 7D BVP analytics
in 7D phase field beating analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.models.level_c.beating.ml.core.bvp_7d_analytics import BVP7DAnalytics


class TestBVP7DAnalytics:
    """Test suite for 7D BVP analytics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analytics = BVP7DAnalytics()
        self.test_phase_features = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.9, 0.1, 0.3, 0.7, 0.5, 0.8, 0.2])
        self.test_features = {
            "spectral_entropy": 0.5,
            "frequency_spacing": 0.3,
            "frequency_bandwidth": 0.7,
            "phase_coherence": 0.8,
            "topological_charge": 0.5
        }
        self.test_coupling_features = {
            "coupling_strength": 0.6,
            "interaction_energy": 0.8,
            "coupling_symmetry": 0.4,
            "nonlinear_strength": 0.7,
            "mixing_degree": 0.3,
            "coupling_efficiency": 0.9,
            "phase_coherence": 0.8,
            "topological_charge": 0.5
        }
    
    def test_initialization(self):
        """Test 7D BVP analytics initialization."""
        assert hasattr(self.analytics, 'analytical_cache')
        assert hasattr(self.analytics, 'prediction_precision')
        assert hasattr(self.analytics, 'phase_field_dimensions')
        assert hasattr(self.analytics, 'bvp_parameters')
        assert self.analytics.prediction_precision == 1e-12
        assert self.analytics.phase_field_dimensions == 7
        assert 'mu' in self.analytics.bvp_parameters
        assert 'beta' in self.analytics.bvp_parameters
        assert 'lambda_param' in self.analytics.bvp_parameters
        assert 'nu' in self.analytics.bvp_parameters
    
    def test_7d_frequency_prediction(self):
        """Test 7D frequency prediction."""
        frequencies = self.analytics.compute_7d_frequency_prediction(
            self.test_phase_features, self.test_features
        )
        
        assert isinstance(frequencies, list)
        assert len(frequencies) == 3
        assert all(isinstance(f, float) for f in frequencies)
        assert all(np.isfinite(f) for f in frequencies)
        assert all(f > 0.0 for f in frequencies)
    
    def test_7d_coupling_prediction(self):
        """Test 7D coupling prediction."""
        coupling = self.analytics.compute_7d_coupling_prediction(
            self.test_phase_features, self.test_coupling_features
        )
        
        assert isinstance(coupling, dict)
        assert len(coupling) == 6
        expected_keys = [
            "coupling_strength", "interaction_energy", "coupling_symmetry",
            "nonlinear_strength", "mixing_degree", "coupling_efficiency"
        ]
        for key in expected_keys:
            assert key in coupling
            assert isinstance(coupling[key], float)
            assert np.isfinite(coupling[key])
            assert coupling[key] >= 0.0
    
    def test_analytical_confidence(self):
        """Test analytical confidence calculation."""
        confidence = self.analytics.compute_analytical_confidence(self.test_features)
        
        assert isinstance(confidence, float)
        assert np.isfinite(confidence)
        assert 0.0 <= confidence <= 1.0
        
        # Test with extreme values
        extreme_features = {
            "phase_coherence": 1.0,
            "topological_charge": 1.0
        }
        extreme_confidence = self.analytics.compute_analytical_confidence(extreme_features)
        assert 0.0 <= extreme_confidence <= 1.0
    
    def test_coupling_analytical_confidence(self):
        """Test coupling analytical confidence calculation."""
        confidence = self.analytics.compute_coupling_analytical_confidence(self.test_coupling_features)
        
        assert isinstance(confidence, float)
        assert np.isfinite(confidence)
        assert 0.0 <= confidence <= 1.0
        
        # Test with extreme values
        extreme_features = {
            "interaction_energy": 1.0,
            "phase_coherence": 1.0
        }
        extreme_confidence = self.analytics.compute_coupling_analytical_confidence(extreme_features)
        assert 0.0 <= extreme_confidence <= 1.0
    
    def test_analytical_feature_importance(self):
        """Test analytical feature importance calculation."""
        importance = self.analytics.compute_analytical_feature_importance(self.test_features)
        
        assert isinstance(importance, dict)
        expected_keys = [
            "spectral_entropy", "frequency_spacing", "frequency_bandwidth",
            "phase_coherence", "topological_charge"
        ]
        for key in expected_keys:
            assert key in importance
            assert isinstance(importance[key], float)
            assert 0.0 <= importance[key] <= 1.0
        
        # Check that importance values sum to 1.0
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 1e-10
    
    def test_coupling_analytical_feature_importance(self):
        """Test coupling analytical feature importance calculation."""
        importance = self.analytics.compute_coupling_analytical_feature_importance(self.test_coupling_features)
        
        assert isinstance(importance, dict)
        expected_keys = [
            "coupling_strength", "interaction_energy", "coupling_symmetry",
            "nonlinear_strength", "mixing_degree", "coupling_efficiency"
        ]
        for key in expected_keys:
            assert key in importance
            assert isinstance(importance[key], float)
            assert 0.0 <= importance[key] <= 1.0
        
        # Check that importance values sum to 1.0
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 1e-10
    
    def test_base_frequency_computation(self):
        """Test base frequency computation."""
        base_freq = self.analytics._compute_base_frequency_7d(0.5, 0.8)
        
        assert isinstance(base_freq, float)
        assert np.isfinite(base_freq)
        assert base_freq > 0.0
        
        # Test with zero values
        base_freq_zero = self.analytics._compute_base_frequency_7d(0.0, 0.0)
        assert base_freq_zero == 0.0
    
    def test_spacing_factor_computation(self):
        """Test spacing factor computation."""
        spacing_factor = self.analytics._compute_spacing_factor_7d(0.3, 0.5)
        
        assert isinstance(spacing_factor, float)
        assert np.isfinite(spacing_factor)
        assert spacing_factor > 0.0
        
        # Test with zero values
        spacing_factor_zero = self.analytics._compute_spacing_factor_7d(0.0, 0.0)
        assert spacing_factor_zero == 0.0
    
    def test_bandwidth_factor_computation(self):
        """Test bandwidth factor computation."""
        bandwidth_factor = self.analytics._compute_bandwidth_factor_7d(0.7, 0.8)
        
        assert isinstance(bandwidth_factor, float)
        assert np.isfinite(bandwidth_factor)
        assert bandwidth_factor > 0.0
        
        # Test with zero values
        bandwidth_factor_zero = self.analytics._compute_bandwidth_factor_7d(0.0, 0.0)
        assert bandwidth_factor_zero == 0.0
    
    def test_coupling_strength_computation(self):
        """Test coupling strength computation."""
        coupling_strength = self.analytics._compute_coupling_strength_7d(0.6, 0.8, 0.5)
        
        assert isinstance(coupling_strength, float)
        assert np.isfinite(coupling_strength)
        assert coupling_strength >= 0.0
    
    def test_interaction_energy_computation(self):
        """Test interaction energy computation."""
        interaction_energy = self.analytics._compute_interaction_energy_7d(0.8, 0.8, 0.5)
        
        assert isinstance(interaction_energy, float)
        assert np.isfinite(interaction_energy)
        assert interaction_energy >= 0.0
    
    def test_coupling_symmetry_computation(self):
        """Test coupling symmetry computation."""
        coupling_symmetry = self.analytics._compute_coupling_symmetry_7d(0.4, 0.8, 0.5)
        
        assert isinstance(coupling_symmetry, float)
        assert np.isfinite(coupling_symmetry)
        assert coupling_symmetry >= 0.0
    
    def test_nonlinear_strength_computation(self):
        """Test nonlinear strength computation."""
        nonlinear_strength = self.analytics._compute_nonlinear_strength_7d(0.7, 0.8, 0.5)
        
        assert isinstance(nonlinear_strength, float)
        assert np.isfinite(nonlinear_strength)
        assert nonlinear_strength >= 0.0
    
    def test_mixing_degree_computation(self):
        """Test mixing degree computation."""
        mixing_degree = self.analytics._compute_mixing_degree_7d(0.3, 0.8, 0.5)
        
        assert isinstance(mixing_degree, float)
        assert np.isfinite(mixing_degree)
        assert mixing_degree >= 0.0
    
    def test_coupling_efficiency_computation(self):
        """Test coupling efficiency computation."""
        coupling_efficiency = self.analytics._compute_coupling_efficiency_7d(0.9, 0.8, 0.5)
        
        assert isinstance(coupling_efficiency, float)
        assert np.isfinite(coupling_efficiency)
        assert coupling_efficiency >= 0.0
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with missing features
        minimal_features = {"spectral_entropy": 0.5}
        frequencies = self.analytics.compute_7d_frequency_prediction(
            self.test_phase_features, minimal_features
        )
        assert len(frequencies) == 3
        assert all(isinstance(f, float) for f in frequencies)
        
        # Test with empty features
        empty_features = {}
        frequencies_empty = self.analytics.compute_7d_frequency_prediction(
            self.test_phase_features, empty_features
        )
        assert len(frequencies_empty) == 3
        assert all(isinstance(f, float) for f in frequencies_empty)
    
    def test_prediction_consistency(self):
        """Test prediction consistency."""
        # Test that predictions are consistent for same input
        frequencies1 = self.analytics.compute_7d_frequency_prediction(
            self.test_phase_features, self.test_features
        )
        frequencies2 = self.analytics.compute_7d_frequency_prediction(
            self.test_phase_features, self.test_features
        )
        
        assert len(frequencies1) == len(frequencies2)
        for f1, f2 in zip(frequencies1, frequencies2):
            assert abs(f1 - f2) < 1e-10
        
        # Test coupling prediction consistency
        coupling1 = self.analytics.compute_7d_coupling_prediction(
            self.test_phase_features, self.test_coupling_features
        )
        coupling2 = self.analytics.compute_7d_coupling_prediction(
            self.test_phase_features, self.test_coupling_features
        )
        
        assert len(coupling1) == len(coupling2)
        for key in coupling1:
            assert abs(coupling1[key] - coupling2[key]) < 1e-10
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large values
        large_features = {
            "spectral_entropy": 100.0,
            "frequency_spacing": 100.0,
            "frequency_bandwidth": 100.0,
            "phase_coherence": 100.0,
            "topological_charge": 100.0
        }
        
        frequencies = self.analytics.compute_7d_frequency_prediction(
            self.test_phase_features, large_features
        )
        assert all(np.isfinite(f) for f in frequencies)
        
        # Test with very small values
        small_features = {
            "spectral_entropy": 1e-10,
            "frequency_spacing": 1e-10,
            "frequency_bandwidth": 1e-10,
            "phase_coherence": 1e-10,
            "topological_charge": 1e-10
        }
        
        frequencies_small = self.analytics.compute_7d_frequency_prediction(
            self.test_phase_features, small_features
        )
        assert all(np.isfinite(f) for f in frequencies_small)
