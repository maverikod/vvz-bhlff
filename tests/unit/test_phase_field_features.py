"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for phase field features.

This module contains comprehensive tests for phase field features
in 7D phase field beating analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.models.level_c.beating.ml.core.phase_field_features import PhaseFieldFeatures


class TestPhaseFieldFeatures:
    """Test suite for phase field features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.features = PhaseFieldFeatures()
        self.test_features = {
            "coupling_symmetry": 0.4,
            "autocorrelation": 0.8,
            "mixing_degree": 0.3,
            "nonlinear_strength": 0.7,
            "interaction_energy": 0.8,
            "coupling_strength": 0.6,
            "frequency_spacing": 0.3,
            "frequency_bandwidth": 0.7
        }
    
    def test_initialization(self):
        """Test phase field features initialization."""
        assert hasattr(self.features, 'feature_cache')
        assert hasattr(self.features, 'computation_precision')
        assert hasattr(self.features, 'phase_field_dimensions')
        assert hasattr(self.features, 'feature_weights')
        assert self.features.computation_precision == 1e-12
        assert self.features.phase_field_dimensions == 7
        assert len(self.features.feature_weights) == 4
        assert 'phase_coherence' in self.features.feature_weights
        assert 'topological_charge' in self.features.feature_weights
        assert 'energy_density' in self.features.feature_weights
        assert 'phase_velocity' in self.features.feature_weights
    
    def test_7d_phase_field_features_computation(self):
        """Test 7D phase field features computation."""
        features_list = self.features.compute_7d_phase_field_features(self.test_features)
        
        assert isinstance(features_list, list)
        assert len(features_list) == 4
        assert all(isinstance(f, float) for f in features_list)
        assert all(np.isfinite(f) for f in features_list)
        
        # Check that features are in expected order
        phase_coherence, topological_charge, energy_density, phase_velocity = features_list
        
        # Phase coherence should be coupling_symmetry * autocorrelation
        expected_phase_coherence = self.test_features["coupling_symmetry"] * self.test_features["autocorrelation"]
        assert abs(phase_coherence - expected_phase_coherence) < 1e-10
        
        # Topological charge should be mixing_degree * nonlinear_strength
        expected_topological_charge = self.test_features["mixing_degree"] * self.test_features["nonlinear_strength"]
        assert abs(topological_charge - expected_topological_charge) < 1e-10
        
        # Energy density should be interaction_energy * coupling_strength
        expected_energy_density = self.test_features["interaction_energy"] * self.test_features["coupling_strength"]
        assert abs(energy_density - expected_energy_density) < 1e-10
        
        # Phase velocity should be frequency_spacing * frequency_bandwidth
        expected_phase_velocity = self.test_features["frequency_spacing"] * self.test_features["frequency_bandwidth"]
        assert abs(phase_velocity - expected_phase_velocity) < 1e-10
    
    def test_phase_coherence_computation(self):
        """Test phase coherence computation."""
        phase_coherence = self.features._compute_phase_coherence(self.test_features)
        
        assert isinstance(phase_coherence, float)
        assert np.isfinite(phase_coherence)
        assert phase_coherence >= 0.0
        
        # Test with zero values
        zero_features = {"coupling_symmetry": 0.0, "autocorrelation": 0.0}
        phase_coherence_zero = self.features._compute_phase_coherence(zero_features)
        assert phase_coherence_zero == 0.0
        
        # Test with missing keys
        minimal_features = {"coupling_symmetry": 0.5}
        phase_coherence_minimal = self.features._compute_phase_coherence(minimal_features)
        assert phase_coherence_minimal == 0.0
    
    def test_topological_charge_computation(self):
        """Test topological charge computation."""
        topological_charge = self.features._compute_topological_charge(self.test_features)
        
        assert isinstance(topological_charge, float)
        assert np.isfinite(topological_charge)
        assert topological_charge >= 0.0
        
        # Test with zero values
        zero_features = {"mixing_degree": 0.0, "nonlinear_strength": 0.0}
        topological_charge_zero = self.features._compute_topological_charge(zero_features)
        assert topological_charge_zero == 0.0
        
        # Test with missing keys
        minimal_features = {"mixing_degree": 0.5}
        topological_charge_minimal = self.features._compute_topological_charge(minimal_features)
        assert topological_charge_minimal == 0.0
    
    def test_energy_density_computation(self):
        """Test energy density computation."""
        energy_density = self.features._compute_energy_density(self.test_features)
        
        assert isinstance(energy_density, float)
        assert np.isfinite(energy_density)
        assert energy_density >= 0.0
        
        # Test with zero values
        zero_features = {"interaction_energy": 0.0, "coupling_strength": 0.0}
        energy_density_zero = self.features._compute_energy_density(zero_features)
        assert energy_density_zero == 0.0
        
        # Test with missing keys
        minimal_features = {"interaction_energy": 0.5}
        energy_density_minimal = self.features._compute_energy_density(minimal_features)
        assert energy_density_minimal == 0.0
    
    def test_phase_velocity_computation(self):
        """Test phase velocity computation."""
        phase_velocity = self.features._compute_phase_velocity(self.test_features)
        
        assert isinstance(phase_velocity, float)
        assert np.isfinite(phase_velocity)
        assert phase_velocity >= 0.0
        
        # Test with zero values
        zero_features = {"frequency_spacing": 0.0, "frequency_bandwidth": 0.0}
        phase_velocity_zero = self.features._compute_phase_velocity(zero_features)
        assert phase_velocity_zero == 0.0
        
        # Test with missing keys
        minimal_features = {"frequency_spacing": 0.5}
        phase_velocity_minimal = self.features._compute_phase_velocity(minimal_features)
        assert phase_velocity_minimal == 0.0
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty features
        empty_features = {}
        features_list = self.features.compute_7d_phase_field_features(empty_features)
        
        assert len(features_list) == 4
        assert all(f == 0.0 for f in features_list)
        
        # Test with None values - should handle gracefully
        none_features = {
            "coupling_symmetry": None,
            "autocorrelation": None,
            "mixing_degree": None,
            "nonlinear_strength": None,
            "interaction_energy": None,
            "coupling_strength": None,
            "frequency_spacing": None,
            "frequency_bandwidth": None
        }
        # This should raise an error or handle None values gracefully
        try:
            features_list_none = self.features.compute_7d_phase_field_features(none_features)
            # If it doesn't raise an error, check the result
            assert len(features_list_none) == 4
            assert all(f == 0.0 for f in features_list_none)
        except TypeError:
            # Expected behavior - None values should cause TypeError
            pass
    
    def test_extreme_values(self):
        """Test with extreme values."""
        # Test with very large values
        large_features = {
            "coupling_symmetry": 1e6,
            "autocorrelation": 1e6,
            "mixing_degree": 1e6,
            "nonlinear_strength": 1e6,
            "interaction_energy": 1e6,
            "coupling_strength": 1e6,
            "frequency_spacing": 1e6,
            "frequency_bandwidth": 1e6
        }
        
        features_list = self.features.compute_7d_phase_field_features(large_features)
        assert len(features_list) == 4
        assert all(np.isfinite(f) for f in features_list)
        assert all(f >= 0.0 for f in features_list)
        
        # Test with very small values
        small_features = {
            "coupling_symmetry": 1e-10,
            "autocorrelation": 1e-10,
            "mixing_degree": 1e-10,
            "nonlinear_strength": 1e-10,
            "interaction_energy": 1e-10,
            "coupling_strength": 1e-10,
            "frequency_spacing": 1e-10,
            "frequency_bandwidth": 1e-10
        }
        
        features_list_small = self.features.compute_7d_phase_field_features(small_features)
        assert len(features_list_small) == 4
        assert all(np.isfinite(f) for f in features_list_small)
        assert all(f >= 0.0 for f in features_list_small)
    
    def test_feature_consistency(self):
        """Test feature computation consistency."""
        # Test that same input gives same output
        features_list1 = self.features.compute_7d_phase_field_features(self.test_features)
        features_list2 = self.features.compute_7d_phase_field_features(self.test_features)
        
        assert len(features_list1) == len(features_list2)
        for f1, f2 in zip(features_list1, features_list2):
            assert abs(f1 - f2) < 1e-10
    
    def test_feature_weights(self):
        """Test feature weights."""
        weights = self.features.feature_weights
        
        assert isinstance(weights, dict)
        assert len(weights) == 4
        
        # Check that weights sum to 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-10
        
        # Check individual weights
        assert 0.0 <= weights['phase_coherence'] <= 1.0
        assert 0.0 <= weights['topological_charge'] <= 1.0
        assert 0.0 <= weights['energy_density'] <= 1.0
        assert 0.0 <= weights['phase_velocity'] <= 1.0
    
    def test_mathematical_foundation(self):
        """Test mathematical foundation of feature computations."""
        # Test phase coherence formula: coupling_symmetry × autocorrelation
        test_symmetry = 0.6
        test_autocorr = 0.4
        test_features = {"coupling_symmetry": test_symmetry, "autocorrelation": test_autocorr}
        
        phase_coherence = self.features._compute_phase_coherence(test_features)
        expected = test_symmetry * test_autocorr
        assert abs(phase_coherence - expected) < 1e-10
        
        # Test topological charge formula: mixing_degree × nonlinear_strength
        test_mixing = 0.5
        test_nonlinear = 0.3
        test_features = {"mixing_degree": test_mixing, "nonlinear_strength": test_nonlinear}
        
        topological_charge = self.features._compute_topological_charge(test_features)
        expected = test_mixing * test_nonlinear
        assert abs(topological_charge - expected) < 1e-10
        
        # Test energy density formula: interaction_energy × coupling_strength
        test_interaction = 0.7
        test_coupling = 0.2
        test_features = {"interaction_energy": test_interaction, "coupling_strength": test_coupling}
        
        energy_density = self.features._compute_energy_density(test_features)
        expected = test_interaction * test_coupling
        assert abs(energy_density - expected) < 1e-10
        
        # Test phase velocity formula: frequency_spacing × frequency_bandwidth
        test_spacing = 0.8
        test_bandwidth = 0.1
        test_features = {"frequency_spacing": test_spacing, "frequency_bandwidth": test_bandwidth}
        
        phase_velocity = self.features._compute_phase_velocity(test_features)
        expected = test_spacing * test_bandwidth
        assert abs(phase_velocity - expected) < 1e-10
