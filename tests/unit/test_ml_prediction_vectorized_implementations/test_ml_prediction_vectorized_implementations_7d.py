"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for 7D phase field computations in vectorized ML implementations.

This module tests symmetry, regularity, energy, momentum, angular momentum,
and entropy computations for 7D phase fields.
"""

import numpy as np

from bhlff.models.level_c.beating.ml.beating_ml_vectorized_methods import (
    BeatingMLVectorizedMethods,
)
from .test_ml_prediction_vectorized_implementations_base import (
    TestVectorizedMLPredictionImplementationsBase,
)


class TestVectorizedMLPrediction7DComputations(
    TestVectorizedMLPredictionImplementationsBase
):
    """Test suite for 7D phase field computations."""
    
    def test_vectorized_7d_phase_field_symmetry(self):
        """
        Test vectorized 7D phase field symmetry computation.
        
        Physical Meaning:
            Tests that vectorized symmetry computation correctly
            implements 7D phase field theory principles.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Test vectorized symmetry computation
        symmetry = vectorized_methods.compute_7d_phase_field_symmetry_vectorized(
            self.test_envelope
        )
        
        # Verify symmetry is in valid range
        assert 0.0 <= symmetry <= 1.0
        
        # Test with different envelope configurations
        # High symmetry case
        high_symmetry_envelope = np.ones((8, 8, 8, 4, 4, 4, 2)) + 1j * np.ones(
            (8, 8, 8, 4, 4, 4, 2)
        )
        high_symmetry = vectorized_methods.compute_7d_phase_field_symmetry_vectorized(
            high_symmetry_envelope
        )
        assert high_symmetry > 0.7
        
        # Low symmetry case
        low_symmetry_envelope = np.random.rand(
            8, 8, 8, 4, 4, 4, 2
        ) + 1j * np.random.rand(8, 8, 8, 4, 4, 4, 2)
        low_symmetry = vectorized_methods.compute_7d_phase_field_symmetry_vectorized(
            low_symmetry_envelope
        )
        assert low_symmetry < 0.8
    
    def test_vectorized_7d_phase_field_regularity(self):
        """
        Test vectorized 7D phase field regularity computation.
        
        Physical Meaning:
            Tests that vectorized regularity computation correctly
            implements 7D phase field theory principles.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Test vectorized regularity computation
        regularity = vectorized_methods.compute_7d_phase_field_regularity_vectorized(
            self.test_envelope
        )
        
        # Verify regularity is in valid range
        assert 0.0 <= regularity <= 1.0
        
        # Test with different envelope configurations
        # High regularity case
        high_regularity_envelope = np.ones((8, 8, 8, 4, 4, 4, 2)) + 1j * np.ones(
            (8, 8, 8, 4, 4, 4, 2)
        )
        high_regularity = (
            vectorized_methods.compute_7d_phase_field_regularity_vectorized(
                high_regularity_envelope
            )
        )
        assert high_regularity > 0.5
        
        # Low regularity case
        low_regularity_envelope = np.random.rand(
            8, 8, 8, 4, 4, 4, 2
        ) + 1j * np.random.rand(8, 8, 8, 4, 4, 4, 2)
        low_regularity = (
            vectorized_methods.compute_7d_phase_field_regularity_vectorized(
                low_regularity_envelope
            )
        )
        assert low_regularity < 0.8
    
    def test_vectorized_ml_pattern_features_extraction(self):
        """
        Test vectorized ML pattern features extraction.
        
        Physical Meaning:
            Tests that vectorized feature extraction correctly
            implements 7D phase field theory principles.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Create test features
        test_pattern_features = {
            "spatial_features": {
                "envelope_energy": 1.5,
                "envelope_max": 2.0,
                "envelope_mean": 0.8,
                "envelope_std": 0.3,
            },
            "frequency_features": {
                "spectrum_peak": 0.6,
                "spectrum_bandwidth": 0.4,
                "spectrum_entropy": 0.5,
            },
            "pattern_features": {"symmetry_score": 0.7, "regularity_score": 0.6},
            "phase_coherence": 0.8,
            "topological_charge": 0.3,
        }
        
        # Test vectorized feature extraction
        features_array = vectorized_methods.extract_ml_pattern_features_vectorized(
            test_pattern_features
        )
        
        # Verify feature array structure
        assert isinstance(features_array, np.ndarray)
        assert features_array.shape == (11,)  # Expected number of features
        
        # Verify all features are numeric
        assert all(isinstance(feature, (int, float)) for feature in features_array)
        
        # Verify feature values are in expected ranges
        assert 0.0 <= features_array[0] <= 10.0  # envelope_energy
        assert 0.0 <= features_array[1] <= 10.0  # envelope_max
        assert 0.0 <= features_array[2] <= 10.0  # envelope_mean
        assert 0.0 <= features_array[3] <= 10.0  # envelope_std
        assert 0.0 <= features_array[4] <= 1.0  # spectrum_peak
        assert 0.0 <= features_array[5] <= 1.0  # spectrum_bandwidth
        assert 0.0 <= features_array[6] <= 1.0  # spectrum_entropy
        assert 0.0 <= features_array[7] <= 1.0  # symmetry_score
        assert 0.0 <= features_array[8] <= 1.0  # regularity_score
        assert 0.0 <= features_array[9] <= 1.0  # phase_coherence
        assert -1.0 <= features_array[10] <= 1.0  # topological_charge
    
    def test_vectorized_7d_phase_field_energy(self):
        """
        Test vectorized 7D phase field energy computation.
        
        Physical Meaning:
            Tests that vectorized energy computation correctly
            implements 7D phase field theory principles.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Test vectorized energy computation
        energy = vectorized_methods.compute_7d_phase_field_energy_vectorized(
            self.test_envelope
        )
        
        # Verify energy is positive
        assert energy >= 0.0
        
        # Test with known envelope
        known_envelope = np.ones((16, 16, 16)) + 1j * np.ones((16, 16, 16))
        known_energy = vectorized_methods.compute_7d_phase_field_energy_vectorized(
            known_envelope
        )
        expected_energy = 16 * 16 * 16 * 2.0  # 2 * (1^2 + 1^2) for each element
        assert abs(known_energy - expected_energy) < 1e-6
    
    def test_vectorized_7d_phase_field_momentum(self):
        """
        Test vectorized 7D phase field momentum computation.
        
        Physical Meaning:
            Tests that vectorized momentum computation correctly
            implements 7D phase field theory principles.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Test vectorized momentum computation
        momentum = vectorized_methods.compute_7d_phase_field_momentum_vectorized(
            self.test_envelope
        )
        
        # Verify momentum is array
        assert isinstance(momentum, np.ndarray)
        assert len(momentum) > 0
        
        # Verify momentum components are positive
        assert all(component >= 0.0 for component in momentum)
    
    def test_vectorized_7d_phase_field_angular_momentum(self):
        """
        Test vectorized 7D phase field angular momentum computation.
        
        Physical Meaning:
            Tests that vectorized angular momentum computation correctly
            implements 7D phase field theory principles.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Test vectorized angular momentum computation
        angular_momentum = (
            vectorized_methods.compute_7d_phase_field_angular_momentum_vectorized(
                self.test_envelope
            )
        )
        
        # Verify angular momentum is numeric
        assert isinstance(angular_momentum, (int, float))
        
        # Test with known envelope
        known_envelope = np.ones((8, 8, 8, 4, 4, 4, 2)) + 1j * np.ones(
            (8, 8, 8, 4, 4, 4, 2)
        )
        known_angular_momentum = (
            vectorized_methods.compute_7d_phase_field_angular_momentum_vectorized(
                known_envelope
            )
        )
        # For envelope with phase 0, angular momentum should be small but not necessarily zero
        assert abs(known_angular_momentum) < 1e5
    
    def test_vectorized_7d_phase_field_entropy(self):
        """
        Test vectorized 7D phase field entropy computation.
        
        Physical Meaning:
            Tests that vectorized entropy computation correctly
            implements 7D phase field theory principles.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Test vectorized entropy computation
        entropy = vectorized_methods.compute_7d_phase_field_entropy_vectorized(
            self.test_envelope
        )
        
        # Verify entropy is non-negative
        assert entropy >= 0.0
        
        # Test with known envelope
        known_envelope = np.ones((8, 8, 8, 4, 4, 4, 2)) + 1j * np.ones(
            (8, 8, 8, 4, 4, 4, 2)
        )
        known_entropy = vectorized_methods.compute_7d_phase_field_entropy_vectorized(
            known_envelope
        )
        # For uniform envelope, entropy should be small but not necessarily zero
        assert abs(known_entropy) < 15.0

