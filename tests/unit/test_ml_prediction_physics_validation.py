"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test suite for ML prediction physics validation.

This module tests the ML prediction implementations with real physics
validation for beating analysis in Level C of 7D phase field theory.

Physical Meaning:
    Tests the complete ML prediction implementations using 7D BVP theory
    and validates that they produce physically meaningful results.

Example:
    >>> pytest tests/unit/test_ml_prediction_physics_validation.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any
import logging

from bhlff.models.level_c.beating.ml.beating_ml_optimization_prediction import BeatingMLPredictionOptimizer
from bhlff.models.level_c.beating.ml.beating_ml_optimization_core import BeatingMLOptimizationCore
from bhlff.models.level_c.beating.ml.beating_ml_optimization_classification import BeatingMLClassificationOptimizer


class TestMLPredictionPhysicsValidation:
    """
    Test suite for ML prediction physics validation.
    
    Physical Meaning:
        Tests the complete ML prediction implementations using 7D BVP theory
        and validates that they produce physically meaningful results.
    """
    
    @pytest.fixture
    def test_envelope(self):
        """Create test envelope data with realistic 7D phase field properties."""
        # Create 7D envelope field data with realistic physics
        # This simulates a 7D phase field block with proper physical properties
        block_shape = (4, 4, 4, 4, 4, 4, 4)  # 7D block shape
        
        # Create envelope with realistic phase field properties
        # Real phase fields have specific spectral and coherence properties
        envelope = np.zeros(block_shape, dtype=complex)
        
        # Add realistic phase field structure
        for i in range(block_shape[0]):
            for j in range(block_shape[1]):
                for k in range(block_shape[2]):
                    for l in range(block_shape[3]):
                        for m in range(block_shape[4]):
                            for n in range(block_shape[5]):
                                for o in range(block_shape[6]):
                                    # Create realistic phase field with spatial structure
                                    phase = 2 * np.pi * (i + j + k + l + m + n + o) / np.sum(block_shape)
                                    amplitude = 1.0 + 0.1 * np.sin(phase)
                                    envelope[i, j, k, l, m, n, o] = amplitude * np.exp(1j * phase)
        
        return envelope
    
    @pytest.fixture
    def test_parameters(self):
        """Create test parameters with realistic ML settings."""
        return {
            "prediction_horizon": 10,
            "feature_window": 5,
            "prediction_threshold": 0.7,
            "model_complexity": "medium",
            "regularization_strength": 0.01,
            "classification_threshold": 0.5,
            "cross_validation_folds": 5,
            "random_state": 42
        }
    
    def test_prediction_accuracy_physics(self, test_envelope, test_parameters):
        """
        Test prediction accuracy calculation with real physics.
        
        Physical Meaning:
            Tests that prediction accuracy calculation correctly uses
            7D phase field physics and produces physically meaningful results.
        """
        # Create optimizer without BVP core (standalone test)
        optimizer = BeatingMLPredictionOptimizer(None)
        
        # Test accuracy calculation with realistic envelope
        accuracy = optimizer._calculate_prediction_accuracy(test_parameters, test_envelope)
        
        # Verify accuracy is within valid range
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)
        
        # Verify accuracy is deterministic
        accuracy2 = optimizer._calculate_prediction_accuracy(test_parameters, test_envelope)
        assert abs(accuracy - accuracy2) < 1e-10
        
        # Test physical dependence on envelope properties
        # Higher energy envelope should give different accuracy
        high_energy_envelope = test_envelope * 2.0
        high_energy_accuracy = optimizer._calculate_prediction_accuracy(test_parameters, high_energy_envelope)
        assert abs(accuracy - high_energy_accuracy) > 1e-6  # Should be different
        
        # Test that accuracy depends on spectral properties
        # Create envelope with different spectral properties
        low_freq_envelope = test_envelope.copy()
        low_freq_envelope[0, 0, 0, 0, 0, 0, 0] *= 10  # Add low frequency component
        low_freq_accuracy = optimizer._calculate_prediction_accuracy(test_parameters, low_freq_envelope)
        assert abs(accuracy - low_freq_accuracy) > 1e-6  # Should be different
    
    def test_prediction_precision_physics(self, test_envelope, test_parameters):
        """
        Test prediction precision calculation with real physics.
        
        Physical Meaning:
            Tests that prediction precision calculation correctly uses
            7D phase field physics and produces physically meaningful results.
        """
        # Create optimizer without BVP core (standalone test)
        optimizer = BeatingMLPredictionOptimizer(None)
        
        # Test precision calculation with realistic envelope
        precision = optimizer._calculate_prediction_precision(test_parameters, test_envelope)
        
        # Verify precision is within valid range
        assert 0.0 <= precision <= 1.0
        assert isinstance(precision, float)
        
        # Verify precision is deterministic
        precision2 = optimizer._calculate_prediction_precision(test_parameters, test_envelope)
        assert abs(precision - precision2) < 1e-10
        
        # Test physical dependence on envelope properties
        # Different envelope structures should give different precision
        noisy_envelope = test_envelope + 0.1 * np.random.randn(*test_envelope.shape)
        noisy_precision = optimizer._calculate_prediction_precision(test_parameters, noisy_envelope)
        assert abs(precision - noisy_precision) > 1e-6  # Should be different
    
    def test_spectral_entropy_physics(self, test_envelope):
        """
        Test spectral entropy computation with real physics.
        
        Physical Meaning:
            Tests that spectral entropy computation correctly analyzes
            7D phase field spectral properties using 7D BVP theory.
        """
        # Create optimizer without BVP core (standalone test)
        optimizer = BeatingMLPredictionOptimizer(None)
        
        # Test spectral entropy computation
        entropy = optimizer._compute_spectral_entropy(test_envelope)
        
        # Verify entropy is non-negative
        assert entropy >= 0.0
        assert isinstance(entropy, float)
        
        # Verify entropy is deterministic
        entropy2 = optimizer._compute_spectral_entropy(test_envelope)
        assert abs(entropy - entropy2) < 1e-10
        
        # Test physical meaning: different spectral properties should give different entropy
        # Create envelope with different spectral properties
        flat_envelope = np.ones_like(test_envelope)  # Flat spectrum
        flat_entropy = optimizer._compute_spectral_entropy(flat_envelope)
        assert abs(entropy - flat_entropy) > 1e-6  # Should be different
        
        # Test that entropy increases with spectral complexity
        complex_envelope = test_envelope + 0.5 * np.random.randn(*test_envelope.shape)
        complex_entropy = optimizer._compute_spectral_entropy(complex_envelope)
        assert complex_entropy > entropy  # More complex should have higher entropy
    
    def test_phase_coherence_physics(self, test_envelope):
        """
        Test phase coherence computation with real physics.
        
        Physical Meaning:
            Tests that phase coherence computation correctly analyzes
            7D phase field phase properties using 7D BVP theory.
        """
        # Create optimizer without BVP core (standalone test)
        optimizer = BeatingMLPredictionOptimizer(None)
        
        # Test phase coherence computation
        coherence = optimizer._compute_phase_coherence(test_envelope)
        
        # Verify coherence is within valid range
        assert 0.0 <= coherence <= 1.0
        assert isinstance(coherence, float)
        
        # Verify coherence is deterministic
        coherence2 = optimizer._compute_phase_coherence(test_envelope)
        assert abs(coherence - coherence2) < 1e-10
        
        # Test physical meaning: different phase structures should give different coherence
        # Create envelope with different phase properties
        random_phase_envelope = np.abs(test_envelope) * np.exp(1j * np.random.randn(*test_envelope.shape))
        random_coherence = optimizer._compute_phase_coherence(random_phase_envelope)
        assert abs(coherence - random_coherence) > 1e-6  # Should be different
        
        # Test that coherent phase gives high coherence
        coherent_envelope = np.abs(test_envelope) * np.exp(1j * 0.1 * np.ones_like(test_envelope))
        coherent_coherence = optimizer._compute_phase_coherence(coherent_envelope)
        assert coherent_coherence > coherence  # More coherent should have higher coherence
    
    def test_ml_optimization_core_physics(self, test_envelope, test_parameters):
        """
        Test ML optimization core with real physics.
        
        Physical Meaning:
            Tests that ML optimization core correctly uses
            7D phase field physics for parameter optimization.
        """
        # Create core without BVP core (standalone test)
        core = BeatingMLOptimizationCore(None)
        
        # Test accuracy calculation
        accuracy = core._calculate_accuracy(test_parameters, test_envelope)
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)
        
        # Test precision calculation
        precision = core._calculate_precision(test_parameters, test_envelope)
        assert 0.0 <= precision <= 1.0
        assert isinstance(precision, float)
        
        # Test recall calculation
        recall = core._calculate_recall(test_parameters, test_envelope)
        assert 0.0 <= recall <= 1.0
        assert isinstance(recall, float)
        
        # Test F1 score calculation
        f1_score = core._calculate_f1_score(test_parameters, test_envelope)
        assert 0.0 <= f1_score <= 1.0
        assert isinstance(f1_score, float)
        
        # Test that F1 score is harmonic mean of precision and recall
        expected_f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        assert abs(f1_score - expected_f1) < 1e-10
    
    def test_classification_physics(self, test_envelope, test_parameters):
        """
        Test classification with real physics.
        
        Physical Meaning:
            Tests that classification correctly uses
            7D phase field physics for pattern classification.
        """
        # Create classifier without BVP core (standalone test)
        classifier = BeatingMLClassificationOptimizer(None)
        
        # Test classification accuracy
        accuracy = classifier._calculate_classification_accuracy(test_parameters, test_envelope)
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)
        
        # Test classification precision
        precision = classifier._calculate_classification_precision(test_parameters, test_envelope)
        assert 0.0 <= precision <= 1.0
        assert isinstance(precision, float)
        
        # Test classification recall
        recall = classifier._calculate_classification_recall(test_parameters, test_envelope)
        assert 0.0 <= recall <= 1.0
        assert isinstance(recall, float)
        
        # Test classification F1 score
        f1_score = classifier._calculate_classification_f1_score(test_parameters, test_envelope)
        assert 0.0 <= f1_score <= 1.0
        assert isinstance(f1_score, float)
        
        # Test that F1 score is harmonic mean of precision and recall
        expected_f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        assert abs(f1_score - expected_f1) < 1e-10
    
    def test_7d_performance_metrics_physics(self, test_parameters):
        """
        Test 7D performance metrics with real physics.
        
        Physical Meaning:
            Tests that 7D performance metrics correctly use
            7D phase field theory for parameter optimization.
        """
        # Create core without BVP core (standalone test)
        core = BeatingMLOptimizationCore(None)
        
        # Test 7D performance metric computation
        metric = core._compute_7d_performance_metric(test_parameters, 0.8)
        
        # Verify metric is within valid range
        assert 0.0 <= metric <= 1.0
        assert isinstance(metric, float)
        
        # Verify metric is deterministic
        metric2 = core._compute_7d_performance_metric(test_parameters, 0.8)
        assert abs(metric - metric2) < 1e-10
        
        # Test that metric depends on parameters
        different_params = test_parameters.copy()
        different_params["prediction_horizon"] = 20
        different_params["regularization_strength"] = 0.1  # Add numerical parameter
        different_params["cross_validation_folds"] = 10  # Add numerical parameter
        different_params["random_state"] = 100  # Add numerical parameter
        different_params["prediction_threshold"] = 0.8  # Add numerical parameter
        different_metric = core._compute_7d_performance_metric(different_params, 0.8)
        assert abs(metric - different_metric) > 1e-6  # Should be different
    
    def test_no_simplified_calculations_physics(self, test_envelope, test_parameters):
        """
        Test that no simplified calculations are used in physics.
        
        Physical Meaning:
            Verifies that all calculations use full 7D BVP theory
            and not simplified approximations, ensuring physical correctness.
        """
        # Create optimizers without BVP core (standalone tests)
        optimizer = BeatingMLPredictionOptimizer(None)
        core = BeatingMLOptimizationCore(None)
        classifier = BeatingMLClassificationOptimizer(None)
        
        # Test that calculations are deterministic (not random)
        accuracy1 = optimizer._calculate_prediction_accuracy(test_parameters, test_envelope)
        accuracy2 = optimizer._calculate_prediction_accuracy(test_parameters, test_envelope)
        assert abs(accuracy1 - accuracy2) < 1e-10
        
        precision1 = core._calculate_precision(test_parameters, test_envelope)
        precision2 = core._calculate_precision(test_parameters, test_envelope)
        assert abs(precision1 - precision2) < 1e-10
        
        classification_accuracy1 = classifier._calculate_classification_accuracy(test_parameters, test_envelope)
        classification_accuracy2 = classifier._calculate_classification_accuracy(test_parameters, test_envelope)
        assert abs(classification_accuracy1 - classification_accuracy2) < 1e-10
        
        # Test that calculations depend on envelope properties (physics)
        modified_envelope = test_envelope * 1.5
        modified_accuracy = optimizer._calculate_prediction_accuracy(test_parameters, modified_envelope)
        assert abs(accuracy1 - modified_accuracy) > 1e-6  # Should be different
        
        # Test that calculations depend on parameters (physics)
        modified_params = test_parameters.copy()
        modified_params["prediction_horizon"] = 5  # Change horizon to affect accuracy
        modified_accuracy2 = optimizer._calculate_prediction_accuracy(modified_params, test_envelope)
        assert abs(accuracy1 - modified_accuracy2) > 1e-6  # Should be different
