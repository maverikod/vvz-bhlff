"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for prediction methods in vectorized ML implementations.

This module tests frequency, coupling, and pattern classification predictions.
"""

import pytest
from unittest.mock import Mock

from bhlff.models.level_c.beating.ml.core.prediction_engine import PredictionEngine
from bhlff.models.level_c.beating.ml.beating_ml_patterns import BeatingMLPatterns
from .test_ml_prediction_vectorized_implementations_base import (
    TestVectorizedMLPredictionImplementationsBase,
)


class TestVectorizedMLPredictionPredictions(
    TestVectorizedMLPredictionImplementationsBase
):
    """Test suite for prediction methods."""
    
    def test_vectorized_frequency_prediction(self):
        """
        Test vectorized frequency prediction implementation.
        
        Physical Meaning:
            Tests that vectorized frequency prediction correctly
            implements 7D phase field theory principles.
        """
        # Create prediction engine with mock components
        mock_model_manager = Mock()
        mock_feature_extractor = Mock()
        
        prediction_engine = PredictionEngine(mock_model_manager, mock_feature_extractor)
        
        # Test vectorized frequency prediction
        result = prediction_engine._predict_frequencies_simple(self.test_features)
        
        # Verify vectorized computation flag
        assert result["vectorized_computation"] is True
        assert result["prediction_method"] == "analytical_7d_bvp_vectorized"
        
        # Verify prediction structure
        assert "predicted_frequencies" in result
        assert "prediction_confidence" in result
        assert "feature_importance" in result
        
        # Verify vectorized computation results
        assert isinstance(result["predicted_frequencies"], list)
        assert len(result["predicted_frequencies"]) == 3
        assert all(
            isinstance(freq, (int, float)) for freq in result["predicted_frequencies"]
        )
        
        # Verify confidence is in valid range
        assert 0.0 <= result["prediction_confidence"] <= 1.0
        
        # Verify feature importance is normalized
        importance_sum = sum(result["feature_importance"].values())
        assert abs(importance_sum - 1.0) < 1e-6
    
    def test_vectorized_coupling_prediction(self):
        """
        Test vectorized coupling prediction implementation.
        
        Physical Meaning:
            Tests that vectorized coupling prediction correctly
            implements 7D phase field theory principles.
        """
        # Create prediction engine with mock components
        mock_model_manager = Mock()
        mock_feature_extractor = Mock()
        
        prediction_engine = PredictionEngine(mock_model_manager, mock_feature_extractor)
        
        # Test vectorized coupling prediction
        result = prediction_engine._predict_coupling_simple(self.test_features)
        
        # Verify vectorized computation flag
        assert result["vectorized_computation"] is True
        assert result["prediction_method"] == "analytical_7d_bvp_vectorized"
        
        # Verify prediction structure
        assert "predicted_coupling" in result
        assert "prediction_confidence" in result
        assert "feature_importance" in result
        
        # Verify coupling prediction structure
        coupling = result["predicted_coupling"]
        assert "coupling_strength" in coupling
        assert "interaction_energy" in coupling
        assert "coupling_symmetry" in coupling
        assert "nonlinear_strength" in coupling
        assert "mixing_degree" in coupling
        assert "coupling_efficiency" in coupling
        
        # Verify all coupling values are numeric
        for key, value in coupling.items():
            assert isinstance(value, (int, float))
        
        # Verify confidence is in valid range
        assert 0.0 <= result["prediction_confidence"] <= 1.0
        
        # Verify feature importance is normalized
        importance_sum = sum(result["feature_importance"].values())
        assert abs(importance_sum - 1.0) < 1e-6
    
    def test_vectorized_pattern_classification(self):
        """
        Test vectorized pattern classification implementation.
        
        Physical Meaning:
            Tests that vectorized pattern classification correctly
            implements 7D phase field theory principles.
        """
        # Create pattern classifier
        pattern_classifier = BeatingMLPatterns(self.mock_bvp_core)
        
        # Test vectorized pattern classification
        result = pattern_classifier.classify_beating_patterns(self.test_envelope)
        
        # Verify classification structure
        assert "pattern_type" in result
        assert "confidence" in result
        assert "classification_method" in result
        
        # Verify pattern type is valid
        valid_patterns = ["symmetric", "regular", "complex", "irregular"]
        assert result["pattern_type"] in valid_patterns
        
        # Verify confidence is in valid range
        assert 0.0 <= result["confidence"] <= 1.0

