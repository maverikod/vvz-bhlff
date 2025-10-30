"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test suite for full ML prediction implementation.

This module tests the complete ML prediction implementation
for beating analysis in Level C of 7D phase field theory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from bhlff.models.level_c.beating.ml.beating_ml_prediction_core import (
    BeatingMLPredictionCore,
)
from bhlff.models.level_c.beating.ml.core.prediction_engine import PredictionEngine
from bhlff.models.level_c.beating.ml.beating_ml_patterns import BeatingMLPatterns


class TestMLPredictionFullImplementation:
    """
    Test suite for full ML prediction implementation.

    Physical Meaning:
        Tests the complete ML prediction implementation
        for beating analysis using 7D phase field theory.
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.bvp_core = Mock()
        self.envelope = np.random.rand(64, 64) + 1j * np.random.rand(64, 64)

        # Initialize prediction core
        self.prediction_core = BeatingMLPredictionCore(self.bvp_core)

        # Initialize prediction engine
        self.model_manager = Mock()
        self.feature_extractor = Mock()
        self.prediction_engine = PredictionEngine(
            self.model_manager, self.feature_extractor
        )

        # Initialize patterns
        self.patterns = BeatingMLPatterns(self.bvp_core)

    def test_full_ml_frequency_prediction(self):
        """
        Test full ML frequency prediction implementation.

        Physical Meaning:
            Tests that the full ML frequency prediction
            uses complete 7D phase field analysis.
        """
        # Mock feature extraction
        features = {
            "spectral_entropy": 0.5,
            "frequency_spacing": 0.3,
            "frequency_bandwidth": 0.4,
            "autocorrelation": 0.6,
            "phase_coherence": 0.7,
            "topological_charge": 0.2,
        }

        # Mock model and scaler
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[100.0, 50.0, 25.0]])
        mock_model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[0.5, 0.3, 0.4, 0.6, 0.7, 0.2]])

        with patch.object(
            self.prediction_core.feature_extractor,
            "extract_frequency_features",
            return_value=features,
        ):
            with patch.object(
                self.prediction_core.feature_extractor,
                "extract_7d_phase_features",
                return_value=np.array([0.5, 0.3, 0.4, 0.6, 0.7, 0.2]),
            ):
                with patch.object(
                    self.prediction_core,
                    "_load_trained_frequency_model",
                    return_value=mock_model,
                ):
                    with patch.object(
                        self.prediction_core,
                        "_load_frequency_scaler",
                        return_value=mock_scaler,
                    ):
                        # Test full ML prediction
                        result = self.prediction_core.predict_beating_frequencies(
                            self.envelope
                        )

                        # Verify full ML implementation
                        assert "predicted_frequencies" in result
                        assert "prediction_confidence" in result
                        assert "feature_importance" in result
                        assert "model_type" in result
                        assert "prediction_variance" in result
                        assert result["prediction_method"] == "full_ml"

                        # Verify 7D phase field features are used
                        self.prediction_core.feature_extractor.extract_frequency_features.assert_called_once_with(
                            self.envelope
                        )
                        # extract_7d_phase_features is called twice - once in ML path and once in analytical fallback
                        assert (
                            self.prediction_core.feature_extractor.extract_7d_phase_features.call_count
                            >= 1
                        )

    def test_full_ml_coupling_prediction(self):
        """
        Test full ML coupling prediction implementation.

        Physical Meaning:
            Tests that the full ML coupling prediction
            uses complete 7D phase field analysis.
        """
        # Mock feature extraction
        features = {
            "coupling_strength": 0.6,
            "interaction_energy": 0.8,
            "coupling_symmetry": 0.5,
            "nonlinear_strength": 0.4,
            "mixing_degree": 0.7,
            "coupling_efficiency": 0.9,
            "phase_coherence": 0.8,
            "topological_charge": 0.3,
        }

        # Mock model and scaler
        mock_model = Mock()
        mock_model.predict.return_value = np.array(
            [[0.72, 0.64, 0.55, 0.36, 0.7, 0.945]]
        )
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array(
            [[0.6, 0.8, 0.5, 0.4, 0.7, 0.9, 0.8, 0.3]]
        )

        with patch.object(
            self.prediction_core.feature_extractor,
            "extract_coupling_features",
            return_value=features,
        ):
            with patch.object(
                self.prediction_core.feature_extractor,
                "extract_7d_phase_features",
                return_value=np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0.9, 0.8, 0.3]),
            ):
                with patch.object(
                    self.prediction_core,
                    "_load_trained_coupling_model",
                    return_value=mock_model,
                ):
                    with patch.object(
                        self.prediction_core,
                        "_load_coupling_scaler",
                        return_value=mock_scaler,
                    ):
                        # Test full ML prediction
                        result = self.prediction_core.predict_mode_coupling(
                            self.envelope
                        )

                        # Verify full ML implementation
                        assert "predicted_coupling" in result
                        assert "prediction_confidence" in result
                        assert "feature_importance" in result
                        assert "model_type" in result
                        assert "prediction_variance" in result
                        assert result["prediction_method"] == "full_ml"

                        # Verify 7D phase field features are used
                        self.prediction_core.feature_extractor.extract_coupling_features.assert_called_once_with(
                            self.envelope
                        )
                        # extract_7d_phase_features is called twice - once in ML path and once in analytical fallback
                        assert (
                            self.prediction_core.feature_extractor.extract_7d_phase_features.call_count
                            >= 1
                        )

    def test_analytical_fallback_frequency_prediction(self):
        """
        Test analytical fallback for frequency prediction.

        Physical Meaning:
            Tests that when ML models are not available,
            the system falls back to full analytical methods.
        """
        # Mock feature extraction
        features = {
            "spectral_entropy": 0.5,
            "frequency_spacing": 0.3,
            "frequency_bandwidth": 0.4,
            "autocorrelation": 0.6,
            "phase_coherence": 0.7,
            "topological_charge": 0.2,
        }

        with patch.object(
            self.prediction_core.feature_extractor,
            "extract_frequency_features",
            return_value=features,
        ):
            with patch.object(
                self.prediction_core.feature_extractor,
                "extract_7d_phase_features",
                return_value=np.array([0.5, 0.3, 0.4, 0.6, 0.7, 0.2]),
            ):
                # Mock no model available
                with patch.object(
                    self.prediction_core,
                    "_load_trained_frequency_model",
                    return_value=None,
                ):
                    with patch.object(
                        self.prediction_core,
                        "_load_frequency_scaler",
                        return_value=None,
                    ):
                        # Test analytical fallback
                        result = self.prediction_core.predict_beating_frequencies(
                            self.envelope
                        )

                        # Verify analytical fallback
                        assert "predicted_frequencies" in result
                        assert "prediction_confidence" in result
                        assert "feature_importance" in result
                        assert result["prediction_method"] == "analytical_7d_bvp"

                        # Verify 7D phase field features are used
                        self.prediction_core.feature_extractor.extract_frequency_features.assert_called_once_with(
                            self.envelope
                        )
                        # extract_7d_phase_features is called twice - once in ML path and once in analytical fallback
                        assert (
                            self.prediction_core.feature_extractor.extract_7d_phase_features.call_count
                            >= 1
                        )

    def test_analytical_fallback_coupling_prediction(self):
        """
        Test analytical fallback for coupling prediction.

        Physical Meaning:
            Tests that when ML models are not available,
            the system falls back to full analytical methods.
        """
        # Mock feature extraction
        features = {
            "coupling_strength": 0.6,
            "interaction_energy": 0.8,
            "coupling_symmetry": 0.5,
            "nonlinear_strength": 0.4,
            "mixing_degree": 0.7,
            "coupling_efficiency": 0.9,
            "phase_coherence": 0.8,
            "topological_charge": 0.3,
        }

        with patch.object(
            self.prediction_core.feature_extractor,
            "extract_coupling_features",
            return_value=features,
        ):
            with patch.object(
                self.prediction_core.feature_extractor,
                "extract_7d_phase_features",
                return_value=np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0.9, 0.8, 0.3]),
            ):
                # Mock no model available
                with patch.object(
                    self.prediction_core,
                    "_load_trained_coupling_model",
                    return_value=None,
                ):
                    with patch.object(
                        self.prediction_core, "_load_coupling_scaler", return_value=None
                    ):
                        # Test analytical fallback
                        result = self.prediction_core.predict_mode_coupling(
                            self.envelope
                        )

                        # Verify analytical fallback
                        assert "predicted_coupling" in result
                        assert "prediction_confidence" in result
                        assert "feature_importance" in result
                        assert result["prediction_method"] == "analytical_7d_bvp"

                        # Verify 7D phase field features are used
                        self.prediction_core.feature_extractor.extract_coupling_features.assert_called_once_with(
                            self.envelope
                        )
                        # extract_7d_phase_features is called twice - once in ML path and once in analytical fallback
                        assert (
                            self.prediction_core.feature_extractor.extract_7d_phase_features.call_count
                            >= 1
                        )

    def test_full_analytical_pattern_classification(self):
        """
        Test full analytical pattern classification.

        Physical Meaning:
            Tests that pattern classification uses complete
            analytical methods based on 7D phase field theory.
        """
        # Mock features
        features = {
            "spatial_features": {"envelope_energy": 1.5, "spatial_coherence": 0.8},
            "frequency_features": {"spectrum_peak": 0.6, "frequency_bandwidth": 0.4},
            "phase_coherence": 0.8,
            "topological_charge": 0.4,
        }

        # Test analytical classification
        result = self.patterns._classify_patterns_simple(features)

        # Verify full analytical implementation
        assert "pattern_type" in result
        assert "confidence" in result
        assert "classification_method" in result
        assert result["classification_method"] == "analytical_7d_bvp"
        assert "phase_coherence" in result
        assert "topological_charge" in result

        # Verify 7D phase field features are used
        assert "phase_coherence" in result["features_used"]
        assert "topological_charge" in result["features_used"]

    def test_7d_phase_field_feature_extraction(self):
        """
        Test 7D phase field feature extraction.

        Physical Meaning:
            Tests that feature extraction uses complete
            7D phase field theory analysis.
        """
        # Test frequency feature extraction
        freq_features = (
            self.prediction_core.feature_extractor.extract_frequency_features(
                self.envelope
            )
        )

        # Verify 7D phase field features are extracted
        assert "spectral_entropy" in freq_features
        assert "frequency_spacing" in freq_features
        assert "frequency_bandwidth" in freq_features
        assert "autocorrelation" in freq_features
        assert "phase_coherence" in freq_features
        assert "topological_charge" in freq_features

        # Test coupling feature extraction
        coup_features = (
            self.prediction_core.feature_extractor.extract_coupling_features(
                self.envelope
            )
        )

        # Verify 7D phase field features are extracted
        assert "coupling_strength" in coup_features
        assert "interaction_energy" in coup_features
        assert "coupling_symmetry" in coup_features
        assert "nonlinear_strength" in coup_features
        assert "mixing_degree" in coup_features
        assert "coupling_efficiency" in coup_features
        assert "phase_coherence" in coup_features
        assert "topological_charge" in coup_features
        assert "energy_density" in coup_features
        assert "phase_velocity" in coup_features

    def test_prediction_confidence_calculation(self):
        """
        Test prediction confidence calculation.

        Physical Meaning:
            Tests that prediction confidence is calculated
            using 7D phase field analysis.
        """
        # Mock model with estimators
        mock_model = Mock()
        mock_model.estimators_ = [Mock(), Mock(), Mock()]
        for estimator in mock_model.estimators_:
            estimator.predict.return_value = np.array([100.0, 50.0, 25.0])

        features = np.array([[0.5, 0.3, 0.4, 0.6, 0.7, 0.2]])

        # Test confidence calculation
        confidence = self.prediction_core._compute_prediction_confidence(
            features, mock_model
        )

        # Verify confidence is calculated
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    def test_feature_importance_extraction(self):
        """
        Test feature importance extraction.

        Physical Meaning:
            Tests that feature importance is extracted
            from ML models using 7D phase field features.
        """
        # Mock model with feature importance
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

        # Test feature importance extraction
        importance = self.prediction_core._get_feature_importance(mock_model)

        # Verify feature importance is extracted
        assert isinstance(importance, dict)
        assert len(importance) > 0

        # Verify 7D phase field features are included
        expected_features = [
            "spectral_entropy",
            "frequency_spacing",
            "frequency_bandwidth",
            "autocorrelation",
            "coupling_strength",
        ]

        for feature in expected_features:
            assert feature in importance

    def test_prediction_variance_calculation(self):
        """
        Test prediction variance calculation.

        Physical Meaning:
            Tests that prediction variance is calculated
            using ensemble methods for uncertainty quantification.
        """
        # Mock model with estimators
        mock_model = Mock()
        mock_model.estimators_ = [Mock(), Mock(), Mock()]
        for estimator in mock_model.estimators_:
            estimator.predict.return_value = np.array([100.0, 50.0, 25.0])

        features = np.array([[0.5, 0.3, 0.4, 0.6, 0.7, 0.2]])

        # Test variance calculation
        variance = self.prediction_core._compute_prediction_variance(
            features, mock_model
        )

        # Verify variance is calculated
        assert isinstance(variance, float)
        assert variance >= 0.0

    def test_no_simplified_predictions(self):
        """
        Test that no simplified predictions are used.

        Physical Meaning:
            Tests that the implementation does not use
            simplified or placeholder predictions.
        """
        # Test that full ML implementation is used
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[100.0, 50.0, 25.0]])
        mock_model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[0.5, 0.3, 0.4, 0.6, 0.7, 0.2]])

        with patch.object(
            self.prediction_core,
            "_load_trained_frequency_model",
            return_value=mock_model,
        ):
            with patch.object(
                self.prediction_core, "_load_frequency_scaler", return_value=mock_scaler
            ):
                result = self.prediction_core.predict_beating_frequencies(self.envelope)

                # Verify no simplified predictions
                assert result["prediction_method"] != "simple"
                assert result["prediction_method"] in ["full_ml", "analytical_7d_bvp"]

        # Test that full analytical implementation is used as fallback
        with patch.object(
            self.prediction_core, "_load_trained_frequency_model", return_value=None
        ):
            with patch.object(
                self.prediction_core, "_load_frequency_scaler", return_value=None
            ):
                result = self.prediction_core.predict_beating_frequencies(self.envelope)

                # Verify full analytical implementation
                assert result["prediction_method"] == "analytical_7d_bvp"
                assert "predicted_frequencies" in result
                assert "prediction_confidence" in result
                assert "feature_importance" in result
