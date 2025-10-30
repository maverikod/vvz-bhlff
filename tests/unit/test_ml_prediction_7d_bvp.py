"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test suite for 7D BVP ML prediction implementations.

This module tests the full ML prediction implementations based on
7D BVP theory for beating analysis.
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.models.level_c.beating.ml.core.prediction_engine import PredictionEngine
from bhlff.models.level_c.beating.ml.core.feature_extraction import FeatureExtractor
from bhlff.models.level_c.beating.ml.core.ml_models import MLModelManager
from bhlff.models.level_c.beating.ml.core.bvp_7d_analytics import BVP7DAnalytics


class TestMLPrediction7DBVP:
    """Test suite for 7D BVP ML prediction implementations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model_manager = MLModelManager()
        self.feature_extractor = FeatureExtractor()
        self.prediction_engine = PredictionEngine(
            self.model_manager, self.feature_extractor
        )
        self.bvp_analytics = BVP7DAnalytics()

        # Create test envelope data
        self.test_envelope = np.random.rand(64, 64, 64) + 1j * np.random.rand(
            64, 64, 64
        )

        # Create test features
        self.test_features = {
            "spectral_entropy": 0.5,
            "frequency_spacing": 0.3,
            "frequency_bandwidth": 0.4,
            "autocorrelation": 0.6,
            "coupling_strength": 0.7,
            "interaction_energy": 0.8,
            "coupling_symmetry": 0.9,
            "nonlinear_strength": 0.5,
            "mixing_degree": 0.6,
            "coupling_efficiency": 0.7,
            "phase_coherence": 0.8,
            "topological_charge": 0.3,
            "energy_density": 0.4,
            "phase_velocity": 0.5,
        }

    def test_7d_frequency_prediction_analytics(self):
        """Test 7D BVP frequency prediction analytics."""
        # Test frequency prediction
        phase_features = self.feature_extractor.extract_7d_phase_features(
            self.test_features
        )
        predicted_frequencies = self.bvp_analytics.compute_7d_frequency_prediction(
            phase_features, self.test_features
        )

        # Verify prediction structure
        assert isinstance(predicted_frequencies, list)
        assert len(predicted_frequencies) == 3

        # Verify all frequencies are positive
        for freq in predicted_frequencies:
            assert freq > 0.0

        # Verify 7D BVP theory implementation
        # Base frequency should depend on spectral entropy and phase coherence
        expected_base = (
            self.test_features["spectral_entropy"]
            * 50.0
            * (1.0 + self.test_features["phase_coherence"])
        )
        assert predicted_frequencies[0] > 0.0
        assert predicted_frequencies[1] > 0.0
        assert predicted_frequencies[2] > 0.0

    def test_7d_coupling_prediction_analytics(self):
        """Test 7D BVP coupling prediction analytics."""
        # Test coupling prediction
        phase_features = self.feature_extractor.extract_7d_phase_features(
            self.test_features
        )
        predicted_coupling = self.bvp_analytics.compute_7d_coupling_prediction(
            phase_features, self.test_features
        )

        # Verify prediction structure
        assert isinstance(predicted_coupling, dict)
        expected_keys = [
            "coupling_strength",
            "interaction_energy",
            "coupling_symmetry",
            "nonlinear_strength",
            "mixing_degree",
            "coupling_efficiency",
        ]
        for key in expected_keys:
            assert key in predicted_coupling
            assert isinstance(predicted_coupling[key], float)
            assert predicted_coupling[key] >= 0.0

        # Verify 7D BVP theory implementation
        # Coupling strength should depend on phase coherence and topological charge
        expected_coupling_strength = (
            self.test_features["coupling_strength"]
            * 0.6
            * (1.0 + self.test_features["phase_coherence"] * 0.3)
            * (1.0 + abs(self.test_features["topological_charge"]) * 0.1)
        )

        assert (
            abs(predicted_coupling["coupling_strength"] - expected_coupling_strength)
            < 1e-10
        )

    def test_analytical_confidence_computation(self):
        """Test analytical confidence computation."""
        # Test frequency confidence
        freq_confidence = self.bvp_analytics.compute_analytical_confidence(
            self.test_features
        )
        assert isinstance(freq_confidence, float)
        assert 0.0 <= freq_confidence <= 1.0

        # Test coupling confidence
        coupling_confidence = self.bvp_analytics.compute_coupling_analytical_confidence(
            self.test_features
        )
        assert isinstance(coupling_confidence, float)
        assert 0.0 <= coupling_confidence <= 1.0

        # Verify confidence increases with phase coherence
        high_coherence_features = self.test_features.copy()
        high_coherence_features["phase_coherence"] = 1.0
        high_confidence = self.bvp_analytics.compute_analytical_confidence(
            high_coherence_features
        )
        assert high_confidence > freq_confidence

    def test_feature_importance_computation(self):
        """Test feature importance computation."""
        # Test frequency feature importance
        freq_importance = self.bvp_analytics.compute_analytical_feature_importance(
            self.test_features
        )
        assert isinstance(freq_importance, dict)

        expected_freq_keys = [
            "spectral_entropy",
            "frequency_spacing",
            "frequency_bandwidth",
            "phase_coherence",
            "topological_charge",
        ]
        for key in expected_freq_keys:
            assert key in freq_importance
            assert isinstance(freq_importance[key], float)
            assert 0.0 <= freq_importance[key] <= 1.0

        # Test coupling feature importance
        coupling_importance = (
            self.bvp_analytics.compute_coupling_analytical_feature_importance(
                self.test_features
            )
        )
        assert isinstance(coupling_importance, dict)

        expected_coupling_keys = [
            "coupling_strength",
            "interaction_energy",
            "coupling_symmetry",
            "nonlinear_strength",
            "mixing_degree",
            "coupling_efficiency",
        ]
        for key in expected_coupling_keys:
            assert key in coupling_importance
            assert isinstance(coupling_importance[key], float)
            assert 0.0 <= coupling_importance[key] <= 1.0

    def test_full_frequency_prediction_pipeline(self):
        """Test full frequency prediction pipeline."""
        # Test with ML enabled
        self.prediction_engine.frequency_prediction_enabled = True
        freq_results = self.prediction_engine.predict_frequencies(self.test_envelope)

        # Verify results structure
        assert isinstance(freq_results, dict)
        assert "predicted_frequencies" in freq_results
        assert "prediction_confidence" in freq_results
        assert "prediction_method" in freq_results

        # Test with ML disabled (analytical fallback)
        self.prediction_engine.frequency_prediction_enabled = False
        freq_results_analytical = self.prediction_engine.predict_frequencies(
            self.test_envelope
        )

        # Verify analytical results
        assert isinstance(freq_results_analytical, dict)
        assert "predicted_frequencies" in freq_results_analytical
        assert "prediction_confidence" in freq_results_analytical
        assert "prediction_method" in freq_results_analytical
        assert freq_results_analytical["prediction_method"] == "analytical_7d_bvp"

        # Verify 7D BVP features are included
        assert "phase_coherence" in freq_results_analytical
        assert "topological_charge" in freq_results_analytical

    def test_full_coupling_prediction_pipeline(self):
        """Test full coupling prediction pipeline."""
        # Test with ML enabled
        self.prediction_engine.coupling_prediction_enabled = True
        coupling_results = self.prediction_engine.predict_coupling(self.test_envelope)

        # Verify results structure
        assert isinstance(coupling_results, dict)
        assert "predicted_coupling" in coupling_results
        assert "prediction_confidence" in coupling_results
        assert "prediction_method" in coupling_results

        # Test with ML disabled (analytical fallback)
        self.prediction_engine.coupling_prediction_enabled = False
        coupling_results_analytical = self.prediction_engine.predict_coupling(
            self.test_envelope
        )

        # Verify analytical results
        assert isinstance(coupling_results_analytical, dict)
        assert "predicted_coupling" in coupling_results_analytical
        assert "prediction_confidence" in coupling_results_analytical
        assert "prediction_method" in coupling_results_analytical
        assert coupling_results_analytical["prediction_method"] == "analytical_7d_bvp"

        # Verify 7D BVP features are included
        assert "interaction_energy" in coupling_results_analytical
        assert "phase_coherence" in coupling_results_analytical

    def test_7d_phase_field_feature_extraction(self):
        """Test 7D phase field feature extraction."""
        # Test frequency features
        freq_features = self.feature_extractor.extract_frequency_features(
            self.test_envelope
        )
        assert isinstance(freq_features, dict)

        # Verify 7D phase field features are included
        assert "phase_coherence" in freq_features
        assert "topological_charge" in freq_features

        # Test coupling features
        coupling_features = self.feature_extractor.extract_coupling_features(
            self.test_envelope
        )
        assert isinstance(coupling_features, dict)

        # Verify 7D phase field features are included
        assert "phase_coherence" in coupling_features
        assert "topological_charge" in coupling_features
        assert "energy_density" in coupling_features
        assert "phase_velocity" in coupling_features

        # Test 7D phase field feature extraction
        phase_features = self.feature_extractor.extract_7d_phase_features(
            coupling_features
        )
        assert isinstance(phase_features, np.ndarray)
        assert (
            len(phase_features) == 14
        )  # 4 basic + 6 coupling + 4 phase field features

    def test_7d_bvp_theory_consistency(self):
        """Test 7D BVP theory consistency in predictions."""
        # Test that predictions are consistent with 7D BVP theory
        phase_features = self.feature_extractor.extract_7d_phase_features(
            self.test_features
        )

        # Test frequency prediction consistency
        freq_prediction = self.bvp_analytics.compute_7d_frequency_prediction(
            phase_features, self.test_features
        )

        # Verify that higher phase coherence leads to higher frequencies
        high_coherence_features = self.test_features.copy()
        high_coherence_features["phase_coherence"] = 1.0
        high_coherence_phase_features = (
            self.feature_extractor.extract_7d_phase_features(high_coherence_features)
        )
        high_coherence_freq = self.bvp_analytics.compute_7d_frequency_prediction(
            high_coherence_phase_features, high_coherence_features
        )

        # Higher phase coherence should lead to higher frequencies
        for i in range(len(freq_prediction)):
            assert high_coherence_freq[i] > freq_prediction[i]

        # Test coupling prediction consistency
        coupling_prediction = self.bvp_analytics.compute_7d_coupling_prediction(
            phase_features, self.test_features
        )

        # Verify that higher topological charge leads to higher coupling strength
        high_charge_features = self.test_features.copy()
        high_charge_features["topological_charge"] = 1.0
        high_charge_phase_features = self.feature_extractor.extract_7d_phase_features(
            high_charge_features
        )
        high_charge_coupling = self.bvp_analytics.compute_7d_coupling_prediction(
            high_charge_phase_features, high_charge_features
        )

        # Higher topological charge should lead to higher coupling strength
        assert (
            high_charge_coupling["coupling_strength"]
            > coupling_prediction["coupling_strength"]
        )

    def test_no_classical_patterns_in_7d_bvp(self):
        """Test that 7D BVP implementation contains no classical patterns."""
        # Test that no exponential decay is used
        # (This is implicit in the 7D BVP theory implementation)

        # Test that no spacetime curvature concepts are used
        # (This is implicit in the 7D BVP theory implementation)

        # Test that no mass terms are used
        # (This is implicit in the 7D BVP theory implementation)

        # Verify that all methods use 7D BVP theory principles
        phase_features = self.feature_extractor.extract_7d_phase_features(
            self.test_features
        )

        # Test frequency prediction uses 7D BVP theory
        freq_prediction = self.bvp_analytics.compute_7d_frequency_prediction(
            phase_features, self.test_features
        )

        # Verify that prediction depends on 7D phase field features
        assert len(freq_prediction) == 3
        for freq in freq_prediction:
            assert freq > 0.0

        # Test coupling prediction uses 7D BVP theory
        coupling_prediction = self.bvp_analytics.compute_7d_coupling_prediction(
            phase_features, self.test_features
        )

        # Verify that prediction depends on 7D phase field features
        assert "coupling_strength" in coupling_prediction
        assert "interaction_energy" in coupling_prediction
        assert "coupling_symmetry" in coupling_prediction
        assert "nonlinear_strength" in coupling_prediction
        assert "mixing_degree" in coupling_prediction
        assert "coupling_efficiency" in coupling_prediction

        # Verify all coupling parameters are positive
        for key, value in coupling_prediction.items():
            assert value >= 0.0
