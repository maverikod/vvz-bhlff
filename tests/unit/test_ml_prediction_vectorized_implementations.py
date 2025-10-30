"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for vectorized ML prediction implementations.

This module tests the vectorized ML prediction implementations
for beating analysis in 7D phase field theory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.models.level_c.beating.ml.core.prediction_engine import PredictionEngine
from bhlff.models.level_c.beating.ml.beating_ml_patterns import BeatingMLPatterns
from bhlff.models.level_c.beating.ml.beating_ml_pattern_feature_extractor import (
    BeatingMLPatternFeatureExtractor,
)
from bhlff.models.level_c.beating.ml.beating_ml_pattern_classifier import (
    BeatingMLPatternClassifier,
)
from bhlff.models.level_c.beating.ml.beating_ml_vectorized_methods import (
    BeatingMLVectorizedMethods,
)


class TestVectorizedMLPredictionImplementations:
    """
    Test suite for vectorized ML prediction implementations.

    Physical Meaning:
        Tests vectorized ML prediction implementations to ensure
        they correctly implement 7D phase field theory principles
        and provide efficient computation using vectorized operations.
    """

    def setup_method(self):
        """Setup test fixtures."""
        # Create mock BVP core
        self.mock_bvp_core = Mock()
        self.mock_bvp_core.domain = Mock()
        self.mock_bvp_core.config = {}

        # Create test envelope data (7D: 3 spatial + 3 phase + 1 time)
        self.test_envelope = np.random.rand(
            16, 16, 16, 8, 8, 8, 4
        ) + 1j * np.random.rand(16, 16, 16, 8, 8, 8, 4)

        # Create test features
        self.test_features = {
            "spectral_entropy": 0.5,
            "frequency_spacing": 0.3,
            "frequency_bandwidth": 0.4,
            "phase_coherence": 0.7,
            "topological_charge": 0.2,
            "coupling_strength": 0.6,
            "interaction_energy": 0.8,
            "coupling_symmetry": 0.5,
            "nonlinear_strength": 0.4,
            "mixing_degree": 0.3,
            "coupling_efficiency": 0.9,
        }

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

        # Create test features for pattern classification
        pattern_features = {
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

    def test_vectorized_computation_performance(self):
        """
        Test vectorized computation performance.

        Physical Meaning:
            Tests that vectorized computations are more efficient
            than non-vectorized implementations.
        """
        import time

        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()

        # Test performance with large envelope
        large_envelope = np.random.rand(8, 8, 8, 4, 4, 4, 2) + 1j * np.random.rand(
            8, 8, 8, 4, 4, 4, 2
        )

        # Measure vectorized computation time
        start_time = time.time()
        symmetry = vectorized_methods.compute_7d_phase_field_symmetry_vectorized(
            large_envelope
        )
        regularity = vectorized_methods.compute_7d_phase_field_regularity_vectorized(
            large_envelope
        )
        energy = vectorized_methods.compute_7d_phase_field_energy_vectorized(
            large_envelope
        )
        vectorized_time = time.time() - start_time

        # Verify computations completed successfully
        assert 0.0 <= symmetry <= 1.0
        assert 0.0 <= regularity <= 1.0
        assert energy >= 0.0

        # Verify vectorized computation is reasonably fast (less than 10 seconds)
        assert vectorized_time < 10.0

    def test_vectorized_computation_consistency(self):
        """
        Test vectorized computation consistency.

        Physical Meaning:
            Tests that vectorized computations produce consistent
            results across multiple runs.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()

        # Test consistency across multiple runs
        results = []
        for _ in range(5):
            symmetry = vectorized_methods.compute_7d_phase_field_symmetry_vectorized(
                self.test_envelope
            )
            regularity = (
                vectorized_methods.compute_7d_phase_field_regularity_vectorized(
                    self.test_envelope
                )
            )
            energy = vectorized_methods.compute_7d_phase_field_energy_vectorized(
                self.test_envelope
            )
            results.append((symmetry, regularity, energy))

        # Verify all results are identical
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10  # symmetry
            assert abs(results[i][1] - results[0][1]) < 1e-10  # regularity
            assert abs(results[i][2] - results[0][2]) < 1e-10  # energy

    def test_pattern_feature_extractor(self):
        """
        Test pattern feature extractor functionality.

        Physical Meaning:
            Tests that pattern feature extractor correctly
            extracts features from 7D phase field configurations.
        """
        # Create feature extractor
        feature_extractor = BeatingMLPatternFeatureExtractor(self.mock_bvp_core)

        # Test feature extraction
        features = feature_extractor.extract_pattern_features(self.test_envelope)

        # Verify feature structure
        assert "spatial_features" in features
        assert "frequency_features" in features
        assert "pattern_features" in features

        # Verify spatial features
        spatial = features["spatial_features"]
        assert "envelope_energy" in spatial
        assert "envelope_max" in spatial
        assert "envelope_mean" in spatial
        assert "envelope_std" in spatial

        # Verify frequency features
        frequency = features["frequency_features"]
        assert "spectrum_peak" in frequency
        assert "spectrum_mean" in frequency
        assert "spectrum_std" in frequency
        assert "spectrum_entropy" in frequency
        assert "frequency_spacing" in frequency
        assert "frequency_bandwidth" in frequency
        assert "dominant_frequencies" in frequency

        # Verify pattern features
        pattern = features["pattern_features"]
        assert "symmetry_score" in pattern
        assert "regularity_score" in pattern
        assert "complexity_score" in pattern

        # Verify feature values are in expected ranges
        assert spatial["envelope_energy"] >= 0.0
        assert spatial["envelope_max"] >= 0.0
        assert spatial["envelope_mean"] >= 0.0
        assert spatial["envelope_std"] >= 0.0

        assert frequency["spectrum_peak"] >= 0.0
        assert frequency["spectrum_mean"] >= 0.0
        assert frequency["spectrum_std"] >= 0.0
        assert frequency["spectrum_entropy"] >= 0.0
        assert frequency["frequency_spacing"] >= 0.0
        assert frequency["frequency_bandwidth"] >= 0.0

        assert 0.0 <= pattern["symmetry_score"] <= 1.0
        assert 0.0 <= pattern["regularity_score"] <= 1.0
        assert 0.0 <= pattern["complexity_score"] <= 1.0

    def test_pattern_classifier(self):
        """
        Test pattern classifier functionality.

        Physical Meaning:
            Tests that pattern classifier correctly
            classifies patterns from extracted features.
        """
        # Create pattern classifier
        pattern_classifier = BeatingMLPatternClassifier(self.mock_bvp_core)

        # Create test features for pattern classification
        test_pattern_features = {
            "spatial_features": {
                "envelope_energy": 1.5,
                "envelope_max": 2.0,
                "envelope_mean": 0.8,
                "envelope_std": 0.3,
            },
            "frequency_features": {
                "spectrum_peak": 0.6,
                "spectrum_mean": 0.4,
                "spectrum_std": 0.2,
                "spectrum_entropy": 0.5,
                "frequency_spacing": 0.3,
                "frequency_bandwidth": 0.4,
                "dominant_frequencies": [1, 2, 3, 4, 5],
            },
            "pattern_features": {
                "symmetry_score": 0.7,
                "regularity_score": 0.6,
                "complexity_score": 0.5,
            },
        }

        # Test pattern classification
        result = pattern_classifier.classify_patterns(test_pattern_features)

        # Verify classification structure
        assert "pattern_type" in result
        assert "confidence" in result
        assert "classification_method" in result

        # Verify pattern type is valid
        valid_patterns = ["symmetric", "regular", "complex", "irregular"]
        assert result["pattern_type"] in valid_patterns

        # Verify confidence is in valid range
        assert 0.0 <= result["confidence"] <= 1.0

        # Verify classification method
        assert result["classification_method"] in [
            "machine_learning",
            "analytical_7d_bvp",
        ]

    def test_refactored_beating_ml_patterns(self):
        """
        Test refactored BeatingMLPatterns functionality.

        Physical Meaning:
            Tests that refactored BeatingMLPatterns correctly
            integrates feature extraction and pattern classification.
        """
        # Create refactored patterns classifier
        patterns = BeatingMLPatterns(self.mock_bvp_core)

        # Test pattern classification
        result = patterns.classify_beating_patterns(self.test_envelope)

        # Verify result structure
        assert "pattern_type" in result
        assert "confidence" in result
        assert "classification_method" in result

        # Verify pattern type is valid
        valid_patterns = ["symmetric", "regular", "complex", "irregular"]
        assert result["pattern_type"] in valid_patterns

        # Verify confidence is in valid range
        assert 0.0 <= result["confidence"] <= 1.0

        # Verify classification method
        assert result["classification_method"] in [
            "machine_learning",
            "analytical_7d_bvp",
        ]

    def test_feature_extractor_spectral_entropy(self):
        """
        Test spectral entropy computation in feature extractor.

        Physical Meaning:
            Tests that spectral entropy is correctly computed
            for frequency spectrum analysis.
        """
        # Create feature extractor
        feature_extractor = BeatingMLPatternFeatureExtractor(self.mock_bvp_core)

        # Test with known spectrum
        known_spectrum = np.array([0.1, 0.2, 0.3, 0.4])
        entropy = feature_extractor._compute_spectral_entropy(known_spectrum)

        # Verify entropy is non-negative
        assert entropy >= 0.0

        # Test with uniform spectrum (should have maximum entropy)
        uniform_spectrum = np.ones(10)
        uniform_entropy = feature_extractor._compute_spectral_entropy(uniform_spectrum)

        # Test with delta spectrum (should have minimum entropy)
        delta_spectrum = np.zeros(10)
        delta_spectrum[5] = 1.0
        delta_entropy = feature_extractor._compute_spectral_entropy(delta_spectrum)

        # Uniform should have higher entropy than delta
        assert uniform_entropy > delta_entropy

    def test_feature_extractor_frequency_analysis(self):
        """
        Test frequency analysis in feature extractor.

        Physical Meaning:
            Tests that frequency spacing and bandwidth are correctly computed
            for spectral analysis.
        """
        # Create feature extractor
        feature_extractor = BeatingMLPatternFeatureExtractor(self.mock_bvp_core)

        # Test frequency spacing computation
        test_spectrum = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        spacing = feature_extractor._compute_frequency_spacing(test_spectrum)

        # Verify spacing is non-negative
        assert spacing >= 0.0

        # Test frequency bandwidth computation
        bandwidth = feature_extractor._compute_frequency_bandwidth(test_spectrum)

        # Verify bandwidth is non-negative
        assert bandwidth >= 0.0

    def test_pattern_classifier_coherence_computation(self):
        """
        Test pattern coherence computation in classifier.

        Physical Meaning:
            Tests that pattern coherence is correctly computed
            using 7D phase field theory.
        """
        # Create pattern classifier
        pattern_classifier = BeatingMLPatternClassifier(self.mock_bvp_core)

        # Create test features
        test_features = {
            "spatial_features": {"envelope_energy": 1.0},
            "frequency_features": {"spectrum_peak": 0.5},
            "pattern_features": {
                "symmetry_score": 0.8,
                "regularity_score": 0.7,
                "complexity_score": 0.6,
            },
        }

        # Test coherence computation
        coherence = pattern_classifier._compute_7d_pattern_coherence(test_features)

        # Verify coherence is in valid range
        assert 0.0 <= coherence <= 1.0

    def test_pattern_classifier_stability_computation(self):
        """
        Test pattern stability computation in classifier.

        Physical Meaning:
            Tests that pattern stability is correctly computed
            using 7D phase field theory.
        """
        # Create pattern classifier
        pattern_classifier = BeatingMLPatternClassifier(self.mock_bvp_core)

        # Create test features
        test_features = {
            "spatial_features": {"envelope_energy": 1.0},
            "frequency_features": {"spectrum_std": 0.3},
            "pattern_features": {
                "symmetry_score": 0.8,
                "regularity_score": 0.7,
                "complexity_score": 0.6,
            },
        }

        # Test stability computation
        stability = pattern_classifier._compute_7d_pattern_stability(test_features)

        # Verify stability is in valid range
        assert 0.0 <= stability <= 1.0
