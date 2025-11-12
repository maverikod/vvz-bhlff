"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for component functionality in vectorized ML implementations.

This module tests feature extractors, classifiers, and pattern analysis components.
"""

import numpy as np

from bhlff.models.level_c.beating.ml.beating_ml_patterns import BeatingMLPatterns
from bhlff.models.level_c.beating.ml.beating_ml_pattern_feature_extractor import (
    BeatingMLPatternFeatureExtractor,
)
from bhlff.models.level_c.beating.ml.beating_ml_pattern_classifier import (
    BeatingMLPatternClassifier,
)
from .test_ml_prediction_vectorized_implementations_base import (
    TestVectorizedMLPredictionImplementationsBase,
)


class TestVectorizedMLPredictionComponents(
    TestVectorizedMLPredictionImplementationsBase
):
    """Test suite for component functionality."""
    
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

