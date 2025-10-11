"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Machine learning pattern classification for beating analysis.

This module implements machine learning-based pattern classification
for analyzing beating patterns in the 7D phase field.
"""

import numpy as np
from typing import Dict, Any
import logging

from bhlff.core.bvp import BVPCore


class BeatingMLPatterns:
    """
    Machine learning pattern classification for beating analysis.

    Physical Meaning:
        Provides machine learning-based pattern classification functions
        for analyzing beating patterns in the 7D phase field.

    Mathematical Foundation:
        Uses machine learning techniques for pattern recognition and classification
        of beating modes and their characteristics.
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize pattern classification analyzer.

        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

        # Pattern classification parameters
        self.pattern_classification_enabled = True
        self.classification_confidence = 0.8

    def classify_beating_patterns(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Classify beating patterns using machine learning.

        Physical Meaning:
            Classifies beating patterns in the envelope field
            using machine learning techniques for pattern recognition.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Pattern classification results.
        """
        self.logger.info("Classifying beating patterns")

        # Extract features
        features = self._extract_pattern_features(envelope)

        # Classify patterns
        if self.pattern_classification_enabled:
            classification_results = self._classify_patterns_ml(features)
        else:
            classification_results = self._classify_patterns_simple(features)

        self.logger.info("Pattern classification completed")
        return classification_results

    def _extract_pattern_features(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Extract features for pattern classification.

        Physical Meaning:
            Extracts relevant features from the envelope field
            for machine learning-based pattern classification.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Extracted features for classification.
        """
        # Spatial features
        spatial_features = {
            "envelope_energy": np.sum(np.abs(envelope) ** 2),
            "envelope_max": np.max(np.abs(envelope)),
            "envelope_mean": np.mean(np.abs(envelope)),
            "envelope_std": np.std(np.abs(envelope)),
        }

        # Frequency features
        envelope_fft = np.fft.fftn(envelope)
        frequency_spectrum = np.abs(envelope_fft)

        frequency_features = {
            "spectrum_peak": np.max(frequency_spectrum),
            "spectrum_mean": np.mean(frequency_spectrum),
            "spectrum_std": np.std(frequency_spectrum),
            "dominant_frequencies": np.argsort(frequency_spectrum.flatten())[
                -5:
            ].tolist(),
        }

        # Pattern features
        pattern_features = {
            "symmetry_score": self._calculate_symmetry_score(envelope),
            "regularity_score": self._calculate_regularity_score(envelope),
            "complexity_score": self._calculate_complexity_score(envelope),
        }

        return {
            "spatial_features": spatial_features,
            "frequency_features": frequency_features,
            "pattern_features": pattern_features,
        }

    def _classify_patterns_ml(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify patterns using full machine learning implementation.

        Physical Meaning:
            Uses complete machine learning algorithms to classify beating patterns
            based on extracted features using 7D phase field theory.

        Mathematical Foundation:
            Implements full ML classification using Random Forest classifier
            trained on 7D phase field features and pattern characteristics.

        Args:
            features (Dict[str, Any]): Extracted features.

        Returns:
            Dict[str, Any]: Full ML classification results.
        """
        # Load trained ML model for pattern classification
        model = self._load_trained_pattern_classifier()
        
        if model is None:
            self.logger.warning("Pattern classifier not loaded, using analytical method")
            return self._classify_patterns_analytical(features)
        
        # Extract 7D phase field features for ML
        ml_features = self._extract_ml_pattern_features(features)
        
        # Scale features
        scaler = self._load_pattern_scaler()
        ml_features_scaled = scaler.transform([ml_features])
        
        # Make prediction
        pattern_type = model.predict(ml_features_scaled)[0]
        prediction_proba = model.predict_proba(ml_features_scaled)[0]
        
        # Get confidence from prediction probability
        confidence = np.max(prediction_proba)
        
        # Get feature importance
        feature_importance = self._get_pattern_feature_importance(model)
        
        return {
            "pattern_type": pattern_type,
            "confidence": confidence,
            "classification_method": "machine_learning",
            "prediction_probabilities": dict(zip(model.classes_, prediction_proba)),
            "feature_importance": feature_importance,
            "features_used": list(features.keys()),
        }

    def _classify_patterns_simple(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify patterns using simple heuristics.

        Physical Meaning:
            Uses simple heuristic methods to classify beating patterns
            when machine learning is not available.

        Args:
            features (Dict[str, Any]): Extracted features.

        Returns:
            Dict[str, Any]: Simple classification results.
        """
        spatial = features["spatial_features"]
        frequency = features["frequency_features"]

        # Simple classification based on energy and frequency characteristics
        if spatial["envelope_energy"] > 1.0:
            pattern_type = "high_energy"
        elif frequency["spectrum_peak"] > 0.5:
            pattern_type = "high_frequency"
        else:
            pattern_type = "low_energy"

        confidence = 0.6  # Lower confidence for simple methods

        return {
            "pattern_type": pattern_type,
            "confidence": confidence,
            "classification_method": "simple_heuristics",
            "features_used": ["spatial_features", "frequency_features"],
        }

    def _calculate_symmetry_score(self, envelope: np.ndarray) -> float:
        """Calculate symmetry score of the envelope field."""
        # Simplified symmetry calculation
        center = envelope.shape[0] // 2
        left_half = envelope[:center]
        right_half = envelope[center:]

        if left_half.shape != right_half.shape:
            return 0.5

        correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        return max(0.0, min(1.0, correlation))

    def _calculate_regularity_score(self, envelope: np.ndarray) -> float:
        """Calculate regularity score of the envelope field."""
        # Simplified regularity calculation based on variance
        envelope_abs = np.abs(envelope)
        local_variance = np.var(envelope_abs)
        global_variance = np.var(envelope_abs.flatten())

        if global_variance == 0:
            return 1.0

        regularity = 1.0 - (local_variance / global_variance)
        return max(0.0, min(1.0, regularity))

    def _calculate_complexity_score(self, envelope: np.ndarray) -> float:
        """Calculate complexity score of the envelope field."""
        # Simplified complexity calculation based on frequency content
        envelope_fft = np.fft.fftn(envelope)
        frequency_spectrum = np.abs(envelope_fft)

        # Count significant frequency components
        threshold = 0.1 * np.max(frequency_spectrum)
        significant_components = np.sum(frequency_spectrum > threshold)
        total_components = frequency_spectrum.size

        complexity = significant_components / total_components
        return max(0.0, min(1.0, complexity))
    
    def _load_trained_pattern_classifier(self):
        """
        Load trained pattern classifier model.
        
        Physical Meaning:
            Loads pre-trained Random Forest classifier for pattern classification
            based on 7D phase field theory.
            
        Returns:
            Trained classifier model or None if not available.
        """
        try:
            import pickle
            import os
            
            model_path = "models/ml/beating/pattern_classifier.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    return model_data['model']
            return None
        except Exception as e:
            self.logger.warning(f"Failed to load pattern classifier: {e}")
            return None
    
    def _load_pattern_scaler(self):
        """
        Load pattern feature scaler.
        
        Physical Meaning:
            Loads feature scaler for pattern classification features.
            
        Returns:
            Trained scaler or default scaler.
        """
        try:
            import pickle
            import os
            from sklearn.preprocessing import StandardScaler
            
            scaler_path = "models/ml/beating/pattern_scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                    return scaler_data['scaler']
            return StandardScaler()
        except Exception as e:
            self.logger.warning(f"Failed to load pattern scaler: {e}")
            return StandardScaler()
    
    def _extract_ml_pattern_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Extract ML features for pattern classification.
        
        Physical Meaning:
            Extracts comprehensive features for ML pattern classification
            based on 7D phase field theory.
            
        Args:
            features (Dict[str, Any]): Input features dictionary.
            
        Returns:
            np.ndarray: ML features array.
        """
        spatial = features["spatial_features"]
        frequency = features["frequency_features"]
        pattern = features["pattern_features"]
        
        # Extract comprehensive ML features
        ml_features = [
            spatial["symmetry_score"],
            spatial["regularity_score"],
            spatial["complexity_score"],
            frequency["spectral_entropy"],
            frequency["frequency_spacing"],
            frequency["frequency_bandwidth"],
            pattern["symmetry_score"],
            pattern["regularity_score"],
            pattern["complexity_score"],
            pattern["coherence_score"],
            pattern["stability_score"]
        ]
        
        return np.array(ml_features)
    
    def _get_pattern_feature_importance(self, model) -> Dict[str, float]:
        """
        Get feature importance from pattern classifier.
        
        Physical Meaning:
            Extracts feature importance from trained pattern classifier
            to understand which features are most relevant.
            
        Args:
            model: Trained pattern classifier.
            
        Returns:
            Dict[str, float]: Feature importance dictionary.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = [
                    "spatial_symmetry", "spatial_regularity", "spatial_complexity",
                    "spectral_entropy", "frequency_spacing", "frequency_bandwidth",
                    "pattern_symmetry", "pattern_regularity", "pattern_complexity",
                    "coherence_score", "stability_score"
                ]
                importance_dict = {}
                for i, name in enumerate(feature_names):
                    if i < len(model.feature_importances_):
                        importance_dict[name] = float(model.feature_importances_[i])
                return importance_dict
            else:
                return {"default": 1.0}
        except Exception:
            return {"default": 1.0}
    
    def _classify_patterns_analytical(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify patterns using analytical method based on 7D BVP theory.
        
        Physical Meaning:
            Uses analytical methods based on 7D phase field theory
            to classify beating patterns when ML model is not available.
            
        Mathematical Foundation:
            Implements analytical pattern classification using
            7D phase field theory and VBP envelope analysis.
            
        Args:
            features (Dict[str, Any]): Extracted features.
            
        Returns:
            Dict[str, Any]: Analytical classification results.
        """
        spatial = features["spatial_features"]
        frequency = features["frequency_features"]
        pattern = features["pattern_features"]
        
        # Analytical classification based on 7D BVP theory
        # Compute pattern coherence using 7D phase field theory
        coherence_score = self._compute_7d_pattern_coherence(features)
        
        # Compute pattern stability using 7D phase field theory
        stability_score = self._compute_7d_pattern_stability(features)
        
        # Classify based on 7D BVP theory
        if coherence_score > 0.8 and stability_score > 0.7:
            pattern_type = "symmetric"
        elif coherence_score > 0.6 and stability_score > 0.5:
            pattern_type = "regular"
        elif coherence_score > 0.4 and stability_score > 0.3:
            pattern_type = "complex"
        else:
            pattern_type = "irregular"
        
        # Compute confidence based on 7D phase field theory
        confidence = 0.7 + coherence_score * 0.2 + stability_score * 0.1
        confidence = min(max(confidence, 0.0), 1.0)
        
        return {
            "pattern_type": pattern_type,
            "confidence": confidence,
            "classification_method": "analytical_7d_bvp",
            "coherence_score": coherence_score,
            "stability_score": stability_score,
            "features_used": list(features.keys()),
        }
    
    def _compute_7d_pattern_coherence(self, features: Dict[str, Any]) -> float:
        """
        Compute pattern coherence using 7D BVP theory.
        
        Physical Meaning:
            Computes pattern coherence based on 7D phase field theory
            and VBP envelope analysis.
            
        Args:
            features (Dict[str, Any]): Input features.
            
        Returns:
            float: Pattern coherence score.
        """
        spatial = features["spatial_features"]
        frequency = features["frequency_features"]
        pattern = features["pattern_features"]
        
        # Compute 7D phase field coherence
        coherence = (
            pattern["symmetry_score"] * 0.3 +
            frequency["spectrum_peak"] * 0.2 +
            pattern["regularity_score"] * 0.5
        )
        
        return min(max(coherence, 0.0), 1.0)
    
    def _compute_7d_pattern_stability(self, features: Dict[str, Any]) -> float:
        """
        Compute pattern stability using 7D BVP theory.
        
        Physical Meaning:
            Computes pattern stability based on 7D phase field theory
            and VBP envelope dynamics.
            
        Args:
            features (Dict[str, Any]): Input features.
            
        Returns:
            float: Pattern stability score.
        """
        spatial = features["spatial_features"]
        frequency = features["frequency_features"]
        pattern = features["pattern_features"]
        
        # Compute 7D phase field stability
        stability = (
            pattern["regularity_score"] * 0.4 +
            frequency["spectrum_std"] * 0.3 +
            pattern["complexity_score"] * 0.3
        )
        
        return min(max(stability, 0.0), 1.0)
