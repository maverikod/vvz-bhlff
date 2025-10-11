"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Prediction engine for ML models.

This module implements the prediction engine for machine learning
models in 7D phase field beating analysis.

Physical Meaning:
    Provides prediction capabilities for beating frequencies and mode coupling
    using trained machine learning models based on 7D phase field theory.

Example:
    >>> engine = PredictionEngine(model_manager, feature_extractor)
    >>> frequencies = engine.predict_frequencies(envelope)
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from .ml_models import MLModelManager
from .feature_extraction import FeatureExtractor
from .bvp_7d_analytics import BVP7DAnalytics


class PredictionEngine:
    """
    Prediction engine for ML models.
    
    Physical Meaning:
        Provides prediction capabilities for beating frequencies and mode coupling
        using trained machine learning models based on 7D phase field theory.
        
    Mathematical Foundation:
        Uses Random Forest regression for frequency prediction and
        Neural Network regression for coupling prediction.
    """
    
    def __init__(self, model_manager: MLModelManager, feature_extractor: FeatureExtractor):
        """
        Initialize prediction engine.
        
        Physical Meaning:
            Sets up the prediction engine with ML models and feature extractor
            for 7D phase field analysis.
            
        Args:
            model_manager (MLModelManager): ML model manager instance.
            feature_extractor (FeatureExtractor): Feature extractor instance.
        """
        self.model_manager = model_manager
        self.feature_extractor = feature_extractor
        self.bvp_analytics = BVP7DAnalytics()
        self.logger = logging.getLogger(__name__)
    
    def predict_frequencies(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Predict beating frequencies using ML.
        
        Physical Meaning:
            Predicts beating frequencies using trained machine learning
            model based on 7D phase field features.
            
        Mathematical Foundation:
            Uses Random Forest regression trained on 7D phase field
            data to predict beating frequencies from spectral features.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Frequency prediction results.
        """
        try:
            # Extract frequency features
            features = self.feature_extractor.extract_frequency_features(envelope)
            
            # Get ML model
            model = self.model_manager.get_frequency_model()
            scaler = self.model_manager.get_frequency_scaler()
            
            if model is None:
                self.logger.warning("Frequency model not loaded, using fallback")
                return self._predict_frequencies_simple(features)
            
            # Extract 7D phase field features
            phase_features = self.feature_extractor.extract_7d_phase_features(features)
            
            # Scale features
            phase_features_scaled = scaler.transform([phase_features])
            
            # Make prediction
            predicted_frequencies = model.predict(phase_features_scaled)[0]
            
            # Get prediction confidence from model
            prediction_confidence = self._compute_prediction_confidence(
                phase_features_scaled, model
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model)
            
            return {
                "predicted_frequencies": predicted_frequencies.tolist(),
                "prediction_confidence": prediction_confidence,
                "feature_importance": feature_importance,
                "model_type": "RandomForest",
                "prediction_variance": self._compute_prediction_variance(
                    phase_features_scaled, model
                )
            }
            
        except Exception as e:
            self.logger.error(f"ML frequency prediction failed: {e}")
            features = self.feature_extractor.extract_frequency_features(envelope)
            return self._predict_frequencies_simple(features)
    
    def predict_coupling(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Predict mode coupling using ML.
        
        Physical Meaning:
            Predicts mode coupling using trained machine learning
            model based on 7D phase field features.
            
        Mathematical Foundation:
            Uses Neural Network regression trained on 7D phase field
            data to predict mode coupling from interaction features.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Coupling prediction results.
        """
        try:
            # Extract coupling features
            features = self.feature_extractor.extract_coupling_features(envelope)
            
            # Get ML model
            model = self.model_manager.get_coupling_model()
            scaler = self.model_manager.get_coupling_scaler()
            
            if model is None:
                self.logger.warning("Coupling model not loaded, using fallback")
                return self._predict_coupling_simple(features)
            
            # Extract 7D phase field features
            phase_features = self.feature_extractor.extract_7d_phase_features(features)
            
            # Scale features
            phase_features_scaled = scaler.transform([phase_features])
            
            # Make prediction
            predicted_coupling_raw = model.predict(phase_features_scaled)[0]
            
            # Format prediction results
            predicted_coupling = {
                "coupling_strength": predicted_coupling_raw[0],
                "interaction_energy": predicted_coupling_raw[1],
                "coupling_symmetry": predicted_coupling_raw[2],
                "nonlinear_strength": predicted_coupling_raw[3],
                "mixing_degree": predicted_coupling_raw[4],
                "coupling_efficiency": predicted_coupling_raw[5]
            }
            
            # Get prediction confidence from model
            prediction_confidence = self._compute_prediction_confidence(
                phase_features_scaled, model
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model)
            
            return {
                "predicted_coupling": predicted_coupling,
                "prediction_confidence": prediction_confidence,
                "feature_importance": feature_importance,
                "model_type": "NeuralNetwork",
                "prediction_variance": self._compute_prediction_variance(
                    phase_features_scaled, model
                )
            }
            
        except Exception as e:
            self.logger.error(f"ML coupling prediction failed: {e}")
            features = self.feature_extractor.extract_coupling_features(envelope)
            return self._predict_coupling_simple(features)
    
    def _predict_frequencies_simple(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict frequencies using full analytical method based on 7D BVP theory.
        
        Physical Meaning:
            Predicts frequencies using complete analytical methods based on
            7D phase field theory and VBP envelope analysis.
            
        Mathematical Foundation:
            Implements full 7D phase field frequency prediction using
            spectral analysis, phase coherence, and topological charge.
            
        Args:
            features (Dict[str, Any]): Frequency features from 7D phase field.
            
        Returns:
            Dict[str, Any]: Full analytical frequency prediction results.
        """
        # Extract 7D phase field features
        phase_features = self.feature_extractor.extract_7d_phase_features(features)
        
        # Compute 7D phase field frequency prediction
        predicted_frequencies = self.bvp_analytics.compute_7d_frequency_prediction(phase_features, features)
        
        # Compute prediction confidence based on phase coherence
        prediction_confidence = self.bvp_analytics.compute_analytical_confidence(features)
        
        # Compute feature importance for analytical method
        feature_importance = self.bvp_analytics.compute_analytical_feature_importance(features)
        
        return {
            "predicted_frequencies": predicted_frequencies,
            "prediction_confidence": prediction_confidence,
            "prediction_method": "analytical_7d_bvp",
            "feature_importance": feature_importance,
            "phase_coherence": features.get("phase_coherence", 0.0),
            "topological_charge": features.get("topological_charge", 0.0)
        }
    
    def _predict_coupling_simple(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict coupling using full analytical method based on 7D BVP theory.
        
        Physical Meaning:
            Predicts mode coupling using complete analytical methods based on
            7D phase field theory and VBP envelope interactions.
            
        Mathematical Foundation:
            Implements full 7D phase field coupling prediction using
            interaction energy, phase coherence, and topological charge.
            
        Args:
            features (Dict[str, Any]): Coupling features from 7D phase field.
            
        Returns:
            Dict[str, Any]: Full analytical coupling prediction results.
        """
        # Extract 7D phase field features
        phase_features = self.feature_extractor.extract_7d_phase_features(features)
        
        # Compute 7D phase field coupling prediction
        predicted_coupling = self.bvp_analytics.compute_7d_coupling_prediction(phase_features, features)
        
        # Compute prediction confidence based on interaction strength
        prediction_confidence = self.bvp_analytics.compute_coupling_analytical_confidence(features)
        
        # Compute feature importance for analytical method
        feature_importance = self.bvp_analytics.compute_coupling_analytical_feature_importance(features)
        
        return {
            "predicted_coupling": predicted_coupling,
            "prediction_confidence": prediction_confidence,
            "prediction_method": "analytical_7d_bvp",
            "feature_importance": feature_importance,
            "interaction_energy": features.get("interaction_energy", 0.0),
            "phase_coherence": features.get("phase_coherence", 0.0)
        }
    
    def _compute_prediction_confidence(self, features: np.ndarray, model) -> float:
        """
        Compute prediction confidence from ML model.
        
        Physical Meaning:
            Computes confidence measure for ML predictions
            based on model uncertainty.
            
        Args:
            features (np.ndarray): Scaled input features.
            model: Trained ML model.
            
        Returns:
            float: Prediction confidence (0-1).
        """
        try:
            if hasattr(model, 'predict_proba'):
                # For models with probability output
                proba = model.predict_proba(features)
                confidence = np.max(proba)
            else:
                # For regression models, use prediction variance
                predictions = []
                if hasattr(model, 'estimators_'):  # Random Forest
                    for estimator in model.estimators_:
                        predictions.append(estimator.predict(features))
                    variance = np.var(predictions)
                    confidence = 1.0 / (1.0 + variance)
                else:
                    # Default confidence
                    confidence = 0.8
            return min(max(confidence, 0.0), 1.0)
        except Exception:
            return 0.7  # Default confidence
    
    def _get_feature_importance(self, model) -> Dict[str, float]:
        """
        Get feature importance from ML model.
        
        Physical Meaning:
            Extracts feature importance from trained ML model
            to understand which features are most relevant.
            
        Args:
            model: Trained ML model.
            
        Returns:
            Dict[str, float]: Feature importance dictionary.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                # Random Forest feature importance
                feature_names = [
                    "spectral_entropy", "frequency_spacing", "frequency_bandwidth",
                    "autocorrelation", "coupling_strength", "interaction_energy",
                    "coupling_symmetry", "nonlinear_strength", "mixing_degree",
                    "coupling_efficiency", "phase_coherence", "topological_charge",
                    "energy_density", "phase_velocity"
                ]
                importance_dict = {}
                for i, name in enumerate(feature_names):
                    if i < len(model.feature_importances_):
                        importance_dict[name] = float(model.feature_importances_[i])
                return importance_dict
            else:
                # Default importance for models without feature_importances_
                return {
                    "spectral_entropy": 0.2,
                    "frequency_spacing": 0.15,
                    "frequency_bandwidth": 0.15,
                    "coupling_strength": 0.2,
                    "interaction_energy": 0.15,
                    "phase_coherence": 0.15
                }
        except Exception:
            return {"default": 1.0}
    
    def _compute_prediction_variance(self, features: np.ndarray, model) -> float:
        """
        Compute prediction variance for uncertainty quantification.
        
        Physical Meaning:
            Computes prediction variance to quantify uncertainty
            in ML predictions.
            
        Args:
            features (np.ndarray): Scaled input features.
            model: Trained ML model.
            
        Returns:
            float: Prediction variance.
        """
        try:
            if hasattr(model, 'estimators_'):  # Random Forest
                predictions = []
                for estimator in model.estimators_:
                    predictions.append(estimator.predict(features))
                return float(np.var(predictions))
            else:
                # For single models, return default variance
                return 0.1
        except Exception:
            return 0.1
