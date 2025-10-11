"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Machine learning prediction core module.

This module implements core ML prediction functionality for beating analysis
in Level C of 7D phase field theory.

Physical Meaning:
    Provides core machine learning-based prediction functions for analyzing
    beating frequencies and mode coupling in the 7D phase field.

Example:
    >>> predictor = BeatingMLPredictionCore(bvp_core)
    >>> frequencies = predictor.predict_beating_frequencies(envelope)
"""

import numpy as np
from typing import Dict, Any
import logging

from bhlff.core.bvp import BVPCore
from .core import MLModelManager, FeatureExtractor, PredictionEngine, MLTrainer


class BeatingMLPredictionCore:
    """
    Machine learning prediction core for beating analysis.
    
    Physical Meaning:
        Provides core machine learning-based prediction functions for analyzing
        beating frequencies and mode coupling in the 7D phase field.
        
    Mathematical Foundation:
        Uses machine learning techniques for frequency prediction and
        mode coupling analysis in beating phenomena.
    """
    
    def __init__(self, bvp_core: BVPCore):
        """
        Initialize prediction analyzer.
        
        Physical Meaning:
            Sets up the ML prediction system with
            appropriate parameters and methods.
            
        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        
        # Prediction parameters
        self.frequency_prediction_enabled = True
        self.coupling_prediction_enabled = True
        self.prediction_confidence = 0.7
        
        # Initialize core components
        self.model_manager = MLModelManager()
        self.feature_extractor = FeatureExtractor()
        self.prediction_engine = PredictionEngine(self.model_manager, self.feature_extractor)
        self.ml_trainer = MLTrainer(self.model_manager)
    
    def predict_beating_frequencies(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Predict beating frequencies using machine learning.
        
        Physical Meaning:
            Predicts beating frequencies using machine learning
            techniques for 7D phase field analysis.
            
        Mathematical Foundation:
            Uses machine learning techniques for frequency prediction
            in beating phenomena.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Frequency prediction results.
        """
        self.logger.info("Starting frequency prediction")
        
        # Use prediction engine for ML prediction
        if self.frequency_prediction_enabled:
            prediction_results = self.prediction_engine.predict_frequencies(envelope)
        else:
            # Fallback to simple prediction
            features = self.feature_extractor.extract_frequency_features(envelope)
            prediction_results = self.prediction_engine._predict_frequencies_simple(features)
        
        # Add confidence and validation
        prediction_results["confidence"] = self.prediction_confidence
        prediction_results["prediction_method"] = "ml" if self.frequency_prediction_enabled else "simple"
        
        self.logger.info("Frequency prediction completed")
        return prediction_results
    
    def predict_mode_coupling(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Predict mode coupling using machine learning.
        
        Physical Meaning:
            Predicts mode coupling using machine learning
            techniques for 7D phase field analysis.
            
        Mathematical Foundation:
            Uses machine learning techniques for mode coupling
            analysis in beating phenomena.
            
        Args:
            envelope (np.ndarray): 7D envelope field data.
            
        Returns:
            Dict[str, Any]: Mode coupling prediction results.
        """
        self.logger.info("Starting mode coupling prediction")
        
        # Use prediction engine for ML prediction
        if self.coupling_prediction_enabled:
            prediction_results = self.prediction_engine.predict_coupling(envelope)
        else:
            # Fallback to simple prediction
            features = self.feature_extractor.extract_coupling_features(envelope)
            prediction_results = self.prediction_engine._predict_coupling_simple(features)
        
        # Add confidence and validation
        prediction_results["confidence"] = self.prediction_confidence
        prediction_results["prediction_method"] = "ml" if self.coupling_prediction_enabled else "simple"
        
        self.logger.info("Mode coupling prediction completed")
        return prediction_results
    
    def train_frequency_model(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Train frequency prediction model using 7D BVP theory.
        
        Physical Meaning:
            Trains Random Forest model for frequency prediction using
            7D phase field theory and VBP envelope configurations.
            
        Mathematical Foundation:
            Uses Random Forest regression trained on synthetic 7D phase field
            data to predict beating frequencies from spectral features.
            
        Args:
            n_samples (int): Number of training samples to generate.
            
        Returns:
            Dict[str, Any]: Training results and model performance.
        """
        self.logger.info(f"Training frequency model with {n_samples} samples")
        return self.ml_trainer.train_frequency_model(n_samples)
    
    def train_coupling_model(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Train coupling prediction model using 7D BVP theory.
        
        Physical Meaning:
            Trains Neural Network model for coupling prediction using
            7D phase field theory and VBP envelope interactions.
            
        Mathematical Foundation:
            Uses Neural Network regression trained on synthetic 7D phase field
            data to predict mode coupling from interaction features.
            
        Args:
            n_samples (int): Number of training samples to generate.
            
        Returns:
            Dict[str, Any]: Training results and model performance.
        """
        self.logger.info(f"Training coupling model with {n_samples} samples")
        return self.ml_trainer.train_coupling_model(n_samples)
    
    def train_all_models(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Train all ML models using 7D BVP theory.
        
        Physical Meaning:
            Trains both frequency and coupling prediction models using
            7D phase field theory and VBP envelope configurations.
            
        Args:
            n_samples (int): Number of training samples to generate.
            
        Returns:
            Dict[str, Any]: Training results for all models.
        """
        self.logger.info(f"Training all models with {n_samples} samples")
        return self.ml_trainer.train_all_models(n_samples)
    
    def validate_models(self, n_samples: int = 200) -> Dict[str, Any]:
        """
        Validate trained ML models.
        
        Physical Meaning:
            Validates trained ML models using independent test data
            to ensure prediction accuracy and reliability.
            
        Args:
            n_samples (int): Number of validation samples to generate.
            
        Returns:
            Dict[str, Any]: Validation results for all models.
        """
        self.logger.info(f"Validating models with {n_samples} samples")
        return self.ml_trainer.validate_models(n_samples)
    
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get model performance metrics.
        
        Physical Meaning:
            Returns performance metrics for trained ML models
            to assess prediction quality and reliability.
            
        Returns:
            Dict[str, Any]: Model performance metrics.
        """
        return self.ml_trainer.get_model_performance()
    
