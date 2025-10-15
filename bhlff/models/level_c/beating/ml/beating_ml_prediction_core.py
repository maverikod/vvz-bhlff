"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Machine learning prediction core module facade.

This module provides the main interface for ML prediction functionality
for beating analysis in Level C of 7D phase field theory.

Physical Meaning:
    Provides machine learning-based prediction functions for analyzing
    beating frequencies and mode coupling in the 7D phase field.

Example:
    >>> predictor = BeatingMLPredictionCore(bvp_core)
    >>> frequencies = predictor.predict_beating_frequencies(envelope)
"""

from bhlff.core.bvp import BVPCore
from .beating_ml_prediction import BeatingMLPredictionCore as CorePredictor


class BeatingMLPredictionCore:
    """
    Machine learning prediction core facade for beating analysis.
    
    Physical Meaning:
        Provides machine learning-based prediction functions for analyzing
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
        # Delegate to core implementation
        self._core = CorePredictor(bvp_core)
    
    def predict_beating_frequencies(self, envelope):
        """Predict beating frequencies using full ML implementation."""
        return self._core.predict_beating_frequencies(envelope)
    
    def predict_mode_coupling(self, envelope):
        """Predict mode coupling using full ML implementation."""
        return self._core.predict_mode_coupling(envelope)
    
    def train_frequency_model(self, n_samples: int = 1000):
        """Train frequency prediction model using 7D BVP theory."""
        return self._core.train_frequency_model(n_samples)
    
    def train_coupling_model(self, n_samples: int = 1000):
        """Train coupling prediction model using 7D BVP theory."""
        return self._core.train_coupling_model(n_samples)
    
    def train_all_models(self, n_samples: int = 1000):
        """Train all ML models using 7D BVP theory."""
        return self._core.train_all_models(n_samples)
    
    def validate_models(self, n_samples: int = 200):
        """Validate trained ML models."""
        return self._core.validate_models(n_samples)
    
    def get_model_performance(self):
        """Get model performance metrics."""
        return self._core.get_model_performance()
