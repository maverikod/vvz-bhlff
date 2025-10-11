"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for ML prediction models.

This module contains comprehensive tests for ML prediction models
in 7D phase field beating analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.models.level_c.beating.ml.beating_ml_prediction_core import BeatingMLPredictionCore
from bhlff.models.level_c.beating.ml.core.training_data_generator import TrainingDataGenerator
from bhlff.models.level_c.beating.ml.core.ml_trainer import MLTrainer
from bhlff.core.bvp import BVPCore


class TestMLPredictionModels:
    """Test suite for ML prediction models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.bvp_core = Mock(spec=BVPCore)
        self.predictor = BeatingMLPredictionCore(self.bvp_core)
        self.data_generator = TrainingDataGenerator()
        self.trainer = MLTrainer()
    
    def test_training_data_generation(self):
        """Test training data generation for ML models."""
        # Test frequency training data generation
        X_freq, y_freq = self.data_generator.generate_frequency_training_data(n_samples=100)
        
        assert X_freq.shape[0] == 100
        assert X_freq.shape[1] == 14  # 14 features
        assert y_freq.shape[0] == 100
        assert y_freq.shape[1] == 3  # 3 frequency predictions
        
        # Test coupling training data generation
        X_coup, y_coup = self.data_generator.generate_coupling_training_data(n_samples=100)
        
        assert X_coup.shape[0] == 100
        assert X_coup.shape[1] == 14  # 14 features
        assert y_coup.shape[0] == 100
        assert y_coup.shape[1] == 6  # 6 coupling parameters
    
    def test_frequency_model_training(self):
        """Test frequency model training."""
        # Train frequency model
        results = self.trainer.train_frequency_model(n_samples=200)
        
        assert "model_type" in results
        assert results["model_type"] == "RandomForest"
        assert "mse" in results
        assert "r2_score" in results
        assert "n_samples" in results
        assert results["n_samples"] == 200
        assert "feature_importance" in results
        
        # Check feature importance
        feature_importance = results["feature_importance"]
        assert len(feature_importance) == 14
        assert all(isinstance(v, float) for v in feature_importance.values())
    
    def test_coupling_model_training(self):
        """Test coupling model training."""
        # Train coupling model
        results = self.trainer.train_coupling_model(n_samples=200)
        
        assert "model_type" in results
        assert results["model_type"] == "NeuralNetwork"
        assert "mse" in results
        assert "r2_score" in results
        assert "n_samples" in results
        assert results["n_samples"] == 200
        assert "n_outputs" in results
        assert results["n_outputs"] == 6
    
    def test_all_models_training(self):
        """Test training all ML models."""
        # Train all models
        results = self.trainer.train_all_models(n_samples=200)
        
        assert "frequency_model" in results
        assert "coupling_model" in results
        assert "training_completed" in results
        assert results["training_completed"] is True
        
        # Check frequency model results
        freq_results = results["frequency_model"]
        assert "model_type" in freq_results
        assert freq_results["model_type"] == "RandomForest"
        
        # Check coupling model results
        coup_results = results["coupling_model"]
        assert "model_type" in coup_results
        assert coup_results["model_type"] == "NeuralNetwork"
    
    def test_model_validation(self):
        """Test model validation."""
        # Train models first
        self.trainer.train_all_models(n_samples=200)
        
        # Validate models
        validation_results = self.trainer.validate_models(n_samples=50)
        
        assert "frequency_model" in validation_results
        assert "coupling_model" in validation_results
        
        # Check frequency model validation
        freq_validation = validation_results["frequency_model"]
        assert "mse" in freq_validation
        assert "r2_score" in freq_validation
        assert "validation_samples" in freq_validation
        assert freq_validation["validation_samples"] == 50
        
        # Check coupling model validation
        coup_validation = validation_results["coupling_model"]
        assert "mse" in coup_validation
        assert "r2_score" in coup_validation
        assert "validation_samples" in coup_validation
        assert coup_validation["validation_samples"] == 50
    
    def test_model_performance_metrics(self):
        """Test model performance metrics."""
        # Train models first
        self.trainer.train_all_models(n_samples=200)
        
        # Get performance metrics
        performance = self.trainer.get_model_performance()
        
        assert "frequency_model" in performance
        assert "coupling_model" in performance
        
        # Check frequency model performance
        freq_performance = performance["frequency_model"]
        assert "model_type" in freq_performance
        assert freq_performance["model_type"] == "RandomForest"
        assert "n_estimators" in freq_performance
        assert "max_depth" in freq_performance
        assert "feature_importance" in freq_performance
        
        # Check coupling model performance
        coup_performance = performance["coupling_model"]
        assert "model_type" in coup_performance
        assert coup_performance["model_type"] == "NeuralNetwork"
        assert "hidden_layers" in coup_performance
        assert "max_iter" in coup_performance
        assert "learning_rate" in coup_performance
    
    def test_prediction_core_training(self):
        """Test prediction core training methods."""
        # Test frequency model training
        results = self.predictor.train_frequency_model(n_samples=200)
        
        assert "model_type" in results
        assert results["model_type"] == "RandomForest"
        assert "n_samples" in results
        assert results["n_samples"] == 200
        
        # Test coupling model training
        results = self.predictor.train_coupling_model(n_samples=200)
        
        assert "model_type" in results
        assert results["model_type"] == "NeuralNetwork"
        assert "n_samples" in results
        assert results["n_samples"] == 200
        
        # Test all models training
        results = self.predictor.train_all_models(n_samples=200)
        
        assert "frequency_model" in results
        assert "coupling_model" in results
        assert "training_completed" in results
        assert results["training_completed"] is True
    
    def test_prediction_core_validation(self):
        """Test prediction core validation methods."""
        # Train models first
        self.predictor.train_all_models(n_samples=200)
        
        # Test validation
        validation_results = self.predictor.validate_models(n_samples=50)
        
        assert "frequency_model" in validation_results
        assert "coupling_model" in validation_results
        
        # Test performance metrics
        performance = self.predictor.get_model_performance()
        
        assert "frequency_model" in performance
        assert "coupling_model" in performance
    
    def test_synthetic_envelope_generation(self):
        """Test synthetic envelope generation."""
        # Generate random phase parameters
        phase_params = self.data_generator._generate_random_phase_params()
        
        assert len(phase_params) == 11  # 11 phase field parameters
        assert all(isinstance(v, float) for v in phase_params.values())
        
        # Generate synthetic envelope
        envelope = self.data_generator._generate_synthetic_envelope(phase_params)
        
        assert isinstance(envelope, np.ndarray)
        assert envelope.shape == (64, 64, 64)  # 3D grid
        assert np.all(np.isfinite(envelope))
    
    def test_feature_extraction(self):
        """Test feature extraction from synthetic data."""
        # Generate random phase parameters
        phase_params = self.data_generator._generate_random_phase_params()
        
        # Generate synthetic envelope
        envelope = self.data_generator._generate_synthetic_envelope(phase_params)
        
        # Extract features
        features = self.data_generator._extract_training_features(envelope, phase_params)
        
        assert len(features) == 14  # 14 features
        assert all(isinstance(f, float) for f in features)
        assert all(np.isfinite(f) for f in features))
    
    def test_target_computation(self):
        """Test target computation for ML models."""
        # Generate random phase parameters
        phase_params = self.data_generator._generate_random_phase_params()
        
        # Compute target frequencies
        target_frequencies = self.data_generator._compute_target_frequencies(phase_params)
        
        assert len(target_frequencies) == 3  # 3 frequency predictions
        assert all(isinstance(f, float) for f in target_frequencies)
        assert all(np.isfinite(f) for f in target_frequencies)
        
        # Compute target coupling
        target_coupling = self.data_generator._compute_target_coupling(phase_params)
        
        assert len(target_coupling) == 6  # 6 coupling parameters
        assert all(isinstance(c, float) for c in target_coupling)
        assert all(np.isfinite(c) for c in target_coupling)
    
    def test_7d_bvp_analytics_integration(self):
        """Test integration with 7D BVP analytics."""
        from bhlff.models.level_c.beating.ml.core.bvp_7d_analytics import BVP7DAnalytics
        
        analytics = BVP7DAnalytics()
        
        # Test frequency prediction
        phase_features = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.9, 0.1, 0.3, 0.7, 0.5, 0.8, 0.2])
        features = {
            "spectral_entropy": 0.5,
            "frequency_spacing": 0.3,
            "frequency_bandwidth": 0.7,
            "phase_coherence": 0.8,
            "topological_charge": 0.5
        }
        
        frequencies = analytics.compute_7d_frequency_prediction(phase_features, features)
        
        assert len(frequencies) == 3
        assert all(isinstance(f, float) for f in frequencies)
        assert all(np.isfinite(f) for f in frequencies)
        
        # Test coupling prediction
        coupling_features = {
            "coupling_strength": 0.6,
            "interaction_energy": 0.8,
            "coupling_symmetry": 0.4,
            "nonlinear_strength": 0.7,
            "mixing_degree": 0.3,
            "coupling_efficiency": 0.9,
            "phase_coherence": 0.8,
            "topological_charge": 0.5
        }
        
        coupling = analytics.compute_7d_coupling_prediction(phase_features, coupling_features)
        
        assert len(coupling) == 6
        assert all(isinstance(c, float) for c in coupling.values())
        assert all(np.isfinite(c) for c in coupling.values())
    
    def test_model_persistence(self):
        """Test model persistence and loading."""
        # Train models
        self.trainer.train_all_models(n_samples=200)
        
        # Check if models are saved
        assert self.trainer.model_manager.frequency_model is not None
        assert self.trainer.model_manager.coupling_model is not None
        assert self.trainer.model_manager.frequency_scaler is not None
        assert self.trainer.model_manager.coupling_scaler is not None
        
        # Test model loading
        new_model_manager = MLTrainer()
        assert new_model_manager.model_manager.frequency_model is None
        assert new_model_manager.model_manager.coupling_model is None
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy of trained models."""
        # Train models with more samples for better accuracy
        self.trainer.train_all_models(n_samples=500)
        
        # Generate test data
        X_freq, y_freq = self.data_generator.generate_frequency_training_data(n_samples=100)
        X_coup, y_coup = self.data_generator.generate_coupling_training_data(n_samples=100)
        
        # Test frequency model accuracy
        if self.trainer.model_manager.frequency_model is not None:
            X_freq_scaled = self.trainer.model_manager.frequency_scaler.transform(X_freq)
            y_freq_pred = self.trainer.model_manager.frequency_model.predict(X_freq_scaled)
            
            # Check prediction shape
            assert y_freq_pred.shape == y_freq.shape
            
            # Check prediction quality (should be reasonable for synthetic data)
            mse = np.mean((y_freq - y_freq_pred) ** 2)
            assert mse < 1000  # Reasonable MSE for synthetic data
        
        # Test coupling model accuracy
        if self.trainer.model_manager.coupling_model is not None:
            X_coup_scaled = self.trainer.model_manager.coupling_scaler.transform(X_coup)
            y_coup_pred = self.trainer.model_manager.coupling_model.predict(X_coup_scaled)
            
            # Check prediction shape
            assert y_coup_pred.shape == y_coup.shape
            
            # Check prediction quality (should be reasonable for synthetic data)
            mse = np.mean((y_coup - y_coup_pred) ** 2)
            assert mse < 1000  # Reasonable MSE for synthetic data
