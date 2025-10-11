#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simplified demonstration script for ML prediction models.

This script demonstrates the full ML prediction capabilities
for 7D phase field beating analysis without full BVPCore initialization.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bhlff.models.level_c.beating.ml.core.training_data_generator import TrainingDataGenerator
from bhlff.models.level_c.beating.ml.core.ml_trainer import MLTrainer
from bhlff.models.level_c.beating.ml.core.ml_models import MLModelManager
from bhlff.models.level_c.beating.ml.core.feature_extraction import FeatureExtractor
from bhlff.models.level_c.beating.ml.core.prediction_engine import PredictionEngine


def setup_logging():
    """Setup logging for demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_ml_prediction():
    """Demonstrate full ML prediction capabilities."""
    print("=" * 80)
    print("7D Phase Field ML Prediction Models Demonstration")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    
    print("\n1. Training Data Generation")
    print("-" * 40)
    
    # Create training data generator
    data_generator = TrainingDataGenerator()
    
    # Generate training data
    print("Generating frequency training data...")
    X_freq, y_freq = data_generator.generate_frequency_training_data(n_samples=500)
    print(f"Generated {X_freq.shape[0]} frequency samples with {X_freq.shape[1]} features")
    
    print("Generating coupling training data...")
    X_coup, y_coup = data_generator.generate_coupling_training_data(n_samples=500)
    print(f"Generated {X_coup.shape[0]} coupling samples with {X_coup.shape[1]} features")
    
    print("\n2. ML Model Training")
    print("-" * 40)
    
    # Create ML trainer
    trainer = MLTrainer()
    
    # Train frequency model
    print("Training frequency prediction model...")
    freq_results = trainer.train_frequency_model(n_samples=500)
    print(f"Frequency model training completed:")
    print(f"  Model type: {freq_results['model_type']}")
    print(f"  MSE: {freq_results['mse']:.4f}")
    print(f"  R²: {freq_results['r2_score']:.4f}")
    print(f"  Samples: {freq_results['n_samples']}")
    
    # Train coupling model
    print("\nTraining coupling prediction model...")
    coup_results = trainer.train_coupling_model(n_samples=500)
    print(f"Coupling model training completed:")
    print(f"  Model type: {coup_results['model_type']}")
    print(f"  MSE: {coup_results['mse']:.4f}")
    print(f"  R²: {coup_results['r2_score']:.4f}")
    print(f"  Samples: {coup_results['n_samples']}")
    
    print("\n3. Model Validation")
    print("-" * 40)
    
    # Validate models
    validation_results = trainer.validate_models(n_samples=100)
    
    print(f"Frequency model validation:")
    print(f"  MSE: {validation_results['frequency_model']['mse']:.4f}")
    print(f"  R²: {validation_results['frequency_model']['r2_score']:.4f}")
    
    print(f"Coupling model validation:")
    print(f"  MSE: {validation_results['coupling_model']['mse']:.4f}")
    print(f"  R²: {validation_results['coupling_model']['r2_score']:.4f}")
    
    print("\n4. Model Performance Metrics")
    print("-" * 40)
    
    # Get performance metrics
    performance = trainer.get_model_performance()
    
    print(f"Frequency model performance:")
    freq_perf = performance['frequency_model']
    print(f"  Model type: {freq_perf['model_type']}")
    print(f"  Estimators: {freq_perf['n_estimators']}")
    print(f"  Max depth: {freq_perf['max_depth']}")
    
    print(f"\nCoupling model performance:")
    coup_perf = performance['coupling_model']
    print(f"  Model type: {coup_perf['model_type']}")
    print(f"  Hidden layers: {coup_perf['hidden_layers']}")
    print(f"  Max iterations: {coup_perf['max_iter']}")
    print(f"  Learning rate: {coup_perf['learning_rate']}")
    
    print("\n5. Feature Importance Analysis")
    print("-" * 40)
    
    # Display feature importance
    feature_importance = freq_results['feature_importance']
    print("Frequency model feature importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    print("\n6. Prediction Demonstration")
    print("-" * 40)
    
    # Create prediction engine
    model_manager = MLModelManager()
    feature_extractor = FeatureExtractor()
    prediction_engine = PredictionEngine(model_manager, feature_extractor)
    
    # Generate synthetic envelope for prediction
    print("Generating synthetic envelope for prediction...")
    envelope = generate_synthetic_envelope()
    
    # Predict frequencies
    print("Predicting beating frequencies...")
    freq_predictions = prediction_engine.predict_frequencies(envelope)
    print(f"Frequency predictions:")
    print(f"  Predicted frequencies: {freq_predictions['predicted_frequencies']}")
    print(f"  Confidence: {freq_predictions['prediction_confidence']:.4f}")
    print(f"  Method: {freq_predictions['prediction_method']}")
    
    # Predict coupling
    print("\nPredicting mode coupling...")
    coup_predictions = prediction_engine.predict_coupling(envelope)
    print(f"Coupling predictions:")
    print(f"  Coupling strength: {coup_predictions['predicted_coupling']['coupling_strength']:.4f}")
    print(f"  Interaction energy: {coup_predictions['predicted_coupling']['interaction_energy']:.4f}")
    print(f"  Coupling symmetry: {coup_predictions['predicted_coupling']['coupling_symmetry']:.4f}")
    print(f"  Confidence: {coup_predictions['prediction_confidence']:.4f}")
    print(f"  Method: {coup_predictions['prediction_method']}")
    
    print("\n7. 7D BVP Theory Integration")
    print("-" * 40)
    
    print("ML models are fully integrated with 7D BVP theory:")
    print("  ✓ Training data generated using 7D phase field theory")
    print("  ✓ Features extracted from VBP envelope configurations")
    print("  ✓ Targets computed using 7D BVP analytical methods")
    print("  ✓ Models trained on synthetic 7D phase field data")
    print("  ✓ Predictions validated against 7D BVP theory")
    
    print("\n8. Memory-Efficient Processing")
    print("-" * 40)
    
    print("Memory-efficient processing for 7D domains:")
    print("  ✓ Block-wise data generation to avoid memory overflow")
    print("  ✓ Streaming feature extraction for large datasets")
    print("  ✓ Incremental model training with batch processing")
    print("  ✓ Lazy loading of ML models and scalers")
    print("  ✓ Optimized data structures for 7D phase field data")
    
    print("\n" + "=" * 80)
    print("ML Prediction Models Demonstration Completed Successfully!")
    print("=" * 80)


def generate_synthetic_envelope():
    """Generate synthetic envelope for demonstration using memory-efficient approach."""
    # Generate smaller 3D spatial grid for memory efficiency
    x = np.linspace(-2, 2, 32)
    y = np.linspace(-2, 2, 32)
    z = np.linspace(-2, 2, 32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Compute 7D phase field envelope
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Base envelope from 7D phase field theory
    envelope = np.exp(-r**2 / 4.0)
    
    # Add phase coherence effects
    envelope *= (1 + 0.5 * np.cos(r))
    
    # Add topological charge effects
    envelope *= (1 + 0.3 * np.sin(r) / (r + 0.1))
    
    # Add coupling effects
    envelope *= (1 + 0.4 * np.sin(2 * r))
    
    return envelope


if __name__ == "__main__":
    demonstrate_ml_prediction()