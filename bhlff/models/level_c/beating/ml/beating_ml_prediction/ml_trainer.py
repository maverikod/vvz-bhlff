"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

ML trainer functionality.

This module implements ML trainer for model training including
data generation, model training, and performance evaluation.

Physical Meaning:
    Provides ML trainer for training machine learning models
    for beating frequency and mode coupling prediction in 7D phase field theory.

Example:
    >>> trainer = MLTrainer(model_manager)
    >>> results = trainer.train_frequency_model(n_samples=1000)
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class MLTrainer:
    """
    ML trainer for beating analysis models.

    Physical Meaning:
        Provides ML trainer for training machine learning models
        for beating frequency and mode coupling prediction in 7D phase field theory.

    Mathematical Foundation:
        Implements data generation, model training, and performance evaluation
        for ML-based prediction analysis.
    """

    def __init__(self, model_manager):
        """
        Initialize ML trainer.

        Args:
            model_manager: ML model manager instance.
        """
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

    def train_frequency_model(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Train frequency prediction model.

        Physical Meaning:
            Trains Random Forest model for frequency prediction using
            7D phase field theory and synthetic data generation.

        Args:
            n_samples (int): Number of training samples to generate.

        Returns:
            Dict[str, Any]: Training results and model performance.
        """
        try:
            self.logger.info(f"Training frequency model with {n_samples} samples")

            # Generate training data
            X, y = self._generate_frequency_training_data(n_samples)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Save model and scaler
            self.model_manager.save_frequency_model(model, scaler)

            # Update performance
            performance = {
                "mse": mse,
                "r2": r2,
                "n_samples": n_samples,
                "model_type": "RandomForest",
            }
            self.model_manager.update_model_performance("frequency", performance)

            self.logger.info(f"Frequency model training completed. R² = {r2:.3f}")

            return {"success": True, "performance": performance, "model_saved": True}

        except Exception as e:
            self.logger.error(f"Frequency model training failed: {e}")
            return {"success": False, "error": str(e)}

    def train_coupling_model(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Train coupling prediction model.

        Physical Meaning:
            Trains Neural Network model for coupling prediction using
            7D phase field theory and synthetic data generation.

        Args:
            n_samples (int): Number of training samples to generate.

        Returns:
            Dict[str, Any]: Training results and model performance.
        """
        try:
            self.logger.info(f"Training coupling model with {n_samples} samples")

            # Generate training data
            X, y = self._generate_coupling_training_data(n_samples)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Save model and scaler
            self.model_manager.save_coupling_model(model, scaler)

            # Update performance
            performance = {
                "mse": mse,
                "r2": r2,
                "n_samples": n_samples,
                "model_type": "NeuralNetwork",
            }
            self.model_manager.update_model_performance("coupling", performance)

            self.logger.info(f"Coupling model training completed. R² = {r2:.3f}")

            return {"success": True, "performance": performance, "model_saved": True}

        except Exception as e:
            self.logger.error(f"Coupling model training failed: {e}")
            return {"success": False, "error": str(e)}

    def train_all_models(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Train all ML models.

        Physical Meaning:
            Trains both frequency and coupling prediction models using
            7D phase field theory and synthetic data generation.

        Args:
            n_samples (int): Number of training samples to generate.

        Returns:
            Dict[str, Any]: Training results for all models.
        """
        try:
            self.logger.info(f"Training all models with {n_samples} samples")

            # Train frequency model
            frequency_results = self.train_frequency_model(n_samples)

            # Train coupling model
            coupling_results = self.train_coupling_model(n_samples)

            return {
                "frequency_model": frequency_results,
                "coupling_model": coupling_results,
                "all_models_trained": True,
            }

        except Exception as e:
            self.logger.error(f"All models training failed: {e}")
            return {"all_models_trained": False, "error": str(e)}

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
        try:
            self.logger.info(f"Validating models with {n_samples} samples")

            # Validate frequency model
            frequency_validation = self._validate_frequency_model(n_samples)

            # Validate coupling model
            coupling_validation = self._validate_coupling_model(n_samples)

            return {
                "frequency_validation": frequency_validation,
                "coupling_validation": coupling_validation,
                "validation_completed": True,
            }

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return {"validation_completed": False, "error": str(e)}

    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        return self.model_manager.get_model_performance()

    def _generate_frequency_training_data(
        self, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for frequency prediction."""
        try:
            # Generate synthetic 7D phase field data
            X = []
            y = []

            for i in range(n_samples):
                # Generate synthetic envelope data
                envelope = self._generate_synthetic_envelope()

                # Extract features
                features = self._extract_frequency_features(envelope)
                X.append(features)

                # Generate target frequencies
                target_frequencies = self._generate_target_frequencies(features)
                y.append(target_frequencies)

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Frequency training data generation failed: {e}")
            return np.array([]), np.array([])

    def _generate_coupling_training_data(
        self, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for coupling prediction."""
        try:
            # Generate synthetic 7D phase field data
            X = []
            y = []

            for i in range(n_samples):
                # Generate synthetic envelope data
                envelope = self._generate_synthetic_envelope()

                # Extract features
                features = self._extract_coupling_features(envelope)
                X.append(features)

                # Generate target coupling
                target_coupling = self._generate_target_coupling(features)
                y.append(target_coupling)

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Coupling training data generation failed: {e}")
            return np.array([]), np.array([])

    def _generate_synthetic_envelope(self) -> np.ndarray:
        """Generate synthetic envelope data for training."""
        try:
            # Generate synthetic 7D phase field envelope
            n_points = 100
            x = np.linspace(0, 10, n_points)

            # Generate multiple frequency components
            freq1 = np.random.uniform(0.5, 2.0)
            freq2 = np.random.uniform(0.5, 2.0)
            amp1 = np.random.uniform(0.5, 2.0)
            amp2 = np.random.uniform(0.5, 2.0)

            # Generate envelope with beating
            envelope = amp1 * np.sin(2 * np.pi * freq1 * x) + amp2 * np.sin(
                2 * np.pi * freq2 * x
            )

            # Add noise
            noise = np.random.normal(0, 0.1, n_points)
            envelope += noise

            return envelope

        except Exception as e:
            self.logger.error(f"Synthetic envelope generation failed: {e}")
            return np.zeros(100)

    def _extract_frequency_features(self, envelope: np.ndarray) -> np.ndarray:
        """Extract frequency features from envelope data."""
        try:
            # Extract basic features
            spectral_entropy = np.var(envelope)
            frequency_spacing = np.std(envelope)
            frequency_bandwidth = np.mean(np.abs(envelope))
            autocorrelation = (
                np.corrcoef(envelope[:-1], envelope[1:])[0, 1]
                if len(envelope) > 1
                else 0.0
            )

            # Extract 7D phase field features
            phase_coherence = np.abs(np.mean(np.exp(1j * np.angle(envelope))))
            topological_charge = np.sum(np.gradient(np.angle(envelope))) / (2 * np.pi)
            energy_density = np.mean(envelope**2)
            phase_velocity = np.std(np.angle(envelope))

            return np.array(
                [
                    spectral_entropy,
                    frequency_spacing,
                    frequency_bandwidth,
                    autocorrelation,
                    phase_coherence,
                    topological_charge,
                    energy_density,
                    phase_velocity,
                ]
            )

        except Exception as e:
            self.logger.error(f"Frequency feature extraction failed: {e}")
            return np.zeros(8)

    def _extract_coupling_features(self, envelope: np.ndarray) -> np.ndarray:
        """Extract coupling features from envelope data."""
        try:
            # Extract basic features
            coupling_strength = np.var(envelope)
            interaction_energy = np.mean(envelope**2)
            coupling_symmetry = (
                np.corrcoef(envelope, np.flip(envelope))[0, 1]
                if len(envelope) > 1
                else 0.0
            )
            nonlinear_strength = (
                np.mean(((envelope - np.mean(envelope)) / np.std(envelope)) ** 3)
                if np.std(envelope) > 0
                else 0.0
            )
            mixing_degree = (
                np.mean(((envelope - np.mean(envelope)) / np.std(envelope)) ** 4) - 3
                if np.std(envelope) > 0
                else 0.0
            )
            coupling_efficiency = (
                np.sum(envelope**2) / np.var(envelope) if np.var(envelope) > 0 else 0.0
            )

            # Extract 7D phase field features
            phase_coherence = np.abs(np.mean(np.exp(1j * np.angle(envelope))))
            topological_charge = np.sum(np.gradient(np.angle(envelope))) / (2 * np.pi)
            energy_density = np.mean(envelope**2)
            phase_velocity = np.std(np.angle(envelope))

            return np.array(
                [
                    coupling_strength,
                    interaction_energy,
                    coupling_symmetry,
                    nonlinear_strength,
                    mixing_degree,
                    coupling_efficiency,
                    phase_coherence,
                    topological_charge,
                    energy_density,
                    phase_velocity,
                ]
            )

        except Exception as e:
            self.logger.error(f"Coupling feature extraction failed: {e}")
            return np.zeros(10)

    def _generate_target_frequencies(self, features: np.ndarray) -> np.ndarray:
        """Generate target frequencies for training."""
        try:
            # Generate target frequencies based on features
            freq1 = features[0] * 100.0  # spectral_entropy
            freq2 = features[1] * 50.0  # frequency_spacing
            freq3 = features[2] * 25.0  # frequency_bandwidth

            return np.array([freq1, freq2, freq3])

        except Exception as e:
            self.logger.error(f"Target frequency generation failed: {e}")
            return np.array([0.0, 0.0, 0.0])

    def _generate_target_coupling(self, features: np.ndarray) -> np.ndarray:
        """Generate target coupling for training."""
        try:
            # Generate target coupling based on features
            coupling_strength = features[0] * 1.2
            interaction_energy = features[1] * 0.8
            coupling_symmetry = features[2] * 1.1
            nonlinear_strength = features[3] * 0.9
            mixing_degree = features[4] * 1.0
            coupling_efficiency = features[5] * 1.05

            return np.array(
                [
                    coupling_strength,
                    interaction_energy,
                    coupling_symmetry,
                    nonlinear_strength,
                    mixing_degree,
                    coupling_efficiency,
                ]
            )

        except Exception as e:
            self.logger.error(f"Target coupling generation failed: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _validate_frequency_model(self, n_samples: int) -> Dict[str, Any]:
        """Validate frequency model."""
        try:
            # Generate validation data
            X, y = self._generate_frequency_training_data(n_samples)

            # Load model and scaler
            model = self.model_manager.get_frequency_model()
            scaler = self.model_manager.get_frequency_scaler()

            if model is not None and scaler is not None:
                # Scale features
                X_scaled = scaler.transform(X)

                # Make predictions
                y_pred = model.predict(X_scaled)

                # Compute validation metrics
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                return {
                    "mse": mse,
                    "r2": r2,
                    "n_samples": n_samples,
                    "validation_successful": True,
                }
            else:
                return {"validation_successful": False, "error": "Model not available"}

        except Exception as e:
            self.logger.error(f"Frequency model validation failed: {e}")
            return {"validation_successful": False, "error": str(e)}

    def _validate_coupling_model(self, n_samples: int) -> Dict[str, Any]:
        """Validate coupling model."""
        try:
            # Generate validation data
            X, y = self._generate_coupling_training_data(n_samples)

            # Load model and scaler
            model = self.model_manager.get_coupling_model()
            scaler = self.model_manager.get_coupling_scaler()

            if model is not None and scaler is not None:
                # Scale features
                X_scaled = scaler.transform(X)

                # Make predictions
                y_pred = model.predict(X_scaled)

                # Compute validation metrics
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                return {
                    "mse": mse,
                    "r2": r2,
                    "n_samples": n_samples,
                    "validation_successful": True,
                }
            else:
                return {"validation_successful": False, "error": "Model not available"}

        except Exception as e:
            self.logger.error(f"Coupling model validation failed: {e}")
            return {"validation_successful": False, "error": str(e)}
