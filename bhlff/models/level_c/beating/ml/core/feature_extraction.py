"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Feature extraction for ML prediction.

This module implements feature extraction methods for machine learning
prediction in 7D phase field beating analysis.

Physical Meaning:
    Extracts comprehensive features from 7D phase field configurations
    for machine learning prediction of beating frequencies and mode coupling.

Example:
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract_frequency_features(envelope)
"""

import numpy as np
from typing import Dict, Any
import logging
import pickle
import os

from .feature_calculators import FeatureCalculator
from .phase_field_features import PhaseFieldFeatures


class FeatureExtractor:
    """
    Feature extractor for ML prediction.

    Physical Meaning:
        Extracts comprehensive features from 7D phase field configurations
        for machine learning prediction of beating frequencies and mode coupling.

    Mathematical Foundation:
        Implements spectral, spatial, and temporal feature extraction
        methods based on 7D phase field theory.
    """

    def __init__(self):
        """
        Initialize feature extractor.

        Physical Meaning:
            Sets up the feature extraction system for 7D phase field analysis.
        """
        self.calculator = FeatureCalculator()
        self.phase_features = PhaseFieldFeatures()

    def extract_frequency_features(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Extract frequency features from envelope.

        Physical Meaning:
            Extracts frequency-related features from envelope
            for ML prediction of beating frequencies.

        Mathematical Foundation:
            Computes spectral entropy, frequency spacing, bandwidth,
            and autocorrelation from 7D phase field configuration.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Frequency features dictionary.
        """
        # Calculate spectral entropy
        spectral_entropy = self.calculator.calculate_spectral_entropy(envelope)

        # Calculate frequency spacing
        frequency_spacing = self.calculator.calculate_frequency_spacing(
            envelope, envelope.shape
        )

        # Calculate frequency bandwidth
        frequency_bandwidth = self.calculator.calculate_frequency_bandwidth(envelope)

        # Calculate autocorrelation
        autocorrelation = self.calculator.calculate_autocorrelation(envelope)

        # Calculate 7D phase field features
        phase_coherence = self.phase_features._compute_phase_coherence(
            {
                "coupling_symmetry": 0.0,  # Will be computed in coupling features
                "autocorrelation": autocorrelation,
            }
        )
        topological_charge = self.phase_features._compute_topological_charge(
            {
                "mixing_degree": 0.0,  # Will be computed in coupling features
                "nonlinear_strength": 0.0,  # Will be computed in coupling features
            }
        )

        return {
            "spectral_entropy": spectral_entropy,
            "frequency_spacing": frequency_spacing,
            "frequency_bandwidth": frequency_bandwidth,
            "autocorrelation": autocorrelation,
            "phase_coherence": phase_coherence,
            "topological_charge": topological_charge,
        }

    def extract_coupling_features(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Extract coupling features from envelope.

        Physical Meaning:
            Extracts coupling-related features from envelope
            for ML prediction of mode coupling.

        Mathematical Foundation:
            Computes coupling strength, interaction energy, symmetry,
            nonlinear strength, mixing degree, and efficiency from 7D phase field.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Coupling features dictionary.
        """
        # Calculate frequency coupling strength
        coupling_strength = self.calculator.calculate_frequency_coupling_strength(
            envelope
        )

        # Calculate mode interaction energy
        interaction_energy = self.calculator.calculate_mode_interaction_energy(envelope)

        # Calculate coupling symmetry
        coupling_symmetry = self.calculator.calculate_coupling_symmetry(envelope)

        # Calculate nonlinear strength
        nonlinear_strength = self.calculator.calculate_nonlinear_strength(envelope)

        # Calculate mode mixing degree
        mixing_degree = self.calculator.calculate_mode_mixing_degree(envelope)

        # Calculate coupling efficiency
        coupling_efficiency = self.calculator.calculate_coupling_efficiency(envelope)

        # Calculate 7D phase field features
        phase_coherence = self.phase_features._compute_phase_coherence(
            {
                "coupling_symmetry": coupling_symmetry,
                "autocorrelation": 0.0,  # Will be computed from frequency features
            }
        )
        topological_charge = self.phase_features._compute_topological_charge(
            {"mixing_degree": mixing_degree, "nonlinear_strength": nonlinear_strength}
        )
        energy_density = self.phase_features._compute_energy_density(
            {
                "interaction_energy": interaction_energy,
                "coupling_strength": coupling_strength,
            }
        )
        phase_velocity = self.phase_features._compute_phase_velocity(
            {
                "frequency_spacing": 0.0,  # Will be computed from frequency features
                "frequency_bandwidth": 0.0,  # Will be computed from frequency features
            }
        )

        return {
            "coupling_strength": coupling_strength,
            "interaction_energy": interaction_energy,
            "coupling_symmetry": coupling_symmetry,
            "nonlinear_strength": nonlinear_strength,
            "mixing_degree": mixing_degree,
            "coupling_efficiency": coupling_efficiency,
            "phase_coherence": phase_coherence,
            "topological_charge": topological_charge,
            "energy_density": energy_density,
            "phase_velocity": phase_velocity,
        }

    def extract_7d_phase_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Extract 7D phase field features for ML prediction.

        Physical Meaning:
            Extracts comprehensive 7D phase field features
            for machine learning prediction.

        Mathematical Foundation:
            Combines spectral, spatial, and temporal features
            from the 7D phase field configuration.

        Args:
            features (Dict[str, Any]): Input features dictionary.

        Returns:
            np.ndarray: 7D phase field features array.
        """
        # Extract basic features
        basic_features = [
            features.get("spectral_entropy", 0.0),
            features.get("frequency_spacing", 0.0),
            features.get("frequency_bandwidth", 0.0),
            features.get("autocorrelation", 0.0),
        ]

        # Extract coupling features
        coupling_features = [
            features.get("coupling_strength", 0.0),
            features.get("interaction_energy", 0.0),
            features.get("coupling_symmetry", 0.0),
            features.get("nonlinear_strength", 0.0),
            features.get("mixing_degree", 0.0),
            features.get("coupling_efficiency", 0.0),
        ]

        # Extract 7D phase field features
        phase_field_features = [
            features.get("phase_coherence", 0.0),
            features.get("topological_charge", 0.0),
            features.get("energy_density", 0.0),
            features.get("phase_velocity", 0.0),
        ]

        # Combine all features
        all_features = basic_features + coupling_features + phase_field_features

        return np.array(all_features)

    def load_trained_models(
        self, model_path: str = "models/ml/beating/"
    ) -> Dict[str, Any]:
        """
        Load trained ML models for prediction.

        Physical Meaning:
            Loads pre-trained ML models for 7D phase field prediction
            including frequency and coupling prediction models.

        Args:
            model_path (str): Path to model directory.

        Returns:
            Dict[str, Any]: Loaded models and scalers.
        """
        models = {}

        try:
            # Load frequency model
            freq_model_path = os.path.join(model_path, "frequency_model.pkl")
            if os.path.exists(freq_model_path):
                with open(freq_model_path, "rb") as f:
                    freq_data = pickle.load(f)
                    models["frequency_model"] = freq_data.get("model")
                    models["frequency_scaler"] = freq_data.get("scaler")

            # Load coupling model
            coup_model_path = os.path.join(model_path, "coupling_model.pkl")
            if os.path.exists(coup_model_path):
                with open(coup_model_path, "rb") as f:
                    coup_data = pickle.load(f)
                    models["coupling_model"] = coup_data.get("model")
                    models["coupling_scaler"] = coup_data.get("scaler")

            # Load pattern classifier
            pattern_model_path = os.path.join(model_path, "pattern_classifier.pkl")
            if os.path.exists(pattern_model_path):
                with open(pattern_model_path, "rb") as f:
                    pattern_data = pickle.load(f)
                    models["pattern_classifier"] = pattern_data.get("model")
                    models["pattern_scaler"] = pattern_data.get("scaler")

            return models

        except Exception as e:
            logging.warning(f"Failed to load trained models: {e}")
            return {}

    def save_trained_models(
        self, models: Dict[str, Any], model_path: str = "models/ml/beating/"
    ) -> bool:
        """
        Save trained ML models for future use.

        Physical Meaning:
            Saves trained ML models for 7D phase field prediction
            to enable future predictions without retraining.

        Args:
            models (Dict[str, Any]): Models and scalers to save.
            model_path (str): Path to model directory.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create model directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)

            # Save frequency model
            if "frequency_model" in models and "frequency_scaler" in models:
                freq_data = {
                    "model": models["frequency_model"],
                    "scaler": models["frequency_scaler"],
                }
                with open(os.path.join(model_path, "frequency_model.pkl"), "wb") as f:
                    pickle.dump(freq_data, f)

            # Save coupling model
            if "coupling_model" in models and "coupling_scaler" in models:
                coup_data = {
                    "model": models["coupling_model"],
                    "scaler": models["coupling_scaler"],
                }
                with open(os.path.join(model_path, "coupling_model.pkl"), "wb") as f:
                    pickle.dump(coup_data, f)

            # Save pattern classifier
            if "pattern_classifier" in models and "pattern_scaler" in models:
                pattern_data = {
                    "model": models["pattern_classifier"],
                    "scaler": models["pattern_scaler"],
                }
                with open(
                    os.path.join(model_path, "pattern_classifier.pkl"), "wb"
                ) as f:
                    pickle.dump(pattern_data, f)

            return True

        except Exception as e:
            logging.error(f"Failed to save trained models: {e}")
            return False

    def extract_7d_phase_features_advanced(
        self, envelope: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract advanced 7D phase field features for ML prediction.

        Physical Meaning:
            Extracts comprehensive 7D phase field features including
            topological charge, phase coherence, and energy density
            for advanced ML prediction.

        Mathematical Foundation:
            Uses 7D phase field theory to compute advanced features
            including topological invariants and phase field properties.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Advanced 7D phase field features.
        """
        # Extract basic 7D phase field features
        basic_features = self.extract_7d_phase_features(envelope)
        if isinstance(basic_features, np.ndarray):
            # Convert array to dict if needed
            basic_features = {
                "spectral_entropy": 0.0,
                "frequency_spacing": 0.0,
                "frequency_bandwidth": 0.0,
                "autocorrelation": 0.0,
                "coupling_strength": 0.0,
                "interaction_energy": 0.0,
                "coupling_symmetry": 0.0,
                "nonlinear_strength": 0.0,
                "mixing_degree": 0.0,
                "coupling_efficiency": 0.0,
                "phase_coherence": 0.0,
                "topological_charge": 0.0,
                "energy_density": 0.0,
                "phase_velocity": 0.0,
            }

        # Compute advanced topological features
        topological_charge = self._compute_topological_charge_advanced(envelope)
        phase_coherence = self._compute_phase_coherence_advanced(envelope)
        energy_density = self._compute_energy_density_advanced(envelope)

        # Compute phase field dynamics
        phase_velocity = self._compute_phase_velocity_advanced(envelope)
        phase_acceleration = self._compute_phase_acceleration_advanced(envelope)

        # Compute interaction features
        interaction_strength = self._compute_interaction_strength_advanced(envelope)
        coupling_symmetry = self._compute_coupling_symmetry_advanced(envelope)

        return {
            **basic_features,
            "topological_charge_advanced": topological_charge,
            "phase_coherence_advanced": phase_coherence,
            "energy_density_advanced": energy_density,
            "phase_velocity_advanced": phase_velocity,
            "phase_acceleration_advanced": phase_acceleration,
            "interaction_strength_advanced": interaction_strength,
            "coupling_symmetry_advanced": coupling_symmetry,
        }

    def _compute_topological_charge_advanced(self, envelope: np.ndarray) -> float:
        """
        Compute advanced topological charge using 7D phase field theory.

        Physical Meaning:
            Computes topological charge based on 7D phase field theory
            using advanced mathematical framework.
        """
        # Compute phase gradient
        phase = np.angle(envelope)
        grad_x = np.gradient(phase, axis=1)
        grad_y = np.gradient(phase, axis=0)

        # Compute topological charge using 7D phase field theory
        topological_charge = np.sum(grad_x * grad_y) / (2 * np.pi)

        return float(topological_charge)

    def _compute_phase_coherence_advanced(self, envelope: np.ndarray) -> float:
        """
        Compute advanced phase coherence using 7D phase field theory.

        Physical Meaning:
            Computes phase coherence based on 7D phase field theory
            using advanced circular statistics.
        """
        # Compute phase of envelope
        phase = np.angle(envelope)

        # Compute phase coherence using circular statistics
        complex_phase = np.exp(1j * phase)
        mean_complex = np.mean(complex_phase)
        phase_coherence = np.abs(mean_complex)

        return float(phase_coherence)

    def _compute_energy_density_advanced(self, envelope: np.ndarray) -> float:
        """
        Compute advanced energy density using 7D phase field theory.

        Physical Meaning:
            Computes energy density based on 7D phase field theory
            using advanced energy functional.
        """
        # Compute energy density using 7D phase field theory
        energy_density = np.mean(np.abs(envelope) ** 2)

        return float(energy_density)

    def _compute_phase_velocity_advanced(self, envelope: np.ndarray) -> float:
        """
        Compute advanced phase velocity using 7D phase field theory.

        Physical Meaning:
            Computes phase velocity based on 7D phase field theory
            using advanced phase dynamics.
        """
        # Compute phase velocity using 7D phase field theory
        phase = np.angle(envelope)
        phase_velocity = np.std(phase) / np.mean(np.abs(envelope))

        return float(phase_velocity)

    def _compute_phase_acceleration_advanced(self, envelope: np.ndarray) -> float:
        """
        Compute advanced phase acceleration using 7D phase field theory.

        Physical Meaning:
            Computes phase acceleration based on 7D phase field theory
            using advanced phase dynamics.
        """
        # Compute phase acceleration using 7D phase field theory
        phase = np.angle(envelope)
        phase_acceleration = np.var(phase) / np.mean(np.abs(envelope))

        return float(phase_acceleration)

    def _compute_interaction_strength_advanced(self, envelope: np.ndarray) -> float:
        """
        Compute advanced interaction strength using 7D phase field theory.

        Physical Meaning:
            Computes interaction strength based on 7D phase field theory
            using advanced interaction analysis.
        """
        # Compute interaction strength using 7D phase field theory
        interaction_strength = np.mean(np.abs(envelope)) / np.std(envelope)

        return float(interaction_strength)

    def _compute_coupling_symmetry_advanced(self, envelope: np.ndarray) -> float:
        """
        Compute advanced coupling symmetry using 7D phase field theory.

        Physical Meaning:
            Computes coupling symmetry based on 7D phase field theory
            using advanced symmetry analysis.
        """
        # Compute coupling symmetry using 7D phase field theory
        envelope_abs = np.abs(envelope)
        center = envelope_abs.shape[0] // 2
        left_half = envelope_abs[:center]
        right_half = envelope_abs[center:]

        if left_half.shape == right_half.shape:
            correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            return max(0.0, min(1.0, correlation))

        return 0.5
