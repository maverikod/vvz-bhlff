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
        frequency_spacing = self.calculator.calculate_frequency_spacing(envelope, envelope.shape)
        
        # Calculate frequency bandwidth
        frequency_bandwidth = self.calculator.calculate_frequency_bandwidth(envelope)
        
        # Calculate autocorrelation
        autocorrelation = self.calculator.calculate_autocorrelation(envelope)
        
        # Calculate 7D phase field features
        phase_coherence = self.phase_features._compute_phase_coherence({
            "coupling_symmetry": 0.0,  # Will be computed in coupling features
            "autocorrelation": autocorrelation
        })
        topological_charge = self.phase_features._compute_topological_charge({
            "mixing_degree": 0.0,  # Will be computed in coupling features
            "nonlinear_strength": 0.0  # Will be computed in coupling features
        })
        
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
        coupling_strength = self.calculator.calculate_frequency_coupling_strength(envelope)
        
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
        phase_coherence = self.phase_features._compute_phase_coherence({
            "coupling_symmetry": coupling_symmetry,
            "autocorrelation": 0.0  # Will be computed from frequency features
        })
        topological_charge = self.phase_features._compute_topological_charge({
            "mixing_degree": mixing_degree,
            "nonlinear_strength": nonlinear_strength
        })
        energy_density = self.phase_features._compute_energy_density({
            "interaction_energy": interaction_energy,
            "coupling_strength": coupling_strength
        })
        phase_velocity = self.phase_features._compute_phase_velocity({
            "frequency_spacing": 0.0,  # Will be computed from frequency features
            "frequency_bandwidth": 0.0  # Will be computed from frequency features
        })
        
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
            features.get("autocorrelation", 0.0)
        ]
        
        # Extract coupling features
        coupling_features = [
            features.get("coupling_strength", 0.0),
            features.get("interaction_energy", 0.0),
            features.get("coupling_symmetry", 0.0),
            features.get("nonlinear_strength", 0.0),
            features.get("mixing_degree", 0.0),
            features.get("coupling_efficiency", 0.0)
        ]
        
        # Extract 7D phase field features
        phase_field_features = [
            features.get("phase_coherence", 0.0),
            features.get("topological_charge", 0.0),
            features.get("energy_density", 0.0),
            features.get("phase_velocity", 0.0)
        ]
        
        # Combine all features
        all_features = basic_features + coupling_features + phase_field_features
        
        return np.array(all_features)
