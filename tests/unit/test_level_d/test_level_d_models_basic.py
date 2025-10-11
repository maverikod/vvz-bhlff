"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic unit tests for Level D models.

This module contains basic unit tests for Level D models,
including Level D models and multi-mode model.

Physical Meaning:
    Tests verify that Level D models correctly implement:
    - Multimode superposition with frame stability analysis
    - Basic model initialization and validation

Example:
    >>> pytest tests/unit/test_level_d/test_level_d_models_basic.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import tempfile
import json
import os

from bhlff.models.level_d import (
    LevelDModels,
    MultiModeModel,
)
from bhlff.core.domain import Domain
from bhlff.models.base.abstract_models import AbstractLevelModels


class TestLevelDModels:
    """Test Level D models functionality."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=10.0, N=16, dimensions=7, N_phi=8, N_t=16, T=1.0)

    @pytest.fixture
    def parameters(self):
        """Create test parameters."""
        return {
            "jaccard_threshold": 0.8,
            "frequency_tolerance": 0.05,
            "mode_threshold": 0.1,
            "stability_threshold": 0.9,
        }

    @pytest.fixture
    def level_d_models(self, domain, parameters):
        """Create Level D models instance."""
        return LevelDModels(domain, parameters)

    def test_initialization(self, domain, parameters):
        """Test Level D models initialization."""
        models = LevelDModels(domain, parameters)
        
        assert models.domain == domain
        assert models.parameters == parameters
        assert isinstance(models, AbstractLevelModels)

    def test_parameter_validation(self, domain):
        """Test parameter validation."""
        # Test with valid parameters
        valid_params = {
            "jaccard_threshold": 0.8,
            "frequency_tolerance": 0.05,
            "mode_threshold": 0.1,
            "stability_threshold": 0.9,
        }
        
        models = LevelDModels(domain, valid_params)
        assert models.parameters == valid_params
        
        # Test with invalid parameters
        invalid_params = {
            "jaccard_threshold": -0.1,  # Invalid negative value
            "frequency_tolerance": 0.05,
            "mode_threshold": 0.1,
            "stability_threshold": 0.9,
        }
        
        with pytest.raises(ValueError):
            LevelDModels(domain, invalid_params)

    def test_domain_compatibility(self, parameters):
        """Test domain compatibility."""
        # Test with compatible domain
        compatible_domain = Domain(L=10.0, N=16, dimensions=7, N_phi=8, N_t=16, T=1.0)
        models = LevelDModels(compatible_domain, parameters)
        assert models.domain == compatible_domain
        
        # Test with incompatible domain
        incompatible_domain = Domain(L=10.0, N=16, dimensions=3)  # Wrong dimensions
        with pytest.raises(ValueError):
            LevelDModels(incompatible_domain, parameters)

    def test_model_creation(self, level_d_models):
        """Test model creation."""
        # Test creating a basic model
        model = level_d_models.create_model("basic")
        assert model is not None
        assert hasattr(model, "domain")
        assert hasattr(model, "parameters")

    def test_model_validation(self, level_d_models):
        """Test model validation."""
        # Test valid model
        model = level_d_models.create_model("basic")
        is_valid = level_d_models.validate_model(model)
        assert is_valid
        
        # Test invalid model
        invalid_model = None
        is_valid = level_d_models.validate_model(invalid_model)
        assert not is_valid

    def test_model_serialization(self, level_d_models):
        """Test model serialization."""
        model = level_d_models.create_model("basic")
        
        # Test serialization
        serialized = level_d_models.serialize_model(model)
        assert isinstance(serialized, dict)
        assert "model_type" in serialized
        assert "parameters" in serialized
        
        # Test deserialization
        deserialized = level_d_models.deserialize_model(serialized)
        assert deserialized is not None
        assert hasattr(deserialized, "domain")
        assert hasattr(deserialized, "parameters")

    def test_model_comparison(self, level_d_models):
        """Test model comparison."""
        model1 = level_d_models.create_model("basic")
        model2 = level_d_models.create_model("basic")
        
        # Test equality
        are_equal = level_d_models.compare_models(model1, model2)
        assert are_equal
        
        # Test with different models
        model3 = level_d_models.create_model("advanced")
        are_equal = level_d_models.compare_models(model1, model3)
        assert not are_equal

    def test_model_optimization(self, level_d_models):
        """Test model optimization."""
        model = level_d_models.create_model("basic")
        
        # Test optimization
        optimized_model = level_d_models.optimize_model(model)
        assert optimized_model is not None
        assert hasattr(optimized_model, "domain")
        assert hasattr(optimized_model, "parameters")

    def test_model_analysis(self, level_d_models):
        """Test model analysis."""
        model = level_d_models.create_model("basic")
        
        # Test analysis
        analysis = level_d_models.analyze_model(model)
        assert isinstance(analysis, dict)
        assert "stability" in analysis
        assert "performance" in analysis
        assert "quality" in analysis

    def test_model_export(self, level_d_models):
        """Test model export."""
        model = level_d_models.create_model("basic")
        
        # Test export to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            level_d_models.export_model(model, filename)
            assert os.path.exists(filename)
            
            # Verify file content
            with open(filename, 'r') as f:
                data = json.load(f)
            assert "model_type" in data
            assert "parameters" in data
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_model_import(self, level_d_models):
        """Test model import."""
        # Create and export a model
        model = level_d_models.create_model("basic")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            level_d_models.export_model(model, filename)
            
            # Test import
            imported_model = level_d_models.import_model(filename)
            assert imported_model is not None
            assert hasattr(imported_model, "domain")
            assert hasattr(imported_model, "parameters")
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestMultiModeModel:
    """Test MultiModeModel functionality."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=10.0, N=16, dimensions=7, N_phi=8, N_t=16, T=1.0)

    @pytest.fixture
    def parameters(self):
        """Create test parameters."""
        return {
            "mode_threshold": 0.1,
            "frequency_tolerance": 0.05,
            "stability_threshold": 0.9,
        }

    @pytest.fixture
    def multi_mode_model(self, domain, parameters):
        """Create MultiModeModel instance."""
        return MultiModeModel(domain, parameters)

    def test_initialization(self, domain, parameters):
        """Test MultiModeModel initialization."""
        model = MultiModeModel(domain, parameters)
        
        assert model.domain == domain
        assert model.parameters == parameters

    def test_mode_creation(self, multi_mode_model):
        """Test mode creation."""
        # Test creating a mode
        mode = multi_mode_model.create_mode(frequency=0.5, amplitude=1.0)
        assert mode is not None
        assert hasattr(mode, "frequency")
        assert hasattr(mode, "amplitude")
        assert mode.frequency == 0.5
        assert mode.amplitude == 1.0

    def test_mode_superposition(self, multi_mode_model):
        """Test mode superposition."""
        # Create multiple modes
        modes = [
            multi_mode_model.create_mode(frequency=0.5, amplitude=1.0),
            multi_mode_model.create_mode(frequency=1.0, amplitude=0.5),
            multi_mode_model.create_mode(frequency=1.5, amplitude=0.3),
        ]
        
        # Test superposition
        superposition = multi_mode_model.compute_superposition(modes)
        assert superposition is not None
        assert hasattr(superposition, "field")
        assert hasattr(superposition, "frequencies")
        assert hasattr(superposition, "amplitudes")

    def test_frame_stability_analysis(self, multi_mode_model):
        """Test frame stability analysis."""
        # Create modes
        modes = [
            multi_mode_model.create_mode(frequency=0.5, amplitude=1.0),
            multi_mode_model.create_mode(frequency=1.0, amplitude=0.5),
        ]
        
        # Test stability analysis
        stability = multi_mode_model.analyze_frame_stability(modes)
        assert isinstance(stability, dict)
        assert "stability_ratio" in stability
        assert "frame_quality" in stability
        assert "stability_score" in stability

    def test_mode_filtering(self, multi_mode_model):
        """Test mode filtering."""
        # Create modes with different amplitudes
        modes = [
            multi_mode_model.create_mode(frequency=0.5, amplitude=1.0),
            multi_mode_model.create_mode(frequency=1.0, amplitude=0.05),  # Below threshold
            multi_mode_model.create_mode(frequency=1.5, amplitude=0.3),
        ]
        
        # Test filtering
        filtered_modes = multi_mode_model.filter_modes(modes, threshold=0.1)
        assert len(filtered_modes) == 2  # Should filter out the low amplitude mode

    def test_frequency_analysis(self, multi_mode_model):
        """Test frequency analysis."""
        # Create modes
        modes = [
            multi_mode_model.create_mode(frequency=0.5, amplitude=1.0),
            multi_mode_model.create_mode(frequency=1.0, amplitude=0.5),
            multi_mode_model.create_mode(frequency=1.5, amplitude=0.3),
        ]
        
        # Test frequency analysis
        freq_analysis = multi_mode_model.analyze_frequencies(modes)
        assert isinstance(freq_analysis, dict)
        assert "dominant_frequency" in freq_analysis
        assert "frequency_spectrum" in freq_analysis
        assert "frequency_distribution" in freq_analysis

    def test_amplitude_analysis(self, multi_mode_model):
        """Test amplitude analysis."""
        # Create modes
        modes = [
            multi_mode_model.create_mode(frequency=0.5, amplitude=1.0),
            multi_mode_model.create_mode(frequency=1.0, amplitude=0.5),
            multi_mode_model.create_mode(frequency=1.5, amplitude=0.3),
        ]
        
        # Test amplitude analysis
        amp_analysis = multi_mode_model.analyze_amplitudes(modes)
        assert isinstance(amp_analysis, dict)
        assert "total_amplitude" in amp_analysis
        assert "amplitude_spectrum" in amp_analysis
        assert "amplitude_distribution" in amp_analysis

    def test_mode_interaction(self, multi_mode_model):
        """Test mode interaction analysis."""
        # Create modes
        modes = [
            multi_mode_model.create_mode(frequency=0.5, amplitude=1.0),
            multi_mode_model.create_mode(frequency=1.0, amplitude=0.5),
        ]
        
        # Test interaction analysis
        interaction = multi_mode_model.analyze_mode_interactions(modes)
        assert isinstance(interaction, dict)
        assert "interaction_strength" in interaction
        assert "coupling_matrix" in interaction
        assert "resonance_conditions" in interaction

    def test_mode_optimization(self, multi_mode_model):
        """Test mode optimization."""
        # Create initial modes
        modes = [
            multi_mode_model.create_mode(frequency=0.5, amplitude=1.0),
            multi_mode_model.create_mode(frequency=1.0, amplitude=0.5),
        ]
        
        # Test optimization
        optimized_modes = multi_mode_model.optimize_modes(modes)
        assert optimized_modes is not None
        assert len(optimized_modes) == len(modes)
        assert all(hasattr(mode, "frequency") for mode in optimized_modes)
        assert all(hasattr(mode, "amplitude") for mode in optimized_modes)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
