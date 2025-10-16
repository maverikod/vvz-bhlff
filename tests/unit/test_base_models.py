"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for base model classes.

This module contains comprehensive tests for base model classes
in the BHLFF framework.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.models.base.abstract_model import AbstractModel
from bhlff.models.base.model_base import ModelBase
from bhlff.models.base.abstract_models import AbstractLevelModels


class ConcreteModel(AbstractModel):
    """Concrete implementation of AbstractModel for testing."""
    
    def analyze(self, data):
        """Concrete implementation of analyze method."""
        return {"result": "test", "data_shape": data.shape if hasattr(data, 'shape') else None}


class ConcreteLevelModel(AbstractLevelModels):
    """Concrete implementation of AbstractLevelModels for testing."""
    
    def analyze_field(self, field):
        """Concrete implementation of analyze_field method."""
        return {"field_shape": field.shape, "field_mean": np.mean(field)}


class TestAbstractModel:
    """Test suite for AbstractModel base class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_domain = Mock()
        self.mock_domain.shape = (64, 64, 64)
        self.mock_domain.L = 1.0
        self.mock_domain.N = 64
        self.mock_domain.dimensions = 3
        
        self.model = ConcreteModel(self.mock_domain)
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.domain == self.mock_domain
        assert hasattr(self.model, 'logger')
        assert self.model.logger.name == 'ConcreteModel'
    
    def test_analyze_method(self):
        """Test analyze method implementation."""
        test_data = np.random.random((32, 32))
        result = self.model.analyze(test_data)
        
        assert isinstance(result, dict)
        assert "result" in result
        assert "data_shape" in result
        assert result["result"] == "test"
        assert result["data_shape"] == (32, 32)
    
    def test_validate_domain(self):
        """Test domain validation."""
        assert self.model.validate_domain() is True
        
        # Test with None domain
        model_no_domain = ConcreteModel(None)
        assert model_no_domain.validate_domain() is False
    
    def test_get_domain_info(self):
        """Test domain information retrieval."""
        domain_info = self.model.get_domain_info()
        
        assert isinstance(domain_info, dict)
        assert "shape" in domain_info
        assert "L" in domain_info
        assert "N" in domain_info
        assert "dimensions" in domain_info
        
        assert domain_info["shape"] == (64, 64, 64)
        assert domain_info["L"] == 1.0
        assert domain_info["N"] == 64
        assert domain_info["dimensions"] == 3
    
    def test_logging_methods(self):
        """Test logging methods."""
        with patch.object(self.model.logger, 'info') as mock_info:
            self.model.log_analysis_start("test_analysis")
            mock_info.assert_called_with("Starting test_analysis analysis")
        
        with patch.object(self.model.logger, 'info') as mock_info, \
             patch.object(self.model.logger, 'debug') as mock_debug:
            results = {"key1": "value1", "key2": "value2"}
            self.model.log_analysis_complete("test_analysis", results)
            mock_info.assert_called_with("test_analysis analysis completed")
            mock_debug.assert_called_with("Results: ['key1', 'key2']")
    
    def test_string_representations(self):
        """Test string representations."""
        str_repr = str(self.model)
        repr_repr = repr(self.model)
        
        assert "ConcreteModel" in str_repr
        assert "domain=" in str_repr
        assert str_repr == repr_repr
    
    def test_abstract_model_cannot_be_instantiated(self):
        """Test that AbstractModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractModel(self.mock_domain)


class TestModelBase:
    """Test suite for ModelBase class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_domain = Mock()
        self.mock_domain.shape = (32, 32, 32)
        self.mock_domain.L = 2.0
        self.mock_domain.N = 32
        self.mock_domain.dimensions = 3
        
        self.model = ModelBase(self.mock_domain)
    
    def test_initialization_with_domain(self):
        """Test model initialization with domain."""
        assert self.model.domain == self.mock_domain
        assert hasattr(self.model, 'logger')
        assert self.model.logger.name == 'ModelBase'
    
    def test_initialization_without_domain(self):
        """Test model initialization without domain."""
        model_no_domain = ModelBase()
        assert model_no_domain.domain is None
        assert hasattr(model_no_domain, 'logger')
    
    def test_validate_domain(self):
        """Test domain validation."""
        assert self.model.validate_domain() is True
        
        model_no_domain = ModelBase()
        assert model_no_domain.validate_domain() is False
    
    def test_get_domain_info(self):
        """Test domain information retrieval."""
        domain_info = self.model.get_domain_info()
        
        assert isinstance(domain_info, dict)
        assert "shape" in domain_info
        assert "L" in domain_info
        assert "N" in domain_info
        assert "dimensions" in domain_info
        
        assert domain_info["shape"] == (32, 32, 32)
        assert domain_info["L"] == 2.0
        assert domain_info["N"] == 32
        assert domain_info["dimensions"] == 3
    
    def test_get_domain_info_no_domain(self):
        """Test domain information retrieval without domain."""
        model_no_domain = ModelBase()
        domain_info = model_no_domain.get_domain_info()
        
        assert isinstance(domain_info, dict)
        assert len(domain_info) == 0
    
    def test_validate_array(self):
        """Test array validation."""
        # Valid array
        valid_array = np.random.random((10, 10))
        assert self.model.validate_array(valid_array) is True
        
        # Invalid array - not numpy array
        invalid_array = [[1, 2, 3], [4, 5, 6]]
        assert self.model.validate_array(invalid_array) is False
        
        # Invalid array - contains non-finite values
        invalid_array = np.array([1.0, np.inf, 3.0])
        assert self.model.validate_array(invalid_array) is False
        
        # Invalid array - contains NaN
        invalid_array = np.array([1.0, np.nan, 3.0])
        assert self.model.validate_array(invalid_array) is False
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        # Valid parameters
        valid_params = {"param1": 1.0, "param2": 2.5, "param3": 10}
        assert self.model.validate_parameters(valid_params) is True
        
        # Invalid parameters - contains inf
        invalid_params = {"param1": 1.0, "param2": np.inf}
        assert self.model.validate_parameters(invalid_params) is False
        
        # Invalid parameters - contains NaN
        invalid_params = {"param1": 1.0, "param2": np.nan}
        assert self.model.validate_parameters(invalid_params) is False
    
    def test_compute_statistics(self):
        """Test statistics computation."""
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = self.model.compute_statistics(test_data)
        
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "rms" in stats
        assert "variance" in stats
        
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["rms"] == pytest.approx(np.sqrt(11.0), rel=1e-10)
    
    def test_compute_statistics_invalid_array(self):
        """Test statistics computation with invalid array."""
        invalid_data = [[1, 2, 3], [4, 5, 6]]  # Not numpy array
        stats = self.model.compute_statistics(invalid_data)
        
        assert isinstance(stats, dict)
        assert len(stats) == 0
    
    def test_logging_methods(self):
        """Test logging methods."""
        with patch.object(self.model.logger, 'info') as mock_info:
            self.model.log_analysis_start("test_analysis")
            mock_info.assert_called_with("Starting test_analysis analysis")
        
        with patch.object(self.model.logger, 'info') as mock_info, \
             patch.object(self.model.logger, 'debug') as mock_debug:
            results = {"key1": "value1", "key2": "value2"}
            self.model.log_analysis_complete("test_analysis", results)
            mock_info.assert_called_with("test_analysis analysis completed")
            mock_debug.assert_called_with("Results: ['key1', 'key2']")
    
    def test_string_representations(self):
        """Test string representations."""
        str_repr = str(self.model)
        repr_repr = repr(self.model)
        
        assert "ModelBase" in str_repr
        assert "domain=" in str_repr
        assert str_repr == repr_repr


class TestAbstractLevelModels:
    """Test suite for AbstractLevelModels base class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_domain = Mock()
        self.mock_domain.shape = (16, 16, 16)
        self.mock_domain.L = 4.0
        self.mock_domain.N = 16
        self.mock_domain.dimensions = 3
        
        self.parameters = {
            "param1": 1.0,
            "param2": 2.5,
            "param3": "test_value"
        }
        
        self.model = ConcreteLevelModel(self.mock_domain, self.parameters)
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.domain == self.mock_domain
        assert self.model.parameters == self.parameters
        assert hasattr(self.model, 'logger')
        assert self.model.logger.name == 'ConcreteLevelModel'
    
    def test_analyze_field_method(self):
        """Test analyze_field method implementation."""
        test_field = np.random.random((8, 8, 8))
        result = self.model.analyze_field(test_field)
        
        assert isinstance(result, dict)
        assert "field_shape" in result
        assert "field_mean" in result
        assert result["field_shape"] == (8, 8, 8)
        assert isinstance(result["field_mean"], float)
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        assert self.model.validate_parameters() is True
    
    def test_get_parameter(self):
        """Test parameter retrieval."""
        # Existing parameter
        param1 = self.model.get_parameter("param1")
        assert param1 == 1.0
        
        # Non-existing parameter with default
        param4 = self.model.get_parameter("param4", "default_value")
        assert param4 == "default_value"
        
        # Non-existing parameter without default
        param5 = self.model.get_parameter("param5")
        assert param5 is None
    
    def test_set_parameter(self):
        """Test parameter setting."""
        with patch.object(self.model.logger, 'debug') as mock_debug:
            self.model.set_parameter("new_param", 42.0)
            
            assert self.model.parameters["new_param"] == 42.0
            mock_debug.assert_called_with("Parameter new_param set to 42.0")
    
    def test_logging_methods(self):
        """Test logging methods."""
        with patch.object(self.model.logger, 'info') as mock_info:
            self.model.log_analysis_start("test_analysis")
            mock_info.assert_called_with("Starting test_analysis analysis")
        
        with patch.object(self.model.logger, 'info') as mock_info, \
             patch.object(self.model.logger, 'debug') as mock_debug:
            results = {"key1": "value1", "key2": "value2"}
            self.model.log_analysis_complete("test_analysis", results)
            mock_info.assert_called_with("test_analysis analysis completed")
            mock_debug.assert_called_with("Results: ['key1', 'key2']")
    
    def test_string_representations(self):
        """Test string representations."""
        str_repr = str(self.model)
        repr_repr = repr(self.model)
        
        assert "ConcreteLevelModel" in str_repr
        assert "domain=" in str_repr
        assert "parameters=" in str_repr
        assert "3 items" in str_repr  # 3 parameters
        
        assert str_repr != repr_repr  # Different representations
        assert "parameters=" in repr_repr
    
    def test_abstract_level_models_cannot_be_instantiated(self):
        """Test that AbstractLevelModels cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractLevelModels(self.mock_domain, self.parameters)


class TestModelIntegration:
    """Test suite for model integration scenarios."""
    
    def test_model_inheritance_hierarchy(self):
        """Test model inheritance hierarchy."""
        mock_domain = Mock()
        mock_domain.shape = (8, 8, 8)
        
        # Test AbstractModel inheritance
        concrete_model = ConcreteModel(mock_domain)
        assert isinstance(concrete_model, AbstractModel)
        
        # Test AbstractLevelModels inheritance
        parameters = {"test": 1.0}
        concrete_level_model = ConcreteLevelModel(mock_domain, parameters)
        assert isinstance(concrete_level_model, AbstractLevelModels)
        
        # Test ModelBase inheritance
        model_base = ModelBase(mock_domain)
        assert isinstance(model_base, ModelBase)
    
    def test_model_polymorphism(self):
        """Test model polymorphism."""
        mock_domain = Mock()
        mock_domain.shape = (4, 4, 4)
        
        models = [
            ConcreteModel(mock_domain),
            ModelBase(mock_domain),
            ConcreteLevelModel(mock_domain, {"test": 1.0})
        ]
        
        for model in models:
            # All models should have domain
            assert hasattr(model, 'domain')
            assert model.domain == mock_domain
            
            # All models should have logger
            assert hasattr(model, 'logger')
            
            # All models should have string representations
            str_repr = str(model)
            assert isinstance(str_repr, str)
            assert len(str_repr) > 0
    
    def test_model_error_handling(self):
        """Test model error handling."""
        mock_domain = Mock()
        mock_domain.shape = (2, 2, 2)
        
        model = ModelBase(mock_domain)
        
        # Test with invalid data types
        with patch.object(model.logger, 'error') as mock_error:
            result = model.compute_statistics("invalid_data")
            assert result == {}
            mock_error.assert_called()
        
        # Test with invalid parameters
        with patch.object(model.logger, 'error') as mock_error:
            result = model.validate_parameters({"param": np.inf})
            assert result is False
            mock_error.assert_called()
    
    def test_model_performance(self):
        """Test model performance with large data."""
        mock_domain = Mock()
        mock_domain.shape = (100, 100, 100)
        
        model = ModelBase(mock_domain)
        
        # Test with large array
        large_array = np.random.random((100, 100, 100))
        
        import time
        start_time = time.time()
        stats = model.compute_statistics(large_array)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
