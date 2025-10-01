"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive tests for BVP constants numerical module.

This module provides comprehensive tests for the BVP constants numerical module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_constants_numerical import BVPConstantsNumerical


class TestBVPConstantsNumericalCoverage:
    """Comprehensive tests for BVP constants numerical module."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(
            L=1.0,
            N=8,
            dimensions=3,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
            },
            "numerical": {
                "precision": "float64",
                "tolerance": 1e-12,
                "max_iterations": 1000,
                "convergence_criteria": 1e-6
            }
        }

    def test_bvp_constants_numerical_creation(self, domain_7d, config):
        """Test BVP constants numerical creation."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test basic properties
        assert hasattr(constants, 'domain')
        assert hasattr(constants, 'config')
        assert constants.domain == domain_7d
        assert constants.config == config

    def test_bvp_constants_numerical_precision(self, domain_7d, config):
        """Test BVP constants numerical precision methods."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test get_precision method
        precision = constants.get_precision()
        assert isinstance(precision, str)
        assert precision in ['float32', 'float64']
        
        # Test set_precision method
        constants.set_precision('float32')
        new_precision = constants.get_precision()
        assert new_precision == 'float32'

    def test_bvp_constants_numerical_tolerance(self, domain_7d, config):
        """Test BVP constants numerical tolerance methods."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test get_tolerance method
        tolerance = constants.get_tolerance()
        assert isinstance(tolerance, float)
        assert tolerance > 0
        
        # Test set_tolerance method
        constants.set_tolerance(1e-10)
        new_tolerance = constants.get_tolerance()
        assert new_tolerance == 1e-10

    def test_bvp_constants_numerical_iterations(self, domain_7d, config):
        """Test BVP constants numerical iteration methods."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test get_max_iterations method
        max_iter = constants.get_max_iterations()
        assert isinstance(max_iter, int)
        assert max_iter > 0
        
        # Test set_max_iterations method
        constants.set_max_iterations(500)
        new_max_iter = constants.get_max_iterations()
        assert new_max_iter == 500

    def test_bvp_constants_numerical_convergence(self, domain_7d, config):
        """Test BVP constants numerical convergence methods."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test get_convergence_criteria method
        criteria = constants.get_convergence_criteria()
        assert isinstance(criteria, float)
        assert criteria > 0
        
        # Test set_convergence_criteria method
        constants.set_convergence_criteria(1e-8)
        new_criteria = constants.get_convergence_criteria()
        assert new_criteria == 1e-8

    def test_bvp_constants_numerical_validation(self, domain_7d, config):
        """Test BVP constants numerical validation."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test validate_numerical_config method
        is_valid = constants.validate_numerical_config()
        assert isinstance(is_valid, bool)
        
        # Test validate_precision method
        precision_valid = constants.validate_precision()
        assert isinstance(precision_valid, bool)
        
        # Test validate_tolerance method
        tolerance_valid = constants.validate_tolerance()
        assert isinstance(tolerance_valid, bool)

    def test_bvp_constants_numerical_properties(self, domain_7d, config):
        """Test BVP constants numerical properties."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test get_numerical_properties method
        properties = constants.get_numerical_properties()
        assert isinstance(properties, dict)
        assert 'precision' in properties
        assert 'tolerance' in properties
        assert 'max_iterations' in properties
        assert 'convergence_criteria' in properties

    def test_bvp_constants_numerical_parameter_access(self, domain_7d, config):
        """Test BVP constants numerical parameter access."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test get_numerical_parameter method
        precision = constants.get_numerical_parameter('precision')
        assert isinstance(precision, str)
        
        # Test set_numerical_parameter method
        constants.set_numerical_parameter('test_param', 42.0)
        test_param = constants.get_numerical_parameter('test_param')
        assert test_param == 42.0

    def test_bvp_constants_numerical_serialization(self, domain_7d, config):
        """Test BVP constants numerical serialization."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test to_dict method
        constants_dict = constants.to_dict()
        assert isinstance(constants_dict, dict)
        assert 'numerical' in constants_dict
        
        # Test from_dict method
        new_constants = BVPConstantsNumerical.from_dict(domain_7d, constants_dict)
        assert isinstance(new_constants, BVPConstantsNumerical)
        assert new_constants.domain == domain_7d

    def test_bvp_constants_numerical_comparison(self, domain_7d, config):
        """Test BVP constants numerical comparison."""
        constants1 = BVPConstantsNumerical(domain_7d, config)
        constants2 = BVPConstantsNumerical(domain_7d, config)
        
        # Test equality
        assert constants1 == constants2
        
        # Test inequality with different numerical config
        different_config = config.copy()
        different_config['numerical']['precision'] = 'float32'
        constants3 = BVPConstantsNumerical(domain_7d, different_config)
        assert constants1 != constants3

    def test_bvp_constants_numerical_string_representation(self, domain_7d, config):
        """Test BVP constants numerical string representation."""
        constants = BVPConstantsNumerical(domain_7d, config)
        
        # Test string representation
        str_repr = str(constants)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        
        # Test repr
        repr_str = repr(constants)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
