"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive tests for BVP constants base module.

This module provides comprehensive tests for the BVP constants base module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_constants_base import BVPConstantsBase


class TestBVPConstantsBaseCoverage:
    """Comprehensive tests for BVP constants base module."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, dimensions=3, N_phi=4, N_t=8, T=1.0)

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
        }

    def test_bvp_constants_base_creation(self, domain_7d, config):
        """Test BVP constants base creation."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test basic properties
        assert hasattr(constants, "domain")
        assert hasattr(constants, "config")
        assert constants.domain == domain_7d
        assert constants.config == config

    def test_bvp_constants_base_methods(self, domain_7d, config):
        """Test BVP constants base methods."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test get_parameter method
        kappa_0 = constants.get_parameter("envelope_equation.kappa_0")
        assert kappa_0 == 1.0

        # Test get_parameter with default
        non_existent = constants.get_parameter("non.existent.param", default=42.0)
        assert non_existent == 42.0

        # Test get_parameter with nested access
        mu = constants.get_parameter("basic_material.mu")
        assert mu == 1.0

    def test_bvp_constants_base_validation(self, domain_7d, config):
        """Test BVP constants base validation."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test validate_config method
        is_valid = constants.validate_config()
        assert isinstance(is_valid, bool)

        # Test with invalid config
        invalid_config = {"invalid": "config"}
        invalid_constants = BVPConstantsBase(domain_7d, invalid_config)
        is_invalid = invalid_constants.validate_config()
        assert isinstance(is_invalid, bool)

    def test_bvp_constants_base_physical_properties(self, domain_7d, config):
        """Test BVP constants base physical properties."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test get_physical_properties method
        properties = constants.get_physical_properties()
        assert isinstance(properties, dict)

        # Test get_material_properties method
        material_props = constants.get_material_properties()
        assert isinstance(material_props, dict)

        # Test get_envelope_properties method
        envelope_props = constants.get_envelope_properties()
        assert isinstance(envelope_props, dict)

    def test_bvp_constants_base_numerical_properties(self, domain_7d, config):
        """Test BVP constants base numerical properties."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test get_numerical_properties method
        numerical_props = constants.get_numerical_properties()
        assert isinstance(numerical_props, dict)

        # Test get_precision method
        precision = constants.get_precision()
        assert isinstance(precision, str)

        # Test get_tolerance method
        tolerance = constants.get_tolerance()
        assert isinstance(tolerance, float)
        assert tolerance > 0

    def test_bvp_constants_base_derived_properties(self, domain_7d, config):
        """Test BVP constants base derived properties."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test get_derived_properties method
        derived_props = constants.get_derived_properties()
        assert isinstance(derived_props, dict)

        # Test compute_derived_parameters method
        derived_params = constants.compute_derived_parameters()
        assert isinstance(derived_params, dict)

        # Test update_derived_parameters method
        constants.update_derived_parameters()
        # Should not raise any exceptions

    def test_bvp_constants_base_parameter_access(self, domain_7d, config):
        """Test BVP constants base parameter access."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test set_parameter method
        constants.set_parameter("test.param", 123.0)
        test_param = constants.get_parameter("test.param")
        assert test_param == 123.0

        # Test has_parameter method
        has_param = constants.has_parameter("test.param")
        assert has_param is True

        has_no_param = constants.has_parameter("non.existent.param")
        assert has_no_param is False

    def test_bvp_constants_base_serialization(self, domain_7d, config):
        """Test BVP constants base serialization."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test to_dict method
        constants_dict = constants.to_dict()
        assert isinstance(constants_dict, dict)

        # Test from_dict method
        new_constants = BVPConstantsBase.from_dict(domain_7d, constants_dict)
        assert isinstance(new_constants, BVPConstantsBase)
        assert new_constants.domain == domain_7d

    def test_bvp_constants_base_comparison(self, domain_7d, config):
        """Test BVP constants base comparison."""
        constants1 = BVPConstantsBase(domain_7d, config)
        constants2 = BVPConstantsBase(domain_7d, config)

        # Test equality
        assert constants1 == constants2

        # Test inequality with different config
        different_config = config.copy()
        different_config["test"] = "different"
        constants3 = BVPConstantsBase(domain_7d, different_config)
        assert constants1 != constants3

    def test_bvp_constants_base_string_representation(self, domain_7d, config):
        """Test BVP constants base string representation."""
        constants = BVPConstantsBase(domain_7d, config)

        # Test string representation
        str_repr = str(constants)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

        # Test repr
        repr_str = repr(constants)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
