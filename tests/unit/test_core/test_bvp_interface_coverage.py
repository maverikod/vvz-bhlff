"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive tests for BVP interface module.

This module provides comprehensive tests for the BVP interface module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_interface import BVPInterface
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


class TestBVPInterfaceCoverage:
    """Comprehensive tests for BVP interface module."""

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
            }
        }

    @pytest.fixture
    def bvp_constants(self, config):
        """Create BVP constants."""
        return BVPConstantsAdvanced(config)

    def test_bvp_interface_creation(self, domain_7d, bvp_constants):
        """Test BVP interface creation."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Test basic properties
        assert hasattr(bvp_interface, 'domain')
        assert hasattr(bvp_interface, 'constants')
        assert bvp_interface.domain == domain_7d
        assert bvp_interface.constants == bvp_constants

    def test_bvp_interface_process_source(self, domain_7d, bvp_constants):
        """Test BVP interface process source method."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Create test source
        source = np.zeros(domain_7d.shape)
        source[domain_7d.N//2, domain_7d.N//2, domain_7d.N//2, :, :, :, :] = 1.0
        
        # Test process_source method
        result = bvp_interface.process_source(source)
        
        # Validate result
        assert isinstance(result, dict)
        assert 'envelope' in result
        assert 'postulates' in result
        assert 'quenches' in result
        assert 'impedance' in result

    def test_bvp_interface_solve_envelope(self, domain_7d, bvp_constants):
        """Test BVP interface solve envelope method."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Create test source
        source = np.zeros(domain_7d.shape)
        source[domain_7d.N//2, domain_7d.N//2, domain_7d.N//2, :, :, :, :] = 1.0
        
        # Test solve_envelope method
        envelope = bvp_interface.solve_envelope(source)
        
        # Validate result
        assert isinstance(envelope, np.ndarray)
        assert envelope.shape == domain_7d.shape
        assert np.all(np.isfinite(envelope))

    def test_bvp_interface_validate_postulates(self, domain_7d, bvp_constants):
        """Test BVP interface validate postulates method."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test validate_postulates method
        postulates = bvp_interface.validate_postulates(envelope)
        
        # Validate result
        assert isinstance(postulates, dict)

    def test_bvp_interface_detect_quenches(self, domain_7d, bvp_constants):
        """Test BVP interface detect quenches method."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test detect_quenches method
        quenches = bvp_interface.detect_quenches(envelope)
        
        # Validate result
        assert isinstance(quenches, np.ndarray)
        assert quenches.shape == domain_7d.shape
        assert np.all((quenches == 0) | (quenches == 1))

    def test_bvp_interface_compute_impedance(self, domain_7d, bvp_constants):
        """Test BVP interface compute impedance method."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test compute_impedance method
        impedance = bvp_interface.compute_impedance(envelope)
        
        # Validate result
        assert isinstance(impedance, np.ndarray)
        assert impedance.shape == domain_7d.shape
        assert np.all(np.isfinite(impedance))

    def test_bvp_interface_get_interface_info(self, domain_7d, bvp_constants):
        """Test BVP interface get interface info method."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Test get_interface_info method
        info = bvp_interface.get_interface_info()
        
        # Validate result
        assert isinstance(info, dict)
        assert 'domain_info' in info
        assert 'constants_info' in info

    def test_bvp_interface_validate_interface(self, domain_7d, bvp_constants):
        """Test BVP interface validate interface method."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Test validate_interface method
        is_valid = bvp_interface.validate_interface()
        
        # Validate result
        assert isinstance(is_valid, bool)

    def test_bvp_interface_serialization(self, domain_7d, bvp_constants):
        """Test BVP interface serialization."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Test to_dict method
        interface_dict = bvp_interface.to_dict()
        assert isinstance(interface_dict, dict)
        
        # Test from_dict method
        new_interface = BVPInterface.from_dict(domain_7d, bvp_constants, interface_dict)
        assert isinstance(new_interface, BVPInterface)
        assert new_interface.domain == domain_7d

    def test_bvp_interface_comparison(self, domain_7d, bvp_constants):
        """Test BVP interface comparison."""
        bvp_interface1 = BVPInterface(domain_7d, bvp_constants)
        bvp_interface2 = BVPInterface(domain_7d, bvp_constants)
        
        # Test equality
        assert bvp_interface1 == bvp_interface2
        
        # Test inequality with different domain
        different_domain = Domain(L=2.0, N=16, dimensions=3, N_phi=8, N_t=16, T=2.0)
        bvp_interface3 = BVPInterface(different_domain, bvp_constants)
        assert bvp_interface1 != bvp_interface3

    def test_bvp_interface_string_representation(self, domain_7d, bvp_constants):
        """Test BVP interface string representation."""
        bvp_interface = BVPInterface(domain_7d, bvp_constants)
        
        # Test string representation
        str_repr = str(bvp_interface)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        
        # Test repr
        repr_str = repr(bvp_interface)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
