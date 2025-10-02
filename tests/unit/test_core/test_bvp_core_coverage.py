"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive tests for BVP core module.

This module provides comprehensive tests for the BVP core module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_core import BVPCore
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


class TestBVPCoreCoverage:
    """Comprehensive tests for BVP core module."""

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

    def test_bvp_core_creation(self, domain_7d, bvp_constants):
        """Test BVP core creation."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Test basic properties
        assert hasattr(bvp_core, 'domain')
        assert hasattr(bvp_core, 'constants')
        assert bvp_core.domain == domain_7d
        assert bvp_core.constants == bvp_constants

    def test_bvp_core_solve_envelope(self, domain_7d, bvp_constants):
        """Test BVP core solve envelope method."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Create test source
        source = np.zeros(domain_7d.shape)
        source[domain_7d.N//2, domain_7d.N//2, domain_7d.N//2, :, :, :, :] = 1.0
        
        # Test solve_envelope method
        envelope = bvp_core.solve_envelope(source)
        
        # Validate result
        assert isinstance(envelope, np.ndarray)
        assert envelope.shape == domain_7d.shape
        assert np.all(np.isfinite(envelope))

    def test_bvp_core_compute_residual(self, domain_7d, bvp_constants):
        """Test BVP core compute residual method."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test compute_residual method
        residual = bvp_core.compute_residual(envelope)
        
        # Validate result
        assert isinstance(residual, np.ndarray)
        assert residual.shape == domain_7d.shape
        assert np.all(np.isfinite(residual))

    def test_bvp_core_compute_jacobian(self, domain_7d, bvp_constants):
        """Test BVP core compute jacobian method."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test compute_jacobian method
        jacobian = bvp_core.compute_jacobian(envelope)
        
        # Validate result
        assert isinstance(jacobian, np.ndarray)
        assert jacobian.shape == domain_7d.shape
        assert np.all(np.isfinite(jacobian))

    def test_bvp_core_compute_energy(self, domain_7d, bvp_constants):
        """Test BVP core compute energy method."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test compute_energy method
        energy = bvp_core.compute_energy(envelope)
        
        # Validate result
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert energy >= 0

    def test_bvp_core_compute_gradient(self, domain_7d, bvp_constants):
        """Test BVP core compute gradient method."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test compute_gradient method
        gradient = bvp_core.compute_gradient(envelope)
        
        # Validate result
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == domain_7d.shape
        assert np.all(np.isfinite(gradient))

    def test_bvp_core_compute_laplacian(self, domain_7d, bvp_constants):
        """Test BVP core compute laplacian method."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test compute_laplacian method
        laplacian = bvp_core.compute_laplacian(envelope)
        
        # Validate result
        assert isinstance(laplacian, np.ndarray)
        assert laplacian.shape == domain_7d.shape
        assert np.all(np.isfinite(laplacian))

    def test_bvp_core_validate_solution(self, domain_7d, bvp_constants):
        """Test BVP core validate solution method."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test validate_solution method
        is_valid = bvp_core.validate_solution(envelope)
        
        # Validate result
        assert isinstance(is_valid, bool)

    def test_bvp_core_get_solution_info(self, domain_7d, bvp_constants):
        """Test BVP core get solution info method."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Create test envelope
        envelope = np.ones(domain_7d.shape)
        
        # Test get_solution_info method
        info = bvp_core.get_solution_info(envelope)
        
        # Validate result
        assert isinstance(info, dict)
        assert 'energy' in info
        assert 'residual_norm' in info
        assert 'convergence' in info

    def test_bvp_core_serialization(self, domain_7d, bvp_constants):
        """Test BVP core serialization."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Test to_dict method
        core_dict = bvp_core.to_dict()
        assert isinstance(core_dict, dict)
        
        # Test from_dict method
        new_core = BVPCore.from_dict(domain_7d, bvp_constants, core_dict)
        assert isinstance(new_core, BVPCore)
        assert new_core.domain == domain_7d

    def test_bvp_core_comparison(self, domain_7d, bvp_constants):
        """Test BVP core comparison."""
        bvp_core1 = BVPCore(domain_7d, bvp_constants)
        bvp_core2 = BVPCore(domain_7d, bvp_constants)
        
        # Test equality
        assert bvp_core1 == bvp_core2
        
        # Test inequality with different domain
        different_domain = Domain(L=2.0, N=16, dimensions=3, N_phi=8, N_t=16, T=2.0)
        bvp_core3 = BVPCore(different_domain, bvp_constants)
        assert bvp_core1 != bvp_core3

    def test_bvp_core_string_representation(self, domain_7d, bvp_constants):
        """Test BVP core string representation."""
        bvp_core = BVPCore(domain_7d, bvp_constants)
        
        # Test string representation
        str_repr = str(bvp_core)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        
        # Test repr
        repr_str = repr(bvp_core)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
