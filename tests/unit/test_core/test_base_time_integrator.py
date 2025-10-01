"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for BaseTimeIntegrator.

This module contains unit tests for the BaseTimeIntegrator class
in the 7D BVP framework, focusing on parameter validation and
abstract base class functionality.

Physical Meaning:
    Tests the abstract base class functionality and parameter validation
    for temporal integrators in the 7D BVP framework.

Mathematical Foundation:
    Tests validate the integration of the dynamic equation:
    ∂a/∂t + ν(-Δ)^β a + λa = s(x,φ,t)
    with various source configurations and parameter combinations.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.time import (
    BaseTimeIntegrator, 
    BVPExponentialIntegrator
)
from bhlff.core.domain import Domain, Parameters


class TestBaseTimeIntegrator:
    """
    Unit tests for BaseTimeIntegrator.
    
    Physical Meaning:
        Tests the abstract base class functionality and parameter validation
        for temporal integrators in the 7D BVP framework.
    """
    
    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)
    
    @pytest.fixture
    def parameters_basic(self):
        """Basic parameters for testing."""
        return Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,
            nu=1.0,
            precision='float64',
            fft_plan='MEASURE',
            tolerance=1e-12
        )
    
    def test_parameter_validation(self, domain_7d, parameters_basic):
        """
        Test parameter validation.
        
        Physical Meaning:
            Validates that the integrator correctly validates physical
            parameters and raises appropriate errors for invalid values.
        """
        # Test valid parameters
        integrator = BVPExponentialIntegrator(domain_7d, parameters_basic)
        assert integrator.is_initialized
        
        # Test invalid nu (negative)
        invalid_params = Parameters(
            mu=1.0, beta=1.0, lambda_param=0.1, nu=-1.0,
            precision='float64', fft_plan='MEASURE', tolerance=1e-12
        )
        with pytest.raises(ValueError, match="Diffusion coefficient ν must be positive"):
            BVPExponentialIntegrator(domain_7d, invalid_params)
        
        # Test invalid beta (out of range)
        invalid_params = Parameters(
            mu=1.0, beta=2.5, lambda_param=0.1, nu=1.0,
            precision='float64', fft_plan='MEASURE', tolerance=1e-12
        )
        with pytest.raises(ValueError, match="Fractional order β must be in \\(0,2\\)"):
            BVPExponentialIntegrator(domain_7d, invalid_params)
        
        # Test invalid lambda (negative)
        invalid_params = Parameters(
            mu=1.0, beta=1.0, lambda_param=-0.1, nu=1.0,
            precision='float64', fft_plan='MEASURE', tolerance=1e-12
        )
        with pytest.raises(ValueError, match="Damping parameter λ must be non-negative"):
            BVPExponentialIntegrator(domain_7d, invalid_params)
    
    def test_abstract_methods(self, domain_7d, parameters_basic):
        """
        Test that abstract methods raise NotImplementedError.
        
        Physical Meaning:
            Validates that the abstract base class properly enforces
            implementation of required methods in subclasses.
        """
        # Create instance of abstract class (should raise error)
        with pytest.raises(TypeError):
            BaseTimeIntegrator(domain_7d, parameters_basic)
    
    def test_domain_validation(self, parameters_basic):
        """
        Test domain validation.
        
        Physical Meaning:
            Validates that the integrator correctly validates the
            computational domain and raises appropriate errors.
        """
        # Test invalid domain (negative L)
        invalid_domain = Domain(L=-1.0, N=8, N_phi=4, N_t=8, dimensions=7)
        with pytest.raises(ValueError, match="Domain size L must be positive"):
            BVPExponentialIntegrator(invalid_domain, parameters_basic)
        
        # Test invalid domain (N too small)
        invalid_domain = Domain(L=1.0, N=1, N_phi=4, N_t=8, dimensions=7)
        with pytest.raises(ValueError, match="Grid size N must be at least 2"):
            BVPExponentialIntegrator(invalid_domain, parameters_basic)
    
    def test_initialization_state(self, domain_7d, parameters_basic):
        """
        Test initialization state tracking.
        
        Physical Meaning:
            Validates that the integrator correctly tracks its
            initialization state and prevents operations before
            proper initialization.
        """
        integrator = BVPExponentialIntegrator(domain_7d, parameters_basic)
        assert integrator.is_initialized
        
        # Test that operations work after initialization
        source = np.random.random(domain_7d.shape) + 1j * np.random.random(domain_7d.shape)
        result = integrator.step(source, 0.01)
        assert result is not None
        assert result.shape == domain_7d.shape
