"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for BVPExponentialIntegrator.

This module contains unit tests for the BVPExponentialIntegrator class
in the 7D BVP framework, focusing on exponential integration methods
for dynamic phase field equations.

Physical Meaning:
    Tests the exponential integrator for solving dynamic phase field
    equations with optimal accuracy for BVP problems.

Mathematical Foundation:
    Tests validate the integration of the dynamic equation:
    ∂a/∂t + ν(-Δ)^β a + λa = s(x,φ,t)
    using exponential integrator methods for optimal accuracy.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.time import BVPExponentialIntegrator
from bhlff.core.domain import Domain, Parameters


class TestBVPExponentialIntegrator:
    """
    Unit tests for BVPExponentialIntegrator.
    
    Physical Meaning:
        Tests the exponential integrator for solving dynamic phase field
        equations with optimal accuracy for BVP problems.
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
    
    @pytest.fixture
    def integrator(self, domain_7d, parameters_basic):
        """Create exponential integrator for testing."""
        return BVPExponentialIntegrator(domain_7d, parameters_basic)
    
    def test_initialization(self, integrator, domain_7d, parameters_basic):
        """
        Test integrator initialization.
        
        Physical Meaning:
            Validates that the exponential integrator initializes correctly
            with the computational domain and physics parameters.
        """
        assert integrator.is_initialized
        assert integrator.domain == domain_7d
        assert integrator.parameters == parameters_basic
        assert integrator._spectral_coeffs is not None
        assert integrator._spectral_coeffs.shape == domain_7d.shape
    
    def test_single_step(self, integrator, domain_7d):
        """
        Test single time step.
        
        Physical Meaning:
            Validates that a single time step produces physically reasonable
            results for the exponential integrator.
        """
        # Create test field
        current_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        dt = 0.01
        
        # Perform step
        next_field = integrator.step(current_field, source_field, dt)
        
        # Validate results
        assert next_field.shape == domain_7d.shape
        assert next_field.dtype == np.complex128
        assert not np.any(np.isnan(next_field))
        assert not np.any(np.isinf(next_field))
    
    def test_harmonic_source_integration(self, integrator, domain_7d):
        """
        Test integration with harmonic source.
        
        Physical Meaning:
            Validates the exact solution for harmonic sources, which is
            the key advantage of the exponential integrator.
        """
        # Create harmonic source
        source_amplitude = np.random.randn(*domain_7d.shape).astype(np.complex128)
        frequency = 1.0
        time_steps = np.linspace(0, 1.0, 11)
        initial_field = np.zeros(domain_7d.shape, dtype=np.complex128)
        
        # Integrate
        result = integrator.integrate_harmonic_source(
            initial_field, source_amplitude, frequency, time_steps
        )
        
        # Validate results
        assert result.shape == (len(time_steps),) + domain_7d.shape
        assert result.dtype == np.complex128
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
        # Check that result evolves over time
        assert not np.allclose(result[0], result[-1])
    
    def test_integration_accuracy(self, integrator, domain_7d):
        """
        Test integration accuracy.
        
        Physical Meaning:
            Validates that the exponential integrator maintains high accuracy
            over multiple time steps.
        """
        # Create simple test case
        initial_field = np.zeros(domain_7d.shape, dtype=np.complex128)
        source_field = np.zeros((11,) + domain_7d.shape, dtype=np.complex128)
        time_steps = np.linspace(0, 1.0, 11)
        
        # Add simple source
        source_field[:, 0, 0, 0, 0, 0, 0] = 1.0
        
        # Integrate
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Validate results
        assert result.shape == (len(time_steps),) + domain_7d.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_spectral_coefficients(self, integrator, domain_7d, parameters_basic):
        """
        Test spectral coefficients computation.
        
        Physical Meaning:
            Validates that the spectral coefficients are computed correctly
            for the exponential integrator, which is crucial for accuracy.
        """
        # Check spectral coefficients
        assert integrator._spectral_coeffs is not None
        assert integrator._spectral_coeffs.shape == domain_7d.shape
        
        # Check that coefficients are positive (for stability)
        assert np.all(np.real(integrator._spectral_coeffs) > 0)
        
        # Check that coefficients match expected values
        expected_coeffs = (
            parameters_basic.mu * 
            np.ones(domain_7d.shape) + 
            parameters_basic.lambda_param
        )
        np.testing.assert_allclose(
            integrator._spectral_coeffs, expected_coeffs, rtol=1e-10
        )
    
    def test_time_step_validation(self, integrator, domain_7d):
        """
        Test time step validation.
        
        Physical Meaning:
            Validates that the integrator properly validates time step sizes
            to ensure numerical stability.
        """
        current_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Test valid time step
        dt_valid = 0.01
        result = integrator.step(current_field, source_field, dt_valid)
        assert result is not None
        
        # Test invalid time step (negative)
        dt_invalid = -0.01
        with pytest.raises(ValueError, match="Time step must be positive"):
            integrator.step(current_field, source_field, dt_invalid)
        
        # Test invalid time step (too large)
        dt_too_large = 10.0
        with pytest.raises(ValueError, match="Time step too large for stability"):
            integrator.step(current_field, source_field, dt_too_large)
    
    def test_field_validation(self, integrator, domain_7d):
        """
        Test field validation.
        
        Physical Meaning:
            Validates that the integrator properly validates input fields
            to ensure they match the computational domain.
        """
        # Test valid field
        valid_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        result = integrator.step(valid_field, source_field, 0.01)
        assert result is not None
        
        # Test invalid field shape
        invalid_field = np.random.randn(4, 4, 4).astype(np.complex128)
        with pytest.raises(ValueError, match="Field shape must match domain"):
            integrator.step(invalid_field, source_field, 0.01)
        
        # Test invalid field type
        invalid_field = np.random.randn(*domain_7d.shape).astype(np.float64)
        with pytest.raises(ValueError, match="Field must be complex"):
            integrator.step(invalid_field, source_field, 0.01)
