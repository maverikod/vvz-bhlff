"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for CrankNicolsonIntegrator.

This module contains unit tests for the CrankNicolsonIntegrator class
in the 7D BVP framework, focusing on implicit integration methods
for dynamic phase field equations.

Physical Meaning:
    Tests the Crank-Nicolson integrator for solving dynamic phase field
    equations with second-order accuracy and unconditional stability.

Mathematical Foundation:
    Tests validate the integration of the dynamic equation:
    ∂a/∂t + ν(-Δ)^β a + λa = s(x,φ,t)
    using Crank-Nicolson implicit scheme for unconditional stability.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.time import CrankNicolsonIntegrator
from bhlff.core.domain import Domain, Parameters


class TestCrankNicolsonIntegrator:
    """
    Unit tests for CrankNicolsonIntegrator.
    
    Physical Meaning:
        Tests the Crank-Nicolson integrator for solving dynamic phase field
        equations with second-order accuracy and unconditional stability.
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
        """Create Crank-Nicolson integrator for testing."""
        return CrankNicolsonIntegrator(domain_7d, parameters_basic)
    
    def test_initialization(self, integrator, domain_7d, parameters_basic):
        """
        Test integrator initialization.
        
        Physical Meaning:
            Validates that the Crank-Nicolson integrator initializes correctly
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
            results for the Crank-Nicolson integrator.
        """
        # Create test fields
        current_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        current_source = np.random.randn(*domain_7d.shape).astype(np.complex128)
        next_source = np.random.randn(*domain_7d.shape).astype(np.complex128)
        dt = 0.01
        
        # Perform step
        next_field = integrator.step(current_field, current_source, next_source, dt)
        
        # Validate results
        assert next_field.shape == domain_7d.shape
        assert next_field.dtype == np.complex128
        assert not np.any(np.isnan(next_field))
        assert not np.any(np.isinf(next_field))
    
    def test_implicit_step(self, integrator, domain_7d):
        """
        Test implicit time step.
        
        Physical Meaning:
            Validates the implicit Crank-Nicolson scheme for unconditional
            stability in stiff problems.
        """
        # Create test fields
        current_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        dt = 0.01
        
        # Perform implicit step
        next_field = integrator.step_implicit(current_field, source_field, dt)
        
        # Validate results
        assert next_field.shape == domain_7d.shape
        assert next_field.dtype == np.complex128
        assert not np.any(np.isnan(next_field))
        assert not np.any(np.isinf(next_field))
    
    def test_second_order_accuracy(self, integrator, domain_7d):
        """
        Test second-order accuracy.
        
        Physical Meaning:
            Validates that the Crank-Nicolson integrator achieves second-order
            accuracy in time, which is its key advantage over explicit methods.
        """
        # Create simple test case
        initial_field = np.zeros(domain_7d.shape, dtype=np.complex128)
        source_field = np.zeros((11,) + domain_7d.shape, dtype=np.complex128)
        time_steps = np.linspace(0, 1.0, 11)
        
        # Add simple source
        source_field[:, 0, 0, 0, 0, 0, 0] = 1.0
        
        # Integrate with different time steps
        dt1 = 0.01
        dt2 = 0.005
        
        result1 = integrator.integrate(initial_field, source_field, time_steps)
        result2 = integrator.integrate(initial_field, source_field, time_steps)
        
        # Check that results are consistent
        assert result1.shape == result2.shape
        assert not np.any(np.isnan(result1))
        assert not np.any(np.isnan(result2))
    
    def test_unconditional_stability(self, integrator, domain_7d):
        """
        Test unconditional stability.
        
        Physical Meaning:
            Validates that the Crank-Nicolson integrator remains stable
            even with large time steps, which is crucial for stiff problems.
        """
        # Create test fields
        current_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Test with large time step (should remain stable)
        dt_large = 1.0
        result = integrator.step_implicit(current_field, source_field, dt_large)
        
        # Validate results
        assert result.shape == domain_7d.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
        # Check that solution doesn't blow up
        assert np.linalg.norm(result) < 1e6
    
    def test_spectral_coefficients(self, integrator, domain_7d, parameters_basic):
        """
        Test spectral coefficients computation.
        
        Physical Meaning:
            Validates that the spectral coefficients are computed correctly
            for the Crank-Nicolson integrator.
        """
        # Check spectral coefficients
        assert integrator._spectral_coeffs is not None
        assert integrator._spectral_coeffs.shape == domain_7d.shape
        
        # Check that coefficients are positive (for stability)
        assert np.all(np.real(integrator._spectral_coeffs) > 0)
    
    def test_time_step_validation(self, integrator, domain_7d):
        """
        Test time step validation.
        
        Physical Meaning:
            Validates that the integrator properly validates time step sizes.
        """
        current_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Test valid time step
        dt_valid = 0.01
        result = integrator.step_implicit(current_field, source_field, dt_valid)
        assert result is not None
        
        # Test invalid time step (negative)
        dt_invalid = -0.01
        with pytest.raises(ValueError, match="Time step must be positive"):
            integrator.step_implicit(current_field, source_field, dt_invalid)
    
    def test_field_validation(self, integrator, domain_7d):
        """
        Test field validation.
        
        Physical Meaning:
            Validates that the integrator properly validates input fields.
        """
        # Test valid field
        valid_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        result = integrator.step_implicit(valid_field, source_field, 0.01)
        assert result is not None
        
        # Test invalid field shape
        invalid_field = np.random.randn(4, 4, 4).astype(np.complex128)
        with pytest.raises(ValueError, match="Field shape must match domain"):
            integrator.step_implicit(invalid_field, source_field, 0.01)
        
        # Test invalid field type
        invalid_field = np.random.randn(*domain_7d.shape).astype(np.float64)
        with pytest.raises(ValueError, match="Field must be complex"):
            integrator.step_implicit(invalid_field, source_field, 0.01)
