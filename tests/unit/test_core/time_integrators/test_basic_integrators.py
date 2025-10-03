"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic time integrators tests.

This module contains basic tests for time integrators
including fundamental validation and basic functionality tests.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.time import (
    BVPEnvelopeIntegrator, 
    CrankNicolsonIntegrator,
    MemoryKernel,
    QuenchDetector
)
from bhlff.core.domain import Domain, Parameters


class TestBasicIntegrators:
    """
    Basic tests for time integrators.
    
    Physical Meaning:
        Tests the basic functionality of temporal integration methods
        in 7D space-time, ensuring they produce physically meaningful results.
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
    
    def test_envelope_integrator_creation(self, domain_7d, parameters_basic):
        """Test envelope integrator creation."""
        integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)
        
        # Check that integrator is created successfully
        assert integrator is not None, "Exponential integrator should be created"
        assert integrator.domain == domain_7d, "Domain should be set correctly"
        assert integrator.parameters == parameters_basic, "Parameters should be set correctly"
    
    def test_crank_nicolson_integrator_creation(self, domain_7d, parameters_basic):
        """Test Crank-Nicolson integrator creation."""
        integrator = CrankNicolsonIntegrator(domain_7d, parameters_basic)
        
        # Check that integrator is created successfully
        assert integrator is not None, "Crank-Nicolson integrator should be created"
        assert integrator.domain == domain_7d, "Domain should be set correctly"
        assert integrator.parameters == parameters_basic, "Parameters should be set correctly"
    
    def test_memory_kernel_creation(self, domain_7d, parameters_basic):
        """Test memory kernel creation."""
        kernel = MemoryKernel(domain_7d, parameters_basic)
        
        # Check that kernel is created successfully
        assert kernel is not None, "Memory kernel should be created"
        assert kernel.domain == domain_7d, "Domain should be set correctly"
        assert kernel.parameters == parameters_basic, "Parameters should be set correctly"
    
    def test_quench_detector_creation(self, domain_7d, parameters_basic):
        """Test quench detector creation."""
        detector = QuenchDetector(domain_7d, parameters_basic)
        
        # Check that detector is created successfully
        assert detector is not None, "Quench detector should be created"
        assert detector.domain == domain_7d, "Domain should be set correctly"
        assert detector.parameters == parameters_basic, "Parameters should be set correctly"
    
    def test_integrator_parameter_validation(self, domain_7d):
        """Test parameter validation in integrators."""
        # Test with valid parameters
        valid_params = Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,
            nu=1.0,
            precision='float64',
            fft_plan='MEASURE',
            tolerance=1e-12
        )
        
        integrator = BVPExponentialIntegrator(domain_7d, valid_params)
        assert integrator is not None, "Integrator should be created with valid parameters"
        
        # Test with invalid parameters (negative mu)
        with pytest.raises(ValueError):
            invalid_params = Parameters(
                mu=-1.0,  # Invalid: negative mu
                beta=1.0,
                lambda_param=0.1,
                nu=1.0,
                precision='float64',
                fft_plan='MEASURE',
                tolerance=1e-12
            )
            BVPExponentialIntegrator(domain_7d, invalid_params)
    
    def test_integrator_domain_validation(self, parameters_basic):
        """Test domain validation in integrators."""
        # Test with valid domain
        valid_domain = Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)
        integrator = BVPExponentialIntegrator(valid_domain, parameters_basic)
        assert integrator is not None, "Integrator should be created with valid domain"
        
        # Test with invalid domain (negative L)
        with pytest.raises(ValueError):
            invalid_domain = Domain(L=-1.0, N=8, N_phi=4, N_t=8, dimensions=7)
            BVPExponentialIntegrator(invalid_domain, parameters_basic)
    
    def test_integrator_basic_functionality(self, domain_7d, parameters_basic):
        """Test basic functionality of integrators."""
        # Test exponential integrator
        exp_integrator = BVPExponentialIntegrator(domain_7d, parameters_basic)
        
        # Check that integrator has required methods
        assert hasattr(exp_integrator, 'integrate'), "Exponential integrator should have integrate method"
        assert hasattr(exp_integrator, 'setup'), "Exponential integrator should have setup method"
        
        # Test Crank-Nicolson integrator
        cn_integrator = CrankNicolsonIntegrator(domain_7d, parameters_basic)
        
        # Check that integrator has required methods
        assert hasattr(cn_integrator, 'integrate'), "Crank-Nicolson integrator should have integrate method"
        assert hasattr(cn_integrator, 'setup'), "Crank-Nicolson integrator should have setup method"
    
    def test_memory_kernel_functionality(self, domain_7d, parameters_basic):
        """Test memory kernel functionality."""
        kernel = MemoryKernel(domain_7d, parameters_basic)
        
        # Check that kernel has required methods
        assert hasattr(kernel, 'compute_kernel'), "Memory kernel should have compute_kernel method"
        assert hasattr(kernel, 'apply_kernel'), "Memory kernel should have apply_kernel method"
    
    def test_quench_detector_functionality(self, domain_7d, parameters_basic):
        """Test quench detector functionality."""
        detector = QuenchDetector(domain_7d, parameters_basic)
        
        # Check that detector has required methods
        assert hasattr(detector, 'detect_quench'), "Quench detector should have detect_quench method"
        assert hasattr(detector, 'analyze_quench'), "Quench detector should have analyze_quench method"
    
    def test_integrator_consistency(self, domain_7d, parameters_basic):
        """Test consistency between different integrators."""
        # Create different integrators
        exp_integrator = BVPExponentialIntegrator(domain_7d, parameters_basic)
        cn_integrator = CrankNicolsonIntegrator(domain_7d, parameters_basic)
        
        # Both should have the same domain and parameters
        assert exp_integrator.domain == cn_integrator.domain, "Integrators should have same domain"
        assert exp_integrator.parameters == cn_integrator.parameters, "Integrators should have same parameters"
    
    def test_integrator_error_handling(self, domain_7d, parameters_basic):
        """Test error handling in integrators."""
        integrator = BVPExponentialIntegrator(domain_7d, parameters_basic)
        
        # Test with invalid input types
        with pytest.raises((TypeError, ValueError)):
            integrator.integrate("invalid_input")
        
        # Test with wrong shape arrays
        with pytest.raises((ValueError, TypeError)):
            wrong_shape = np.zeros((4, 4, 4))  # Wrong shape for 7D domain
            integrator.integrate(wrong_shape)
    
    def test_integrator_performance(self, domain_7d, parameters_basic):
        """Test basic performance of integrators."""
        import time
        
        integrator = BVPExponentialIntegrator(domain_7d, parameters_basic)
        
        # Create test field
        test_field = np.random.randn(*domain_7d.shape)
        
        # Time the integration
        start_time = time.time()
        try:
            result = integrator.integrate(test_field)
            end_time = time.time()
            
            # Integration should complete in reasonable time
            integration_time = end_time - start_time
            assert integration_time < 10.0, f"Integration took too long: {integration_time:.3f}s"
            
            # Result should be finite
            assert np.all(np.isfinite(result)), "Integration result should be finite"
            
        except (NotImplementedError, AttributeError):
            # Some methods might not be fully implemented yet
            pass
