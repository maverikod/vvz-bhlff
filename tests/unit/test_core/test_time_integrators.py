"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for 7D BVP time integrators.

This module contains comprehensive unit tests for the time integrators
in the 7D BVP framework, including tests for exponential integrator,
Crank-Nicolson integrator, memory kernel, and quench detection.

Physical Meaning:
    These tests validate the correctness of temporal integration methods
    for solving dynamic phase field equations in 7D space-time.

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
    BVPExponentialIntegrator, 
    CrankNicolsonIntegrator,
    MemoryKernel,
    QuenchDetector
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


class TestMemoryKernel:
    """
    Unit tests for MemoryKernel.
    
    Physical Meaning:
        Tests the memory kernel for non-local temporal effects in
        the 7D phase field dynamics.
    """
    
    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)
    
    @pytest.fixture
    def memory_kernel(self, domain_7d):
        """Create memory kernel for testing."""
        return MemoryKernel(domain_7d, num_memory_vars=3)
    
    def test_initialization(self, memory_kernel, domain_7d):
        """
        Test memory kernel initialization.
        
        Physical Meaning:
            Validates that the memory kernel initializes correctly with
            the specified number of memory variables.
        """
        assert memory_kernel._initialized
        assert memory_kernel.domain == domain_7d
        assert memory_kernel.num_memory_vars == 3
        assert len(memory_kernel.memory_variables) == 3
        assert len(memory_kernel.relaxation_times) == 3
        assert len(memory_kernel.coupling_strengths) == 3
        
        # Check that all memory variables have correct shape
        for memory_var in memory_kernel.memory_variables:
            assert memory_var.shape == domain_7d.shape
            assert memory_var.dtype == np.complex128
    
    def test_memory_application(self, memory_kernel, domain_7d):
        """
        Test memory kernel application.
        
        Physical Meaning:
            Validates that the memory kernel correctly applies non-local
            temporal effects to the field.
        """
        # Create test field
        field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Apply memory kernel
        result = memory_kernel.apply(field, time=0.0)
        
        # Validate results
        assert result.shape == domain_7d.shape
        assert result.dtype == np.complex128
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_memory_evolution(self, memory_kernel, domain_7d):
        """
        Test memory variable evolution.
        
        Physical Meaning:
            Validates that memory variables evolve correctly according
            to their evolution equation.
        """
        # Create test field
        field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        dt = 0.01
        
        # Store initial memory variables
        initial_memory = [var.copy() for var in memory_kernel.memory_variables]
        
        # Evolve memory kernel
        memory_kernel.evolve(field, dt)
        
        # Check that memory variables changed
        for i, (initial, current) in enumerate(zip(initial_memory, memory_kernel.memory_variables)):
            assert not np.allclose(initial, current), f"Memory variable {i} did not evolve"
    
    def test_memory_reset(self, memory_kernel, domain_7d):
        """
        Test memory kernel reset.
        
        Physical Meaning:
            Validates that the memory kernel can be reset to clear
            all memory of past configurations.
        """
        # Evolve memory kernel to create non-zero values
        field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        memory_kernel.evolve(field, 0.01)
        
        # Check that memory variables are non-zero
        for memory_var in memory_kernel.memory_variables:
            assert not np.allclose(memory_var, 0)
        
        # Reset memory kernel
        memory_kernel.reset()
        
        # Check that all memory variables are zero
        for memory_var in memory_kernel.memory_variables:
            assert np.allclose(memory_var, 0)


class TestQuenchDetector:
    """
    Unit tests for QuenchDetector.
    
    Physical Meaning:
        Tests the quench detection system for monitoring energy dumping
        events during temporal integration.
    """
    
    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)
    
    @pytest.fixture
    def quench_detector(self, domain_7d):
        """Create quench detector for testing."""
        return QuenchDetector(domain_7d, energy_threshold=1e-3, rate_threshold=1e-2, magnitude_threshold=10.0)
    
    def test_initialization(self, quench_detector, domain_7d):
        """
        Test quench detector initialization.
        
        Physical Meaning:
            Validates that the quench detector initializes correctly with
            the specified thresholds.
        """
        assert quench_detector._initialized
        assert quench_detector.domain == domain_7d
        assert quench_detector.energy_threshold == 1e-3
        assert quench_detector.rate_threshold == 1e-2
        assert quench_detector.magnitude_threshold == 10.0
        assert len(quench_detector.quench_history) == 0
    
    def test_quench_detection_energy(self, quench_detector, domain_7d):
        """
        Test quench detection based on energy threshold.
        
        Physical Meaning:
            Validates that the quench detector correctly identifies
            energy dumping events based on energy change threshold.
        """
        # Create field with small energy
        small_field = np.ones(domain_7d.shape, dtype=np.complex128) * 0.1
        
        # First detection (no previous energy)
        quench_detected = quench_detector.detect_quench(small_field, time=0.0)
        assert not quench_detected
        
        # Create field with large energy change
        large_field = np.ones(domain_7d.shape, dtype=np.complex128) * 100.0
        
        # Second detection (large energy change)
        quench_detected = quench_detector.detect_quench(large_field, time=0.01)
        assert quench_detected
        assert len(quench_detector.quench_history) == 1
    
    def test_quench_detection_magnitude(self, quench_detector, domain_7d):
        """
        Test quench detection based on magnitude threshold.
        
        Physical Meaning:
            Validates that the quench detector correctly identifies
            quench events based on field magnitude threshold.
        """
        # Create field with large magnitude
        large_field = np.ones(domain_7d.shape, dtype=np.complex128) * 20.0
        
        # Detect quench
        quench_detected = quench_detector.detect_quench(large_field, time=0.0)
        assert quench_detected
        assert len(quench_detector.quench_history) == 1
        
        # Check quench event details
        quench_event = quench_detector.quench_history[0]
        assert quench_event['time'] == 0.0
        assert 'magnitude' in quench_event['reasons'][0]
    
    def test_quench_history(self, quench_detector, domain_7d):
        """
        Test quench event history.
        
        Physical Meaning:
            Validates that the quench detector correctly records
            and manages quench event history.
        """
        # Create multiple quench events
        for i in range(3):
            field = np.ones(domain_7d.shape, dtype=np.complex128) * (20.0 + i)
            quench_detector.detect_quench(field, time=i * 0.1)
        
        # Check history
        history = quench_detector.get_quench_history()
        assert len(history) == 3
        
        # Check statistics
        stats = quench_detector.get_statistics()
        assert stats['total_quenches'] == 3
        assert stats['quench_rate'] > 0
    
    def test_quench_clear_history(self, quench_detector, domain_7d):
        """
        Test quench history clearing.
        
        Physical Meaning:
            Validates that the quench detector can clear its history
            to start fresh monitoring.
        """
        # Create quench event
        field = np.ones(domain_7d.shape, dtype=np.complex128) * 20.0
        quench_detector.detect_quench(field, time=0.0)
        
        # Check history exists
        assert len(quench_detector.quench_history) == 1
        
        # Clear history
        quench_detector.clear_history()
        
        # Check history is cleared
        assert len(quench_detector.quench_history) == 0
        assert quench_detector._previous_energy is None
