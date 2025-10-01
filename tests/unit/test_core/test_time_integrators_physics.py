"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for 7D BVP time integrators.

This module contains physical validation tests for the time integrators
in the 7D BVP framework, testing against analytical solutions and
physical principles.

Physical Meaning:
    These tests validate the physical correctness of temporal integration
    methods by comparing against analytical solutions and checking
    physical conservation laws.

Mathematical Foundation:
    Tests validate against analytical solutions for:
    1. Harmonic sources: s(x,t) = s₀(x)e^(-iωt)
    2. Decay solutions: a(x,t) = a₀(x)e^(-λt)
    3. Energy conservation and dissipation
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.time import (
    BVPExponentialIntegrator, 
    CrankNicolsonIntegrator,
    MemoryKernel,
    QuenchDetector
)
from bhlff.core.domain import Domain, Parameters


class TestPhysicalValidation:
    """
    Physical validation tests for time integrators.
    
    Physical Meaning:
        Tests the physical correctness of temporal integration methods
        by comparing against analytical solutions and checking physical
        conservation laws in 7D space-time.
    """
    
    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)
    
    @pytest.fixture
    def parameters_decay(self):
        """Parameters for decay test."""
        return Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.5,  # Strong damping
            nu=1.0,
            precision='float64',
            fft_plan='MEASURE',
            tolerance=1e-12
        )
    
    @pytest.fixture
    def parameters_harmonic(self):
        """Parameters for harmonic test."""
        return Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,  # Weak damping
            nu=1.0,
            precision='float64',
            fft_plan='MEASURE',
            tolerance=1e-12
        )
    
    def test_exponential_decay_analytical(self, domain_7d, parameters_decay):
        """
        Test exponential decay against analytical solution.
        
        Physical Meaning:
            Validates that the exponential integrator correctly reproduces
            the analytical decay solution: a(x,t) = a₀(x)e^(-λt)
            for the case with no source and strong damping.
            
        Mathematical Foundation:
            For λ > 0 and s(x,t) = 0, the analytical solution is:
            a(x,t) = a₀(x)e^(-λt)
        """
        # Create exponential integrator
        integrator = BVPExponentialIntegrator(domain_7d, parameters_decay)
        
        # Create initial field
        initial_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Create zero source
        time_steps = np.linspace(0, 2.0, 21)
        source_field = np.zeros((len(time_steps),) + domain_7d.shape, dtype=np.complex128)
        
        # Integrate
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Compare with analytical solution
        lambda_param = parameters_decay.lambda_param
        for i, t in enumerate(time_steps):
            analytical = initial_field * np.exp(-lambda_param * t)
            numerical = result[i]
            
            # Check relative error
            relative_error = np.linalg.norm(numerical - analytical) / np.linalg.norm(analytical)
            assert relative_error < 1e-10, f"Decay test failed at t={t:.3f}, error={relative_error:.2e}"
    
    def test_harmonic_source_analytical(self, domain_7d, parameters_harmonic):
        """
        Test harmonic source against analytical solution.
        
        Physical Meaning:
            Validates that the exponential integrator correctly reproduces
            the analytical solution for harmonic sources:
            s(x,t) = s₀(x)e^(-iωt)
            
        Mathematical Foundation:
            For harmonic source, the analytical solution is:
            â(k,t) = â₀(k)e^(-(ν|k|^(2β)+λ)t) + ŝ₀(k)/(ν|k|^(2β)+λ+iω)(1-e^(-(ν|k|^(2β)+λ+iω)t))
        """
        # Create exponential integrator
        integrator = BVPExponentialIntegrator(domain_7d, parameters_harmonic)
        
        # Create harmonic source
        source_amplitude = np.random.randn(*domain_7d.shape).astype(np.complex128)
        frequency = 1.0
        time_steps = np.linspace(0, 1.0, 11)
        initial_field = np.zeros(domain_7d.shape, dtype=np.complex128)
        
        # Integrate using exact method
        result_exact = integrator.integrate_harmonic_source(
            initial_field, source_amplitude, frequency, time_steps
        )
        
        # Create source field for general integration
        source_field = np.zeros((len(time_steps),) + domain_7d.shape, dtype=np.complex128)
        for i, t in enumerate(time_steps):
            source_field[i] = source_amplitude * np.exp(-1j * frequency * t)
        
        # Integrate using general method
        result_general = integrator.integrate(initial_field, source_field, time_steps)
        
        # Compare results
        for i in range(len(time_steps)):
            relative_error = np.linalg.norm(result_exact[i] - result_general[i]) / np.linalg.norm(result_exact[i])
            assert relative_error < 1e-10, f"Harmonic test failed at step {i}, error={relative_error:.2e}"
    
    def test_energy_conservation_no_damping(self, domain_7d):
        """
        Test energy conservation for undamped system.
        
        Physical Meaning:
            Validates that energy is conserved in the absence of damping
            and sources, as expected from the physical principles.
            
        Mathematical Foundation:
            For λ = 0 and s(x,t) = 0, the energy should be conserved:
            E(t) = ∫ |a(x,t)|² dx = constant
        """
        # Create parameters with no damping
        parameters = Parameters(
            mu=1.0,
            beta=1.0,
            lambda_param=0.0,  # No damping
            nu=1.0,
            precision='float64',
            fft_plan='MEASURE',
            tolerance=1e-12
        )
        
        # Create integrator
        integrator = BVPExponentialIntegrator(domain_7d, parameters)
        
        # Create initial field
        initial_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Create zero source
        time_steps = np.linspace(0, 1.0, 11)
        source_field = np.zeros((len(time_steps),) + domain_7d.shape, dtype=np.complex128)
        
        # Integrate
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Calculate energy at each time step
        energies = []
        for i in range(len(time_steps)):
            energy = np.sum(np.abs(result[i]) ** 2)
            energies.append(energy)
        
        # Check energy conservation
        initial_energy = energies[0]
        for i, energy in enumerate(energies):
            relative_error = abs(energy - initial_energy) / initial_energy
            assert relative_error < 1e-10, f"Energy conservation failed at step {i}, error={relative_error:.2e}"
    
    def test_energy_dissipation_with_damping(self, domain_7d, parameters_decay):
        """
        Test energy dissipation with damping.
        
        Physical Meaning:
            Validates that energy decreases over time in the presence
            of damping, as expected from the physical principles.
            
        Mathematical Foundation:
            For λ > 0, the energy should decrease over time:
            E(t) = E₀e^(-2λt)
        """
        # Create integrator
        integrator = BVPExponentialIntegrator(domain_7d, parameters_decay)
        
        # Create initial field
        initial_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Create zero source
        time_steps = np.linspace(0, 2.0, 21)
        source_field = np.zeros((len(time_steps),) + domain_7d.shape, dtype=np.complex128)
        
        # Integrate
        result = integrator.integrate(initial_field, source_field, time_steps)
        
        # Calculate energy at each time step
        energies = []
        for i in range(len(time_steps)):
            energy = np.sum(np.abs(result[i]) ** 2)
            energies.append(energy)
        
        # Check energy dissipation
        initial_energy = energies[0]
        lambda_param = parameters_decay.lambda_param
        
        for i, (t, energy) in enumerate(zip(time_steps, energies)):
            # Analytical energy: E(t) = E₀e^(-2λt)
            analytical_energy = initial_energy * np.exp(-2 * lambda_param * t)
            relative_error = abs(energy - analytical_energy) / analytical_energy
            assert relative_error < 1e-8, f"Energy dissipation failed at t={t:.3f}, error={relative_error:.2e}"
    
    def test_crank_nicolson_stability(self, domain_7d, parameters_harmonic):
        """
        Test Crank-Nicolson stability for large time steps.
        
        Physical Meaning:
            Validates that the Crank-Nicolson integrator remains stable
            for large time steps, demonstrating its unconditional stability.
            
        Mathematical Foundation:
            Crank-Nicolson scheme is unconditionally stable, meaning
            it should remain stable for any time step size.
        """
        # Create Crank-Nicolson integrator
        integrator = CrankNicolsonIntegrator(domain_7d, parameters_harmonic)
        
        # Create test fields
        current_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        current_source = np.random.randn(*domain_7d.shape).astype(np.complex128)
        next_source = np.random.randn(*domain_7d.shape).astype(np.complex128)
        
        # Test with large time step
        dt = 1.0  # Large time step
        
        # Perform step
        next_field = integrator.step(current_field, current_source, next_source, dt)
        
        # Check stability (no NaN or Inf values)
        assert not np.any(np.isnan(next_field)), "Crank-Nicolson became unstable with large time step"
        assert not np.any(np.isinf(next_field)), "Crank-Nicolson became unstable with large time step"
        
        # Check that result is reasonable
        assert np.linalg.norm(next_field) < 1e6, "Crank-Nicolson result is unreasonably large"
    
    def test_memory_kernel_physics(self, domain_7d):
        """
        Test memory kernel physical behavior.
        
        Physical Meaning:
            Validates that the memory kernel correctly implements
            non-local temporal effects with proper relaxation behavior.
            
        Mathematical Foundation:
            Memory variables should evolve according to:
            ∂mⱼ/∂t + (1/τⱼ) mⱼ = field
            leading to exponential relaxation with time constant τⱼ.
        """
        # Create memory kernel
        memory_kernel = MemoryKernel(domain_7d, num_memory_vars=2)
        
        # Set specific relaxation times
        taus = [0.1, 1.0]
        memory_kernel.set_relaxation_times(taus)
        
        # Create constant field
        field = np.ones(domain_7d.shape, dtype=np.complex128)
        
        # Evolve memory kernel
        dt = 0.01
        time_steps = np.arange(0, 2.0, dt)
        memory_evolution = []
        
        for t in time_steps:
            memory_evolution.append([var.copy() for var in memory_kernel.memory_variables])
            memory_kernel.evolve(field, dt)
        
        # Check relaxation behavior
        for i, tau in enumerate(taus):
            # Get memory variable evolution
            memory_values = [mem[i] for mem in memory_evolution]
            
            # Check that memory variable approaches steady state
            # Steady state: m = τ * field
            steady_state = tau * field
            final_memory = memory_values[-1]
            
            # Check convergence to steady state
            relative_error = np.linalg.norm(final_memory - steady_state) / np.linalg.norm(steady_state)
            assert relative_error < 0.1, f"Memory variable {i} did not converge to steady state, error={relative_error:.2e}"
    
    def test_quench_detection_physics(self, domain_7d):
        """
        Test quench detection physical behavior.
        
        Physical Meaning:
            Validates that the quench detector correctly identifies
            physical energy dumping events based on energy thresholds.
            
        Mathematical Foundation:
            Quench detection should trigger when:
            - Energy change exceeds threshold
            - Rate of energy change exceeds threshold
            - Field magnitude exceeds threshold
        """
        # Create quench detector with low thresholds
        quench_detector = QuenchDetector(
            domain_7d, 
            energy_threshold=1e-6,
            rate_threshold=1e-5,
            magnitude_threshold=5.0
        )
        
        # Test gradual energy increase (should not trigger quench)
        field_small = np.ones(domain_7d.shape, dtype=np.complex128) * 0.1
        quench_detected = quench_detector.detect_quench(field_small, time=0.0)
        assert not quench_detected, "Quench detected for small field"
        
        # Test sudden energy increase (should trigger quench)
        field_large = np.ones(domain_7d.shape, dtype=np.complex128) * 10.0
        quench_detected = quench_detector.detect_quench(field_large, time=0.01)
        assert quench_detected, "Quench not detected for large field"
        
        # Check quench event details
        assert len(quench_detector.quench_history) == 1
        quench_event = quench_detector.quench_history[0]
        assert quench_event['time'] == 0.01
        assert 'magnitude' in quench_event['reasons'][0]
    
    def test_integrator_consistency(self, domain_7d, parameters_harmonic):
        """
        Test consistency between different integrators.
        
        Physical Meaning:
            Validates that different integrators produce consistent
            results for the same problem, ensuring physical correctness.
            
        Mathematical Foundation:
            Both integrators should converge to the same solution
            as the time step size decreases.
        """
        # Create integrators
        exp_integrator = BVPExponentialIntegrator(domain_7d, parameters_harmonic)
        cn_integrator = CrankNicolsonIntegrator(domain_7d, parameters_harmonic)
        
        # Create test case
        initial_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        dt = 0.001  # Small time step
        
        # Perform single step with both integrators
        exp_result = exp_integrator.step(initial_field, source_field, dt)
        cn_result = cn_integrator.step(initial_field, source_field, source_field, dt)
        
        # Compare results
        relative_error = np.linalg.norm(exp_result - cn_result) / np.linalg.norm(exp_result)
        assert relative_error < 1e-6, f"Integrator consistency failed, error={relative_error:.2e}"
    
    def test_7d_dimensionality(self, domain_7d, parameters_harmonic):
        """
        Test 7D dimensionality handling.
        
        Physical Meaning:
            Validates that the integrators correctly handle the 7D
            nature of the phase field, including all spatial, phase,
            and temporal dimensions.
            
        Mathematical Foundation:
            The 7D wave vector should be computed correctly:
            |k|² = |k_x|² + |k_φ|² + k_t²
        """
        # Create integrator
        integrator = BVPExponentialIntegrator(domain_7d, parameters_harmonic)
        
        # Check that spectral coefficients have correct shape
        assert integrator._spectral_coeffs.shape == domain_7d.shape
        
        # Check that all dimensions are handled
        assert len(integrator._spectral_coeffs.shape) == 7
        
        # Test integration with 7D field
        initial_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        source_field = np.random.randn(*domain_7d.shape).astype(np.complex128)
        dt = 0.01
        
        # Perform step
        result = integrator.step(initial_field, source_field, dt)
        
        # Check that result maintains 7D structure
        assert result.shape == domain_7d.shape
        assert len(result.shape) == 7
        
        # Check that all dimensions are non-trivial
        for i in range(7):
            assert result.shape[i] > 1, f"Dimension {i} is trivial"
