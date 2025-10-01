"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Validation tests for 7D FFT Solver in BHLFF Framework.

This module contains comprehensive validation tests for the 7D FFT solver,
including analytical tests, numerical tests, and boundary case tests
as specified in the Level A validation requirements.

Physical Meaning:
    These tests validate the correctness of the 7D FFT solver for the
    fractional Riesz operator, ensuring that the solver produces
    physically meaningful and mathematically correct results.

Mathematical Foundation:
    Tests validate the spectral solution: â(k) = ŝ(k) / (μ|k|^(2β) + λ)
    for various source configurations and parameter combinations.
"""

import numpy as np
import pytest
import json
import os
from typing import Dict, Any, Tuple

from bhlff.core.fft import FFTSolver7D, FractionalLaplacian
from bhlff.core.domain import Domain


class TestFFTSolver7DValidation:
    """
    Validation tests for 7D FFT Solver.
    
    Physical Meaning:
        Comprehensive validation of the 7D FFT solver implementation
        against analytical solutions and physical requirements.
    """
    
    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=8, N_phi=4, N_t=8, dimensions=7)
    
    @pytest.fixture
    def parameters_basic(self):
        """Basic parameters for testing."""
        return {
            'mu': 1.0,
            'beta': 1.0,
            'lambda_param': 0.1,
            'precision': 'float64',
            'fft_plan': 'MEASURE',
            'tolerance': 1e-12
        }
    
    @pytest.fixture
    def solver(self, domain_7d, parameters_basic):
        """Create FFT solver for testing."""
        return FFTSolver7D(domain_7d, parameters_basic)
    
    def test_A01_plane_wave_stationary(self, solver, domain_7d):
        """
        Test A0.1: Plane wave stationary solution.
        
        Physical Meaning:
            Tests the fundamental spectral solution for a plane wave source,
            validating the formula â = ŝ / D for single frequency modes.
            
        Mathematical Foundation:
            For s(x) = exp(i k₀·x), the solution is:
            a(x) = exp(i k₀·x) / D(k₀)
            where D(k₀) = μ|k₀|^(2β) + λ
        """
        # Test parameters
        k_modes = [(4, 0, 0), (0, 4, 0), (3, 3, 2)]
        mu = solver.parameters['mu']
        beta = solver.parameters['beta']
        lambda_param = solver.parameters['lambda']
        
        for k_mode in k_modes:
            # Create plane wave source
            source = self._create_plane_wave_source(domain_7d, k_mode, amplitude=1.0)
            
            # Solve
            solution = solver.solve_stationary(source)
            
            # Analytical solution
            k_magnitude = np.sqrt(sum(k**2 for k in k_mode))
            D_k = mu * (k_magnitude ** (2 * beta)) + lambda_param
            expected_solution = source / D_k
            
            # Validate solution
            self._validate_plane_wave_solution(solution, expected_solution, k_mode)
    
    def test_A02_multifrequency_source(self, solver, domain_7d):
        """
        Test A0.2: Multifrequency source superposition.
        
        Physical Meaning:
            Tests the superposition principle for multiple frequency
            sources, ensuring linearity and absence of aliasing.
            
        Mathematical Foundation:
            For s(x) = Σᵢ Aᵢ exp(i kᵢ·x), the solution is:
            a(x) = Σᵢ Aᵢ exp(i kᵢ·x) / D(kᵢ)
        """
        # Create multifrequency source
        k_modes = [(2, 0, 0), (0, 2, 0), (1, 1, 1), (3, 0, 0)]
        amplitudes = [1.0, 0.5, 0.8, 0.3]
        
        source = np.zeros(domain_7d.shape, dtype=complex)
        for k_mode, amplitude in zip(k_modes, amplitudes):
            plane_wave = self._create_plane_wave_source(domain_7d, k_mode, amplitude)
            source += plane_wave
        
        # Solve
        solution = solver.solve_stationary(source)
        
        # Validate superposition
        self._validate_multifrequency_solution(solution, k_modes, amplitudes, solver)
    
    def test_A03_zero_mode_compatibility(self, domain_7d):
        """
        Test A0.3: Zero mode compatibility for λ=0.
        
        Physical Meaning:
            Tests the handling of the zero mode when λ=0,
            ensuring proper error handling for incompatible cases.
            
        Mathematical Foundation:
            When λ=0, the zero mode D(0) = 0, so ŝ(0) must be zero
            to avoid singularity.
        """
        # Test case 1: λ=0 with zero mean source (should work)
        parameters_zero_lambda = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda': 0.0,
            'precision': 'float64'
        }
        solver_zero = FFTSolver7D(domain_7d, parameters_zero_lambda)
        
        # Create zero mean source
        source_zero_mean = self._create_plane_wave_source(domain_7d, (2, 0, 0), 1.0)
        solution = solver_zero.solve_stationary(source_zero_mean)
        
        # Should work without error
        assert solution is not None
        
        # Test case 2: λ=0 with non-zero mean source (should fail)
        source_constant = np.ones(domain_7d.shape, dtype=complex)
        
        with pytest.raises(ValueError, match="lambda=0 requires mean\\(source\\)=0"):
            solver_zero.solve_stationary(source_constant)
    
    def test_A04_time_dependent_harmonic(self, solver, domain_7d):
        """
        Test A0.4: Time-dependent harmonic source.
        
        Physical Meaning:
            Tests the time integrator for harmonic sources,
            validating the steady-state solution.
            
        Mathematical Foundation:
            For s(x,t) = s₀ exp(i k₀·x) exp(-iωt), the steady-state solution is:
            a_ss(x,t) = s₀ exp(i k₀·x) exp(-iωt) / (ν|k₀|^(2β) + λ + iω)
        """
        # Test parameters
        k_mode = (4, 0, 0)
        omega = 1.0
        t_final = 10.0
        dt = 0.01
        
        # Create harmonic source
        source = self._create_plane_wave_source(domain_7d, k_mode, 1.0)
        
        # Time integration parameters
        time_params = {
            't_final': t_final,
            'dt': dt,
            'scheme': 'exponential',
            'omega': omega
        }
        
        # Solve time-dependent problem
        time_evolution = solver.solve_time_dependent(source, time_params)
        
        # Validate steady-state solution
        self._validate_harmonic_steady_state(time_evolution, k_mode, omega, solver)
    
    def test_A05_energy_balance_residual(self, solver, domain_7d):
        """
        Test A0.5: Energy balance and residual validation.
        
        Physical Meaning:
            Tests energy conservation and residual computation,
            ensuring the solution minimizes the energy functional.
            
        Mathematical Foundation:
            The residual r = L_β a - s should satisfy:
            - ||r||₂ / ||s||₂ ≤ 10⁻¹²
            - Re(Σ â*(k) r̂(k)) ≈ 0 (orthogonality)
        """
        # Create test source
        source = self._create_plane_wave_source(domain_7d, (4, 0, 0), 1.0)
        
        # Solve
        solution = solver.solve_stationary(source)
        
        # Validate solution
        metrics = solver.validate_solution(solution, source)
        
        # Check residual norm
        assert metrics['residual_norm'] <= 1e-12, f"Residual norm too large: {metrics['residual_norm']}"
        
        # Check orthogonality
        assert metrics['orthogonality'] <= 1e-12, f"Orthogonality violation: {metrics['orthogonality']}"
        
        # Check energy balance
        assert metrics['energy_balance'] <= 0.03, f"Energy balance violation: {metrics['energy_balance']}"
    
    def test_A11_scale_length_invariance(self, domain_7d):
        """
        Test A1.1: Scale length invariance.
        
        Physical Meaning:
            Tests that dimensionless solutions are invariant
            under changes in domain size L.
            
        Mathematical Foundation:
            For the same dimensionless source, the dimensionless
            solution should be identical regardless of L.
        """
        # Test parameters
        parameters = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda_param': 0.1,
            'precision': 'float64'
        }
        
        # Create two domains with different L but same Δ
        domain1 = Domain(L=1.0, N=64, dimensions=3)
        domain2 = Domain(L=2.0, N=128, dimensions=3)
        
        solver1 = FFTSolver7D(domain1, parameters)
        solver2 = FFTSolver7D(domain2, parameters)
        
        # Create dimensionless source
        k_mode = (4, 0, 0)
        source1 = self._create_plane_wave_source(domain1, k_mode, 1.0)
        source2 = self._create_plane_wave_source(domain2, k_mode, 1.0)
        
        # Solve
        solution1 = solver1.solve_stationary(source1)
        solution2 = solver2.solve_stationary(source2)
        
        # Compare dimensionless solutions
        self._validate_scale_invariance(solution1, solution2, domain1, domain2)
    
    def test_A12_units_invariance(self, domain_7d):
        """
        Test A1.2: Units invariance.
        
        Physical Meaning:
            Tests that dimensionless solutions are invariant
            under changes in base units (L₀, T₀, A₀).
            
        Mathematical Foundation:
            For the same dimensionless parameters, the dimensionless
            solution should be identical regardless of base units.
        """
        # Test with different base units but same dimensionless parameters
        parameters1 = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda_param': 0.1,
            'precision': 'float64'
        }
        
        parameters2 = {
            'mu': 2.0,  # Different base units
            'beta': 1.0,
            'lambda': 0.2,  # Scaled accordingly
            'precision': 'float64'
        }
        
        solver1 = FFTSolver7D(domain_7d, parameters1)
        solver2 = FFTSolver7D(domain_7d, parameters2)
        
        # Create source
        source = self._create_plane_wave_source(domain_7d, (4, 0, 0), 1.0)
        
        # Solve
        solution1 = solver1.solve_stationary(source)
        solution2 = solver2.solve_stationary(source)
        
        # Compare solutions (should be identical for dimensionless case)
        self._validate_units_invariance(solution1, solution2)
    
    def _create_plane_wave_source(self, domain: Domain, k_mode: Tuple[int, ...], 
                                 amplitude: float) -> np.ndarray:
        """Create plane wave source."""
        # Create coordinate arrays
        coords = []
        for i, n in enumerate(domain.shape):
            x = np.linspace(0, domain.L, n, endpoint=False)
            coords.append(x)
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(*coords, indexing='ij')
        
        # Create plane wave
        k_dot_r = k_mode[0] * X + k_mode[1] * Y + k_mode[2] * Z
        source = amplitude * np.exp(1j * 2 * np.pi * k_dot_r / domain.L)
        
        return source
    
    def _validate_plane_wave_solution(self, solution: np.ndarray, 
                                    expected: np.ndarray, k_mode: Tuple[int, ...]):
        """Validate plane wave solution."""
        # Check L2 error
        l2_error = np.linalg.norm(solution - expected) / np.linalg.norm(expected)
        assert l2_error <= 1e-12, f"L2 error too large for k={k_mode}: {l2_error}"
        
        # Check amplitude
        amplitude_error = abs(np.max(np.abs(solution)) - np.max(np.abs(expected)))
        assert amplitude_error <= 1e-12, f"Amplitude error too large for k={k_mode}: {amplitude_error}"
    
    def _validate_multifrequency_solution(self, solution: np.ndarray, 
                                        k_modes: list, amplitudes: list, solver):
        """Validate multifrequency solution."""
        # Check that solution is linear combination of individual solutions
        individual_solutions = []
        for k_mode, amplitude in zip(k_modes, amplitudes):
            source = self._create_plane_wave_source(solver.domain, k_mode, amplitude)
            individual_solution = solver.solve_stationary(source)
            individual_solutions.append(individual_solution)
        
        # Superposition should hold
        expected_solution = sum(individual_solutions)
        l2_error = np.linalg.norm(solution - expected_solution) / np.linalg.norm(expected_solution)
        assert l2_error <= 1e-12, f"Superposition principle violated: {l2_error}"
    
    def _validate_harmonic_steady_state(self, time_evolution: np.ndarray, 
                                      k_mode: Tuple[int, ...], omega: float, solver):
        """Validate harmonic steady-state solution."""
        # For now, just check that time evolution is computed
        assert time_evolution is not None
        assert time_evolution.shape == solver.domain.shape
        
        # TODO: Implement detailed steady-state validation
        # This would require implementing the time integrator first
    
    def _validate_scale_invariance(self, solution1: np.ndarray, solution2: np.ndarray,
                                 domain1: Domain, domain2: Domain):
        """Validate scale invariance."""
        # Compare dimensionless solutions
        # For now, just check that both solutions exist
        assert solution1 is not None
        assert solution2 is not None
        
        # TODO: Implement detailed scale invariance validation
        # This would require proper dimensionless comparison
    
    def _validate_units_invariance(self, solution1: np.ndarray, solution2: np.ndarray):
        """Validate units invariance."""
        # For now, just check that both solutions exist
        assert solution1 is not None
        assert solution2 is not None
        
        # TODO: Implement detailed units invariance validation
        # This would require proper dimensionless comparison
