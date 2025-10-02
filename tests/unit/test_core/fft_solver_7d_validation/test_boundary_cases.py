"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Boundary case tests for 7D FFT Solver.

This module contains boundary case tests including edge cases,
extreme parameter values, and error condition tests.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.fft import FFTSolver7D, FractionalLaplacian
from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters


class TestBoundaryCases:
    """
    Boundary case tests for 7D FFT Solver.
    
    Physical Meaning:
        Tests edge cases and extreme conditions to ensure
        robust behavior of the 7D FFT solver implementation.
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
            precision='float64',
            fft_plan='MEASURE',
            tolerance=1e-12
        )
    
    @pytest.fixture
    def solver(self, domain_7d, parameters_basic):
        """Create FFT solver for testing."""
        return FFTSolver7D(domain_7d, parameters_basic)
    
    def test_C01_zero_source(self, solver, domain_7d):
        """
        Test C0.1: Zero source test.
        
        Physical Meaning:
            Tests that the solver correctly handles zero source,
            which should produce zero solution.
        """
        # Create zero source
        source = np.zeros(domain_7d.shape)
        
        # Solve
        solution = solver.solve_stationary(source)
        
        # Solution should be zero
        assert np.allclose(solution, 0.0, atol=1e-15), "Zero source should produce zero solution"
    
    def test_C02_very_small_source(self, solver, domain_7d):
        """
        Test C0.2: Very small source test.
        
        Physical Meaning:
            Tests that the solver handles very small source values
            without numerical issues.
        """
        # Create very small source
        source = np.full(domain_7d.shape, 1e-15)
        
        # Solve
        solution = solver.solve_stationary(source)
        
        # Solution should be finite and proportional to source
        assert np.all(np.isfinite(solution)), "Solution should be finite for very small source"
        assert np.allclose(solution, source / solver.parameters.lambda_param, rtol=1e-10), "Solution should be proportional to source"
    
    def test_C03_very_large_source(self, solver, domain_7d):
        """
        Test C0.3: Very large source test.
        
        Physical Meaning:
            Tests that the solver handles very large source values
            without overflow or numerical instability.
        """
        # Create very large source
        source = np.full(domain_7d.shape, 1e15)
        
        # Solve
        solution = solver.solve_stationary(source)
        
        # Solution should be finite
        assert np.all(np.isfinite(solution)), "Solution should be finite for very large source"
        assert not np.any(np.isinf(solution)), "Solution should not contain Inf for very large source"
    
    def test_C04_extreme_parameters(self, domain_7d):
        """
        Test C0.4: Extreme parameter values test.
        
        Physical Meaning:
            Tests that the solver handles extreme parameter values
            without breaking or producing invalid results.
        """
        # Test extreme parameter combinations
        extreme_cases = [
            {'mu': 1e-10, 'beta': 0.01, 'lambda_param': 1e-10},  # Very small
            {'mu': 1e10, 'beta': 1.99, 'lambda_param': 1e10},    # Very large
            {'mu': 0.0, 'beta': 1.0, 'lambda_param': 0.1},       # Zero mu
            {'mu': 1.0, 'beta': 0.0, 'lambda_param': 0.1},       # Zero beta
        ]
        
        for case in extreme_cases:
            try:
                # Create solver with extreme parameters
                test_params = Parameters(
                    mu=case['mu'],
                    beta=case['beta'],
                    lambda_param=case['lambda_param'],
                    precision='float64',
                    fft_plan='MEASURE',
                    tolerance=1e-12
                )
                test_solver = FFTSolver7D(domain_7d, test_params)
                
                # Create test source
                source = np.ones(domain_7d.shape)
                
                # Solve
                solution = test_solver.solve_stationary(source)
                
                # Check that solution is finite
                assert np.all(np.isfinite(solution)), f"Solution should be finite for extreme case {case}"
                
            except (ValueError, RuntimeError) as e:
                # Some extreme cases may legitimately fail
                # This is acceptable as long as the error is reasonable
                assert "invalid" in str(e).lower() or "out of range" in str(e).lower(), f"Unexpected error for case {case}: {e}"
    
    def test_C05_singular_conditions(self, domain_7d):
        """
        Test C0.5: Singular conditions test.
        
        Physical Meaning:
            Tests that the solver handles singular or near-singular
            conditions appropriately.
        """
        # Test near-singular conditions
        singular_cases = [
            {'mu': 1.0, 'beta': 1.0, 'lambda_param': 1e-15},  # Near-zero damping
            {'mu': 1e-15, 'beta': 1.0, 'lambda_param': 0.1},  # Near-zero diffusion
        ]
        
        for case in singular_cases:
            test_params = Parameters(
                mu=case['mu'],
                beta=case['beta'],
                lambda_param=case['lambda_param'],
                precision='float64',
                fft_plan='MEASURE',
                tolerance=1e-12
            )
            test_solver = FFTSolver7D(domain_7d, test_params)
            
            # Create test source
            source = np.ones(domain_7d.shape)
            
            # Solve
            solution = test_solver.solve_stationary(source)
            
            # Solution should be finite (solver should handle singularities)
            assert np.all(np.isfinite(solution)), f"Solution should be finite for singular case {case}"
    
    def test_C06_memory_usage(self, solver, domain_7d):
        """
        Test C0.6: Memory usage test.
        
        Physical Meaning:
            Tests that the solver doesn't consume excessive memory
            for reasonable problem sizes.
        """
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and solve multiple problems
        for i in range(10):
            source = np.random.randn(*domain_7d.shape)
            solution = solver.solve_stationary(source)
            del source, solution  # Explicit cleanup
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100.0, f"Memory usage increased too much: {memory_increase:.1f}MB"
    
    def test_C07_error_handling(self, domain_7d):
        """
        Test C0.7: Error handling test.
        
        Physical Meaning:
            Tests that the solver properly handles invalid inputs
            and produces appropriate error messages.
        """
        # Test invalid domain
        with pytest.raises((ValueError, TypeError)):
            invalid_domain = "invalid_domain"
            FFTSolver7D(invalid_domain, Parameters())
        
        # Test invalid parameters
        with pytest.raises((ValueError, TypeError)):
            invalid_params = "invalid_params"
            FFTSolver7D(domain_7d, invalid_params)
        
        # Test negative parameters
        with pytest.raises(ValueError):
            negative_params = Parameters(
                mu=-1.0,  # Negative mu should raise error
                beta=1.0,
                lambda_param=0.1,
                precision='float64',
                fft_plan='MEASURE',
                tolerance=1e-12
            )
            FFTSolver7D(domain_7d, negative_params)
    
    def test_C08_performance_benchmark(self, solver, domain_7d):
        """
        Test C0.8: Performance benchmark test.
        
        Physical Meaning:
            Tests that the solver performs reasonably well
            for typical problem sizes.
        """
        import time
        
        # Create test source
        source = np.random.randn(*domain_7d.shape)
        
        # Time the solution
        start_time = time.time()
        solution = solver.solve_stationary(source)
        end_time = time.time()
        
        solve_time = end_time - start_time
        
        # Solution should complete in reasonable time (less than 1 second for this size)
        assert solve_time < 1.0, f"Solution took too long: {solve_time:.3f}s"
        
        # Solution should be finite
        assert np.all(np.isfinite(solution)), "Solution should be finite"
    
    def test_C09_consistency_across_runs(self, solver, domain_7d):
        """
        Test C0.9: Consistency across multiple runs.
        
        Physical Meaning:
            Tests that the solver produces consistent results
            across multiple runs with the same input.
        """
        # Create test source
        source = np.random.randn(*domain_7d.shape)
        
        # Solve multiple times
        solutions = []
        for i in range(5):
            solution = solver.solve_stationary(source)
            solutions.append(solution)
        
        # All solutions should be identical
        for i in range(1, len(solutions)):
            relative_diff = np.linalg.norm(solutions[0] - solutions[i]) / np.linalg.norm(solutions[0])
            assert relative_diff < 1e-15, f"Solutions should be identical across runs, diff={relative_diff}"
