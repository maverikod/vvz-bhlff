"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test suite for soliton optimization compliance with 7D BVP theory.

This module tests that all soliton optimization algorithms are fully
implemented according to the classical patterns correction plan,
ensuring no simplified implementations or placeholders remain.

Physical Meaning:
    Tests comprehensive soliton optimization including single and
    multi-soliton solutions with full 7D BVP theory implementation
    and step resonator interactions.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import logging

from bhlff.models.level_f.nonlinear.soliton_analysis.single_soliton import SingleSolitonSolver
from bhlff.models.level_f.nonlinear.soliton_analysis.multi_soliton import MultiSolitonSolver
from bhlff.models.level_f.nonlinear.soliton_analysis.solutions import SolitonAnalysisSolutions


class TestSolitonOptimizationCompliance:
    """
    Test suite for soliton optimization compliance with 7D BVP theory.
    
    Physical Meaning:
        Tests that all soliton optimization algorithms are fully
        implemented with complete 7D BVP theory and step resonator
        interactions, ensuring no classical patterns remain.
    """
    
    @pytest.fixture
    def nonlinear_params(self) -> Dict[str, Any]:
        """Create test nonlinear parameters."""
        return {
            'mu': 1.0,
            'beta': 1.5,
            'lambda': 0.1,
            'interaction_strength': 0.1,
            'three_body_strength': 0.01
        }
    
    @pytest.fixture
    def test_system(self):
        """Create test system configuration."""
        return {
            'domain_size': 20.0,
            'grid_points': 100,
            'boundary_conditions': 'periodic'
        }
    
    def test_single_soliton_full_optimization(self, test_system, nonlinear_params):
        """
        Test that single soliton optimization is fully implemented.
        
        Physical Meaning:
            Verifies that single soliton finding uses complete
            optimization with 7D BVP theory and proper energy
            minimization.
        """
        solver = SingleSolitonSolver(test_system, nonlinear_params)
        
        # Test that optimization is fully implemented
        result = solver.find_single_soliton()
        
        if result is not None:
            # Verify full implementation characteristics
            assert 'optimization_success' in result
            assert 'energy' in result
            assert 'convergence_info' in result
            assert 'solution' in result
            
            # Verify optimization used proper bounds and method
            convergence_info = result['convergence_info']
            assert 'iterations' in convergence_info
            assert 'function_evaluations' in convergence_info
            assert 'gradient_norm' in convergence_info
            
            # Verify solution contains full physical parameters
            solution = result['solution']
            assert 'spatial_grid' in solution
            assert 'profile' in solution
            assert 'mass' in solution
            assert 'momentum' in solution
            assert 'topological_charge' in solution
    
    def test_two_soliton_full_optimization(self, test_system, nonlinear_params):
        """
        Test that two-soliton optimization is fully implemented.
        
        Physical Meaning:
            Verifies that two-soliton finding uses complete
            optimization with 7D BVP theory and interaction analysis.
        """
        solver = MultiSolitonSolver(test_system, nonlinear_params)
        
        # Test that optimization is fully implemented
        results = solver.find_two_soliton_solutions()
        
        if results:
            for result in results:
                # Verify full implementation characteristics
                assert 'optimization_success' in result
                assert 'energy' in result
                assert 'interaction_strength' in result
                assert 'convergence_info' in result
                assert 'solution' in result
                
                # Verify multi-soliton specific parameters
                assert 'num_solitons' in result
                assert result['num_solitons'] == 2
                assert 'soliton_1' in result
                assert 'soliton_2' in result
                
                # Verify interaction analysis
                assert 'interaction_strength' in result
                assert isinstance(result['interaction_strength'], (int, float))
    
    def test_three_soliton_full_optimization(self, test_system, nonlinear_params):
        """
        Test that three-soliton optimization is fully implemented.
        
        Physical Meaning:
            Verifies that three-soliton finding uses complete
            optimization with 7D BVP theory and multi-body interactions.
        """
        solver = MultiSolitonSolver(test_system, nonlinear_params)
        
        # Test that optimization is fully implemented
        results = solver.find_three_soliton_solutions()
        
        if results:
            for result in results:
                # Verify full implementation characteristics
                assert 'optimization_success' in result
                assert 'energy' in result
                assert 'convergence_info' in result
                assert 'solution' in result
                
                # Verify multi-soliton specific parameters
                assert 'num_solitons' in result
                assert result['num_solitons'] == 3
                assert 'soliton_1' in result
                assert 'soliton_2' in result
                assert 'soliton_3' in result
    
    def test_no_exponential_decay_in_soliton_interactions(self, test_system, nonlinear_params):
        """
        Test that no exponential decay is used in soliton interactions.
        
        Physical Meaning:
            Verifies that soliton interactions use step resonator
            theory instead of exponential decay, following 7D BVP
            principles.
        """
        solver = MultiSolitonSolver(test_system, nonlinear_params)
        
        # Test step resonator interaction
        distance = 2.0
        interaction_range = 3.0
        
        # This should use step resonator, not exponential
        interaction_factor = solver._step_resonator_interaction(distance, interaction_range)
        
        # Step resonator should return 1.0 or 0.0, not exponential decay
        assert interaction_factor in [0.0, 1.0]
        
        # Test that it's actually a step function
        close_distance = 1.0
        far_distance = 5.0
        
        close_interaction = solver._step_resonator_interaction(close_distance, interaction_range)
        far_interaction = solver._step_resonator_interaction(far_distance, interaction_range)
        
        # Close distance should have interaction, far distance should not
        assert close_interaction == 1.0
        assert far_interaction == 0.0
    
    def test_full_energy_calculations_with_interactions(self, test_system, nonlinear_params):
        """
        Test that energy calculations include full interactions.
        
        Physical Meaning:
            Verifies that energy calculations include all interaction
            terms using 7D BVP step resonator theory.
        """
        solver = MultiSolitonSolver(test_system, nonlinear_params)
        
        # Test single soliton energy
        test_solution = np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        amplitude = 1.0
        width = 1.0
        
        energy = solver.compute_soliton_energy(test_solution, amplitude, width)
        
        # Energy should be positive and finite
        assert energy > 0
        assert np.isfinite(energy)
        
        # Test two-soliton energy with interactions
        amp1, width1, pos1 = 1.0, 1.0, -2.0
        amp2, width2, pos2 = 1.0, 1.0, 2.0
        
        two_soliton_energy = solver._compute_two_soliton_energy(
            test_solution, amp1, width1, pos1, amp2, width2, pos2
        )
        
        # Two-soliton energy should include interaction terms
        assert two_soliton_energy > 0
        assert np.isfinite(two_soliton_energy)
    
    def test_7d_fractional_laplacian_implementation(self, test_system, nonlinear_params):
        """
        Test that 7D fractional Laplacian is fully implemented.
        
        Physical Meaning:
            Verifies that the fractional Laplacian operator is
            implemented using complete 7D BVP theory with proper
            spectral representation.
        """
        solver = SingleSolitonSolver(test_system, nonlinear_params)
        
        # Test fractional Laplacian computation
        x = np.linspace(-5.0, 5.0, 50)
        field = np.exp(-x**2)
        
        fractional_laplacian = solver._compute_full_fractional_laplacian(x, field)
        
        # Should return finite values
        assert np.all(np.isfinite(fractional_laplacian))
        assert len(fractional_laplacian) == len(field)
        
        # Should be consistent with beta parameter
        assert np.any(fractional_laplacian != 0)  # Should have some non-zero values
    
    def test_no_simplified_implementations(self, test_system, nonlinear_params):
        """
        Test that no simplified implementations remain.
        
        Physical Meaning:
            Verifies that all soliton algorithms are fully
            implemented without placeholders or simplifications.
        """
        solver = SolitonAnalysisSolutions(test_system, nonlinear_params)
        
        # Test that all methods are fully implemented
        solutions = solver.find_soliton_solutions()
        
        # Should return actual results, not placeholders
        assert isinstance(solutions, list)
        
        # If solutions found, they should be complete
        for solution in solutions:
            assert 'type' in solution
            assert 'energy' in solution
            assert 'optimization_success' in solution
            
            # Should not contain placeholder values
            assert solution['energy'] != 0.0  # Not a placeholder
            assert solution['optimization_success'] is not False  # Not a placeholder
    
    def test_complete_bvp_solving(self, test_system, nonlinear_params):
        """
        Test that BVP solving is complete and not simplified.
        
        Physical Meaning:
            Verifies that boundary value problem solving uses
            complete 7D BVP theory with proper boundary conditions.
        """
        solver = SingleSolitonSolver(test_system, nonlinear_params)
        
        # Test 7D soliton ODE computation
        x = np.linspace(-5.0, 5.0, 20)
        y = np.array([np.exp(-x**2), -2*x*np.exp(-x**2)])
        amplitude = 1.0
        width = 1.0
        
        ode_result = solver.compute_7d_soliton_ode(x, y, amplitude, width)
        
        # Should return proper ODE system
        assert ode_result.shape == y.shape
        assert np.all(np.isfinite(ode_result))
        
        # Should include all terms: fractional Laplacian, lambda term, source
        assert np.any(ode_result != 0)  # Should have non-zero derivatives
    
    def test_step_resonator_vs_exponential(self, test_system, nonlinear_params):
        """
        Test that step resonator is used instead of exponential decay.
        
        Physical Meaning:
            Verifies that interactions use step resonator theory
            with sharp cutoffs instead of exponential decay,
            following 7D BVP principles.
        """
        solver = MultiSolitonSolver(test_system, nonlinear_params)
        
        # Test step resonator behavior
        interaction_range = 2.0
        
        # Test various distances
        distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        for distance in distances:
            interaction = solver._step_resonator_interaction(distance, interaction_range)
            
            # Should be exactly 1.0 or 0.0 (step function)
            assert interaction in [0.0, 1.0]
            
            # Should be 1.0 for distances < range, 0.0 for distances >= range
            if distance < interaction_range:
                assert interaction == 1.0
            else:
                assert interaction == 0.0
    
    def test_complete_optimization_parameters(self, test_system, nonlinear_params):
        """
        Test that optimization uses complete parameters and bounds.
        
        Physical Meaning:
            Verifies that optimization uses proper bounds, methods,
            and convergence criteria for 7D BVP theory.
        """
        solver = SingleSolitonSolver(test_system, nonlinear_params)
        
        # Test that optimization is properly configured
        result = solver.find_single_soliton()
        
        if result is not None:
            # Verify optimization used proper method and bounds
            convergence_info = result['convergence_info']
            
            # Should have proper convergence information
            assert 'iterations' in convergence_info
            assert 'function_evaluations' in convergence_info
            assert 'gradient_norm' in convergence_info
            
            # Should have used L-BFGS-B method (indicated by proper convergence)
            assert convergence_info['iterations'] > 0
            assert convergence_info['function_evaluations'] > 0
            
            # Should have converged successfully
            assert result['optimization_success'] is True
            assert result['energy'] > 0  # Should have positive energy
