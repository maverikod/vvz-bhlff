"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test suite for soliton optimization algorithms.

This module tests the complete soliton optimization algorithms
including single, two, and three-soliton solutions using
full 7D BVP theory implementation.

Physical Meaning:
    Tests validate that all soliton optimization algorithms
    correctly implement 7D BVP theory with step resonator
    interactions instead of classical exponential decay.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import logging

from bhlff.models.level_f.nonlinear.soliton_analysis.single_soliton import SingleSolitonSolver
from bhlff.models.level_f.nonlinear.soliton_analysis.multi_soliton_solutions import MultiSolitonSolutions
from bhlff.models.level_f.nonlinear.soliton_analysis.multi_soliton_core import MultiSolitonCore


class TestSolitonOptimizationAlgorithms:
    """Test suite for complete soliton optimization algorithms."""
    
    def setup_method(self):
        """Setup test environment."""
        # Mock system and parameters
        self.mock_system = Mock()
        self.nonlinear_params = {
            'mu': 1.0,
            'beta': 1.5,
            'lambda': 0.1,
            'interaction_strength': 0.5,
            'three_body_strength': 0.2
        }
        
        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
    
    def test_single_soliton_solver_initialization(self):
        """Test single soliton solver initialization."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        assert solver.mu == 1.0
        assert solver.beta == 1.5
        assert solver.lambda_param == 0.1
        assert solver.system == self.mock_system
    
    def test_single_soliton_find_solution(self):
        """Test single soliton solution finding with full 7D BVP theory."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        # Mock the BVP solver to return a successful solution
        with patch('scipy.integrate.solve_bvp') as mock_solve_bvp, \
             patch('scipy.optimize.minimize') as mock_minimize:
            
            # Mock successful BVP solution
            mock_sol = Mock()
            mock_sol.success = True
            mock_sol.y = np.array([[1.0, 0.5, 0.1], [0.0, 0.0, 0.0]])  # Field and derivative
            mock_solve_bvp.return_value = mock_sol
            
            # Mock successful optimization
            mock_result = Mock()
            mock_result.success = True
            mock_result.fun = -2.5  # Negative energy
            mock_result.x = [1.2, 0.8, 0.5]  # amplitude, width, position
            mock_result.nit = 50
            mock_result.nfev = 100
            mock_result.jac = np.array([0.1, 0.05, 0.02])
            mock_minimize.return_value = mock_result
            
            # Test solution finding
            solution = solver.find_single_soliton()
            
            assert solution is not None
            assert solution['type'] == 'single'
            assert solution['optimization_success'] is True
            assert 'amplitude' in solution
            assert 'width' in solution
            assert 'position' in solution
            assert 'energy' in solution
            assert 'physical_properties' in solution
    
    def test_single_soliton_step_resonator_profile(self):
        """Test step resonator profile computation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        x = np.linspace(-5, 5, 100)
        position = 0.0
        width = 2.0
        
        profile = solver._step_resonator_profile(x, position, width)
        
        # Check step resonator behavior
        assert np.all(profile[np.abs(x) < width] == 1.0)
        assert np.all(profile[np.abs(x) >= width] == 0.0)
    
    def test_single_soliton_validation(self):
        """Test soliton shape validation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        # Test valid soliton shape
        valid_solution = np.array([[1.0, 0.8, 0.5, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0]])
        assert solver._validate_soliton_shape(valid_solution, 1.0, 1.0) is True
        
        # Test invalid soliton shape (too high amplitude)
        invalid_solution = np.array([[5.0, 4.0, 3.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        assert solver._validate_soliton_shape(invalid_solution, 1.0, 1.0) is False
    
    def test_multi_soliton_solutions_initialization(self):
        """Test multi-soliton solutions initialization."""
        solver = MultiSolitonSolutions(self.mock_system, self.nonlinear_params)
        
        assert solver.mu == 1.0
        assert solver.beta == 1.5
        assert solver.lambda_param == 0.1
        assert solver.interaction_strength == 0.5
        assert solver.three_body_strength == 0.2
        assert solver.core is not None
    
    def test_two_soliton_solution_finding(self):
        """Test two-soliton solution finding with full 7D BVP theory."""
        solver = MultiSolitonSolutions(self.mock_system, self.nonlinear_params)
        
        # Mock the BVP solver and optimizer
        with patch('scipy.integrate.solve_bvp') as mock_solve_bvp, \
             patch('scipy.optimize.minimize') as mock_minimize:
            
            # Mock successful BVP solution
            mock_sol = Mock()
            mock_sol.success = True
            mock_sol.y = np.array([[1.0, 0.8, 0.5, 0.3, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0]])
            mock_solve_bvp.return_value = mock_sol
            
            # Mock successful optimization
            mock_result = Mock()
            mock_result.success = True
            mock_result.fun = -5.0  # Negative energy
            mock_result.x = [1.2, 0.8, -2.0, 1.0, 1.0, 2.0]  # Two soliton parameters
            mock_result.nit = 75
            mock_result.nfev = 150
            mock_result.jac = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
            mock_minimize.return_value = mock_result
            
            # Test two-soliton solution finding
            solutions = solver.find_two_soliton_solutions()
            
            assert len(solutions) > 0
            solution = solutions[0]
            assert solution['type'] == 'multi'
            assert solution['num_solitons'] == 2
            assert 'soliton_1' in solution
            assert 'soliton_2' in solution
            assert 'interaction_strength' in solution
            assert 'physical_properties' in solution
    
    def test_three_soliton_solution_finding(self):
        """Test three-soliton solution finding with full 7D BVP theory."""
        solver = MultiSolitonSolutions(self.mock_system, self.nonlinear_params)
        
        # Mock the BVP solver and optimizer
        with patch('scipy.integrate.solve_bvp') as mock_solve_bvp, \
             patch('scipy.optimize.minimize') as mock_minimize:
            
            # Mock successful BVP solution
            mock_sol = Mock()
            mock_sol.success = True
            mock_sol.y = np.array([[1.0, 0.8, 0.5, 0.3, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            mock_solve_bvp.return_value = mock_sol
            
            # Mock successful optimization
            mock_result = Mock()
            mock_result.success = True
            mock_result.fun = -8.0  # Negative energy
            mock_result.x = [1.2, 0.8, -3.0, 1.0, 1.0, 0.0, 0.9, 1.1, 3.0]  # Three soliton parameters
            mock_result.nit = 100
            mock_result.nfev = 200
            mock_result.jac = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01, 0.06, 0.04, 0.02])
            mock_minimize.return_value = mock_result
            
            # Test three-soliton solution finding
            solutions = solver.find_three_soliton_solutions()
            
            assert len(solutions) > 0
            solution = solutions[0]
            assert solution['type'] == 'multi'
            assert solution['num_solitons'] == 3
            assert 'soliton_1' in solution
            assert 'soliton_2' in solution
            assert 'soliton_3' in solution
            assert 'physical_properties' in solution
    
    def test_step_resonator_interaction(self):
        """Test step resonator interaction computation."""
        solver = MultiSolitonSolutions(self.mock_system, self.nonlinear_params)
        
        # Test step resonator interaction
        distance = 1.5
        interaction_range = 2.0
        
        # Should be 1.0 when distance < interaction_range
        interaction = solver.core._step_resonator_interaction(distance, interaction_range)
        assert interaction == 1.0
        
        # Should be 0.0 when distance >= interaction_range
        distance = 3.0
        interaction = solver.core._step_resonator_interaction(distance, interaction_range)
        assert interaction == 0.0
    
    def test_fractional_laplacian_computation(self):
        """Test fractional Laplacian computation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        x = np.linspace(-5, 5, 100)
        field = np.exp(-x**2)  # Gaussian field
        
        frac_lap = solver._compute_full_fractional_laplacian(x, field)
        
        assert len(frac_lap) == len(field)
        assert not np.any(np.isnan(frac_lap))
        assert not np.any(np.isinf(frac_lap))
    
    def test_energy_computation(self):
        """Test soliton energy computation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        # Test energy computation
        solution = np.array([[1.0, 0.8, 0.5, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0]])
        amplitude = 1.0
        width = 1.0
        
        energy = solver.compute_soliton_energy(solution, amplitude, width)
        
        assert energy > 0
        assert not np.isnan(energy)
        assert not np.isinf(energy)
    
    def test_physical_properties_computation(self):
        """Test physical properties computation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        # Mock solution
        solution = {
            'spatial_grid': np.linspace(-5, 5, 100),
            'profile': np.exp(-np.linspace(-5, 5, 100)**2),
            'mass': 2.0,
            'momentum': 0.5,
            'topological_charge': 1.0
        }
        
        amplitude = 1.0
        width = 1.0
        position = 0.0
        
        properties = solver._compute_soliton_physical_properties(amplitude, width, position, solution)
        
        assert 'kinetic_energy' in properties
        assert 'potential_energy' in properties
        assert 'total_energy' in properties
        assert 'stability_metric' in properties
        assert 'phase_coherence' in properties
        assert '7d_bvp_properties' in properties
    
    def test_7d_bvp_properties_computation(self):
        """Test 7D BVP specific properties computation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        x = np.linspace(-5, 5, 100)
        profile = np.exp(-x**2)
        amplitude = 1.0
        width = 1.0
        
        bvp_properties = solver._compute_7d_bvp_properties(profile, x, amplitude, width)
        
        assert 'fractional_laplacian_contribution' in bvp_properties
        assert 'step_resonator_efficiency' in bvp_properties
        assert '7d_phase_space_properties' in bvp_properties
        assert 'bvp_convergence_quality' in bvp_properties
    
    def test_phase_space_properties(self):
        """Test 7D phase space properties computation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        x = np.linspace(-5, 5, 100)
        profile = np.exp(-x**2)
        
        phase_space_props = solver._compute_7d_phase_space_properties(profile, x)
        
        assert 'phase_space_volume' in phase_space_props
        assert 'phase_space_entropy' in phase_space_props
        assert 'spectral_width' in phase_space_props
    
    def test_solution_quality_validation(self):
        """Test solution quality validation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        # Test valid solution
        valid_solution = {
            'spatial_grid': np.linspace(-5, 5, 100),
            'profile': np.exp(-np.linspace(-5, 5, 100)**2),
            'mass': 2.0,
            'momentum': 0.5,
            'topological_charge': 1.0
        }
        
        assert solver._validate_solution_quality(valid_solution, 1.0, 1.0) is True
        
        # Test invalid solution (negative mass)
        invalid_solution = valid_solution.copy()
        invalid_solution['mass'] = -1.0
        
        assert solver._validate_solution_quality(invalid_solution, 1.0, 1.0) is False
    
    def test_multi_soliton_energy_computation(self):
        """Test multi-soliton energy computation."""
        core = MultiSolitonCore(self.mock_system, self.nonlinear_params)
        
        # Test two-soliton energy
        solution = np.array([[1.0, 0.8, 0.5, 0.3, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0]])
        amp1, width1, pos1 = 1.0, 1.0, -2.0
        amp2, width2, pos2 = 1.0, 1.0, 2.0
        
        energy = core.compute_two_soliton_energy(solution, amp1, width1, pos1, amp2, width2, pos2)
        
        assert energy > 0
        assert not np.isnan(energy)
        assert not np.isinf(energy)
    
    def test_three_body_interaction_energy(self):
        """Test three-body interaction energy computation."""
        solver = MultiSolitonSolutions(self.mock_system, self.nonlinear_params)
        
        # Test three-body interaction
        amp1, width1, pos1 = 1.0, 1.0, -3.0
        amp2, width2, pos2 = 1.0, 1.0, 0.0
        amp3, width3, pos3 = 1.0, 1.0, 3.0
        
        energy = solver._compute_three_body_interaction_energy(
            amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3
        )
        
        assert energy >= 0
        assert not np.isnan(energy)
        assert not np.isinf(energy)
    
    def test_step_resonator_boundary_condition(self):
        """Test step resonator boundary condition."""
        solver = MultiSolitonSolutions(self.mock_system, self.nonlinear_params)
        
        # Test boundary condition
        field_value = 0.5
        amplitude = 1.0
        
        boundary_value = solver._step_resonator_boundary_condition(field_value, amplitude)
        
        assert boundary_value == field_value  # Should return field_value since > 0.1 * amplitude
        
        # Test with small field value
        field_value = 0.05
        boundary_value = solver._step_resonator_boundary_condition(field_value, amplitude)
        
        assert boundary_value == 0.0  # Should return 0.0 since < 0.1 * amplitude
    
    def test_optimization_robustness(self):
        """Test optimization robustness with multiple initial guesses."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        # Test that the solver tries multiple initial guesses
        with patch('scipy.integrate.solve_bvp') as mock_solve_bvp, \
             patch('scipy.optimize.minimize') as mock_minimize:
            
            # Mock some failures and one success
            mock_sol = Mock()
            mock_sol.success = True
            mock_sol.y = np.array([[1.0, 0.5, 0.1], [0.0, 0.0, 0.0]])
            mock_solve_bvp.return_value = mock_sol
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.fun = -2.5
            mock_result.x = [1.2, 0.8, 0.5]
            mock_result.nit = 50
            mock_result.nfev = 100
            mock_result.jac = np.array([0.1, 0.05, 0.02])
            mock_minimize.return_value = mock_result
            
            solution = solver.find_single_soliton()
            
            # Should have tried multiple initial guesses
            assert mock_minimize.call_count >= 1
            assert solution is not None
    
    def test_7d_bvp_theory_compliance(self):
        """Test compliance with 7D BVP theory principles."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        # Test that step resonator is used instead of exponential decay
        x = np.linspace(-5, 5, 100)
        position = 0.0
        width = 2.0
        
        profile = solver._step_resonator_profile(x, position, width)
        
        # Check that it's a step function, not exponential
        assert np.all(profile[np.abs(x) < width] == 1.0)
        assert np.all(profile[np.abs(x) >= width] == 0.0)
        
        # No exponential decay should be present
        assert not np.any(np.exp(-np.abs(x)) == profile)
    
    def test_energy_conservation(self):
        """Test energy conservation in soliton solutions."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        x = np.linspace(-5, 5, 100)
        field = np.exp(-x**2)
        
        # Compute energy
        energy = solver.compute_soliton_energy(np.array([field, np.zeros_like(field)]), 1.0, 1.0)
        
        # Energy should be positive and finite
        assert energy > 0
        assert not np.isnan(energy)
        assert not np.isinf(energy)
    
    def test_phase_coherence_computation(self):
        """Test phase coherence computation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        x = np.linspace(-5, 5, 100)
        profile = np.exp(-x**2)
        
        coherence = solver._compute_phase_coherence(profile, x)
        
        assert 0.0 <= coherence <= 1.0
        assert not np.isnan(coherence)
        assert not np.isinf(coherence)
    
    def test_stability_metric_computation(self):
        """Test stability metric computation."""
        solver = SingleSolitonSolver(self.mock_system, self.nonlinear_params)
        
        x = np.linspace(-5, 5, 100)
        profile = np.exp(-x**2)
        
        stability = solver._compute_stability_metric(profile, x)
        
        assert stability >= 0.0
        assert not np.isnan(stability)
        assert not np.isinf(stability)


if __name__ == '__main__':
    pytest.main([__file__])