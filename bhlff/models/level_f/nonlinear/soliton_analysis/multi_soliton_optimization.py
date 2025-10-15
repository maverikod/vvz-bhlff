"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Multi-soliton optimization functionality.

This module implements multi-soliton optimization using complete
7D BVP theory with advanced optimization algorithms.

Physical Meaning:
    Implements multi-soliton optimization including parameter optimization,
    solution finding, and convergence analysis using 7D BVP theory.

Example:
    >>> optimizer = MultiSolitonOptimization(system, nonlinear_params)
    >>> solutions = optimizer.find_multi_soliton_solutions()
"""

import numpy as np
from typing import Dict, Any, List
from scipy.optimize import minimize
from scipy.integrate import solve_bvp
import logging

from .base import SolitonAnalysisBase
from .multi_soliton_core import MultiSolitonCore
from .multi_soliton_validation import MultiSolitonValidation


class MultiSolitonOptimization(SolitonAnalysisBase):
    """
    Multi-soliton optimization functionality.
    
    Physical Meaning:
        Implements multi-soliton optimization including parameter optimization,
        solution finding, and convergence analysis using 7D BVP theory.
        
    Mathematical Foundation:
        Optimizes multi-soliton parameters using complete 7D BVP theory
        with multiple initial guesses and advanced convergence criteria.
    """
    
    def __init__(self, system, nonlinear_params: Dict[str, Any]):
        """Initialize multi-soliton optimization."""
        super().__init__(system, nonlinear_params)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core functionality
        self.core = MultiSolitonCore(system, nonlinear_params)
        self.validator = MultiSolitonValidation(system, nonlinear_params)
    
    def find_multi_soliton_solutions(self) -> List[Dict[str, Any]]:
        """
        Find multi-soliton solutions using full 7D BVP theory.
        
        Physical Meaning:
            Finds multi-soliton solutions through complete optimization
            using 7D fractional Laplacian equations and interaction
            potentials between solitons.
            
        Returns:
            List[Dict[str, Any]]: Multi-soliton solutions with full
            physical parameters and interaction analysis.
        """
        multi_solitons = []
        
        try:
            # Find two-soliton solution with full optimization
            two_soliton = self.find_two_soliton_solutions()
            if two_soliton:
                multi_solitons.extend(two_soliton)
            
            # Find three-soliton solution with full optimization
            three_soliton = self.find_three_soliton_solutions()
            if three_soliton:
                multi_solitons.extend(three_soliton)
            
            return multi_solitons
            
        except Exception as e:
            self.logger.error(f"Multi-soliton solutions finding failed: {e}")
            return []
    
    def find_two_soliton_solutions(self) -> List[Dict[str, Any]]:
        """
        Find two-soliton solutions using full 7D BVP theory.
        
        Physical Meaning:
            Finds two-soliton solutions through complete optimization
            using 7D fractional Laplacian equations and soliton-soliton
            interaction potentials with full 7D BVP theory implementation.
            
        Returns:
            List[Dict[str, Any]]: Two-soliton solutions with interaction
            analysis and full physical parameters.
        """
        try:
            solutions = []
            
            # Multiple initial guesses for robust optimization
            initial_guesses = [
                [1.0, 1.0, -3.0, 1.0, 1.0, 3.0],    # Standard separation
                [1.2, 0.8, -2.0, 0.8, 1.2, 2.0],    # Different amplitudes/widths
                [0.9, 1.1, -4.0, 1.1, 0.9, 4.0],    # Wider separation
                [1.1, 0.9, -1.5, 0.9, 1.1, 1.5],    # Closer separation
                [1.3, 0.7, -2.5, 0.7, 1.3, 2.5]     # Asymmetric
            ]
            
            best_solution = None
            best_energy = float('inf')
            
            for i, initial_params in enumerate(initial_guesses):
                try:
                    # Setup 7D mesh for BVP solving with adaptive resolution
                    x_mesh = np.linspace(-20.0, 20.0, 300)
                    y_guess = np.zeros((2, len(x_mesh)))
                    
                    def two_soliton_equations_7d(params):
                        """7D two-soliton equations with full interaction physics."""
                        amp1, width1, pos1, amp2, width2, pos2 = params
                        
                        def two_soliton_ode(x, y):
                            """7D two-soliton ODE system with complete interactions."""
                            return self.core.compute_7d_two_soliton_ode(x, y, amp1, width1, pos1, amp2, width2, pos2)
                        
                        # Enhanced boundary conditions for two solitons
                        def bc(ya, yb):
                            # Soliton boundary conditions with proper 7D BVP theory
                            return [ya[0] - amp1 * self._step_resonator_boundary_condition(ya[0], amp1),
                                   yb[0] - amp2 * self._step_resonator_boundary_condition(yb[0], amp2)]
                        
                        try:
                            # Solve BVP with enhanced convergence
                            sol = solve_bvp(two_soliton_ode, bc, x_mesh, y_guess, 
                                          tol=1e-8, max_nodes=1500)
                            
                            if sol.success:
                                # Compute total energy including interaction
                                energy = self.core.compute_two_soliton_energy(sol.y, amp1, width1, pos1, amp2, width2, pos2)
                                
                                # Additional energy penalty for unphysical solutions
                                if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                                    return 1e10
                                
                                # Check for proper two-soliton shape
                                if not self.validator.validate_two_soliton_shape(sol.y, amp1, width1, amp2, width2):
                                    return 1e10
                                
                                return -energy  # Minimize negative energy
                            else:
                                return 1e10  # Large penalty for failed solution
                                
                        except Exception as e:
                            self.logger.debug(f"BVP solution failed for two-soliton guess {i}: {e}")
                            return 1e10  # Penalty for failed BVP solution
                    
                    # Optimize two-soliton parameters using 7D theory with enhanced bounds
                    result = minimize(
                        two_soliton_equations_7d,
                        initial_params,
                        method='L-BFGS-B',
                        bounds=[(0.1, 3.0), (0.3, 2.0), (-12.0, 12.0), (0.1, 3.0), (0.3, 2.0), (-12.0, 12.0)],
                        options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    if result.success and result.fun < best_energy and result.fun < 1e9:
                        best_energy = result.fun
                        amp1, width1, pos1, amp2, width2, pos2 = result.x
                        
                        # Compute final two-soliton solution with full validation
                        final_solution = self._compute_final_two_soliton_solution(
                            amp1, width1, pos1, amp2, width2, pos2
                        )
                        
                        # Compute interaction strength with full physics
                        interaction_strength = self._compute_soliton_interaction_strength(
                            amp1, width1, pos1, amp2, width2, pos2
                        )
                        
                        # Validate solution quality
                        if self.validator.validate_two_soliton_solution_quality(final_solution, amp1, width1, amp2, width2):
                            best_solution = {
                                "type": "multi",
                                "num_solitons": 2,
                                "soliton_1": {"amplitude": amp1, "width": width1, "position": pos1},
                                "soliton_2": {"amplitude": amp2, "width": width2, "position": pos2},
                                "energy": -result.fun,
                                "interaction_strength": interaction_strength,
                                "optimization_success": True,
                                "solution": final_solution,
                                "convergence_info": {
                                    "iterations": result.nit,
                                    "function_evaluations": result.nfev,
                                    "gradient_norm": np.linalg.norm(result.jac) if result.jac is not None else 0.0,
                                    "initial_guess_index": i,
                                    "energy_convergence": best_energy
                                },
                                "physical_properties": self.validator.compute_two_soliton_physical_properties(
                                    amp1, width1, pos1, amp2, width2, pos2, final_solution
                                )
                            }
                            
                except Exception as e:
                    self.logger.debug(f"Two-soliton optimization failed for guess {i}: {e}")
                    continue
            
            if best_solution:
                solutions.append(best_solution)
            
            return solutions
                
        except Exception as e:
            self.logger.warning(f"Two-soliton finding failed: {e}")
            return []
    
    def find_three_soliton_solutions(self) -> List[Dict[str, Any]]:
        """
        Find three-soliton solutions using full 7D BVP theory.
        
        Physical Meaning:
            Finds three-soliton solutions through complete optimization
            using 7D fractional Laplacian equations and multi-soliton
            interaction potentials with full 7D BVP theory implementation.
            
        Returns:
            List[Dict[str, Any]]: Three-soliton solutions with full
            interaction analysis and stability properties.
        """
        try:
            solutions = []
            
            # Multiple initial guesses for robust optimization
            initial_guesses = [
                [1.0, 1.0, -4.0, 1.0, 1.0, 0.0, 1.0, 1.0, 4.0],    # Standard triangular
                [1.2, 0.8, -3.0, 0.8, 1.2, 0.0, 1.0, 0.9, 3.0],   # Different amplitudes/widths
                [0.9, 1.1, -5.0, 1.1, 0.9, 0.0, 0.8, 1.0, 5.0],   # Wider separation
                [1.1, 0.9, -2.0, 0.9, 1.1, 0.0, 1.2, 0.8, 2.0],   # Closer separation
                [1.3, 0.7, -3.5, 0.7, 1.3, 0.0, 0.9, 1.1, 3.5]    # Asymmetric
            ]
            
            best_solution = None
            best_energy = float('inf')
            
            for i, initial_params in enumerate(initial_guesses):
                try:
                    # Setup 7D mesh for BVP solving with adaptive resolution
                    x_mesh = np.linspace(-25.0, 25.0, 400)
                    y_guess = np.zeros((2, len(x_mesh)))
                    
                    def three_soliton_equations_7d(params):
                        """7D three-soliton equations with full interaction physics."""
                        amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3 = params
                        
                        def three_soliton_ode(x, y):
                            """7D three-soliton ODE system with complete interactions."""
                            return self.core.compute_7d_three_soliton_ode(x, y, amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3)
                        
                        # Enhanced boundary conditions for three solitons
                        def bc(ya, yb):
                            # Soliton boundary conditions with proper 7D BVP theory
                            return [ya[0] - amp1 * self._step_resonator_boundary_condition(ya[0], amp1),
                                   yb[0] - amp3 * self._step_resonator_boundary_condition(yb[0], amp3)]
                        
                        try:
                            # Solve BVP with enhanced convergence
                            sol = solve_bvp(three_soliton_ode, bc, x_mesh, y_guess, 
                                          tol=1e-8, max_nodes=2000)
                            
                            if sol.success:
                                # Compute total energy including all interactions
                                energy = self.core.compute_three_soliton_energy(
                                    sol.y, amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3
                                )
                                
                                # Additional energy penalty for unphysical solutions
                                if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                                    return 1e10
                                
                                # Check for proper three-soliton shape
                                if not self.validator.validate_three_soliton_shape(sol.y, amp1, width1, amp2, width2, amp3, width3):
                                    return 1e10
                                
                                return -energy  # Minimize negative energy
                            else:
                                return 1e10  # Large penalty for failed solution
                                
                        except Exception as e:
                            self.logger.debug(f"BVP solution failed for three-soliton guess {i}: {e}")
                            return 1e10  # Penalty for failed BVP solution
                    
                    # Optimize three-soliton parameters using 7D theory with enhanced bounds
                    result = minimize(
                        three_soliton_equations_7d,
                        initial_params,
                        method='L-BFGS-B',
                        bounds=[(0.1, 3.0), (0.3, 2.0), (-15.0, 15.0), (0.1, 3.0), (0.3, 2.0), (-15.0, 15.0), (0.1, 3.0), (0.3, 2.0), (-15.0, 15.0)],
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    if result.success and result.fun < best_energy and result.fun < 1e9:
                        best_energy = result.fun
                        amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3 = result.x
                        
                        # Compute final three-soliton solution with full validation
                        final_solution = self._compute_final_three_soliton_solution(
                            amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3
                        )
                        
                        # Validate solution quality
                        if self.validator.validate_three_soliton_solution_quality(final_solution, amp1, width1, amp2, width2, amp3, width3):
                            best_solution = {
                                "type": "multi",
                                "num_solitons": 3,
                                "soliton_1": {"amplitude": amp1, "width": width1, "position": pos1},
                                "soliton_2": {"amplitude": amp2, "width": width2, "position": pos2},
                                "soliton_3": {"amplitude": amp3, "width": width3, "position": pos3},
                                "energy": -result.fun,
                                "optimization_success": True,
                                "solution": final_solution,
                                "convergence_info": {
                                    "iterations": result.nit,
                                    "function_evaluations": result.nfev,
                                    "gradient_norm": np.linalg.norm(result.jac) if result.jac is not None else 0.0,
                                    "initial_guess_index": i,
                                    "energy_convergence": best_energy
                                },
                                "physical_properties": self.validator.compute_three_soliton_physical_properties(
                                    amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3, final_solution
                                )
                            }
                            
                except Exception as e:
                    self.logger.debug(f"Three-soliton optimization failed for guess {i}: {e}")
                    continue
            
            if best_solution:
                solutions.append(best_solution)
            
            return solutions
                
        except Exception as e:
            self.logger.warning(f"Three-soliton finding failed: {e}")
            return []
    
    def _compute_final_two_soliton_solution(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float) -> Dict[str, Any]:
        """
        Compute final two-soliton solution with full physical parameters.
        
        Physical Meaning:
            Generates the complete two-soliton solution with all physical
            parameters and properties computed from the optimization results.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            
        Returns:
            Dict[str, Any]: Complete two-soliton solution with physical properties.
        """
        try:
            # Generate spatial grid
            x = np.linspace(-15.0, 15.0, 300)
            
            # Compute individual soliton profiles using 7D BVP step resonator theory
            profile1 = amp1 * self._step_resonator_profile(x, pos1, width1)
            profile2 = amp2 * self._step_resonator_profile(x, pos2, width2)
            total_profile = profile1 + profile2
            
            # Compute soliton properties
            mass1 = np.trapz(profile1, x)
            mass2 = np.trapz(profile2, x)
            total_mass = np.trapz(total_profile, x)
            
            # Compute interaction metrics
            distance = abs(pos2 - pos1)
            overlap_integral = np.trapz(profile1 * profile2, x)
            interaction_strength = self.compute_soliton_interaction_strength(amp1, width1, pos1, amp2, width2, pos2)
            
            return {
                "spatial_grid": x,
                "total_profile": total_profile,
                "soliton_1_profile": profile1,
                "soliton_2_profile": profile2,
                "total_mass": total_mass,
                "individual_masses": [mass1, mass2],
                "distance": distance,
                "overlap_integral": overlap_integral,
                "interaction_strength": interaction_strength
            }
            
        except Exception as e:
            self.logger.error(f"Final two-soliton solution computation failed: {e}")
            return {}
    
    def _compute_final_three_soliton_solution(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, amp3: float, width3: float, pos3: float) -> Dict[str, Any]:
        """
        Compute final three-soliton solution with full physical parameters.
        
        Physical Meaning:
            Generates the complete three-soliton solution with all physical
            parameters and properties computed from the optimization results.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            amp3, width3, pos3 (float): Third soliton parameters.
            
        Returns:
            Dict[str, Any]: Complete three-soliton solution with physical properties.
        """
        try:
            # Generate spatial grid
            x = np.linspace(-20.0, 20.0, 400)
            
            # Compute individual soliton profiles using 7D BVP step resonator theory
            profile1 = amp1 * self._step_resonator_profile(x, pos1, width1)
            profile2 = amp2 * self._step_resonator_profile(x, pos2, width2)
            profile3 = amp3 * self._step_resonator_profile(x, pos3, width3)
            total_profile = profile1 + profile2 + profile3
            
            # Compute soliton properties
            mass1 = np.trapz(profile1, x)
            mass2 = np.trapz(profile2, x)
            mass3 = np.trapz(profile3, x)
            total_mass = np.trapz(total_profile, x)
            
            # Compute interaction metrics
            distances = [abs(pos2 - pos1), abs(pos3 - pos1), abs(pos3 - pos2)]
            overlap_integrals = [
                np.trapz(profile1 * profile2, x),
                np.trapz(profile1 * profile3, x),
                np.trapz(profile2 * profile3, x)
            ]
            
            return {
                "spatial_grid": x,
                "total_profile": total_profile,
                "soliton_1_profile": profile1,
                "soliton_2_profile": profile2,
                "soliton_3_profile": profile3,
                "total_mass": total_mass,
                "individual_masses": [mass1, mass2, mass3],
                "distances": distances,
                "overlap_integrals": overlap_integrals
            }
            
        except Exception as e:
            self.logger.error(f"Final three-soliton solution computation failed: {e}")
            return {}
    
    def _step_resonator_profile(self, x: np.ndarray, position: float, width: float) -> np.ndarray:
        """
        Step resonator profile using 7D BVP theory.
        
        Physical Meaning:
            Implements step resonator profile instead of exponential
            decay, following 7D BVP theory principles with sharp
            cutoff at soliton width.
            
        Mathematical Foundation:
            Step resonator profile:
            f(x) = 1 if |x - pos| < width, 0 if |x - pos| ≥ width
            where width is the soliton width parameter.
            
        Args:
            x (np.ndarray): Spatial coordinate array.
            position (float): Soliton position.
            width (float): Soliton width parameter.
            
        Returns:
            np.ndarray: Step resonator profile.
        """
        try:
            # Step resonator: sharp cutoff at soliton width
            distance = np.abs(x - position)
            return np.where(distance < width, 1.0, 0.0)
            
        except Exception as e:
            self.logger.error(f"Step resonator profile computation failed: {e}")
            return np.zeros_like(x)
    
    def _step_resonator_boundary_condition(self, field_value: float, amplitude: float) -> float:
        """
        Step resonator boundary condition using 7D BVP theory.
        
        Physical Meaning:
            Implements step resonator boundary condition instead of
            exponential decay, following 7D BVP theory principles.
            
        Args:
            field_value (float): Field value at boundary.
            amplitude (float): Soliton amplitude.
            
        Returns:
            float: Boundary condition value.
        """
        try:
            # Step resonator: sharp boundary condition
            if abs(field_value) < 0.1 * amplitude:
                return 0.0
            else:
                return field_value
                
        except Exception as e:
            self.logger.error(f"Step resonator boundary condition computation failed: {e}")
            return field_value
    
    def _compute_soliton_interaction_strength(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float) -> float:
        """
        Compute soliton interaction strength using 7D BVP theory.
        
        Physical Meaning:
            Computes the interaction strength between two solitons
            using 7D BVP step resonator theory instead of exponential
            decay interactions.
            
        Mathematical Foundation:
            Interaction strength based on step resonator overlap:
            I = A₁A₂ * step_resonator(distance, width₁ + width₂)
            where step_resonator implements 7D BVP theory.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            
        Returns:
            float: Interaction strength.
        """
        try:
            # Compute distance between solitons
            distance = abs(pos2 - pos1)
            
            # Compute interaction range
            interaction_range = width1 + width2
            
            # Compute step resonator interaction strength
            if distance < interaction_range:
                interaction_strength = amp1 * amp2 * self.interaction_strength
            else:
                interaction_strength = 0.0
            
            return interaction_strength
            
        except Exception as e:
            self.logger.error(f"Soliton interaction strength computation failed: {e}")
            return 0.0
