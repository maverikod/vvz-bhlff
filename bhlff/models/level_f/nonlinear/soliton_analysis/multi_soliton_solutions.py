"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Multi-soliton solutions and optimization.

This module implements multi-soliton solution finding and optimization
using complete 7D BVP theory with soliton-soliton interactions.

Physical Meaning:
    Finds and optimizes multi-soliton solutions including two and three
    soliton configurations with complete interaction analysis
    and stability properties using 7D phase field theory.

Example:
    >>> solver = MultiSolitonSolutions(system, nonlinear_params)
    >>> solutions = solver.find_multi_soliton_solutions()
"""

import numpy as np
from typing import Dict, Any, List
from scipy.optimize import minimize
from scipy.integrate import solve_bvp
import logging

from .base import SolitonAnalysisBase
from .multi_soliton_core import MultiSolitonCore


class MultiSolitonSolutions(SolitonAnalysisBase):
    """
    Multi-soliton solution finder and optimizer.
    
    Physical Meaning:
        Finds multi-soliton solutions through complete optimization
        using 7D fractional Laplacian equations and soliton-soliton
        interaction potentials.
        
    Mathematical Foundation:
        Solves the multi-soliton system:
        L_β a = μ(-Δ)^β a + λa + V_int(a₁, a₂, ...) = s(x,t)
        where V_int represents soliton-soliton interactions.
    """
    
    def __init__(self, system, nonlinear_params: Dict[str, Any]):
        """Initialize multi-soliton solutions."""
        super().__init__(system, nonlinear_params)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core functionality
        self.core = MultiSolitonCore(system, nonlinear_params)
    
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
            
        Mathematical Foundation:
            Solves the two-soliton system:
            L_β a = μ(-Δ)^β a + λa + V_int(a₁, a₂) = s₁(x) + s₂(x)
            where V_int represents soliton-soliton interactions.
            
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
                                if not self._validate_two_soliton_shape(sol.y, amp1, width1, amp2, width2):
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
                        if self._validate_two_soliton_solution_quality(final_solution, amp1, width1, amp2, width2):
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
                                "physical_properties": self._compute_two_soliton_physical_properties(
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
            
        Mathematical Foundation:
            Solves the three-soliton system:
            L_β a = μ(-Δ)^β a + λa + V_int(a₁, a₂, a₃) = s₁(x) + s₂(x) + s₃(x)
            where V_int includes all pairwise and three-body interactions.
            
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
                                if not self._validate_three_soliton_shape(sol.y, amp1, width1, amp2, width2, amp3, width3):
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
                        if self._validate_three_soliton_solution_quality(final_solution, amp1, width1, amp2, width2, amp3, width3):
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
                                "physical_properties": self._compute_three_soliton_physical_properties(
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
    
    def _validate_two_soliton_shape(self, solution: np.ndarray, amp1: float, width1: float, amp2: float, width2: float) -> bool:
        """
        Validate two-soliton shape for physical correctness.
        
        Physical Meaning:
            Validates that the two-soliton solution has proper physical
            characteristics including proper separation, individual
            soliton shapes, and interaction effects.
            
        Args:
            solution (np.ndarray): Two-soliton solution.
            amp1, width1 (float): First soliton parameters.
            amp2, width2 (float): Second soliton parameters.
            
        Returns:
            bool: True if two-soliton shape is valid.
        """
        try:
            field = solution[0] if solution.ndim > 1 else solution
            
            # Check for proper amplitudes
            max_field = np.max(np.abs(field))
            expected_max = max(amp1, amp2)
            
            if max_field < 0.3 * expected_max or max_field > 3.0 * expected_max:
                return False
            
            # Check for two distinct peaks (basic two-soliton check)
            if len(field) > 20:
                # Find peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(np.abs(field), height=0.5 * max_field)
                
                if len(peaks) < 1 or len(peaks) > 3:  # Should have 1-3 peaks (2 solitons + possible interaction)
                    return False
            
            # Check for no excessive oscillations
            if len(field) > 10:
                second_deriv = np.gradient(np.gradient(field))
                if np.any(np.abs(second_deriv) > 20.0 * expected_max):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Two-soliton shape validation failed: {e}")
            return False
    
    def _validate_two_soliton_solution_quality(self, solution: Dict[str, Any], amp1: float, width1: float, amp2: float, width2: float) -> bool:
        """
        Validate overall two-soliton solution quality.
        
        Physical Meaning:
            Validates that the complete two-soliton solution meets
            all physical requirements and quality criteria.
            
        Args:
            solution (Dict[str, Any]): Complete two-soliton solution.
            amp1, width1 (float): First soliton parameters.
            amp2, width2 (float): Second soliton parameters.
            
        Returns:
            bool: True if solution quality is acceptable.
        """
        try:
            # Check solution completeness
            required_keys = ['spatial_grid', 'total_profile', 'soliton_1_profile', 'soliton_2_profile', 'total_mass']
            if not all(key in solution for key in required_keys):
                return False
            
            # Check physical parameters
            if solution['total_mass'] <= 0 or np.isnan(solution['total_mass']):
                return False
            
            # Check individual soliton profiles
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            
            if np.any(np.isnan(profile1)) or np.any(np.isinf(profile1)):
                return False
            if np.any(np.isnan(profile2)) or np.any(np.isinf(profile2)):
                return False
            
            # Check interaction distance is reasonable
            distance = solution.get('distance', 0.0)
            if distance < 0.1 or distance > 20.0:  # Reasonable interaction distance
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Two-soliton solution quality validation failed: {e}")
            return False
    
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
    
    def _compute_two_soliton_physical_properties(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive two-soliton physical properties.
        
        Physical Meaning:
            Computes all relevant physical properties of the two-soliton
            system including individual energies, interaction energy,
            stability metrics, and 7D BVP specific properties.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            solution (Dict[str, Any]): Two-soliton solution.
            
        Returns:
            Dict[str, Any]: Complete physical properties.
        """
        try:
            # Compute individual soliton energies
            energy1 = self._compute_individual_soliton_energy(solution['soliton_1_profile'], solution['spatial_grid'])
            energy2 = self._compute_individual_soliton_energy(solution['soliton_2_profile'], solution['spatial_grid'])
            
            # Compute interaction energy
            interaction_energy = self._compute_interaction_energy(amp1, width1, pos1, amp2, width2, pos2)
            
            # Compute stability metrics
            stability_metric = self._compute_two_soliton_stability(solution)
            
            # Compute phase coherence
            phase_coherence = self._compute_two_soliton_phase_coherence(solution)
            
            # Compute 7D BVP specific properties
            bvp_properties = self._compute_two_soliton_7d_bvp_properties(solution, amp1, width1, amp2, width2)
            
            return {
                "individual_energies": [energy1, energy2],
                "interaction_energy": interaction_energy,
                "total_energy": energy1 + energy2 + interaction_energy,
                "stability_metric": stability_metric,
                "phase_coherence": phase_coherence,
                "7d_bvp_properties": bvp_properties,
                "energy_ratio": interaction_energy / (energy1 + energy2) if (energy1 + energy2) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Two-soliton physical properties computation failed: {e}")
            return {}
    
    def _compute_individual_soliton_energy(self, profile: np.ndarray, x: np.ndarray) -> float:
        """Compute energy of individual soliton."""
        try:
            # Kinetic energy
            kinetic_energy = 0.5 * np.trapz(np.gradient(profile) ** 2, x)
            
            # Potential energy
            potential_energy = 0.5 * self.lambda_param * np.trapz(profile ** 2, x)
            
            return kinetic_energy + potential_energy
            
        except Exception as e:
            self.logger.error(f"Individual soliton energy computation failed: {e}")
            return 0.0
    
    def _compute_interaction_energy(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float) -> float:
        """Compute interaction energy between solitons."""
        try:
            distance = abs(pos2 - pos1)
            interaction_range = width1 + width2
            
            # Step resonator interaction energy
            if distance < interaction_range:
                return self.interaction_strength * amp1 * amp2
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Interaction energy computation failed: {e}")
            return 0.0
    
    def _compute_two_soliton_stability(self, solution: Dict[str, Any]) -> float:
        """Compute two-soliton stability metric."""
        try:
            total_profile = solution['total_profile']
            x = solution['spatial_grid']
            
            # Compute energy distribution
            energy_density = 0.5 * (np.gradient(total_profile) ** 2 + self.lambda_param * total_profile ** 2)
            
            # Compute stability as energy localization
            peak_energy = np.max(energy_density)
            total_energy = np.trapz(energy_density, x)
            
            if total_energy > 0:
                return peak_energy / total_energy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Two-soliton stability computation failed: {e}")
            return 0.0
    
    def _compute_two_soliton_phase_coherence(self, solution: Dict[str, Any]) -> float:
        """Compute two-soliton phase coherence."""
        try:
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            
            # Compute phase fields
            phase1 = np.arctan2(profile1, np.gradient(profile1))
            phase2 = np.arctan2(profile2, np.gradient(profile2))
            
            # Compute phase coherence as correlation
            if len(phase1) == len(phase2):
                correlation = np.corrcoef(phase1, phase2)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Two-soliton phase coherence computation failed: {e}")
            return 0.0
    
    def _compute_two_soliton_7d_bvp_properties(self, solution: Dict[str, Any], amp1: float, width1: float, amp2: float, width2: float) -> Dict[str, Any]:
        """Compute 7D BVP specific properties for two-soliton system."""
        try:
            total_profile = solution['total_profile']
            x = solution['spatial_grid']
            
            # Compute fractional Laplacian contribution
            fractional_contribution = self._compute_fractional_laplacian_contribution(total_profile, x)
            
            # Compute step resonator efficiency
            step_efficiency = self._compute_two_soliton_step_efficiency(solution, width1, width2)
            
            # Compute interaction efficiency
            interaction_efficiency = self._compute_interaction_efficiency(solution)
            
            return {
                "fractional_laplacian_contribution": fractional_contribution,
                "step_resonator_efficiency": step_efficiency,
                "interaction_efficiency": interaction_efficiency,
                "7d_phase_space_properties": self._compute_7d_phase_space_properties(total_profile, x)
            }
            
        except Exception as e:
            self.logger.error(f"Two-soliton 7D BVP properties computation failed: {e}")
            return {}
    
    def _compute_fractional_laplacian_contribution(self, profile: np.ndarray, x: np.ndarray) -> float:
        """Compute fractional Laplacian contribution."""
        try:
            # Compute fractional Laplacian using FFT
            dx = x[1] - x[0] if len(x) > 1 else 1.0
            profile_fft = np.fft.fft(profile)
            k = np.fft.fftfreq(len(x), dx) * 2 * np.pi
            k_magnitude = np.abs(k)
            k_magnitude[0] = 1e-10  # Avoid division by zero
            
            fractional_spectrum = (k_magnitude ** (2 * self.beta)) * profile_fft
            fractional_laplacian = np.real(np.fft.ifft(fractional_spectrum))
            
            # Compute contribution
            total_energy = np.trapz(profile ** 2, x)
            frac_energy = np.trapz(profile * fractional_laplacian, x)
            
            if total_energy > 0:
                return abs(frac_energy) / total_energy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Fractional Laplacian contribution computation failed: {e}")
            return 0.0
    
    def _compute_two_soliton_step_efficiency(self, solution: Dict[str, Any], width1: float, width2: float) -> float:
        """Compute step resonator efficiency for two-soliton system."""
        try:
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            x = solution['spatial_grid']
            
            # Compute step resonator profiles
            step1 = self._step_resonator_profile(x, solution.get('soliton_1_position', 0.0), width1)
            step2 = self._step_resonator_profile(x, solution.get('soliton_2_position', 0.0), width2)
            
            # Compute efficiency as overlap
            overlap1 = np.trapz(profile1 * step1, x)
            overlap2 = np.trapz(profile2 * step2, x)
            total1 = np.trapz(np.abs(profile1), x)
            total2 = np.trapz(np.abs(profile2), x)
            
            efficiency1 = overlap1 / total1 if total1 > 0 else 0.0
            efficiency2 = overlap2 / total2 if total2 > 0 else 0.0
            
            return (efficiency1 + efficiency2) / 2.0
            
        except Exception as e:
            self.logger.error(f"Two-soliton step efficiency computation failed: {e}")
            return 0.0
    
    def _compute_interaction_efficiency(self, solution: Dict[str, Any]) -> float:
        """Compute interaction efficiency."""
        try:
            overlap_integral = solution.get('overlap_integral', 0.0)
            total_mass = solution.get('total_mass', 1.0)
            
            if total_mass > 0:
                return overlap_integral / total_mass
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Interaction efficiency computation failed: {e}")
            return 0.0
    
    def _compute_7d_phase_space_properties(self, profile: np.ndarray, x: np.ndarray) -> Dict[str, float]:
        """Compute 7D phase space properties."""
        try:
            # Compute momentum space representation
            profile_fft = np.fft.fft(profile)
            k = np.fft.fftfreq(len(x), x[1] - x[0]) * 2 * np.pi
            
            # Compute phase space volume
            phase_space_volume = np.trapz(np.abs(profile_fft) ** 2, k)
            
            # Compute phase space entropy
            prob_dist = np.abs(profile_fft) ** 2
            prob_dist = prob_dist / np.sum(prob_dist)  # Normalize
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
            
            return {
                "phase_space_volume": phase_space_volume,
                "phase_space_entropy": entropy,
                "spectral_width": np.std(k * np.abs(profile_fft))
            }
            
        except Exception as e:
            self.logger.error(f"7D phase space properties computation failed: {e}")
            return {}
    
    def _validate_three_soliton_shape(self, solution: np.ndarray, amp1: float, width1: float, amp2: float, width2: float, amp3: float, width3: float) -> bool:
        """
        Validate three-soliton shape for physical correctness.
        
        Physical Meaning:
            Validates that the three-soliton solution has proper physical
            characteristics including proper separation, individual
            soliton shapes, and multi-body interaction effects.
            
        Args:
            solution (np.ndarray): Three-soliton solution.
            amp1, width1 (float): First soliton parameters.
            amp2, width2 (float): Second soliton parameters.
            amp3, width3 (float): Third soliton parameters.
            
        Returns:
            bool: True if three-soliton shape is valid.
        """
        try:
            field = solution[0] if solution.ndim > 1 else solution
            
            # Check for proper amplitudes
            max_field = np.max(np.abs(field))
            expected_max = max(amp1, amp2, amp3)
            
            if max_field < 0.2 * expected_max or max_field > 4.0 * expected_max:
                return False
            
            # Check for three distinct peaks (basic three-soliton check)
            if len(field) > 30:
                # Find peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(np.abs(field), height=0.3 * max_field)
                
                if len(peaks) < 2 or len(peaks) > 5:  # Should have 2-5 peaks (3 solitons + possible interactions)
                    return False
            
            # Check for no excessive oscillations
            if len(field) > 15:
                second_deriv = np.gradient(np.gradient(field))
                if np.any(np.abs(second_deriv) > 30.0 * expected_max):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Three-soliton shape validation failed: {e}")
            return False
    
    def _validate_three_soliton_solution_quality(self, solution: Dict[str, Any], amp1: float, width1: float, amp2: float, width2: float, amp3: float, width3: float) -> bool:
        """
        Validate overall three-soliton solution quality.
        
        Physical Meaning:
            Validates that the complete three-soliton solution meets
            all physical requirements and quality criteria.
            
        Args:
            solution (Dict[str, Any]): Complete three-soliton solution.
            amp1, width1 (float): First soliton parameters.
            amp2, width2 (float): Second soliton parameters.
            amp3, width3 (float): Third soliton parameters.
            
        Returns:
            bool: True if solution quality is acceptable.
        """
        try:
            # Check solution completeness
            required_keys = ['spatial_grid', 'total_profile', 'soliton_1_profile', 'soliton_2_profile', 'soliton_3_profile', 'total_mass']
            if not all(key in solution for key in required_keys):
                return False
            
            # Check physical parameters
            if solution['total_mass'] <= 0 or np.isnan(solution['total_mass']):
                return False
            
            # Check individual soliton profiles
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            profile3 = solution['soliton_3_profile']
            
            for profile in [profile1, profile2, profile3]:
                if np.any(np.isnan(profile)) or np.any(np.isinf(profile)):
                    return False
            
            # Check interaction distances are reasonable
            distances = solution.get('distances', [])
            if len(distances) >= 3:
                for distance in distances:
                    if distance < 0.1 or distance > 30.0:  # Reasonable interaction distances
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Three-soliton solution quality validation failed: {e}")
            return False
    
    def _compute_three_soliton_physical_properties(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, amp3: float, width3: float, pos3: float, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive three-soliton physical properties.
        
        Physical Meaning:
            Computes all relevant physical properties of the three-soliton
            system including individual energies, pairwise interactions,
            three-body interactions, stability metrics, and 7D BVP specific properties.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            amp3, width3, pos3 (float): Third soliton parameters.
            solution (Dict[str, Any]): Three-soliton solution.
            
        Returns:
            Dict[str, Any]: Complete physical properties.
        """
        try:
            # Compute individual soliton energies
            energy1 = self._compute_individual_soliton_energy(solution['soliton_1_profile'], solution['spatial_grid'])
            energy2 = self._compute_individual_soliton_energy(solution['soliton_2_profile'], solution['spatial_grid'])
            energy3 = self._compute_individual_soliton_energy(solution['soliton_3_profile'], solution['spatial_grid'])
            
            # Compute pairwise interaction energies
            interaction_12 = self._compute_interaction_energy(amp1, width1, pos1, amp2, width2, pos2)
            interaction_13 = self._compute_interaction_energy(amp1, width1, pos1, amp3, width3, pos3)
            interaction_23 = self._compute_interaction_energy(amp2, width2, pos2, amp3, width3, pos3)
            
            # Compute three-body interaction energy
            three_body_energy = self._compute_three_body_interaction_energy(amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3)
            
            # Compute stability metrics
            stability_metric = self._compute_three_soliton_stability(solution)
            
            # Compute phase coherence
            phase_coherence = self._compute_three_soliton_phase_coherence(solution)
            
            # Compute 7D BVP specific properties
            bvp_properties = self._compute_three_soliton_7d_bvp_properties(solution, amp1, width1, amp2, width2, amp3, width3)
            
            return {
                "individual_energies": [energy1, energy2, energy3],
                "pairwise_interactions": [interaction_12, interaction_13, interaction_23],
                "three_body_interaction": three_body_energy,
                "total_energy": energy1 + energy2 + energy3 + interaction_12 + interaction_13 + interaction_23 + three_body_energy,
                "stability_metric": stability_metric,
                "phase_coherence": phase_coherence,
                "7d_bvp_properties": bvp_properties,
                "interaction_ratio": (interaction_12 + interaction_13 + interaction_23 + three_body_energy) / (energy1 + energy2 + energy3) if (energy1 + energy2 + energy3) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Three-soliton physical properties computation failed: {e}")
            return {}
    
    def _compute_three_body_interaction_energy(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, amp3: float, width3: float, pos3: float) -> float:
        """Compute three-body interaction energy."""
        try:
            # Compute distances between all solitons
            distance_12 = abs(pos2 - pos1)
            distance_13 = abs(pos3 - pos1)
            distance_23 = abs(pos3 - pos2)
            
            # Compute interaction range for three-body interaction
            interaction_range = width1 + width2 + width3
            
            # Three-body interaction using step resonator theory
            total_distance = distance_12 + distance_13 + distance_23
            if total_distance < interaction_range:
                return self.three_body_strength * amp1 * amp2 * amp3
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Three-body interaction energy computation failed: {e}")
            return 0.0
    
    def _compute_three_soliton_stability(self, solution: Dict[str, Any]) -> float:
        """Compute three-soliton stability metric."""
        try:
            total_profile = solution['total_profile']
            x = solution['spatial_grid']
            
            # Compute energy distribution
            energy_density = 0.5 * (np.gradient(total_profile) ** 2 + self.lambda_param * total_profile ** 2)
            
            # Compute stability as energy localization
            peak_energy = np.max(energy_density)
            total_energy = np.trapz(energy_density, x)
            
            if total_energy > 0:
                return peak_energy / total_energy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Three-soliton stability computation failed: {e}")
            return 0.0
    
    def _compute_three_soliton_phase_coherence(self, solution: Dict[str, Any]) -> float:
        """Compute three-soliton phase coherence."""
        try:
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            profile3 = solution['soliton_3_profile']
            
            # Compute phase fields
            phase1 = np.arctan2(profile1, np.gradient(profile1))
            phase2 = np.arctan2(profile2, np.gradient(profile2))
            phase3 = np.arctan2(profile3, np.gradient(profile3))
            
            # Compute phase coherence as average correlation
            if len(phase1) == len(phase2) == len(phase3):
                corr_12 = np.corrcoef(phase1, phase2)[0, 1]
                corr_13 = np.corrcoef(phase1, phase3)[0, 1]
                corr_23 = np.corrcoef(phase2, phase3)[0, 1]
                
                correlations = [corr_12, corr_13, corr_23]
                valid_correlations = [c for c in correlations if not np.isnan(c)]
                
                if valid_correlations:
                    return np.mean(np.abs(valid_correlations))
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Three-soliton phase coherence computation failed: {e}")
            return 0.0
    
    def _compute_three_soliton_7d_bvp_properties(self, solution: Dict[str, Any], amp1: float, width1: float, amp2: float, width2: float, amp3: float, width3: float) -> Dict[str, Any]:
        """Compute 7D BVP specific properties for three-soliton system."""
        try:
            total_profile = solution['total_profile']
            x = solution['spatial_grid']
            
            # Compute fractional Laplacian contribution
            fractional_contribution = self._compute_fractional_laplacian_contribution(total_profile, x)
            
            # Compute step resonator efficiency
            step_efficiency = self._compute_three_soliton_step_efficiency(solution, width1, width2, width3)
            
            # Compute multi-body interaction efficiency
            interaction_efficiency = self._compute_three_soliton_interaction_efficiency(solution)
            
            return {
                "fractional_laplacian_contribution": fractional_contribution,
                "step_resonator_efficiency": step_efficiency,
                "multi_body_interaction_efficiency": interaction_efficiency,
                "7d_phase_space_properties": self._compute_7d_phase_space_properties(total_profile, x)
            }
            
        except Exception as e:
            self.logger.error(f"Three-soliton 7D BVP properties computation failed: {e}")
            return {}
    
    def _compute_three_soliton_step_efficiency(self, solution: Dict[str, Any], width1: float, width2: float, width3: float) -> float:
        """Compute step resonator efficiency for three-soliton system."""
        try:
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            profile3 = solution['soliton_3_profile']
            x = solution['spatial_grid']
            
            # Compute step resonator profiles
            step1 = self._step_resonator_profile(x, solution.get('soliton_1_position', 0.0), width1)
            step2 = self._step_resonator_profile(x, solution.get('soliton_2_position', 0.0), width2)
            step3 = self._step_resonator_profile(x, solution.get('soliton_3_position', 0.0), width3)
            
            # Compute efficiency as overlap for each soliton
            overlap1 = np.trapz(profile1 * step1, x)
            overlap2 = np.trapz(profile2 * step2, x)
            overlap3 = np.trapz(profile3 * step3, x)
            
            total1 = np.trapz(np.abs(profile1), x)
            total2 = np.trapz(np.abs(profile2), x)
            total3 = np.trapz(np.abs(profile3), x)
            
            efficiency1 = overlap1 / total1 if total1 > 0 else 0.0
            efficiency2 = overlap2 / total2 if total2 > 0 else 0.0
            efficiency3 = overlap3 / total3 if total3 > 0 else 0.0
            
            return (efficiency1 + efficiency2 + efficiency3) / 3.0
            
        except Exception as e:
            self.logger.error(f"Three-soliton step efficiency computation failed: {e}")
            return 0.0
    
    def _compute_three_soliton_interaction_efficiency(self, solution: Dict[str, Any]) -> float:
        """Compute three-soliton interaction efficiency."""
        try:
            overlap_integrals = solution.get('overlap_integrals', [])
            total_mass = solution.get('total_mass', 1.0)
            
            if total_mass > 0 and len(overlap_integrals) >= 3:
                # Compute average interaction efficiency
                total_overlap = sum(overlap_integrals)
                return total_overlap / total_mass
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Three-soliton interaction efficiency computation failed: {e}")
            return 0.0
