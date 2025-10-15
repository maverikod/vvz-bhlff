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
            interaction potentials.
            
        Returns:
            List[Dict[str, Any]]: Two-soliton solutions with interaction
            analysis and full physical parameters.
        """
        try:
            solutions = []
            
            # Setup 7D mesh for BVP solving
            x_mesh = np.linspace(-15.0, 15.0, 200)
            y_guess = np.zeros((2, len(x_mesh)))
            
            def two_soliton_equations_7d(params):
                """7D two-soliton equations with interaction."""
                amp1, width1, pos1, amp2, width2, pos2 = params
                
                def two_soliton_ode(x, y):
                    """7D two-soliton ODE system with interactions."""
                    return self.core.compute_7d_two_soliton_ode(x, y, amp1, width1, pos1, amp2, width2, pos2)
                
                # Boundary conditions for two solitons
                def bc(ya, yb):
                    return [ya[0] - amp1, yb[0] - amp2]
                
                try:
                    # Solve BVP
                    sol = solve_bvp(two_soliton_ode, bc, x_mesh, y_guess)
                    
                    if sol.success:
                        # Compute total energy including interaction
                        energy = self.core.compute_two_soliton_energy(sol.y, amp1, width1, pos1, amp2, width2, pos2)
                        return -energy  # Minimize negative energy
                    else:
                        return 1e10  # Large penalty for failed solution
                        
                except Exception:
                    return 1e10  # Penalty for failed BVP solution
            
            # Initial guess for two-soliton parameters
            initial_params = [1.0, 1.0, -3.0, 1.0, 1.0, 3.0]
            
            # Optimize two-soliton parameters using 7D theory
            result = minimize(
                two_soliton_equations_7d,
                initial_params,
                method='L-BFGS-B',
                bounds=[(0.1, 3.0), (0.5, 3.0), (-10.0, 10.0), (0.1, 3.0), (0.5, 3.0), (-10.0, 10.0)],
                options={'maxiter': 150, 'ftol': 1e-9}
            )
            
            if result.success and result.fun < 1e9:
                amp1, width1, pos1, amp2, width2, pos2 = result.x
                
                # Compute final two-soliton solution
                final_solution = self._compute_final_two_soliton_solution(
                    amp1, width1, pos1, amp2, width2, pos2
                )
                
                # Compute interaction strength
                interaction_strength = self.compute_soliton_interaction_strength(
                    amp1, width1, pos1, amp2, width2, pos2
                )
                
                solution = {
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
                        "gradient_norm": np.linalg.norm(result.jac) if result.jac is not None else 0.0
                    }
                }
                
                solutions.append(solution)
            
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
            interaction potentials.
            
        Returns:
            List[Dict[str, Any]]: Three-soliton solutions with full
            interaction analysis and stability properties.
        """
        try:
            solutions = []
            
            # Setup 7D mesh for BVP solving
            x_mesh = np.linspace(-20.0, 20.0, 300)
            y_guess = np.zeros((2, len(x_mesh)))
            
            def three_soliton_equations_7d(params):
                """7D three-soliton equations with interactions."""
                amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3 = params
                
                def three_soliton_ode(x, y):
                    """7D three-soliton ODE system with interactions."""
                    return self.core.compute_7d_three_soliton_ode(x, y, amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3)
                
                # Boundary conditions for three solitons
                def bc(ya, yb):
                    return [ya[0] - amp1, yb[0] - amp3]
                
                try:
                    # Solve BVP
                    sol = solve_bvp(three_soliton_ode, bc, x_mesh, y_guess)
                    
                    if sol.success:
                        # Compute total energy including all interactions
                        energy = self.core.compute_three_soliton_energy(
                            sol.y, amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3
                        )
                        return -energy  # Minimize negative energy
                    else:
                        return 1e10  # Large penalty for failed solution
                        
                except Exception:
                    return 1e10  # Penalty for failed BVP solution
            
            # Initial guess for three-soliton parameters
            initial_params = [1.0, 1.0, -4.0, 1.0, 1.0, 0.0, 1.0, 1.0, 4.0]
            
            # Optimize three-soliton parameters using 7D theory
            result = minimize(
                three_soliton_equations_7d,
                initial_params,
                method='L-BFGS-B',
                bounds=[(0.1, 3.0), (0.5, 3.0), (-10.0, 10.0), (0.1, 3.0), (0.5, 3.0), (-10.0, 10.0), (0.1, 3.0), (0.5, 3.0), (-10.0, 10.0)],
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if result.success and result.fun < 1e9:
                amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3 = result.x
                
                # Compute final three-soliton solution
                final_solution = self._compute_final_three_soliton_solution(
                    amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3
                )
                
                solution = {
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
                        "gradient_norm": np.linalg.norm(result.jac) if result.jac is not None else 0.0
                    }
                }
                
                solutions.append(solution)
            
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
            
            # Compute individual soliton profiles
            profile1 = amp1 * np.exp(-((x - pos1) ** 2) / (2 * width1 ** 2))
            profile2 = amp2 * np.exp(-((x - pos2) ** 2) / (2 * width2 ** 2))
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
            
            # Compute individual soliton profiles
            profile1 = amp1 * np.exp(-((x - pos1) ** 2) / (2 * width1 ** 2))
            profile2 = amp2 * np.exp(-((x - pos2) ** 2) / (2 * width2 ** 2))
            profile3 = amp3 * np.exp(-((x - pos3) ** 2) / (2 * width3 ** 2))
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
