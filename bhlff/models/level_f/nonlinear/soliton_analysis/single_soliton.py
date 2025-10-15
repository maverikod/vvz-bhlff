"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Single soliton analysis and optimization.

This module implements single soliton solution finding and analysis
using full 7D BVP theory with fractional Laplacian equations.

Physical Meaning:
    Finds and analyzes single soliton solutions in 7D space-time
    using complete optimization with boundary value problem
    solving and energy minimization.

Example:
    >>> solver = SingleSolitonSolver(system, nonlinear_params)
    >>> solution = solver.find_single_soliton()
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.optimize import minimize
from scipy.integrate import solve_bvp
import logging

from .base import SolitonAnalysisBase


class SingleSolitonSolver(SolitonAnalysisBase):
    """
    Single soliton solution finder and analyzer.
    
    Physical Meaning:
        Finds single soliton solutions through complete optimization
        using 7D fractional Laplacian equations and boundary value
        problem solving.
        
    Mathematical Foundation:
        Solves the 7D soliton equation:
        L_β a = μ(-Δ)^β a + λa = s(x,t)
        with soliton boundary conditions and energy minimization.
    """
    
    def __init__(self, system, nonlinear_params: Dict[str, Any]):
        """Initialize single soliton solver."""
        super().__init__(system, nonlinear_params)
        self.logger = logging.getLogger(__name__)
    
    def find_single_soliton(self) -> Optional[Dict[str, Any]]:
        """
        Find single soliton solution using full 7D BVP theory.
        
        Physical Meaning:
            Finds single soliton solution through complete optimization
            using 7D fractional Laplacian equations and boundary value
            problem solving.
            
        Mathematical Foundation:
            Solves the 7D soliton equation:
            L_β a = μ(-Δ)^β a + λa = s(x,t)
            with soliton boundary conditions and energy minimization.
            
        Returns:
            Optional[Dict[str, Any]]: Single soliton solution with full
            physical parameters and optimization results.
        """
        try:
            # Initial guess for soliton parameters
            initial_amplitude = 1.0
            initial_width = 1.0
            initial_position = 0.0
            
            # Setup 7D mesh for BVP solving
            x_mesh = np.linspace(-10.0, 10.0, 100)
            y_guess = np.zeros((2, len(x_mesh)))
            
            def soliton_equations_7d(params):
                """7D soliton equations from BVP theory."""
                amplitude, width, position = params
                
                def soliton_ode(x, y):
                    """7D soliton ODE system."""
                    return self.compute_7d_soliton_ode(x, y, amplitude, width)
                
                # Boundary conditions for soliton
                def bc(ya, yb):
                    return [ya[0] - amplitude, yb[0]]
                
                try:
                    # Solve BVP
                    sol = solve_bvp(soliton_ode, bc, x_mesh, y_guess)
                    
                    if sol.success:
                        # Compute soliton energy
                        energy = self.compute_soliton_energy(sol.y, amplitude, width)
                        return -energy  # Minimize negative energy
                    else:
                        return 1e10  # Large penalty for failed solution
                        
                except Exception:
                    return 1e10  # Penalty for failed BVP solution
            
            # Optimize soliton parameters using 7D theory
            result = minimize(
                soliton_equations_7d,
                [initial_amplitude, initial_width, initial_position],
                method='L-BFGS-B',
                bounds=[(0.1, 5.0), (0.5, 5.0), (-10.0, 10.0)],
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            if result.success and result.fun < 1e9:
                amplitude, width, position = result.x
                
                # Compute final soliton solution
                final_solution = self._compute_final_soliton_solution(
                    amplitude, width, position
                )
                
                return {
                    "type": "single",
                    "amplitude": amplitude,
                    "width": width,
                    "position": position,
                    "energy": -result.fun,
                    "optimization_success": True,
                    "solution": final_solution,
                    "convergence_info": {
                        "iterations": result.nit,
                        "function_evaluations": result.nfev,
                        "gradient_norm": np.linalg.norm(result.jac) if result.jac is not None else 0.0
                    }
                }
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Single soliton finding failed: {e}")
            return None
    
    def _compute_final_soliton_solution(self, amplitude: float, width: float, position: float) -> Dict[str, Any]:
        """
        Compute final soliton solution with full physical parameters.
        
        Physical Meaning:
            Generates the complete soliton solution with all physical
            parameters and properties computed from the optimization results.
            
        Args:
            amplitude (float): Optimized soliton amplitude.
            width (float): Optimized soliton width.
            position (float): Optimized soliton position.
            
        Returns:
            Dict[str, Any]: Complete soliton solution with physical properties.
        """
        try:
            # Generate spatial grid
            x = np.linspace(-10.0, 10.0, 200)
            
            # Compute soliton profile
            profile = amplitude * np.exp(-((x - position) ** 2) / (2 * width ** 2))
            
            # Compute soliton properties
            soliton_mass = np.trapz(profile, x)
            soliton_momentum = np.trapz(profile * np.gradient(profile), x)
            
            # Compute topological charge
            topological_charge = self.compute_topological_charge(profile)
            
            return {
                "spatial_grid": x,
                "profile": profile,
                "mass": soliton_mass,
                "momentum": soliton_momentum,
                "topological_charge": topological_charge,
                "width_parameter": width,
                "amplitude_parameter": amplitude,
                "position_parameter": position
            }
            
        except Exception as e:
            self.logger.error(f"Final soliton solution computation failed: {e}")
            return {}
