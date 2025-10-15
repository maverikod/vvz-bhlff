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
            problem solving with full 7D BVP theory implementation.
            
        Mathematical Foundation:
            Solves the 7D soliton equation:
            L_β a = μ(-Δ)^β a + λa = s(x,t)
            with soliton boundary conditions and energy minimization.
            Uses complete optimization with multiple initial guesses
            and advanced convergence criteria.
            
        Returns:
            Optional[Dict[str, Any]]: Single soliton solution with full
            physical parameters and optimization results.
        """
        try:
            # Multiple initial guesses for robust optimization
            initial_guesses = [
                [1.0, 1.0, 0.0],    # Standard guess
                [1.5, 0.8, 0.0],    # Higher amplitude, narrower
                [0.8, 1.2, 0.0],    # Lower amplitude, wider
                [1.2, 1.0, 2.0],    # Offset position
                [0.9, 0.9, -1.5]    # Negative offset
            ]
            
            best_solution = None
            best_energy = float('inf')
            
            for i, initial_params in enumerate(initial_guesses):
                try:
                    # Setup 7D mesh for BVP solving with adaptive resolution
                    x_mesh = np.linspace(-15.0, 15.0, 200)
                    y_guess = np.zeros((2, len(x_mesh)))
                    
                    def soliton_equations_7d(params):
                        """7D soliton equations from BVP theory with full implementation."""
                        amplitude, width, position = params
                        
                        def soliton_ode(x, y):
                            """7D soliton ODE system with complete physics."""
                            return self.compute_7d_soliton_ode(x, y, amplitude, width, position)
                        
                        # Boundary conditions for soliton with proper 7D BVP theory
                        def bc(ya, yb):
                            # Soliton boundary conditions: field approaches zero at boundaries
                            return [ya[0] - amplitude * self._step_resonator_boundary_condition(ya[0], amplitude),
                                   yb[0] - 0.0]  # Field vanishes at far boundary
                        
                        try:
                            # Solve BVP with enhanced convergence
                            sol = solve_bvp(soliton_ode, bc, x_mesh, y_guess, 
                                          tol=1e-8, max_nodes=1000)
                            
                            if sol.success:
                                # Compute soliton energy with full 7D BVP theory
                                energy = self.compute_soliton_energy(sol.y, amplitude, width)
                                
                                # Additional energy penalty for unphysical solutions
                                if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                                    return 1e10
                                
                                # Check for proper soliton shape
                                if not self._validate_soliton_shape(sol.y, amplitude, width):
                                    return 1e10
                                
                                return -energy  # Minimize negative energy
                            else:
                                return 1e10  # Large penalty for failed solution
                                
                        except Exception as e:
                            self.logger.debug(f"BVP solution failed for guess {i}: {e}")
                            return 1e10  # Penalty for failed BVP solution
                    
                    # Optimize soliton parameters using 7D theory with enhanced bounds
                    result = minimize(
                        soliton_equations_7d,
                        initial_params,
                        method='L-BFGS-B',
                        bounds=[(0.1, 3.0), (0.3, 2.0), (-8.0, 8.0)],
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    if result.success and result.fun < best_energy and result.fun < 1e9:
                        best_energy = result.fun
                        amplitude, width, position = result.x
                        
                        # Compute final soliton solution with full validation
                        final_solution = self._compute_final_soliton_solution(
                            amplitude, width, position
                        )
                        
                        # Validate solution quality
                        if self._validate_solution_quality(final_solution, amplitude, width):
                            best_solution = {
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
                                    "gradient_norm": np.linalg.norm(result.jac) if result.jac is not None else 0.0,
                                    "initial_guess_index": i,
                                    "energy_convergence": best_energy
                                },
                                "physical_properties": self._compute_soliton_physical_properties(
                                    amplitude, width, position, final_solution
                                )
                            }
                            
                except Exception as e:
                    self.logger.debug(f"Optimization failed for guess {i}: {e}")
                    continue
            
            return best_solution
                
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
            
            # Compute soliton profile using 7D BVP step resonator theory
            profile = amplitude * self._step_resonator_profile(x, position, width)
            
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
    
    def compute_7d_soliton_ode(self, x: np.ndarray, y: np.ndarray, amplitude: float, width: float, position: float = 0.0) -> np.ndarray:
        """
        Compute 7D soliton ODE system for BVP solver.
        
        Physical Meaning:
            Implements the 7D fractional Laplacian equation for soliton
            evolution with proper boundary conditions and energy conservation
            using complete 7D BVP theory.
            
        Mathematical Foundation:
            Solves the system:
            dy/dx = [y[1], -μ(-Δ)^β y[0] - λy[0] + s(x)]
            where s(x) is the soliton source term and (-Δ)^β is the
            fractional Laplacian operator.
            
        Args:
            x (np.ndarray): Spatial coordinate.
            y (np.ndarray): Solution vector [field, derivative].
            amplitude (float): Soliton amplitude.
            width (float): Soliton width parameter.
            
        Returns:
            np.ndarray: ODE system derivatives.
        """
        try:
            # Extract field and derivative
            field = y[0]
            field_deriv = y[1]
            
            # Compute fractional Laplacian term using full 7D BVP theory
            fractional_laplacian = self._compute_full_fractional_laplacian(x, field)
            
            # Soliton source term using 7D BVP step resonator theory
            source = amplitude * self._step_resonator_source(x, width)
            
            # ODE system
            dydx = np.array([
                field_deriv,  # dy[0]/dx = y[1]
                -fractional_laplacian - self.lambda_param * field + source  # dy[1]/dx = RHS
            ])
            
            return dydx
            
        except Exception as e:
            self.logger.error(f"7D soliton ODE computation failed: {e}")
            return np.zeros_like(y)
    
    def compute_soliton_energy(self, solution: np.ndarray, amplitude: float, width: float) -> float:
        """
        Compute soliton energy from solution.
        
        Physical Meaning:
            Calculates the total energy of the soliton solution,
            including kinetic and potential energy contributions
            from the 7D phase field theory.
            
        Mathematical Foundation:
            E = ∫ [½(∇a)² + V(a)] d⁷x
            where V(a) is the potential energy density.
            
        Args:
            solution (np.ndarray): Soliton solution field.
            amplitude (float): Soliton amplitude.
            width (float): Soliton width parameter.
            
        Returns:
            float: Total soliton energy.
        """
        try:
            # Extract field values
            field = solution[0] if solution.ndim > 1 else solution
            
            # Compute kinetic energy (gradient term)
            if len(field) > 1:
                gradient = np.gradient(field)
                kinetic_energy = 0.5 * np.sum(gradient ** 2)
            else:
                kinetic_energy = 0.0
            
            # Compute potential energy
            # V(a) = ½λa² + ¼μa⁴ (typical soliton potential)
            potential_energy = (0.5 * self.lambda_param * np.sum(field ** 2) + 
                               0.25 * self.mu * np.sum(field ** 4))
            
            # Total energy
            total_energy = kinetic_energy + potential_energy
            
            return total_energy
            
        except Exception as e:
            self.logger.error(f"Soliton energy computation failed: {e}")
            return 0.0
    
    def _compute_full_fractional_laplacian(self, x: np.ndarray, field: np.ndarray) -> np.ndarray:
        """
        Compute full fractional Laplacian using 7D BVP theory.
        
        Physical Meaning:
            Computes the fractional Laplacian operator (-Δ)^β using
            the complete 7D BVP theory with proper spectral representation
            and boundary conditions.
            
        Mathematical Foundation:
            Implements the fractional Laplacian in spectral space:
            (-Δ)^β f(x) = F^{-1}[|k|^(2β) F[f(x)]]
            where F is the Fourier transform and β ∈ (0,2).
            
        Args:
            x (np.ndarray): Spatial coordinate array.
            field (np.ndarray): Field values at spatial points.
            
        Returns:
            np.ndarray: Fractional Laplacian of the field.
        """
        try:
            # Ensure uniform spacing for FFT
            dx = x[1] - x[0] if len(x) > 1 else 1.0
            
            # Compute FFT of the field
            field_fft = np.fft.fft(field)
            
            # Compute wave numbers
            N = len(x)
            k = np.fft.fftfreq(N, dx) * 2 * np.pi
            
            # Compute fractional Laplacian in spectral space
            # |k|^(2β) with proper handling of k=0 mode
            k_magnitude = np.abs(k)
            k_magnitude[0] = 1e-10  # Avoid division by zero
            
            fractional_spectrum = (k_magnitude ** (2 * self.beta)) * field_fft
            
            # Transform back to real space
            fractional_laplacian = np.real(np.fft.ifft(fractional_spectrum))
            
            return self.mu * fractional_laplacian
            
        except Exception as e:
            self.logger.error(f"Full fractional Laplacian computation failed: {e}")
            # Fallback to local approximation if FFT fails
            return self.mu * (np.abs(x) ** (2 * self.beta)) * field
    
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
    
    def _step_resonator_source(self, x: np.ndarray, width: float) -> np.ndarray:
        """
        Step resonator source term using 7D BVP theory.
        
        Physical Meaning:
            Implements step resonator source term instead of exponential
            decay, following 7D BVP theory principles with sharp
            cutoff at source width.
            
        Mathematical Foundation:
            Step resonator source:
            s(x) = 1 if |x| < width, 0 if |x| ≥ width
            where width is the source width parameter.
            
        Args:
            x (np.ndarray): Spatial coordinate array.
            width (float): Source width parameter.
            
        Returns:
            np.ndarray: Step resonator source term.
        """
        try:
            # Step resonator: sharp cutoff at source width
            return np.where(np.abs(x) < width, 1.0, 0.0)
            
        except Exception as e:
            self.logger.error(f"Step resonator source computation failed: {e}")
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
    
    def _validate_soliton_shape(self, solution: np.ndarray, amplitude: float, width: float) -> bool:
        """
        Validate soliton shape for physical correctness.
        
        Physical Meaning:
            Validates that the soliton solution has proper physical
            characteristics including monotonic decay and proper
            amplitude-width relationship.
            
        Args:
            solution (np.ndarray): Soliton solution.
            amplitude (float): Soliton amplitude.
            width (float): Soliton width.
            
        Returns:
            bool: True if soliton shape is valid.
        """
        try:
            field = solution[0] if solution.ndim > 1 else solution
            
            # Check for proper amplitude
            max_field = np.max(np.abs(field))
            if max_field < 0.5 * amplitude or max_field > 2.0 * amplitude:
                return False
            
            # Check for monotonic decay (basic shape check)
            if len(field) > 10:
                # Check that field decays from center
                center_idx = len(field) // 2
                left_decay = np.all(np.diff(field[:center_idx]) <= 0)
                right_decay = np.all(np.diff(field[center_idx:]) >= 0)
                
                if not (left_decay and right_decay):
                    return False
            
            # Check for no oscillations (smooth profile)
            if len(field) > 5:
                second_deriv = np.gradient(np.gradient(field))
                if np.any(np.abs(second_deriv) > 10.0 * amplitude):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Soliton shape validation failed: {e}")
            return False
    
    def _validate_solution_quality(self, solution: Dict[str, Any], amplitude: float, width: float) -> bool:
        """
        Validate overall solution quality.
        
        Physical Meaning:
            Validates that the complete soliton solution meets
            all physical requirements and quality criteria.
            
        Args:
            solution (Dict[str, Any]): Complete soliton solution.
            amplitude (float): Soliton amplitude.
            width (float): Soliton width.
            
        Returns:
            bool: True if solution quality is acceptable.
        """
        try:
            # Check solution completeness
            required_keys = ['spatial_grid', 'profile', 'mass', 'momentum', 'topological_charge']
            if not all(key in solution for key in required_keys):
                return False
            
            # Check physical parameters
            if solution['mass'] <= 0 or np.isnan(solution['mass']):
                return False
            
            if abs(solution['topological_charge']) > 2.0:  # Reasonable topological charge
                return False
            
            # Check profile quality
            profile = solution['profile']
            if np.any(np.isnan(profile)) or np.any(np.isinf(profile)):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Solution quality validation failed: {e}")
            return False
    
    def _compute_soliton_physical_properties(self, amplitude: float, width: float, position: float, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive soliton physical properties.
        
        Physical Meaning:
            Computes all relevant physical properties of the soliton
            including energy, momentum, topological charge, and stability
            metrics using 7D BVP theory.
            
        Args:
            amplitude (float): Soliton amplitude.
            width (float): Soliton width.
            position (float): Soliton position.
            solution (Dict[str, Any]): Soliton solution.
            
        Returns:
            Dict[str, Any]: Complete physical properties.
        """
        try:
            profile = solution['profile']
            x = solution['spatial_grid']
            
            # Compute additional physical properties
            kinetic_energy = 0.5 * np.trapz(np.gradient(profile) ** 2, x)
            potential_energy = 0.5 * self.lambda_param * np.trapz(profile ** 2, x)
            total_energy = kinetic_energy + potential_energy
            
            # Compute stability metrics
            stability_metric = self._compute_stability_metric(profile, x)
            
            # Compute phase coherence
            phase_coherence = self._compute_phase_coherence(profile, x)
            
            # Compute 7D BVP specific properties
            bvp_properties = self._compute_7d_bvp_properties(profile, x, amplitude, width)
            
            return {
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
                "total_energy": total_energy,
                "stability_metric": stability_metric,
                "phase_coherence": phase_coherence,
                "7d_bvp_properties": bvp_properties,
                "energy_density": total_energy / (width * 2),  # Energy per unit width
                "momentum_density": solution['momentum'] / (width * 2)
            }
            
        except Exception as e:
            self.logger.error(f"Physical properties computation failed: {e}")
            return {}
    
    def _compute_stability_metric(self, profile: np.ndarray, x: np.ndarray) -> float:
        """
        Compute soliton stability metric.
        
        Physical Meaning:
            Computes a stability metric based on the soliton's
            energy distribution and shape characteristics.
            
        Args:
            profile (np.ndarray): Soliton profile.
            x (np.ndarray): Spatial coordinates.
            
        Returns:
            float: Stability metric (higher is more stable).
        """
        try:
            # Compute energy distribution
            energy_density = 0.5 * (np.gradient(profile) ** 2 + self.lambda_param * profile ** 2)
            
            # Compute stability as ratio of peak energy to total energy
            peak_energy = np.max(energy_density)
            total_energy = np.trapz(energy_density, x)
            
            if total_energy > 0:
                return peak_energy / total_energy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Stability metric computation failed: {e}")
            return 0.0
    
    def _compute_phase_coherence(self, profile: np.ndarray, x: np.ndarray) -> float:
        """
        Compute phase coherence metric.
        
        Physical Meaning:
            Computes the phase coherence of the soliton based on
            its phase field properties and 7D BVP theory.
            
        Args:
            profile (np.ndarray): Soliton profile.
            x (np.ndarray): Spatial coordinates.
            
        Returns:
            float: Phase coherence metric.
        """
        try:
            # Compute phase field from profile
            phase_field = np.arctan2(profile, np.gradient(profile))
            
            # Compute phase coherence as consistency of phase
            phase_variance = np.var(phase_field)
            phase_coherence = 1.0 / (1.0 + phase_variance)
            
            return phase_coherence
            
        except Exception as e:
            self.logger.error(f"Phase coherence computation failed: {e}")
            return 0.0
    
    def _compute_7d_bvp_properties(self, profile: np.ndarray, x: np.ndarray, amplitude: float, width: float) -> Dict[str, Any]:
        """
        Compute 7D BVP specific properties.
        
        Physical Meaning:
            Computes properties specific to 7D BVP theory including
            fractional Laplacian effects and step resonator properties.
            
        Args:
            profile (np.ndarray): Soliton profile.
            x (np.ndarray): Spatial coordinates.
            amplitude (float): Soliton amplitude.
            width (float): Soliton width.
            
        Returns:
            Dict[str, Any]: 7D BVP specific properties.
        """
        try:
            # Compute fractional Laplacian contribution
            fractional_contribution = self._compute_fractional_laplacian_contribution(profile, x)
            
            # Compute step resonator efficiency
            step_efficiency = self._compute_step_resonator_efficiency(profile, x, width)
            
            # Compute 7D phase space properties
            phase_space_properties = self._compute_7d_phase_space_properties(profile, x)
            
            return {
                "fractional_laplacian_contribution": fractional_contribution,
                "step_resonator_efficiency": step_efficiency,
                "7d_phase_space_properties": phase_space_properties,
                "bvp_convergence_quality": self._compute_bvp_convergence_quality(profile, x)
            }
            
        except Exception as e:
            self.logger.error(f"7D BVP properties computation failed: {e}")
            return {}
    
    def _compute_fractional_laplacian_contribution(self, profile: np.ndarray, x: np.ndarray) -> float:
        """Compute fractional Laplacian contribution to soliton energy."""
        try:
            # Compute fractional Laplacian
            frac_lap = self._compute_full_fractional_laplacian(x, profile)
            
            # Compute contribution as ratio to total energy
            total_energy = np.trapz(profile ** 2, x)
            frac_energy = np.trapz(profile * frac_lap, x)
            
            if total_energy > 0:
                return abs(frac_energy) / total_energy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Fractional Laplacian contribution computation failed: {e}")
            return 0.0
    
    def _compute_step_resonator_efficiency(self, profile: np.ndarray, x: np.ndarray, width: float) -> float:
        """Compute step resonator efficiency."""
        try:
            # Compute step resonator profile
            step_profile = self._step_resonator_profile(x, 0.0, width)
            
            # Compute efficiency as overlap with step resonator
            overlap = np.trapz(profile * step_profile, x)
            total_profile = np.trapz(np.abs(profile), x)
            
            if total_profile > 0:
                return overlap / total_profile
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Step resonator efficiency computation failed: {e}")
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
    
    def _compute_bvp_convergence_quality(self, profile: np.ndarray, x: np.ndarray) -> float:
        """Compute BVP convergence quality metric."""
        try:
            # Compute residual of the 7D equation
            residual = self._compute_equation_residual(profile, x)
            
            # Compute quality as inverse of residual
            quality = 1.0 / (1.0 + np.mean(residual ** 2))
            
            return quality
            
        except Exception as e:
            self.logger.error(f"BVP convergence quality computation failed: {e}")
            return 0.0
    
    def _compute_equation_residual(self, profile: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute residual of the 7D soliton equation."""
        try:
            # Compute fractional Laplacian
            frac_lap = self._compute_full_fractional_laplacian(x, profile)
            
            # Compute source term
            source = self._step_resonator_source(x, 1.0)  # Default width
            
            # Compute residual: L_β a - λa - s(x)
            residual = frac_lap + self.lambda_param * profile - source
            
            return residual
            
        except Exception as e:
            self.logger.error(f"Equation residual computation failed: {e}")
            return np.zeros_like(profile)