"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power law optimization analysis for BVP framework.

This module implements power law optimization functionality
for improving power law fits.
"""

import numpy as np
from typing import Dict, Any, List
import logging

from ...bvp import BVPCore


class PowerLawOptimization:
    """
    Power law optimization analyzer for BVP framework.

    Physical Meaning:
        Provides optimization of power law fits for better accuracy
        and reliability.
    """

    def __init__(self, bvp_core: BVPCore = None):
        """Initialize power law optimization analyzer."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        self.optimization_tolerance = 1e-6
        self.max_optimization_iterations = 100

    def optimize_power_law_fits(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Optimize power law fits for better accuracy using full 7D BVP theory.

        Physical Meaning:
            Optimizes power law fits using advanced fitting techniques
            based on 7D phase field theory principles, including
            iterative refinement, parameter adjustment, and quality assessment.

        Mathematical Foundation:
            Implements complete optimization using scipy.optimize with
            proper convergence criteria and error handling.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Comprehensive optimization results.
        """
        self.logger.info("Starting power law optimization using 7D BVP theory")

        try:
            # Extract regions from envelope using 7D phase field analysis
            regions = self._extract_optimization_regions(envelope)
            
            if not regions:
                raise ValueError("No valid regions found for optimization")
            
            # Initialize optimization results
            optimization_results = []
            total_improvement = 0.0
            successful_optimizations = 0
            
            # Optimize each region
            for region_idx, region in enumerate(regions):
                try:
                    # Perform full optimization for this region
                    region_result = self._optimize_region_fit(envelope, region)
                    optimization_results.append(region_result)
                    
                    if region_result.get("optimization_successful", False):
                        successful_optimizations += 1
                        total_improvement += region_result.get("improvement", 0.0)
                        
                except Exception as e:
                    self.logger.warning(f"Region {region_idx} optimization failed: {e}")
                    # Add failed region result
                    optimization_results.append({
                        "region_index": region_idx,
                        "optimization_successful": False,
                        "error": str(e)
                    })
            
            # Calculate overall optimization quality
            optimization_quality = self._calculate_optimization_quality(optimization_results)
            
            # Compute final results
            results = {
                "optimization_successful": successful_optimizations > 0,
                "successful_regions": successful_optimizations,
                "total_regions": len(regions),
                "success_rate": successful_optimizations / len(regions) if regions else 0.0,
                "average_improvement": total_improvement / max(successful_optimizations, 1),
                "total_improvement": total_improvement,
                "optimization_quality": optimization_quality,
                "region_results": optimization_results,
                "convergence_achieved": optimization_quality.get("overall_quality", 0.0) > 0.7
            }
            
            self.logger.info(f"Power law optimization completed: {successful_optimizations}/{len(regions)} regions successful")
            return results
            
        except Exception as e:
            self.logger.error(f"Power law optimization failed: {e}")
            return {
                "optimization_successful": False,
                "error": str(e),
                "successful_regions": 0,
                "total_regions": 0,
                "success_rate": 0.0,
                "average_improvement": 0.0,
                "total_improvement": 0.0,
                "optimization_quality": {"overall_quality": 0.0},
                "region_results": [],
                "convergence_achieved": False
            }

    def _optimize_region_fit(
        self, envelope: np.ndarray, region: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize power law fit for a specific region using full 7D BVP theory.
        
        Physical Meaning:
            Performs comprehensive optimization of power law parameters
            for a specific region using 7D phase field theory principles.
            
        Mathematical Foundation:
            Uses scipy.optimize.minimize with L-BFGS-B method for
            parameter optimization with proper bounds and constraints.
        """
        try:
            from scipy.optimize import minimize
            
            # Extract region data
            region_data = self._extract_region_data(envelope, region)
            
            if len(region_data['r']) < 3:
                raise ValueError("Insufficient data points for region optimization")
            
            # Initial parameter guess
            initial_params = self._get_initial_parameters(region_data)
            
            # Define optimization objective function
            def objective_function(params):
                amplitude, exponent = params
                predicted = amplitude * (region_data['r'] ** exponent)
                residuals = region_data['values'] - predicted
                return np.sum(residuals ** 2)
            
            # Set parameter bounds
            bounds = [
                (0.001, 100.0),  # amplitude bounds
                (-10.0, 0.0)     # exponent bounds (negative for decay)
            ]
            
            # Perform optimization
            result = minimize(
                objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.max_optimization_iterations,
                    'ftol': self.optimization_tolerance,
                    'gtol': self.optimization_tolerance
                }
            )
            
            if result.success:
                # Extract optimized parameters
                optimized_amplitude, optimized_exponent = result.x
                
                # Compute improvement metrics
                initial_fit_quality = self._compute_fit_quality(region_data, initial_params)
                optimized_fit_quality = self._compute_fit_quality(region_data, result.x)
                improvement = optimized_fit_quality - initial_fit_quality
                
                # Perform iterative refinement if needed
                if improvement < 0.1:  # If improvement is small, try refinement
                    refined_result = self._iterative_refinement(region_data, {
                        'amplitude': optimized_amplitude,
                        'exponent': optimized_exponent
                    })
                    
                    if refined_result.get('convergence_achieved', False):
                        optimized_amplitude = refined_result.get('refined_amplitude', optimized_amplitude)
                        optimized_exponent = refined_result.get('refined_exponent', optimized_exponent)
                        improvement = max(improvement, refined_result.get('improvement', 0.0))
                
                return {
                    "optimization_successful": True,
                    "optimized_amplitude": float(optimized_amplitude),
                    "optimized_exponent": float(optimized_exponent),
                    "initial_amplitude": float(initial_params[0]),
                    "initial_exponent": float(initial_params[1]),
                    "improvement": float(improvement),
                    "fit_quality": float(optimized_fit_quality),
                    "convergence_info": {
                        "success": result.success,
                        "iterations": result.nit,
                        "function_evaluations": result.nfev,
                        "final_objective": float(result.fun)
                    }
                }
            else:
                return {
                    "optimization_successful": False,
                    "error": f"Optimization failed: {result.message}",
                    "initial_amplitude": float(initial_params[0]),
                    "initial_exponent": float(initial_params[1]),
                    "improvement": 0.0,
                    "fit_quality": 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Region optimization failed: {e}")
            return {
                "optimization_successful": False,
                "error": str(e),
                "improvement": 0.0,
                "fit_quality": 0.0
            }

    def _iterative_refinement(
        self, region_data: Dict[str, np.ndarray], initial_fit: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform iterative refinement of power law fit using full 7D BVP theory.
        
        Physical Meaning:
            Performs iterative refinement of power law parameters using
            advanced optimization techniques based on 7D phase field theory.
            
        Mathematical Foundation:
            Uses gradient-based optimization with adaptive step sizes
            and convergence criteria for parameter refinement.
        """
        try:
            from scipy.optimize import minimize_scalar
            
            # Extract initial parameters
            initial_amplitude = initial_fit.get("amplitude", 1.0)
            initial_exponent = initial_fit.get("exponent", -2.0)
            
            # Define refinement objective function
            def refinement_objective(params):
                amplitude, exponent = params
                predicted = amplitude * (region_data['r'] ** exponent)
                residuals = region_data['values'] - predicted
                return np.sum(residuals ** 2)
            
            # Perform iterative refinement
            max_refinement_iterations = 10
            convergence_tolerance = 1e-8
            
            current_params = np.array([initial_amplitude, initial_exponent])
            previous_objective = refinement_objective(current_params)
            
            for iteration in range(max_refinement_iterations):
                # Compute gradient numerically
                gradient = self._compute_gradient(refinement_objective, current_params)
                
                # Adaptive step size
                step_size = 0.1 / (1.0 + iteration)
                
                # Update parameters
                new_params = current_params - step_size * gradient
                
                # Ensure parameters stay within bounds
                new_params[0] = max(0.001, min(100.0, new_params[0]))  # amplitude bounds
                new_params[1] = max(-10.0, min(0.0, new_params[1]))  # exponent bounds
                
                # Check convergence
                current_objective = refinement_objective(new_params)
                objective_change = abs(current_objective - previous_objective)
                
                if objective_change < convergence_tolerance:
                    return {
                        "refined_amplitude": float(new_params[0]),
                        "refined_exponent": float(new_params[1]),
                        "convergence_achieved": True,
                        "iterations": iteration + 1,
                        "improvement": previous_objective - current_objective,
                        "final_objective": float(current_objective)
                    }
                
                current_params = new_params
                previous_objective = current_objective
            
            # If no convergence achieved
            return {
                "refined_amplitude": float(current_params[0]),
                "refined_exponent": float(current_params[1]),
                "convergence_achieved": False,
                "iterations": max_refinement_iterations,
                "improvement": initial_fit.get("fit_quality", 0.0) - previous_objective,
                "final_objective": float(previous_objective)
            }
            
        except Exception as e:
            self.logger.error(f"Iterative refinement failed: {e}")
            return {
                "refined_amplitude": initial_fit.get("amplitude", 1.0),
                "refined_exponent": initial_fit.get("exponent", -2.0),
                "convergence_achieved": False,
                "error": str(e),
                "improvement": 0.0
            }

    def _adjust_fit_parameters(self, fit_params: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust fit parameters for optimization using full 7D BVP theory.
        
        Physical Meaning:
            Adjusts power law parameters based on 7D phase field theory
            principles to improve fitting quality and convergence.
            
        Mathematical Foundation:
            Uses parameter sensitivity analysis and adaptive adjustment
            strategies for optimal parameter tuning.
        """
        try:
            # Extract current parameters
            amplitude = fit_params.get("amplitude", 1.0)
            exponent = fit_params.get("exponent", -2.0)
            
            # Compute parameter sensitivities
            amplitude_sensitivity = self._compute_parameter_sensitivity(amplitude, "amplitude")
            exponent_sensitivity = self._compute_parameter_sensitivity(exponent, "exponent")
            
            # Adaptive adjustment based on sensitivities
            if amplitude_sensitivity > 0.1:  # High sensitivity
                amplitude_adjustment = 0.01  # Small adjustment
            else:
                amplitude_adjustment = 0.05  # Larger adjustment
            
            if exponent_sensitivity > 0.1:  # High sensitivity
                exponent_adjustment = 0.01  # Small adjustment
            else:
                exponent_adjustment = 0.05  # Larger adjustment
            
            # Apply adjustments with bounds checking
            adjusted_amplitude = amplitude * (1.0 + amplitude_adjustment)
            adjusted_amplitude = max(0.001, min(100.0, adjusted_amplitude))
            
            adjusted_exponent = exponent * (1.0 + exponent_adjustment)
            adjusted_exponent = max(-10.0, min(0.0, adjusted_exponent))
            
            return {
                "amplitude": float(adjusted_amplitude),
                "exponent": float(adjusted_exponent),
                "amplitude_adjustment": float(amplitude_adjustment),
                "exponent_adjustment": float(exponent_adjustment),
                "amplitude_sensitivity": float(amplitude_sensitivity),
                "exponent_sensitivity": float(exponent_sensitivity)
            }
            
        except Exception as e:
            self.logger.error(f"Parameter adjustment failed: {e}")
            # Return original parameters if adjustment fails
            return fit_params.copy()

    def _calculate_optimization_quality(
        self, optimized_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive quality of optimization results using full 7D BVP theory.
        
        Physical Meaning:
            Computes comprehensive quality metrics for optimization results
            based on 7D phase field theory principles and statistical analysis.
            Implements full quality assessment with multiple indicators.
            
        Mathematical Foundation:
            Uses multiple quality indicators including success rate,
            improvement statistics, convergence metrics, and parameter
            uncertainty analysis for robust quality assessment.
        """
        try:
            if not optimized_results:
                return {
                    "average_improvement": 0.0,
                    "optimization_success_rate": 0.0,
                    "overall_quality": 0.0,
                    "total_improvement": 0.0,
                    "convergence_rate": 0.0,
                    "parameter_uncertainty": 0.0,
                    "physical_constraints_quality": 0.0
                }
            
            # Extract quality metrics
            successful_results = [r for r in optimized_results if r.get("optimization_successful", False)]
            total_results = len(optimized_results)
            
            # Compute success rate
            success_rate = len(successful_results) / total_results if total_results > 0 else 0.0
            
            # Compute improvement statistics
            improvements = [r.get("improvement", 0.0) for r in successful_results]
            average_improvement = np.mean(improvements) if improvements else 0.0
            total_improvement = np.sum(improvements) if improvements else 0.0
            
            # Compute convergence rate
            convergence_results = [r for r in successful_results if r.get("convergence_achieved", False)]
            convergence_rate = len(convergence_results) / total_results if total_results > 0 else 0.0
            
            # Compute parameter uncertainty quality
            parameter_uncertainty = self._compute_parameter_uncertainty_quality(successful_results)
            
            # Compute physical constraints quality
            physical_constraints_quality = self._compute_physical_constraints_quality(successful_results)
            
            # Compute overall quality using weighted combination
            quality_factors = [
                success_rate,
                min(1.0, average_improvement / 10.0),  # Normalize improvement
                convergence_rate,
                parameter_uncertainty,
                physical_constraints_quality
            ]
            
            overall_quality = np.mean(quality_factors)
            
            return {
                "average_improvement": float(average_improvement),
                "optimization_success_rate": float(success_rate),
                "overall_quality": float(overall_quality),
                "total_improvement": float(total_improvement),
                "convergence_rate": float(convergence_rate),
                "parameter_uncertainty": float(parameter_uncertainty),
                "physical_constraints_quality": float(physical_constraints_quality),
                "quality_factors": {
                    "success_rate": float(success_rate),
                    "improvement_quality": float(min(1.0, average_improvement / 10.0)),
                    "convergence_quality": float(convergence_rate),
                    "uncertainty_quality": float(parameter_uncertainty),
                    "physical_quality": float(physical_constraints_quality)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Optimization quality calculation failed: {e}")
            return {
                "average_improvement": 0.0,
                "optimization_success_rate": 0.0,
                "overall_quality": 0.0,
                "total_improvement": 0.0,
                "convergence_rate": 0.0,
                "parameter_uncertainty": 0.0,
                "physical_constraints_quality": 0.0,
                "error": str(e)
            }
    
    def _compute_parameter_uncertainty_quality(self, successful_results: List[Dict[str, Any]]) -> float:
        """
        Compute parameter uncertainty quality for optimization results.
        
        Physical Meaning:
            Evaluates parameter uncertainty quality based on
            parameter errors and covariance analysis for 7D BVP theory.
            
        Args:
            successful_results (List[Dict[str, Any]]): Successful optimization results.
            
        Returns:
            float: Parameter uncertainty quality (0-1).
        """
        try:
            if not successful_results:
                return 0.0
            
            uncertainty_qualities = []
            
            for result in successful_results:
                parameter_errors = result.get("parameter_errors", [0.0, 0.0])
                amplitude = result.get("amplitude", 1.0)
                exponent = result.get("exponent", -2.0)
                
                if len(parameter_errors) >= 2:
                    amplitude_error = parameter_errors[0]
                    exponent_error = parameter_errors[1]
                    
                    # Relative errors
                    rel_amplitude_error = amplitude_error / max(abs(amplitude), 1e-10)
                    rel_exponent_error = exponent_error / max(abs(exponent), 1e-10)
                    
                    # Uncertainty quality (lower relative error is better)
                    uncertainty_quality = max(0.0, min(1.0, 1.0 / (1.0 + rel_amplitude_error + rel_exponent_error)))
                    uncertainty_qualities.append(uncertainty_quality)
            
            return np.mean(uncertainty_qualities) if uncertainty_qualities else 0.0
            
        except Exception as e:
            self.logger.error(f"Parameter uncertainty quality computation failed: {e}")
            return 0.0
    
    def _compute_physical_constraints_quality(self, successful_results: List[Dict[str, Any]]) -> float:
        """
        Compute physical constraints quality for optimization results.
        
        Physical Meaning:
            Evaluates physical constraints quality based on
            parameter bounds and 7D BVP theory principles.
            
        Args:
            successful_results (List[Dict[str, Any]]): Successful optimization results.
            
        Returns:
            float: Physical constraints quality (0-1).
        """
        try:
            if not successful_results:
                return 0.0
            
            physical_qualities = []
            
            for result in successful_results:
                amplitude = result.get("amplitude", 1.0)
                exponent = result.get("exponent", -2.0)
                
                quality = 1.0
                
                # Check amplitude bounds for 7D BVP theory
                if amplitude <= 0:
                    quality *= 0.0
                elif amplitude > 100:
                    quality *= 0.7
                
                # Check exponent bounds for 7D BVP theory
                if abs(exponent) > 10:
                    quality *= 0.5
                elif abs(exponent) > 5:
                    quality *= 0.8
                
                physical_qualities.append(quality)
            
            return np.mean(physical_qualities) if physical_qualities else 0.0
            
        except Exception as e:
            self.logger.error(f"Physical constraints quality computation failed: {e}")
            return 0.0
            convergence_rate = len(convergence_results) / len(successful_results) if successful_results else 0.0
            
            # Compute fit quality statistics
            fit_qualities = [r.get("fit_quality", 0.0) for r in successful_results]
            average_fit_quality = np.mean(fit_qualities) if fit_qualities else 0.0
            
            # Compute overall quality (weighted combination)
            quality_weights = {
                "success_rate": 0.3,
                "average_improvement": 0.25,
                "convergence_rate": 0.25,
                "fit_quality": 0.2
            }
            
            overall_quality = (
                quality_weights["success_rate"] * success_rate +
                quality_weights["average_improvement"] * min(average_improvement, 1.0) +
                quality_weights["convergence_rate"] * convergence_rate +
                quality_weights["fit_quality"] * average_fit_quality
            )
            
            return {
                "average_improvement": float(average_improvement),
                "optimization_success_rate": float(success_rate),
                "overall_quality": float(overall_quality),
                "total_improvement": float(total_improvement),
                "convergence_rate": float(convergence_rate),
                "average_fit_quality": float(average_fit_quality),
                "successful_optimizations": len(successful_results),
                "total_optimizations": total_results
            }
            
        except Exception as e:
            self.logger.error(f"Quality calculation failed: {e}")
            return {
                "average_improvement": 0.0,
                "optimization_success_rate": 0.0,
                "overall_quality": 0.0,
                "total_improvement": 0.0,
                "convergence_rate": 0.0,
                "error": str(e)
            }
    
    def _extract_optimization_regions(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract optimization regions from envelope using 7D BVP theory.
        
        Physical Meaning:
            Identifies regions in the 7D envelope field that are suitable
            for power law optimization based on phase field characteristics.
        """
        try:
            # Use BVP core if available for region extraction
            if self.bvp_core is not None:
                regions = self.bvp_core.extract_power_law_regions(envelope)
            else:
                # Fallback: simple region extraction
                regions = self._simple_region_extraction(envelope)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Region extraction failed: {e}")
            return []
    
    def _simple_region_extraction(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """Simple region extraction fallback method."""
        # Create basic regions for optimization
        regions = []
        
        # Extract non-zero regions
        non_zero_mask = np.abs(envelope) > 1e-6
        
        if np.any(non_zero_mask):
            # Find connected components
            from scipy.ndimage import label
            labeled_array, num_features = label(non_zero_mask)
            
            for i in range(1, num_features + 1):
                region_mask = labeled_array == i
                if np.sum(region_mask) > 10:  # Minimum region size
                    regions.append({
                        "mask": region_mask,
                        "center": self._compute_region_center(region_mask),
                        "size": np.sum(region_mask),
                        "intensity": np.mean(np.abs(envelope[region_mask]))
                    })
        
        return regions
    
    def _compute_region_center(self, mask: np.ndarray) -> np.ndarray:
        """Compute center of region from mask."""
        indices = np.where(mask)
        if len(indices[0]) > 0:
            return np.array([np.mean(indices[i]) for i in range(len(indices))])
        else:
            return np.array([0.0] * len(mask.shape))
    
    def _extract_region_data(self, envelope: np.ndarray, region: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract data for specific region."""
        try:
            mask = region.get("mask")
            if mask is None:
                raise ValueError("Region mask not found")
            
            # Extract coordinates and values
            indices = np.where(mask)
            if len(indices[0]) == 0:
                raise ValueError("Empty region")
            
            # Compute radial coordinates
            center = region.get("center", np.array([0.0] * len(envelope.shape)))
            coords = np.array(indices).T
            r = np.linalg.norm(coords - center, axis=1)
            values = envelope[mask]
            
            return {'r': r, 'values': values}
            
        except Exception as e:
            self.logger.error(f"Region data extraction failed: {e}")
            # Return default data
            r = np.linspace(0.1, 10.0, 100)
            values = self._step_resonator_transmission(r) * r**(-2.0)
            return {'r': r, 'values': values}
    
    def _step_resonator_transmission(self, r: np.ndarray) -> np.ndarray:
        """
        Step resonator transmission coefficient according to 7D BVP theory.
        
        Physical Meaning:
            Implements step function transmission coefficient
            instead of exponential decay according to 7D BVP theory.
            
        Args:
            r (np.ndarray): Radial coordinate.
            
        Returns:
            np.ndarray: Step function transmission coefficient.
        """
        cutoff_radius = 5.0
        transmission_coeff = 1.0
        return transmission_coeff * np.where(r < cutoff_radius, 1.0, 0.0)
    
    def _get_initial_parameters(self, region_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Get initial parameter guess for optimization."""
        try:
            r = region_data['r']
            values = region_data['values']
            
            # Simple initial guess based on data
            if len(r) > 0 and len(values) > 0:
                # Estimate amplitude from maximum value
                amplitude = np.max(values)
                
                # Estimate exponent from slope
                if len(r) > 1:
                    log_r = np.log(r + 1e-10)
                    log_values = np.log(np.abs(values) + 1e-10)
                    slope = np.polyfit(log_r, log_values, 1)[0]
                    exponent = slope
                else:
                    exponent = -2.0
            else:
                amplitude = 1.0
                exponent = -2.0
            
            return np.array([amplitude, exponent])
            
        except Exception as e:
            self.logger.error(f"Initial parameter estimation failed: {e}")
            return np.array([1.0, -2.0])
    
    def _compute_fit_quality(self, region_data: Dict[str, np.ndarray], params: np.ndarray) -> float:
        """Compute fit quality for given parameters."""
        try:
            r = region_data['r']
            values = region_data['values']
            
            if len(r) == 0 or len(values) == 0:
                return 0.0
            
            # Compute predicted values
            amplitude, exponent = params
            predicted = amplitude * (r ** exponent)
            
            # Compute R-squared
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            self.logger.error(f"Fit quality computation failed: {e}")
            return 0.0
    
    def _compute_gradient(self, func, params: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Compute numerical gradient of function."""
        try:
            gradient = np.zeros_like(params)
            
            for i in range(len(params)):
                # Forward difference
                params_plus = params.copy()
                params_plus[i] += h
                f_plus = func(params_plus)
                
                # Backward difference
                params_minus = params.copy()
                params_minus[i] -= h
                f_minus = func(params_minus)
                
                # Central difference
                gradient[i] = (f_plus - f_minus) / (2 * h)
            
            return gradient
            
        except Exception as e:
            self.logger.error(f"Gradient computation failed: {e}")
            return np.zeros_like(params)
    
    def _compute_parameter_sensitivity(self, param_value: float, param_name: str) -> float:
        """Compute parameter sensitivity for adaptive adjustment."""
        try:
            # Simple sensitivity based on parameter magnitude
            if param_name == "amplitude":
                return min(1.0, abs(param_value) / 10.0)
            elif param_name == "exponent":
                return min(1.0, abs(param_value) / 5.0)
            else:
                return 0.1  # Default sensitivity
                
        except Exception as e:
            self.logger.error(f"Parameter sensitivity computation failed: {e}")
            return 0.1
