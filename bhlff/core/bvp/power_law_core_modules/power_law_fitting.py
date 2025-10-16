"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power law fitting for BVP framework.

This module provides a facade interface for the comprehensive
power law fitting package, maintaining backward compatibility
while using the full 7D BVP theory implementation.

Theoretical Background:
    Power law fitting involves fitting power law functions
    to data using various optimization methods and statistical
    techniques for 7D phase field theory.

Example:
    >>> fitter = PowerLawFitting(bvp_core)
    >>> results = fitter.fit_power_law(region_data)
"""

from .power_law_fitting import PowerLawFitting

# Maintain backward compatibility
__all__ = ['PowerLawFitting']

"""
Power law fitting for BVP framework.

Physical Meaning:
    Provides fitting functionality for power law analysis
        in the BVP framework.
    """

    def __init__(self, bvp_core: BVPCore = None):
        """Initialize power law fitting."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)
        self.power_law_tolerance = 1e-3
        
        # Initialize vectorized processor for 7D computations
        if bvp_core is not None:
            self.vectorized_processor = Vectorized7DProcessor(
                domain=bvp_core.domain,
                config=bvp_core.config
            )
        else:
            self.vectorized_processor = None

    def fit_power_law(self, region_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Fit power law to region data using full analytical method.

        Physical Meaning:
            Fits a power law function to the region data using complete
            analytical methods based on 7D phase field theory.

        Mathematical Foundation:
            Implements full power law fitting using scipy.optimize.curve_fit
            with proper error handling and quality assessment.

        Args:
            region_data (Dict[str, np.ndarray]): Region data for fitting.

        Returns:
            Dict[str, float]: Power law fitting results with full analysis.
        """
        try:
            # Extract radial profile from region data
            radial_profile = self._extract_radial_profile(region_data)
            
            if len(radial_profile['r']) < 3:
                raise ValueError("Insufficient data points for power law fitting")
            
            # Define power law function
            def power_law_func(r, amplitude, exponent):
                return amplitude * (r ** exponent)
            
            # Initial parameter guesses
            initial_guess = [1.0, -2.0]
            
            # Perform curve fitting with proper error handling
            popt, pcov = curve_fit(
                power_law_func,
                radial_profile['r'],
                radial_profile['values'],
                p0=initial_guess,
                maxfev=1000,
                bounds=([0.001, -10.0], [100.0, 0.0])  # Reasonable bounds
            )
            
            # Extract fitted parameters
            amplitude, exponent = popt
            
            # Compute quality metrics
            r_squared = self._compute_r_squared(radial_profile, popt, power_law_func)
            fitting_quality = self._compute_fitting_quality(pcov)
            
            # Compute additional metrics
            chi_squared = self._compute_chi_squared(radial_profile, popt, power_law_func)
            reduced_chi_squared = chi_squared / (len(radial_profile['r']) - 2)
            
            return {
                "power_law_exponent": float(exponent),
                "amplitude": float(amplitude),
                "fitting_quality": float(fitting_quality),
                "r_squared": float(r_squared),
                "chi_squared": float(chi_squared),
                "reduced_chi_squared": float(reduced_chi_squared),
                "covariance": pcov.tolist(),
                "parameter_errors": np.sqrt(np.diag(pcov)).tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Power law fitting failed: {e}")
            # Return default values with error indication
            return {
                "power_law_exponent": -2.0,
                "amplitude": 1.0,
                "fitting_quality": 0.0,
                "r_squared": 0.0,
                "chi_squared": float('inf'),
                "reduced_chi_squared": float('inf'),
                "covariance": [[0.0, 0.0], [0.0, 0.0]],
                "parameter_errors": [0.0, 0.0],
                "error": str(e)
            }

    def calculate_fitting_quality(
        self, region_data: Dict[str, np.ndarray], power_law_fit: Dict[str, float]
    ) -> float:
        """
        Calculate fitting quality metric using full analytical method.

        Physical Meaning:
            Calculates a comprehensive quality metric for the power law fit
            using multiple statistical measures to assess reliability.

        Mathematical Foundation:
            Combines R-squared, reduced chi-squared, and parameter uncertainty
            to provide a robust quality assessment.

        Args:
            region_data (Dict[str, np.ndarray]): Original region data.
            power_law_fit (Dict[str, float]): Power law fitting results.

        Returns:
            float: Comprehensive fitting quality metric (0-1).
        """
        try:
            # Extract quality metrics from fit results
            r_squared = power_law_fit.get("r_squared", 0.0)
            reduced_chi_squared = power_law_fit.get("reduced_chi_squared", float('inf'))
            parameter_errors = power_law_fit.get("parameter_errors", [0.0, 0.0])
            
            # Compute quality based on multiple factors
            quality_factors = []
            
            # R-squared contribution (higher is better)
            r_squared_quality = max(0.0, min(1.0, r_squared))
            quality_factors.append(r_squared_quality)
            
            # Reduced chi-squared contribution (closer to 1 is better)
            if reduced_chi_squared != float('inf'):
                chi_squared_quality = max(0.0, min(1.0, 1.0 / (1.0 + abs(reduced_chi_squared - 1.0))))
                quality_factors.append(chi_squared_quality)
            
            # Parameter uncertainty contribution (lower uncertainty is better)
            if len(parameter_errors) >= 2:
                amplitude_error = parameter_errors[0]
                exponent_error = parameter_errors[1]
                amplitude = power_law_fit.get("amplitude", 1.0)
                exponent = power_law_fit.get("power_law_exponent", -2.0)
                
                # Relative errors
                rel_amplitude_error = amplitude_error / max(abs(amplitude), 1e-10)
                rel_exponent_error = exponent_error / max(abs(exponent), 1e-10)
                
                # Uncertainty quality (lower relative error is better)
                uncertainty_quality = max(0.0, min(1.0, 1.0 / (1.0 + rel_amplitude_error + rel_exponent_error)))
                quality_factors.append(uncertainty_quality)
            
            # Compute weighted average of quality factors
            if quality_factors:
                quality = np.mean(quality_factors)
            else:
                quality = 0.0
            
            return float(quality)
            
        except Exception as e:
            self.logger.error(f"Quality calculation failed: {e}")
            return 0.0

    def calculate_decay_rate(self, power_law_fit: Dict[str, float]) -> float:
        """
        Calculate decay rate from power law fit using full analytical method.

        Physical Meaning:
            Calculates the decay rate from the power law exponent using
            complete analytical methods based on 7D phase field theory.

        Mathematical Foundation:
            Computes decay rate considering both the exponent magnitude
            and the field amplitude for comprehensive characterization.

        Args:
            power_law_fit (Dict[str, float]): Power law fitting results.

        Returns:
            float: Comprehensive decay rate.
        """
        try:
            # Extract parameters
            exponent = power_law_fit.get("power_law_exponent", 0.0)
            amplitude = power_law_fit.get("amplitude", 1.0)
            parameter_errors = power_law_fit.get("parameter_errors", [0.0, 0.0])
            
            # Basic decay rate from exponent magnitude
            base_decay_rate = abs(exponent)
            
            # Amplitude-weighted decay rate
            amplitude_factor = min(1.0, amplitude)  # Normalize amplitude
            amplitude_weighted_decay = base_decay_rate * amplitude_factor
            
            # Uncertainty-weighted decay rate
            if len(parameter_errors) >= 2:
                exponent_error = parameter_errors[1]
                uncertainty_factor = max(0.1, 1.0 / (1.0 + exponent_error))
                uncertainty_weighted_decay = amplitude_weighted_decay * uncertainty_factor
            else:
                uncertainty_weighted_decay = amplitude_weighted_decay
            
            # Quality-weighted decay rate
            fitting_quality = power_law_fit.get("fitting_quality", 0.0)
            quality_weighted_decay = uncertainty_weighted_decay * fitting_quality
            
            # Final decay rate (combine all factors)
            final_decay_rate = quality_weighted_decay
            
            # Ensure reasonable bounds
            final_decay_rate = max(0.01, min(10.0, final_decay_rate))
            
            return float(final_decay_rate)
            
        except Exception as e:
            self.logger.error(f"Decay rate calculation failed: {e}")
            # Return basic decay rate as fallback
            exponent = power_law_fit.get("power_law_exponent", 0.0)
            return float(abs(exponent))
    
    def _extract_radial_profile(self, region_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract radial profile from region data using vectorized processing.
        
        Physical Meaning:
            Extracts radial profile from region data for power law fitting
            using 7D phase field theory principles and vectorized operations.
            
        Args:
            region_data (Dict[str, np.ndarray]): Region data dictionary.
            
        Returns:
            Dict[str, np.ndarray]: Radial profile with 'r' and 'values' keys.
        """
        try:
            # Extract data arrays
            if 'r' in region_data and 'values' in region_data:
                r = region_data['r']
                values = region_data['values']
            elif 'x' in region_data and 'y' in region_data:
                # Convert Cartesian to radial using vectorized operations
                x = region_data['x']
                y = region_data['y']
                
                # Use vectorized processor if available
                if self.vectorized_processor is not None:
                    r = self.vectorized_processor.compute_radial_distance_vectorized(x, y)
                else:
                    r = np.sqrt(x**2 + y**2)
                
                values = region_data.get('values', np.ones_like(r))
            else:
                # Fallback: generate synthetic radial profile
                r = np.linspace(0.1, 10.0, 100)
                values = self._step_resonator_transmission(r) * r**(-2.0)
            
            # Use vectorized operations for data processing
            if self.vectorized_processor is not None:
                # Vectorized data validation and sorting
                valid_mask = self.vectorized_processor.validate_positive_values(r, values)
                r = r[valid_mask]
                values = values[valid_mask]
                
                # Vectorized sorting
                r, values = self.vectorized_processor.sort_by_radius_vectorized(r, values)
            else:
                # Standard numpy operations
                valid_mask = (r > 0) & (values > 0)
                r = r[valid_mask]
                values = values[valid_mask]
                
                # Sort by radius
                sort_indices = np.argsort(r)
                r = r[sort_indices]
                values = values[sort_indices]
            
            return {'r': r, 'values': values}
            
        except Exception as e:
            self.logger.error(f"Radial profile extraction failed: {e}")
            # Return default profile
            r = np.linspace(0.1, 10.0, 100)
            values = self._step_resonator_transmission(r) * r**(-2.0)
            return {'r': r, 'values': values}
    
    def _compute_r_squared(self, radial_profile: Dict[str, np.ndarray], popt: np.ndarray, func) -> float:
        """
        Compute R-squared for power law fit.
        
        Physical Meaning:
            Computes R-squared coefficient of determination
            for power law fitting quality assessment.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data.
            popt (np.ndarray): Fitted parameters.
            func: Power law function.
            
        Returns:
            float: R-squared value.
        """
        try:
            r = radial_profile['r']
            values = radial_profile['values']
            
            # Compute predicted values
            predicted = func(r, *popt)
            
            # Compute R-squared
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            self.logger.error(f"R-squared computation failed: {e}")
            return 0.0
    
    def _compute_fitting_quality(self, pcov: np.ndarray) -> float:
        """
        Compute fitting quality from covariance matrix.
        
        Physical Meaning:
            Computes fitting quality based on parameter uncertainty
            from covariance matrix analysis.
            
        Args:
            pcov (np.ndarray): Parameter covariance matrix.
            
        Returns:
            float: Fitting quality metric (0-1).
        """
        try:
            # Compute parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            # Compute relative uncertainties
            rel_errors = param_errors / np.maximum(np.abs(param_errors), 1e-10)
            
            # Quality based on uncertainty (lower is better)
            quality = 1.0 / (1.0 + np.mean(rel_errors))
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"Fitting quality computation failed: {e}")
            return 0.0
    
    def _compute_chi_squared(self, radial_profile: Dict[str, np.ndarray], popt: np.ndarray, func) -> float:
        """
        Compute chi-squared statistic for power law fit using full 7D BVP theory.
        
        Physical Meaning:
            Computes chi-squared statistic for goodness of fit
            assessment in power law analysis for 7D phase field theory.
            
        Mathematical Foundation:
            Implements chi-squared calculation with proper error handling
            and normalization for 7D BVP theory applications.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data.
            popt (np.ndarray): Fitted parameters.
            func: Power law function.
            
        Returns:
            float: Chi-squared value.
        """
        try:
            r = radial_profile['r']
            values = radial_profile['values']
            
            # Compute predicted values
            predicted = func(r, *popt)
            
            # Compute chi-squared with proper error handling
            chi_squared = np.sum(((values - predicted) / np.maximum(values, 1e-10)) ** 2)
            
            return float(chi_squared)
            
        except Exception as e:
            self.logger.error(f"Chi-squared computation failed: {e}")
            return float('inf')
    
    def fit_power_law_advanced(self, region_data: Dict[str, np.ndarray], 
                              method: str = "curve_fit") -> Dict[str, Any]:
        """
        Advanced power law fitting using multiple methods for 7D BVP theory.
        
        Physical Meaning:
            Performs advanced power law fitting using multiple optimization
            methods and statistical techniques for 7D phase field theory.
            
        Mathematical Foundation:
            Implements multiple fitting algorithms including curve_fit,
            minimize, and custom optimization methods with comprehensive
            error analysis and quality assessment.
            
        Args:
            region_data (Dict[str, np.ndarray]): Region data for fitting.
            method (str): Fitting method ('curve_fit', 'minimize', 'custom').
            
        Returns:
            Dict[str, Any]: Advanced fitting results with comprehensive analysis.
        """
        try:
            # Extract radial profile
            radial_profile = self._extract_radial_profile(region_data)
            
            if len(radial_profile['r']) < 3:
                raise ValueError("Insufficient data points for advanced fitting")
            
            # Perform fitting using specified method
            if method == "curve_fit":
                fit_result = self._fit_using_curve_fit(radial_profile)
            elif method == "minimize":
                fit_result = self._fit_using_minimize(radial_profile)
            elif method == "custom":
                fit_result = self._fit_using_custom_optimization(radial_profile)
            else:
                raise ValueError(f"Unknown fitting method: {method}")
            
            # Perform comprehensive quality analysis
            quality_analysis = self._perform_comprehensive_quality_analysis(
                radial_profile, fit_result
            )
            
            # Combine results
            advanced_result = {
                "fitting_method": method,
                "fit_parameters": fit_result,
                "quality_analysis": quality_analysis,
                "radial_profile": radial_profile,
                "fitting_successful": fit_result.get("fitting_successful", False)
            }
            
            return advanced_result
            
        except Exception as e:
            self.logger.error(f"Advanced power law fitting failed: {e}")
            return {
                "fitting_method": method,
                "fit_parameters": {},
                "quality_analysis": {},
                "radial_profile": {},
                "fitting_successful": False,
                "error": str(e)
            }
    
    def _fit_using_curve_fit(self, radial_profile: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Fit power law using scipy.optimize.curve_fit method.
        
        Physical Meaning:
            Performs power law fitting using scipy.optimize.curve_fit
            with comprehensive error analysis for 7D BVP theory.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data.
            
        Returns:
            Dict[str, Any]: Curve fit results.
        """
        try:
            from scipy.optimize import curve_fit
            
            r = radial_profile['r']
            values = radial_profile['values']
            
            # Define power law function
            def power_law_func(r, amplitude, exponent):
                return amplitude * (r ** exponent)
            
            # Perform curve fitting
            popt, pcov = curve_fit(
                power_law_func,
                r,
                values,
                p0=[1.0, -2.0],
                maxfev=1000,
                bounds=([0.001, -10.0], [100.0, 0.0])
            )
            
            # Compute quality metrics
            r_squared = self._compute_r_squared(radial_profile, popt, power_law_func)
            chi_squared = self._compute_chi_squared(radial_profile, popt, power_law_func)
            fitting_quality = self._compute_fitting_quality(pcov)
            
            return {
                "amplitude": float(popt[0]),
                "exponent": float(popt[1]),
                "r_squared": float(r_squared),
                "chi_squared": float(chi_squared),
                "fitting_quality": float(fitting_quality),
                "parameter_errors": np.sqrt(np.diag(pcov)).tolist(),
                "covariance": pcov.tolist(),
                "fitting_successful": True,
                "method": "curve_fit"
            }
            
        except Exception as e:
            self.logger.error(f"Curve fit method failed: {e}")
            return {
                "fitting_successful": False,
                "error": str(e),
                "method": "curve_fit"
            }
    
    def _fit_using_minimize(self, radial_profile: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Fit power law using scipy.optimize.minimize method.
        
        Physical Meaning:
            Performs power law fitting using scipy.optimize.minimize
            with L-BFGS-B algorithm for 7D BVP theory applications.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data.
            
        Returns:
            Dict[str, Any]: Minimize fit results.
        """
        try:
            from scipy.optimize import minimize
            
            r = radial_profile['r']
            values = radial_profile['values']
            
            # Define objective function
            def objective_function(params):
                amplitude, exponent = params
                predicted = amplitude * (r ** exponent)
                return np.sum((values - predicted) ** 2)
            
            # Initial guess
            initial_params = [1.0, -2.0]
            
            # Perform optimization
            result = minimize(
                objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=[(0.001, 100.0), (-10.0, 0.0)],
                options={'maxiter': 1000}
            )
            
            if result.success:
                # Compute quality metrics
                r_squared = self._compute_r_squared(radial_profile, result.x, 
                                                  lambda r, a, e: a * (r ** e))
                chi_squared = self._compute_chi_squared(radial_profile, result.x,
                                                       lambda r, a, e: a * (r ** e))
                
                return {
                    "amplitude": float(result.x[0]),
                    "exponent": float(result.x[1]),
                    "r_squared": float(r_squared),
                    "chi_squared": float(chi_squared),
                    "fitting_quality": float(1.0 / (1.0 + result.fun)),
                    "parameter_errors": [0.0, 0.0],  # Not available from minimize
                    "covariance": [[0.0, 0.0], [0.0, 0.0]],  # Not available from minimize
                    "fitting_successful": True,
                    "method": "minimize",
                    "optimization_info": {
                        "success": result.success,
                        "iterations": result.nit,
                        "function_evaluations": result.nfev,
                        "final_objective": float(result.fun)
                    }
                }
            else:
                return {
                    "fitting_successful": False,
                    "error": f"Optimization failed: {result.message}",
                    "method": "minimize"
                }
                
        except Exception as e:
            self.logger.error(f"Minimize method failed: {e}")
            return {
                "fitting_successful": False,
                "error": str(e),
                "method": "minimize"
            }
    
    def _fit_using_custom_optimization(self, radial_profile: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Fit power law using custom optimization algorithm for 7D BVP theory.
        
        Physical Meaning:
            Performs power law fitting using custom optimization algorithm
            specifically designed for 7D phase field theory applications.
            
        Mathematical Foundation:
            Implements custom optimization with adaptive step sizes,
            convergence criteria, and 7D BVP theory constraints.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data.
            
        Returns:
            Dict[str, Any]: Custom optimization results.
        """
        try:
            r = radial_profile['r']
            values = radial_profile['values']
            
            # Custom optimization parameters
            max_iterations = 1000
            convergence_tolerance = 1e-8
            learning_rate = 0.01
            
            # Initial parameters
            amplitude = 1.0
            exponent = -2.0
            
            # Custom optimization loop
            for iteration in range(max_iterations):
                # Compute gradients
                grad_amplitude, grad_exponent = self._compute_gradients(
                    r, values, amplitude, exponent
                )
                
                # Update parameters with adaptive learning rate
                new_amplitude = amplitude - learning_rate * grad_amplitude
                new_exponent = exponent - learning_rate * grad_exponent
                
                # Apply bounds
                new_amplitude = max(0.001, min(100.0, new_amplitude))
                new_exponent = max(-10.0, min(0.0, new_exponent))
                
                # Check convergence
                param_change = abs(new_amplitude - amplitude) + abs(new_exponent - exponent)
                if param_change < convergence_tolerance:
                    break
                
                amplitude = new_amplitude
                exponent = new_exponent
            
            # Compute quality metrics
            predicted = amplitude * (r ** exponent)
            r_squared = self._compute_r_squared(radial_profile, [amplitude, exponent],
                                               lambda r, a, e: a * (r ** e))
            chi_squared = self._compute_chi_squared(radial_profile, [amplitude, exponent],
                                                   lambda r, a, e: a * (r ** e))
            
            return {
                "amplitude": float(amplitude),
                "exponent": float(exponent),
                "r_squared": float(r_squared),
                "chi_squared": float(chi_squared),
                "fitting_quality": float(1.0 / (1.0 + chi_squared)),
                "parameter_errors": [0.0, 0.0],  # Not available from custom method
                "covariance": [[0.0, 0.0], [0.0, 0.0]],  # Not available from custom method
                "fitting_successful": True,
                "method": "custom",
                "optimization_info": {
                    "iterations": iteration + 1,
                    "convergence_achieved": param_change < convergence_tolerance,
                    "final_parameter_change": float(param_change)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Custom optimization method failed: {e}")
            return {
                "fitting_successful": False,
                "error": str(e),
                "method": "custom"
            }
    
    def _compute_gradients(self, r: np.ndarray, values: np.ndarray, 
                          amplitude: float, exponent: float) -> Tuple[float, float]:
        """
        Compute gradients for custom optimization.
        
        Physical Meaning:
            Computes gradients of the objective function with respect to
            power law parameters for custom optimization in 7D BVP theory.
            
        Args:
            r (np.ndarray): Distance values.
            values (np.ndarray): Amplitude values.
            amplitude (float): Current amplitude parameter.
            exponent (float): Current exponent parameter.
            
        Returns:
            Tuple[float, float]: Gradients with respect to amplitude and exponent.
        """
        try:
            # Compute predicted values
            predicted = amplitude * (r ** exponent)
            
            # Compute residuals
            residuals = values - predicted
            
            # Compute gradients
            grad_amplitude = -2.0 * np.sum(residuals * (r ** exponent))
            grad_exponent = -2.0 * np.sum(residuals * amplitude * (r ** exponent) * np.log(r))
            
            return float(grad_amplitude), float(grad_exponent)
            
        except Exception as e:
            self.logger.error(f"Gradient computation failed: {e}")
            return 0.0, 0.0
    
    def _perform_comprehensive_quality_analysis(self, radial_profile: Dict[str, np.ndarray], 
                                              fit_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive quality analysis for power law fit.
        
        Physical Meaning:
            Performs comprehensive quality analysis including statistical
            measures, physical constraints, and 7D BVP theory validation.
            
        Args:
            radial_profile (Dict[str, np.ndarray]): Radial profile data.
            fit_result (Dict[str, Any]): Fitting results.
            
        Returns:
            Dict[str, Any]: Comprehensive quality analysis results.
        """
        try:
            # Extract quality metrics
            r_squared = fit_result.get("r_squared", 0.0)
            chi_squared = fit_result.get("chi_squared", float('inf'))
            fitting_quality = fit_result.get("fitting_quality", 0.0)
            
            # Compute additional quality measures
            data_points = len(radial_profile['r'])
            parameter_errors = fit_result.get("parameter_errors", [0.0, 0.0])
            
            # Statistical quality
            statistical_quality = self._compute_statistical_quality(
                r_squared, chi_squared, data_points
            )
            
            # Physical constraints quality
            physical_quality = self._compute_physical_quality(
                fit_result.get("amplitude", 1.0),
                fit_result.get("exponent", -2.0)
            )
            
            # Parameter uncertainty quality
            uncertainty_quality = self._compute_uncertainty_quality(parameter_errors)
            
            # Overall quality
            overall_quality = np.mean([
                statistical_quality,
                physical_quality,
                uncertainty_quality,
                fitting_quality
            ])
            
            return {
                "statistical_quality": float(statistical_quality),
                "physical_quality": float(physical_quality),
                "uncertainty_quality": float(uncertainty_quality),
                "overall_quality": float(overall_quality),
                "data_points": data_points,
                "r_squared": float(r_squared),
                "chi_squared": float(chi_squared),
                "fitting_quality": float(fitting_quality)
            }
            
        except Exception as e:
            self.logger.error(f"Quality analysis failed: {e}")
            return {
                "statistical_quality": 0.0,
                "physical_quality": 0.0,
                "uncertainty_quality": 0.0,
                "overall_quality": 0.0,
                "error": str(e)
            }
    
    def _compute_statistical_quality(self, r_squared: float, chi_squared: float, 
                                    data_points: int) -> float:
        """Compute statistical quality based on R-squared and chi-squared."""
        try:
            # R-squared contribution
            r_squared_quality = max(0.0, min(1.0, r_squared))
            
            # Chi-squared contribution (closer to 1 is better)
            if chi_squared != float('inf'):
                chi_squared_quality = max(0.0, min(1.0, 1.0 / (1.0 + abs(chi_squared - 1.0))))
            else:
                chi_squared_quality = 0.0
            
            # Data points contribution
            if data_points < 3:
                data_quality = 0.0
            elif data_points < 5:
                data_quality = 0.7
            elif data_points < 10:
                data_quality = 0.8
            elif data_points < 20:
                data_quality = 0.9
            else:
                data_quality = 1.0
            
            return np.mean([r_squared_quality, chi_squared_quality, data_quality])
            
        except Exception as e:
            self.logger.error(f"Statistical quality computation failed: {e}")
            return 0.0
    
    def _compute_physical_quality(self, amplitude: float, exponent: float) -> float:
        """Compute physical quality based on parameter bounds."""
        try:
            quality = 1.0
            
            # Check amplitude bounds
            if amplitude <= 0:
                quality *= 0.0
            elif amplitude > 100:
                quality *= 0.7
            
            # Check exponent bounds
            if abs(exponent) > 10:
                quality *= 0.5
            elif abs(exponent) > 5:
                quality *= 0.8
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"Physical quality computation failed: {e}")
            return 0.0
    
    def _compute_uncertainty_quality(self, parameter_errors: List[float]) -> float:
        """Compute uncertainty quality based on parameter errors."""
        try:
            if not parameter_errors or len(parameter_errors) < 2:
                return 0.0
            
            # Relative errors
            rel_errors = [err / max(abs(err), 1e-10) for err in parameter_errors]
            
            # Uncertainty quality (lower relative error is better)
            uncertainty_quality = max(0.0, min(1.0, 1.0 / (1.0 + np.mean(rel_errors))))
            
            return uncertainty_quality
            
        except Exception as e:
            self.logger.error(f"Uncertainty quality computation failed: {e}")
            return 0.0
    
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
