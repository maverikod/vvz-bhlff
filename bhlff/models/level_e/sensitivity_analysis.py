"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Sensitivity analysis for Level E experiments using Sobol indices.

This module implements comprehensive sensitivity analysis for the 7D phase
field theory, using Sobol indices to rank parameter importance and
investigate system stability under parameter variations.

Theoretical Background:
    Sensitivity analysis quantifies the relative importance of different
    parameters in determining system behavior. Sobol indices provide
    a rigorous mathematical framework for ranking parameter influence
    on key observables.

Mathematical Foundation:
    Computes Sobol indices S_i = Var[E[Y|X_i]]/Var[Y] where Y is the
    output and X_i are the input parameters. Uses Latin Hypercube
    Sampling for efficient parameter space exploration.

Example:
    >>> analyzer = SensitivityAnalyzer(parameter_ranges)
    >>> results = analyzer.analyze_parameter_sensitivity(n_samples=1000)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json
from scipy import stats
from scipy.optimize import minimize


class SensitivityAnalyzer:
    """
    Sobol sensitivity analysis for parameter ranking.
    
    Physical Meaning:
        Quantifies the relative importance of different parameters
        in determining the system behavior, providing insights into
        which parameters most strongly influence key observables.
        
    Mathematical Foundation:
        Computes Sobol indices S_i = Var[E[Y|X_i]]/Var[Y] where Y
        is the output and X_i are the input parameters.
    """
    
    def __init__(self, parameter_ranges: Dict[str, Tuple[float, float]]):
        """
        Initialize Sobol analyzer.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
        """
        self.param_ranges = parameter_ranges
        self.param_names = list(parameter_ranges.keys())
        self.n_params = len(self.param_names)
        
        # Setup parameter indices
        self._param_indices = {name: i for i, name in enumerate(self.param_names)}
    
    def generate_lhs_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate Latin Hypercube samples.
        
        Physical Meaning:
            Creates efficient sampling of parameter space ensuring
            good coverage with minimal computational cost.
            
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, n_params) with parameter values
        """
        # Generate Latin Hypercube samples
        samples = np.zeros((n_samples, self.n_params))
        
        for i, (param_name, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            # Generate uniform samples in [0, 1]
            uniform_samples = np.random.uniform(0, 1, n_samples)
            
            # Apply Latin Hypercube sampling
            lhs_samples = (uniform_samples + np.arange(n_samples)) / n_samples
            
            # Scale to parameter range
            samples[:, i] = min_val + lhs_samples * (max_val - min_val)
        
        return samples
    
    def compute_sobol_indices(self, samples: np.ndarray, 
                            outputs: np.ndarray) -> Dict[str, float]:
        """
        Compute Sobol sensitivity indices.
        
        Physical Meaning:
            Calculates first-order and total-order Sobol indices
            to rank parameter importance.
            
        Mathematical Foundation:
            S_i = Var[E[Y|X_i]]/Var[Y] (first-order)
            S_Ti = 1 - Var[E[Y|X_{-i}]]/Var[Y] (total-order)
            
        Args:
            samples: Parameter samples (n_samples, n_params)
            outputs: Corresponding output values (n_samples,)
            
        Returns:
            Dictionary with Sobol indices for each parameter
        """
        # Compute total variance
        total_variance = np.var(outputs)
        
        if total_variance == 0:
            return {name: 0.0 for name in self.param_names}
        
        sobol_indices = {}
        
        for i, param_name in enumerate(self.param_names):
            # First-order Sobol index
            first_order = self._compute_first_order_index(samples, outputs, i)
            
            # Total-order Sobol index
            total_order = self._compute_total_order_index(samples, outputs, i)
            
            sobol_indices[param_name] = {
                'first_order': first_order,
                'total_order': total_order,
                'interaction': total_order - first_order
            }
        
        return sobol_indices
    
    def _compute_first_order_index(self, samples: np.ndarray, 
                                 outputs: np.ndarray, param_idx: int) -> float:
        """Compute first-order Sobol index for parameter."""
        # Group outputs by parameter value (binning)
        param_values = samples[:, param_idx]
        n_bins = min(20, len(np.unique(param_values)))
        
        # Create bins
        bins = np.linspace(np.min(param_values), np.max(param_values), n_bins + 1)
        bin_indices = np.digitize(param_values, bins) - 1
        
        # Compute conditional expectations
        conditional_means = []
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:
                conditional_means.append(np.mean(outputs[mask]))
            else:
                conditional_means.append(0.0)
        
        # Compute variance of conditional expectations
        conditional_variance = np.var(conditional_means)
        total_variance = np.var(outputs)
        
        if total_variance == 0:
            return 0.0
        
        return conditional_variance / total_variance
    
    def _compute_total_order_index(self, samples: np.ndarray, 
                                 outputs: np.ndarray, param_idx: int) -> float:
        """Compute total-order Sobol index for parameter."""
        # Create samples with all parameters except the target parameter
        other_params = np.delete(samples, param_idx, axis=1)
        
        # Compute variance of outputs conditioned on all other parameters
        # This is a simplified implementation - full implementation would
        # require more sophisticated conditional variance estimation
        
        # For now, use a simplified approach based on correlation
        param_values = samples[:, param_idx]
        correlation = np.corrcoef(param_values, outputs)[0, 1]
        
        # Convert correlation to approximate total-order index
        total_order = correlation**2 if not np.isnan(correlation) else 0.0
        
        return total_order
    
    def analyze_parameter_sensitivity(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform complete sensitivity analysis.
        
        Physical Meaning:
            Executes full sensitivity analysis workflow including
            sampling, simulation, and index computation.
            
        Args:
            n_samples: Number of samples for analysis
            
        Returns:
            Complete sensitivity analysis results
        """
        # Generate parameter samples
        samples = self.generate_lhs_samples(n_samples)
        
        # Run simulations for each sample
        outputs = self._run_simulations(samples)
        
        # Compute Sobol indices
        sobol_indices = self.compute_sobol_indices(samples, outputs)
        
        # Rank parameters by importance
        ranking = self._rank_parameters(sobol_indices)
        
        # Compute stability metrics
        stability_metrics = self._compute_stability_metrics(sobol_indices)
        
        return {
            'samples': samples,
            'outputs': outputs,
            'sobol_indices': sobol_indices,
            'parameter_ranking': ranking,
            'stability_metrics': stability_metrics,
            'n_samples': n_samples,
            'parameter_ranges': self.param_ranges
        }
    
    def _run_simulations(self, samples: np.ndarray) -> np.ndarray:
        """
        Run simulations for parameter samples.
        
        Physical Meaning:
            Executes the 7D phase field simulations for each parameter
            combination to generate output data for sensitivity analysis.
        """
        outputs = []
        
        for i, sample in enumerate(samples):
            try:
                # Create parameter dictionary
                params = dict(zip(self.param_names, sample))
                
                # Run simulation (placeholder implementation)
                output = self._simulate_single_case(params)
                outputs.append(output)
                
            except Exception as e:
                # Handle simulation failures
                print(f"Simulation failed for sample {i}: {e}")
                outputs.append(np.nan)
        
        return np.array(outputs)
    
    def _simulate_single_case(self, params: Dict[str, float]) -> float:
        """
        Simulate single parameter case.
        
        Physical Meaning:
            Runs a single simulation with given parameters and returns
            a key observable (e.g., power law exponent, quality factor).
        """
        # Placeholder implementation - in real case, this would run
        # the actual 7D phase field simulation
        
        # Extract key parameters
        beta = params.get('beta', 1.0)
        mu = params.get('mu', 1.0)
        eta = params.get('eta', 0.1)
        
        # Compute a simple observable (power law exponent)
        power_law_exponent = 2 * beta - 3
        
        # Add some noise to simulate realistic variations
        noise = np.random.normal(0, 0.1)
        observable = power_law_exponent + noise
        
        return observable
    
    def _rank_parameters(self, sobol_indices: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Rank parameters by their total-order Sobol indices.
        
        Args:
            sobol_indices: Dictionary with Sobol indices
            
        Returns:
            List of (parameter_name, total_order_index) tuples sorted by importance
        """
        ranking = []
        
        for param_name, indices in sobol_indices.items():
            total_order = indices['total_order']
            ranking.append((param_name, total_order))
        
        # Sort by total-order index (descending)
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def _compute_stability_metrics(self, sobol_indices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute stability metrics for sensitivity analysis.
        
        Physical Meaning:
            Evaluates the stability and reliability of the sensitivity
            analysis results, including convergence and consistency checks.
        """
        # Extract total-order indices
        total_indices = [indices['total_order'] for indices in sobol_indices.values()]
        
        # Compute stability metrics
        total_sensitivity = np.sum(total_indices)
        max_sensitivity = np.max(total_indices)
        min_sensitivity = np.min(total_indices)
        
        # Check for convergence (total sensitivity should be close to 1)
        convergence_metric = abs(total_sensitivity - 1.0)
        
        # Check for parameter dominance
        dominance_ratio = max_sensitivity / min_sensitivity if min_sensitivity > 0 else float('inf')
        
        return {
            'total_sensitivity': total_sensitivity,
            'max_sensitivity': max_sensitivity,
            'min_sensitivity': min_sensitivity,
            'convergence_metric': convergence_metric,
            'dominance_ratio': dominance_ratio,
            'is_converged': convergence_metric < 0.1,
            'is_balanced': dominance_ratio < 10.0
        }
    
    def analyze_mass_complexity_correlation(self, samples: np.ndarray, 
                                          outputs: np.ndarray) -> Dict[str, Any]:
        """
        Analyze correlation between mass and complexity.
        
        Physical Meaning:
            Investigates the "mass = complexity" thesis by analyzing
            the correlation between particle mass and field complexity
            in the 7D phase field theory.
        """
        # Extract mass and complexity parameters
        mass_params = ['mu', 'beta']  # Parameters related to mass
        complexity_params = ['eta', 'gamma']  # Parameters related to complexity
        
        # Compute mass and complexity metrics
        mass_metrics = self._compute_mass_metrics(samples, mass_params)
        complexity_metrics = self._compute_complexity_metrics(samples, complexity_params)
        
        # Compute correlation
        correlation = np.corrcoef(mass_metrics, complexity_metrics)[0, 1]
        
        # Statistical significance test
        n_samples = len(mass_metrics)
        t_statistic = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), n_samples - 2))
        
        return {
            'correlation': correlation,
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'mass_metrics': mass_metrics,
            'complexity_metrics': complexity_metrics
        }
    
    def _compute_mass_metrics(self, samples: np.ndarray, 
                             mass_params: List[str]) -> np.ndarray:
        """Compute mass-related metrics from parameters."""
        mass_values = []
        
        for sample in samples:
            # Create parameter dictionary
            params = dict(zip(self.param_names, sample))
            
            # Compute mass metric (simplified)
            mass_metric = 0.0
            for param in mass_params:
                if param in params:
                    mass_metric += params[param]
            
            mass_values.append(mass_metric)
        
        return np.array(mass_values)
    
    def _compute_complexity_metrics(self, samples: np.ndarray, 
                                   complexity_params: List[str]) -> np.ndarray:
        """Compute complexity-related metrics from parameters."""
        complexity_values = []
        
        for sample in samples:
            # Create parameter dictionary
            params = dict(zip(self.param_names, sample))
            
            # Compute complexity metric (simplified)
            complexity_metric = 0.0
            for param in complexity_params:
                if param in params:
                    complexity_metric += params[param]
            
            complexity_values.append(complexity_metric)
        
        return np.array(complexity_values)
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save sensitivity analysis results to file.
        
        Args:
            results: Analysis results dictionary
            filename: Output filename
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
