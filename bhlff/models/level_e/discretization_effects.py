"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Discretization effects analysis for Level E experiments.

This module implements comprehensive analysis of discretization and
finite-size effects in the 7D phase field theory, investigating
how numerical discretization and finite domain size affect accuracy
and reliability of computational results.

Theoretical Background:
    Discretization effects analysis investigates how numerical
    discretization and finite domain size affect the accuracy and
    reliability of computational results. This is crucial for
    establishing convergence and optimal computational parameters.

Mathematical Foundation:
    Analyzes convergence rates: p = log(|e_h1|/|e_h2|)/log(h1/h2)
    where e_h is the error at grid spacing h. Investigates effects
    of finite domain size on long-range interactions.

Example:
    >>> analyzer = DiscretizationAnalyzer(reference_config)
    >>> results = analyzer.analyze_grid_convergence(grid_sizes)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json


class DiscretizationAnalyzer:
    """
    Analysis of discretization and finite-size effects.
    
    Physical Meaning:
        Investigates how numerical discretization and finite
        domain size affect the accuracy and reliability of
        computational results.
    """
    
    def __init__(self, reference_config: Dict[str, Any]):
        """
        Initialize discretization analyzer.
        
        Args:
            reference_config: Reference configuration for comparison
        """
        self.reference_config = reference_config
        self._setup_convergence_metrics()
    
    def _setup_convergence_metrics(self) -> None:
        """Setup metrics for convergence analysis."""
        self.convergence_metrics = [
            'power_law_exponent',
            'topological_charge',
            'energy',
            'quality_factor',
            'stability'
        ]
    
    def analyze_grid_convergence(self, grid_sizes: List[int]) -> Dict[str, Any]:
        """
        Analyze convergence with grid refinement.
        
        Physical Meaning:
            Investigates how results change as the computational
            grid is refined, establishing convergence rates and
            optimal grid sizes.
            
        Mathematical Foundation:
            Computes convergence rate: p = log(|e_h1|/|e_h2|)/log(h1/h2)
            where e_h is the error at grid spacing h.
            
        Args:
            grid_sizes: List of grid sizes to test
            
        Returns:
            Convergence analysis results
        """
        results = {}
        
        for grid_size in grid_sizes:
            print(f"Analyzing grid size: {grid_size}")
            
            # Create configuration with specified grid size
            config = self._create_grid_config(grid_size)
            
            # Run simulation
            output = self._run_simulation(config)
            
            # Compute metrics
            metrics = self._compute_metrics(output)
            
            results[grid_size] = {
                'config': config,
                'output': output,
                'metrics': metrics
            }
        
        # Analyze convergence
        convergence_analysis = self._analyze_convergence(results)
        
        # Recommend optimal grid size
        recommended_size = self._recommend_grid_size(convergence_analysis)
        
        return {
            'grid_results': results,
            'convergence_analysis': convergence_analysis,
            'recommended_grid_size': recommended_size
        }
    
    def _create_grid_config(self, grid_size: int) -> Dict[str, Any]:
        """Create configuration with specified grid size."""
        config = self.reference_config.copy()
        config['N'] = grid_size
        
        # Adjust domain size if needed to maintain resolution
        if 'L' in config:
            # Keep physical resolution approximately constant
            base_N = config.get('base_N', 256)
            if base_N != grid_size:
                scale_factor = base_N / grid_size
                config['L'] *= scale_factor
        
        return config
    
    def _run_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run simulation with given configuration.
        
        Physical Meaning:
            Executes the 7D phase field simulation with specified
            discretization parameters and returns key observables.
        """
        # Placeholder implementation - in real case, this would run
        # the actual 7D phase field simulation
        
        # Extract key parameters
        N = config.get('N', 256)
        L = config.get('L', 20.0)
        beta = config.get('beta', 1.0)
        mu = config.get('mu', 1.0)
        
        # Compute observables with grid-dependent effects
        dx = L / N  # Grid spacing
        
        # Power law exponent (should be grid-independent)
        power_law_exponent = 2 * beta - 3
        
        # Topological charge (may have discretization errors)
        topological_charge = 1.0 + np.random.normal(0, 0.01 * dx)
        
        # Energy (scales with grid resolution)
        energy = mu * beta * (1 + 0.1 * dx)
        
        # Quality factor (may depend on resolution)
        quality_factor = mu / (0.1 + 0.01 * dx)
        
        # Stability (should be grid-independent)
        stability = 1.0 if beta > 0.5 else 0.0
        
        return {
            'power_law_exponent': power_law_exponent,
            'topological_charge': topological_charge,
            'energy': energy,
            'quality_factor': quality_factor,
            'stability': stability,
            'grid_spacing': dx,
            'grid_size': N
        }
    
    def _compute_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Compute convergence metrics from simulation output."""
        metrics = {}
        
        for metric in self.convergence_metrics:
            if metric in output:
                metrics[metric] = output[metric]
        
        return metrics
    
    def _analyze_convergence(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze convergence behavior.
        
        Physical Meaning:
            Computes convergence rates and identifies optimal
            grid sizes for different observables.
        """
        grid_sizes = sorted(results.keys())
        convergence_rates = {}
        convergence_quality = {}
        
        for metric in self.convergence_metrics:
            if metric in results[grid_sizes[0]]['metrics']:
                # Extract values for this metric
                values = [results[grid_size]['metrics'][metric] for grid_size in grid_sizes]
                
                # Compute convergence rate
                convergence_rate = self._compute_convergence_rate(grid_sizes, values)
                convergence_rates[metric] = convergence_rate
                
                # Assess convergence quality
                quality = self._assess_convergence_quality(values)
                convergence_quality[metric] = quality
        
        # Overall convergence analysis
        overall_convergence = self._analyze_overall_convergence(convergence_rates)
        
        return {
            'convergence_rates': convergence_rates,
            'convergence_quality': convergence_quality,
            'overall_convergence': overall_convergence,
            'grid_sizes': grid_sizes
        }
    
    def _compute_convergence_rate(self, grid_sizes: List[int], 
                                values: List[float]) -> float:
        """
        Compute convergence rate for a metric.
        
        Mathematical Foundation:
            p = log(|e_h1|/|e_h2|)/log(h1/h2) where e_h is the error
            at grid spacing h.
        """
        if len(values) < 2:
            return 0.0
        
        # Use finest grid as reference
        reference_value = values[-1]
        errors = [abs(v - reference_value) for v in values]
        
        # Compute convergence rate
        convergence_rates = []
        for i in range(len(errors) - 1):
            if errors[i] > 0 and errors[i+1] > 0:
                h1 = 1.0 / grid_sizes[i]
                h2 = 1.0 / grid_sizes[i+1]
                rate = np.log(errors[i] / errors[i+1]) / np.log(h1 / h2)
                convergence_rates.append(rate)
        
        return np.mean(convergence_rates) if convergence_rates else 0.0
    
    def _assess_convergence_quality(self, values: List[float]) -> Dict[str, Any]:
        """Assess quality of convergence."""
        if len(values) < 2:
            return {'quality': 'insufficient_data', 'score': 0.0}
        
        # Compute relative changes
        relative_changes = []
        for i in range(len(values) - 1):
            if values[i+1] != 0:
                rel_change = abs(values[i] - values[i+1]) / abs(values[i+1])
                relative_changes.append(rel_change)
        
        # Assess convergence quality
        max_change = max(relative_changes) if relative_changes else 0.0
        mean_change = np.mean(relative_changes) if relative_changes else 0.0
        
        if max_change < 0.01:
            quality = 'excellent'
            score = 1.0
        elif max_change < 0.05:
            quality = 'good'
            score = 0.8
        elif max_change < 0.1:
            quality = 'fair'
            score = 0.6
        else:
            quality = 'poor'
            score = 0.3
        
        return {
            'quality': quality,
            'score': score,
            'max_change': max_change,
            'mean_change': mean_change
        }
    
    def _analyze_overall_convergence(self, convergence_rates: Dict[str, float]) -> Dict[str, Any]:
        """Analyze overall convergence behavior."""
        rates = list(convergence_rates.values())
        
        if not rates:
            return {'overall_rate': 0.0, 'quality': 'unknown'}
        
        # Compute overall convergence rate
        overall_rate = np.mean(rates)
        
        # Assess overall quality
        if overall_rate > 2.0:
            quality = 'excellent'
        elif overall_rate > 1.0:
            quality = 'good'
        elif overall_rate > 0.5:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'overall_rate': overall_rate,
            'quality': quality,
            'individual_rates': convergence_rates
        }
    
    def _recommend_grid_size(self, convergence_analysis: Dict[str, Any]) -> int:
        """Recommend optimal grid size based on convergence analysis."""
        grid_sizes = convergence_analysis['grid_sizes']
        convergence_rates = convergence_analysis['convergence_rates']
        
        # Find grid size where convergence is good but not excessive
        for grid_size in grid_sizes:
            # Check if convergence is good at this grid size
            good_convergence = True
            for metric, rate in convergence_rates.items():
                if rate < 1.0:  # Poor convergence
                    good_convergence = False
                    break
            
            if good_convergence:
                return grid_size
        
        # Default to largest grid size if no good convergence found
        return max(grid_sizes)
    
    def analyze_domain_size_effects(self, domain_sizes: List[float]) -> Dict[str, Any]:
        """
        Analyze effects of finite domain size.
        
        Physical Meaning:
            Investigates how the finite computational domain
            affects results, particularly for long-range
            interactions and boundary effects.
            
        Args:
            domain_sizes: List of domain sizes to test
            
        Returns:
            Domain size analysis results
        """
        results = {}
        
        for domain_size in domain_sizes:
            print(f"Analyzing domain size: {domain_size}")
            
            # Create configuration with specified domain size
            config = self._create_domain_config(domain_size)
            
            # Run simulation
            output = self._run_simulation(config)
            
            # Compute metrics
            metrics = self._compute_metrics(output)
            
            results[domain_size] = {
                'config': config,
                'output': output,
                'metrics': metrics
            }
        
        # Analyze domain size effects
        domain_analysis = self._analyze_domain_effects(results)
        
        return {
            'domain_results': results,
            'domain_analysis': domain_analysis
        }
    
    def _create_domain_config(self, domain_size: float) -> Dict[str, Any]:
        """Create configuration with specified domain size."""
        config = self.reference_config.copy()
        config['L'] = domain_size
        
        return config
    
    def _analyze_domain_effects(self, results: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effects of domain size on results."""
        domain_sizes = sorted(results.keys())
        domain_effects = {}
        
        for metric in self.convergence_metrics:
            if metric in results[domain_sizes[0]]['metrics']:
                # Extract values for this metric
                values = [results[domain_size]['metrics'][metric] for domain_size in domain_sizes]
                
                # Analyze domain size dependence
                dependence = self._analyze_domain_dependence(domain_sizes, values)
                domain_effects[metric] = dependence
        
        # Overall domain size analysis
        overall_analysis = self._analyze_overall_domain_effects(domain_effects)
        
        return {
            'domain_effects': domain_effects,
            'overall_analysis': overall_analysis,
            'domain_sizes': domain_sizes
        }
    
    def _analyze_domain_dependence(self, domain_sizes: List[float], 
                                 values: List[float]) -> Dict[str, Any]:
        """Analyze dependence of metric on domain size."""
        if len(values) < 2:
            return {'dependence': 'insufficient_data', 'slope': 0.0}
        
        # Compute slope of values vs domain size
        slope = np.polyfit(domain_sizes, values, 1)[0]
        
        # Assess dependence
        if abs(slope) < 0.01:
            dependence = 'independent'
        elif abs(slope) < 0.1:
            dependence = 'weak'
        elif abs(slope) < 1.0:
            dependence = 'moderate'
        else:
            dependence = 'strong'
        
        # Compute correlation safely
        try:
            correlation = np.corrcoef(domain_sizes, values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        return {
            'dependence': dependence,
            'slope': slope,
            'correlation': correlation
        }
    
    def _analyze_overall_domain_effects(self, domain_effects: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall domain size effects."""
        dependencies = [effects['dependence'] for effects in domain_effects.values()]
        
        # Count different types of dependence
        independent_count = dependencies.count('independent')
        weak_count = dependencies.count('weak')
        moderate_count = dependencies.count('moderate')
        strong_count = dependencies.count('strong')
        
        # Overall assessment
        if independent_count > len(dependencies) / 2:
            overall_dependence = 'independent'
        elif weak_count > len(dependencies) / 2:
            overall_dependence = 'weak'
        elif moderate_count > len(dependencies) / 2:
            overall_dependence = 'moderate'
        else:
            overall_dependence = 'strong'
        
        return {
            'overall_dependence': overall_dependence,
            'independent_count': independent_count,
            'weak_count': weak_count,
            'moderate_count': moderate_count,
            'strong_count': strong_count
        }
    
    def analyze_time_step_stability(self, time_steps: List[float]) -> Dict[str, Any]:
        """
        Analyze stability with respect to time step.
        
        Physical Meaning:
            Investigates numerical stability of time integration
            schemes and optimal time step selection.
            
        Args:
            time_steps: List of time steps to test
            
        Returns:
            Time step stability analysis
        """
        results = {}
        
        for dt in time_steps:
            print(f"Analyzing time step: {dt}")
            
            # Create configuration with specified time step
            config = self._create_time_step_config(dt)
            
            # Run simulation
            output = self._run_simulation(config)
            
            # Compute metrics
            metrics = self._compute_metrics(output)
            
            results[dt] = {
                'config': config,
                'output': output,
                'metrics': metrics
            }
        
        # Analyze time step stability
        stability_analysis = self._analyze_time_step_stability(results)
        
        return {
            'time_step_results': results,
            'stability_analysis': stability_analysis
        }
    
    def _create_time_step_config(self, dt: float) -> Dict[str, Any]:
        """Create configuration with specified time step."""
        config = self.reference_config.copy()
        config['dt'] = dt
        
        return config
    
    def _analyze_time_step_stability(self, results: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time step stability."""
        time_steps = sorted(results.keys())
        stability_metrics = {}
        
        for metric in self.convergence_metrics:
            if metric in results[time_steps[0]]['metrics']:
                # Extract values for this metric
                values = [results[dt]['metrics'][metric] for dt in time_steps]
                
                # Analyze stability
                stability = self._analyze_metric_stability(time_steps, values)
                stability_metrics[metric] = stability
        
        # Overall stability analysis
        overall_stability = self._analyze_overall_stability(stability_metrics)
        
        return {
            'stability_metrics': stability_metrics,
            'overall_stability': overall_stability,
            'time_steps': time_steps
        }
    
    def _analyze_metric_stability(self, time_steps: List[float], 
                                values: List[float]) -> Dict[str, Any]:
        """Analyze stability of a metric with respect to time step."""
        if len(values) < 2:
            return {'stability': 'insufficient_data', 'score': 0.0}
        
        # Compute relative changes
        relative_changes = []
        for i in range(len(values) - 1):
            if values[i+1] != 0:
                rel_change = abs(values[i] - values[i+1]) / abs(values[i+1])
                relative_changes.append(rel_change)
        
        # Assess stability
        max_change = max(relative_changes) if relative_changes else 0.0
        mean_change = np.mean(relative_changes) if relative_changes else 0.0
        
        if max_change < 0.01:
            stability = 'excellent'
            score = 1.0
        elif max_change < 0.05:
            stability = 'good'
            score = 0.8
        elif max_change < 0.1:
            stability = 'fair'
            score = 0.6
        else:
            stability = 'poor'
            score = 0.3
        
        return {
            'stability': stability,
            'score': score,
            'max_change': max_change,
            'mean_change': mean_change
        }
    
    def _analyze_overall_stability(self, stability_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall time step stability."""
        scores = [metrics['score'] for metrics in stability_metrics.values()]
        
        if not scores:
            return {'overall_score': 0.0, 'stability': 'unknown'}
        
        overall_score = np.mean(scores)
        
        if overall_score > 0.8:
            stability = 'excellent'
        elif overall_score > 0.6:
            stability = 'good'
        elif overall_score > 0.4:
            stability = 'fair'
        else:
            stability = 'poor'
        
        return {
            'overall_score': overall_score,
            'stability': stability,
            'individual_scores': {metric: metrics['score'] for metric, metrics in stability_metrics.items()}
        }
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save discretization analysis results to file.
        
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
