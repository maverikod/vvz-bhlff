"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Performance analysis for Level E experiments.

This module implements comprehensive performance analysis for the 7D phase
field theory, optimizing the balance between computational cost and
accuracy, and creating regression test suites.

Theoretical Background:
    Performance analysis investigates the relationship between computational
    cost and accuracy in the 7D phase field simulations. This is crucial
    for practical applications where computational resources are limited.

Mathematical Foundation:
    Analyzes scaling behavior: T(N) ~ N^α where T is computation time
    and N is problem size. Optimizes accuracy vs cost trade-offs.

Example:
    >>> analyzer = PerformanceAnalyzer(config)
    >>> results = analyzer.analyze_performance()
"""

import numpy as np
import time
import psutil
import json
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import curve_fit


class PerformanceAnalyzer:
    """
    Performance analysis for computational efficiency.
    
    Physical Meaning:
        Analyzes the relationship between computational cost and
        accuracy in the 7D phase field simulations, providing
        optimization recommendations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._setup_performance_metrics()
        self._setup_benchmark_cases()
    
    def _setup_performance_metrics(self) -> None:
        """Setup performance metrics for analysis."""
        self.performance_metrics = [
            'execution_time',
            'memory_usage',
            'cpu_usage',
            'accuracy',
            'convergence_rate'
        ]
    
    def _setup_benchmark_cases(self) -> None:
        """Setup benchmark test cases."""
        self.benchmark_cases = {
            'single_soliton': self._benchmark_single_soliton,
            'defect_pair': self._benchmark_defect_pair,
            'multi_defect_system': self._benchmark_multi_defect_system
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis.
        
        Physical Meaning:
            Analyzes computational performance across different
            problem sizes and configurations, providing optimization
            recommendations.
            
        Returns:
            Complete performance analysis results
        """
        # Analyze scaling behavior
        scaling_analysis = self._analyze_scaling_behavior()
        
        # Analyze accuracy vs cost trade-offs
        accuracy_cost_analysis = self._analyze_accuracy_cost_tradeoffs()
        
        # Run benchmark tests
        benchmark_results = self._run_benchmark_tests()
        
        # Analyze memory usage
        memory_analysis = self._analyze_memory_usage()
        
        # Optimize parameters
        optimization_results = self._optimize_parameters()
        
        return {
            'scaling_analysis': scaling_analysis,
            'accuracy_cost_analysis': accuracy_cost_analysis,
            'benchmark_results': benchmark_results,
            'memory_analysis': memory_analysis,
            'optimization_results': optimization_results
        }
    
    def _analyze_scaling_behavior(self) -> Dict[str, Any]:
        """Analyze computational scaling behavior."""
        # Test different problem sizes
        grid_sizes = [64, 128, 256, 512, 1024]
        domain_sizes = [10.0, 20.0, 40.0, 80.0]
        time_ranges = [1.0, 5.0, 10.0, 20.0]
        
        scaling_results = {}
        
        # Grid size scaling
        grid_scaling = self._test_grid_scaling(grid_sizes)
        scaling_results['grid_scaling'] = grid_scaling
        
        # Domain size scaling
        domain_scaling = self._test_domain_scaling(domain_sizes)
        scaling_results['domain_scaling'] = domain_scaling
        
        # Time range scaling
        time_scaling = self._test_time_scaling(time_ranges)
        scaling_results['time_scaling'] = time_scaling
        
        # Overall scaling analysis
        overall_scaling = self._analyze_overall_scaling(scaling_results)
        scaling_results['overall_scaling'] = overall_scaling
        
        return scaling_results
    
    def _test_grid_scaling(self, grid_sizes: List[int]) -> Dict[str, Any]:
        """Test scaling with grid size."""
        results = []
        
        for grid_size in grid_sizes:
            print(f"Testing grid size: {grid_size}")
            
            # Create configuration
            config = self.config.copy()
            config['N'] = grid_size
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run simulation
            self._run_simulation(config)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            results.append({
                'grid_size': grid_size,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'cpu_usage': psutil.cpu_percent()
            })
        
        # Fit scaling law
        grid_sizes_array = np.array([r['grid_size'] for r in results])
        times_array = np.array([r['execution_time'] for r in results])
        
        # Fit power law: T(N) = a * N^b
        def power_law(N, a, b):
            return a * np.power(N, b)
        
        try:
            popt, pcov = curve_fit(power_law, grid_sizes_array, times_array)
            scaling_exponent = popt[1]
            scaling_coefficient = popt[0]
        except:
            scaling_exponent = 0.0
            scaling_coefficient = 0.0
        
        return {
            'results': results,
            'scaling_exponent': scaling_exponent,
            'scaling_coefficient': scaling_coefficient,
            'fitted_curve': power_law(grid_sizes_array, scaling_coefficient, scaling_exponent).tolist()
        }
    
    def _test_domain_scaling(self, domain_sizes: List[float]) -> Dict[str, Any]:
        """Test scaling with domain size."""
        results = []
        
        for domain_size in domain_sizes:
            print(f"Testing domain size: {domain_size}")
            
            # Create configuration
            config = self.config.copy()
            config['L'] = domain_size
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run simulation
            self._run_simulation(config)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            results.append({
                'domain_size': domain_size,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'cpu_usage': psutil.cpu_percent()
            })
        
        # Fit scaling law
        domain_sizes_array = np.array([r['domain_size'] for r in results])
        times_array = np.array([r['execution_time'] for r in results])
        
        # Fit power law: T(L) = a * L^b
        def power_law(L, a, b):
            return a * np.power(L, b)
        
        try:
            popt, pcov = curve_fit(power_law, domain_sizes_array, times_array)
            scaling_exponent = popt[1]
            scaling_coefficient = popt[0]
        except:
            scaling_exponent = 0.0
            scaling_coefficient = 0.0
        
        return {
            'results': results,
            'scaling_exponent': scaling_exponent,
            'scaling_coefficient': scaling_coefficient,
            'fitted_curve': power_law(domain_sizes_array, scaling_coefficient, scaling_exponent).tolist()
        }
    
    def _test_time_scaling(self, time_ranges: List[float]) -> Dict[str, Any]:
        """Test scaling with time range."""
        results = []
        
        for time_range in time_ranges:
            print(f"Testing time range: {time_range}")
            
            # Create configuration
            config = self.config.copy()
            config['t_max'] = time_range
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run simulation
            self._run_simulation(config)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            results.append({
                'time_range': time_range,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'cpu_usage': psutil.cpu_percent()
            })
        
        # Fit scaling law
        time_ranges_array = np.array([r['time_range'] for r in results])
        times_array = np.array([r['execution_time'] for r in results])
        
        # Fit power law: T(t) = a * t^b
        def power_law(t, a, b):
            return a * np.power(t, b)
        
        try:
            popt, pcov = curve_fit(power_law, time_ranges_array, times_array)
            scaling_exponent = popt[1]
            scaling_coefficient = popt[0]
        except:
            scaling_exponent = 0.0
            scaling_coefficient = 0.0
        
        return {
            'results': results,
            'scaling_exponent': scaling_exponent,
            'scaling_coefficient': scaling_coefficient,
            'fitted_curve': power_law(time_ranges_array, scaling_coefficient, scaling_exponent).tolist()
        }
    
    def _analyze_overall_scaling(self, scaling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall scaling behavior."""
        # Extract scaling exponents
        grid_exponent = scaling_results['grid_scaling']['scaling_exponent']
        domain_exponent = scaling_results['domain_scaling']['scaling_exponent']
        time_exponent = scaling_results['time_scaling']['scaling_exponent']
        
        # Overall scaling assessment
        if grid_exponent > 3.0:
            grid_assessment = 'poor'
        elif grid_exponent > 2.0:
            grid_assessment = 'fair'
        elif grid_exponent > 1.0:
            grid_assessment = 'good'
        else:
            grid_assessment = 'excellent'
        
        if domain_exponent > 2.0:
            domain_assessment = 'poor'
        elif domain_exponent > 1.5:
            domain_assessment = 'fair'
        elif domain_exponent > 1.0:
            domain_assessment = 'good'
        else:
            domain_assessment = 'excellent'
        
        if time_exponent > 2.0:
            time_assessment = 'poor'
        elif time_exponent > 1.5:
            time_assessment = 'fair'
        elif time_exponent > 1.0:
            time_assessment = 'good'
        else:
            time_assessment = 'excellent'
        
        return {
            'grid_scaling': {
                'exponent': grid_exponent,
                'assessment': grid_assessment
            },
            'domain_scaling': {
                'exponent': domain_exponent,
                'assessment': domain_assessment
            },
            'time_scaling': {
                'exponent': time_exponent,
                'assessment': time_assessment
            },
            'overall_assessment': self._compute_overall_assessment([grid_assessment, domain_assessment, time_assessment])
        }
    
    def _compute_overall_assessment(self, assessments: List[str]) -> str:
        """Compute overall assessment from individual assessments."""
        if 'poor' in assessments:
            return 'poor'
        elif 'fair' in assessments:
            return 'fair'
        elif 'good' in assessments:
            return 'good'
        else:
            return 'excellent'
    
    def _analyze_accuracy_cost_tradeoffs(self) -> Dict[str, Any]:
        """Analyze accuracy vs cost trade-offs."""
        # Test different accuracy levels
        accuracy_levels = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        
        results = []
        
        for accuracy in accuracy_levels:
            print(f"Testing accuracy level: {accuracy}")
            
            # Create configuration with specified accuracy
            config = self.config.copy()
            config['tolerance'] = accuracy
            config['max_iterations'] = int(1000 / accuracy)  # Scale iterations with accuracy
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run simulation
            simulation_result = self._run_simulation(config)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Compute actual accuracy achieved
            actual_accuracy = self._compute_actual_accuracy(simulation_result)
            
            results.append({
                'target_accuracy': accuracy,
                'actual_accuracy': actual_accuracy,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'cost_accuracy_ratio': execution_time / actual_accuracy
            })
        
        # Analyze trade-offs
        tradeoff_analysis = self._analyze_tradeoffs(results)
        
        return {
            'results': results,
            'tradeoff_analysis': tradeoff_analysis
        }
    
    def _compute_actual_accuracy(self, simulation_result: Dict[str, Any]) -> float:
        """Compute actual accuracy achieved in simulation."""
        # Placeholder implementation - in real case, this would compute
        # the actual accuracy based on convergence criteria
        
        # Simulate accuracy based on tolerance
        tolerance = simulation_result.get('tolerance', 0.001)
        actual_accuracy = tolerance * np.random.uniform(0.5, 1.5)
        
        return actual_accuracy
    
    def _analyze_tradeoffs(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze accuracy vs cost trade-offs."""
        # Extract data
        target_accuracies = [r['target_accuracy'] for r in results]
        actual_accuracies = [r['actual_accuracy'] for r in results]
        execution_times = [r['execution_time'] for r in results]
        cost_accuracy_ratios = [r['cost_accuracy_ratio'] for r in results]
        
        # Compute efficiency metrics
        efficiency_scores = []
        for i in range(len(results)):
            # Efficiency = accuracy / cost
            efficiency = actual_accuracies[i] / execution_times[i]
            efficiency_scores.append(efficiency)
        
        # Find optimal point
        optimal_index = np.argmax(efficiency_scores)
        optimal_accuracy = target_accuracies[optimal_index]
        optimal_time = execution_times[optimal_index]
        
        # Analyze trends
        accuracy_trend = np.polyfit(target_accuracies, actual_accuracies, 1)[0]
        cost_trend = np.polyfit(target_accuracies, execution_times, 1)[0]
        
        return {
            'efficiency_scores': efficiency_scores,
            'optimal_accuracy': optimal_accuracy,
            'optimal_time': optimal_time,
            'accuracy_trend': accuracy_trend,
            'cost_trend': cost_trend,
            'recommendations': self._generate_optimization_recommendations(results, optimal_index)
        }
    
    def _generate_optimization_recommendations(self, results: List[Dict[str, Any]], 
                                             optimal_index: int) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check if optimal point is at boundary
        if optimal_index == 0:
            recommendations.append("Consider testing even higher accuracy levels")
        elif optimal_index == len(results) - 1:
            recommendations.append("Consider testing lower accuracy levels for faster execution")
        
        # Check efficiency distribution
        efficiency_scores = [r['cost_accuracy_ratio'] for r in results]
        if np.std(efficiency_scores) < 0.1:
            recommendations.append("Efficiency is relatively uniform across accuracy levels")
        else:
            recommendations.append("Significant efficiency variations detected")
        
        # Check memory usage
        memory_usage = [r['memory_usage'] for r in results]
        if max(memory_usage) > 1000:  # 1GB threshold
            recommendations.append("High memory usage detected, consider optimization")
        
        return recommendations
    
    def _run_benchmark_tests(self) -> Dict[str, Any]:
        """Run benchmark tests for regression testing."""
        benchmark_results = {}
        
        for case_name, benchmark_function in self.benchmark_cases.items():
            print(f"Running benchmark: {case_name}")
            
            # Run benchmark
            start_time = time.time()
            result = benchmark_function()
            end_time = time.time()
            
            benchmark_results[case_name] = {
                'result': result,
                'execution_time': end_time - start_time,
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
            }
        
        return benchmark_results
    
    def _benchmark_single_soliton(self) -> Dict[str, Any]:
        """Benchmark single soliton simulation."""
        # Placeholder implementation
        return {
            'energy': 1.0,
            'topological_charge': 1.0,
            'stability': True
        }
    
    def _benchmark_defect_pair(self) -> Dict[str, Any]:
        """Benchmark defect pair simulation."""
        # Placeholder implementation
        return {
            'interaction_energy': 0.5,
            'separation': 2.0,
            'annihilation_time': 10.0
        }
    
    def _benchmark_multi_defect_system(self) -> Dict[str, Any]:
        """Benchmark multi-defect system simulation."""
        # Placeholder implementation
        return {
            'total_energy': 5.0,
            'defect_count': 4,
            'equilibrium_time': 20.0
        }
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        # Test memory usage for different problem sizes
        grid_sizes = [64, 128, 256, 512]
        memory_usage = []
        
        for grid_size in grid_sizes:
            config = self.config.copy()
            config['N'] = grid_size
            
            # Measure memory usage
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run simulation
            self._run_simulation(config)
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            memory_usage.append({
                'grid_size': grid_size,
                'memory_usage': end_memory - start_memory
            })
        
        # Analyze memory scaling
        grid_sizes_array = np.array([m['grid_size'] for m in memory_usage])
        memory_array = np.array([m['memory_usage'] for m in memory_usage])
        
        # Fit memory scaling law
        def memory_law(N, a, b):
            return a * np.power(N, b)
        
        try:
            popt, pcov = curve_fit(memory_law, grid_sizes_array, memory_array)
            memory_scaling_exponent = popt[1]
            memory_scaling_coefficient = popt[0]
        except:
            memory_scaling_exponent = 0.0
            memory_scaling_coefficient = 0.0
        
        return {
            'memory_usage': memory_usage,
            'scaling_exponent': memory_scaling_exponent,
            'scaling_coefficient': memory_scaling_coefficient,
            'fitted_curve': memory_law(grid_sizes_array, memory_scaling_coefficient, memory_scaling_exponent).tolist()
        }
    
    def _optimize_parameters(self) -> Dict[str, Any]:
        """Optimize parameters for best performance."""
        # Test different parameter combinations
        parameter_combinations = [
            {'N': 256, 'dt': 0.01, 'tolerance': 0.001},
            {'N': 512, 'dt': 0.005, 'tolerance': 0.0005},
            {'N': 128, 'dt': 0.02, 'tolerance': 0.002},
            {'N': 384, 'dt': 0.015, 'tolerance': 0.0015}
        ]
        
        optimization_results = []
        
        for params in parameter_combinations:
            config = self.config.copy()
            config.update(params)
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run simulation
            simulation_result = self._run_simulation(config)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Compute efficiency score
            efficiency = self._compute_efficiency_score(simulation_result, execution_time, memory_usage)
            
            optimization_results.append({
                'parameters': params,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'efficiency_score': efficiency,
                'simulation_result': simulation_result
            })
        
        # Find optimal parameters
        optimal_index = np.argmax([r['efficiency_score'] for r in optimization_results])
        optimal_parameters = optimization_results[optimal_index]
        
        return {
            'optimization_results': optimization_results,
            'optimal_parameters': optimal_parameters,
            'recommendations': self._generate_parameter_recommendations(optimization_results, optimal_index)
        }
    
    def _compute_efficiency_score(self, simulation_result: Dict[str, Any], 
                                execution_time: float, memory_usage: float) -> float:
        """Compute efficiency score for parameter combination."""
        # Placeholder implementation - in real case, this would compute
        # a comprehensive efficiency score based on accuracy, time, and memory
        
        # Simple efficiency score
        accuracy = simulation_result.get('accuracy', 0.001)
        
        # Avoid division by zero
        if execution_time <= 0 or memory_usage <= 0:
            return 0.0
        
        efficiency = accuracy / (execution_time * memory_usage)
        
        return efficiency
    
    def _generate_parameter_recommendations(self, optimization_results: List[Dict[str, Any]], 
                                          optimal_index: int) -> List[str]:
        """Generate parameter optimization recommendations."""
        recommendations = []
        
        optimal_result = optimization_results[optimal_index]
        optimal_params = optimal_result['parameters']
        
        # Analyze parameter sensitivity
        for param_name in ['N', 'dt', 'tolerance']:
            param_values = [r['parameters'][param_name] for r in optimization_results]
            param_efficiencies = [r['efficiency_score'] for r in optimization_results]
            
            # Check if parameter has significant impact
            if np.std(param_efficiencies) > 0.1:
                recommendations.append(f"Parameter {param_name} has significant impact on efficiency")
            
            # Check if optimal value is at boundary
            if optimal_params[param_name] == min(param_values):
                recommendations.append(f"Consider testing lower values of {param_name}")
            elif optimal_params[param_name] == max(param_values):
                recommendations.append(f"Consider testing higher values of {param_name}")
        
        return recommendations
    
    def _run_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run simulation with given configuration.
        
        Physical Meaning:
            Executes the 7D phase field simulation with specified
            parameters and returns key observables.
        """
        # Placeholder implementation - in real case, this would run
        # the actual 7D phase field simulation
        
        # Simulate some computation time based on parameters
        N = config.get('N', 256)
        dt = config.get('dt', 0.01)
        tolerance = config.get('tolerance', 0.001)
        
        # Simulate computation time
        time.sleep(0.01 * (N / 256) * (0.01 / dt) * (0.001 / tolerance))
        
        # Return simulated results
        return {
            'accuracy': tolerance * np.random.uniform(0.5, 1.5),
            'energy': np.random.uniform(0.8, 1.2),
            'topological_charge': 1.0 + np.random.normal(0, 0.01),
            'convergence_rate': np.random.uniform(0.8, 1.0)
        }
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save performance analysis results to file.
        
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
