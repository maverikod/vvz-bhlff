"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Time step stability analysis for Level E experiments.

This module implements time step stability analysis for the 7D phase field theory,
investigating numerical stability of time integration schemes.

Theoretical Background:
    Time step stability analysis investigates numerical stability of time integration
    schemes and optimal time step selection.

Mathematical Foundation:
    Analyzes stability of observables with respect to time step variations,
    establishing stability boundaries and optimal time step selection.

Example:
    >>> analyzer = TimeStabilityAnalyzer(reference_config)
    >>> results = analyzer.analyze_time_step_stability(time_steps)
"""

import numpy as np
from typing import Dict, Any, List


class TimeStabilityAnalyzer:
    """
    Time step stability analysis for discretization effects.

    Physical Meaning:
        Investigates numerical stability of time integration
        schemes and optimal time step selection.
    """

    def __init__(self, reference_config: Dict[str, Any]):
        """
        Initialize time stability analyzer.

        Args:
            reference_config: Reference configuration for comparison
        """
        self.reference_config = reference_config
        self._setup_convergence_metrics()

    def _setup_convergence_metrics(self) -> None:
        """Setup metrics for convergence analysis."""
        self.convergence_metrics = [
            "power_law_exponent",
            "topological_charge",
            "energy",
            "quality_factor",
            "stability",
        ]

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

            results[dt] = {"config": config, "output": output, "metrics": metrics}

        # Analyze time step stability
        stability_analysis = self._analyze_time_step_stability(results)

        return {"time_step_results": results, "stability_analysis": stability_analysis}

    def _create_time_step_config(self, dt: float) -> Dict[str, Any]:
        """Create configuration with specified time step."""
        config = self.reference_config.copy()
        config["dt"] = dt

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
        N = config.get("N", 256)
        L = config.get("L", 20.0)
        beta = config.get("beta", 1.0)
        mu = config.get("mu", 1.0)

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
            "power_law_exponent": power_law_exponent,
            "topological_charge": topological_charge,
            "energy": energy,
            "quality_factor": quality_factor,
            "stability": stability,
            "grid_spacing": dx,
            "grid_size": N,
        }

    def _compute_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Compute convergence metrics from simulation output."""
        metrics = {}

        for metric in self.convergence_metrics:
            if metric in output:
                metrics[metric] = output[metric]

        return metrics

    def _analyze_time_step_stability(
        self, results: Dict[float, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze time step stability."""
        time_steps = sorted(results.keys())
        stability_metrics = {}

        for metric in self.convergence_metrics:
            if metric in results[time_steps[0]]["metrics"]:
                # Extract values for this metric
                values = [results[dt]["metrics"][metric] for dt in time_steps]

                # Analyze stability
                stability = self._analyze_metric_stability(time_steps, values)
                stability_metrics[metric] = stability

        # Overall stability analysis
        overall_stability = self._analyze_overall_stability(stability_metrics)

        return {
            "stability_metrics": stability_metrics,
            "overall_stability": overall_stability,
            "time_steps": time_steps,
        }

    def _analyze_metric_stability(
        self, time_steps: List[float], values: List[float]
    ) -> Dict[str, Any]:
        """Analyze stability of a metric with respect to time step."""
        if len(values) < 2:
            return {"stability": "insufficient_data", "score": 0.0}

        # Compute relative changes
        relative_changes = []
        for i in range(len(values) - 1):
            if values[i + 1] != 0:
                rel_change = abs(values[i] - values[i + 1]) / abs(values[i + 1])
                relative_changes.append(rel_change)

        # Assess stability
        max_change = max(relative_changes) if relative_changes else 0.0
        mean_change = np.mean(relative_changes) if relative_changes else 0.0

        if max_change < 0.01:
            stability = "excellent"
            score = 1.0
        elif max_change < 0.05:
            stability = "good"
            score = 0.8
        elif max_change < 0.1:
            stability = "fair"
            score = 0.6
        else:
            stability = "poor"
            score = 0.3

        return {
            "stability": stability,
            "score": score,
            "max_change": max_change,
            "mean_change": mean_change,
        }

    def _analyze_overall_stability(
        self, stability_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze overall time step stability."""
        scores = [metrics["score"] for metrics in stability_metrics.values()]

        if not scores:
            return {"overall_score": 0.0, "stability": "unknown"}

        overall_score = np.mean(scores)

        if overall_score > 0.8:
            stability = "excellent"
        elif overall_score > 0.6:
            stability = "good"
        elif overall_score > 0.4:
            stability = "fair"
        else:
            stability = "poor"

        return {
            "overall_score": overall_score,
            "stability": stability,
            "individual_scores": {
                metric: metrics["score"]
                for metric, metrics in stability_metrics.items()
            },
        }
