"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Domain size effects analysis for Level E experiments.

This module implements domain size effects analysis for the 7D phase field theory,
investigating how the finite computational domain affects results.

Theoretical Background:
    Domain size effects analysis investigates how the finite computational domain
    affects results, particularly for long-range interactions and boundary effects.

Mathematical Foundation:
    Analyzes dependence of observables on domain size L, investigating
    finite-size scaling and boundary effects.

Example:
    >>> analyzer = DomainEffectsAnalyzer(reference_config)
    >>> results = analyzer.analyze_domain_size_effects(domain_sizes)
"""

import numpy as np
from typing import Dict, Any, List


class DomainEffectsAnalyzer:
    """
    Domain size effects analysis for discretization effects.

    Physical Meaning:
        Investigates how the finite computational domain
        affects results, particularly for long-range
        interactions and boundary effects.
    """

    def __init__(self, reference_config: Dict[str, Any]):
        """
        Initialize domain effects analyzer.

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
                "config": config,
                "output": output,
                "metrics": metrics,
            }

        # Analyze domain size effects
        domain_analysis = self._analyze_domain_effects(results)

        return {"domain_results": results, "domain_analysis": domain_analysis}

    def _create_domain_config(self, domain_size: float) -> Dict[str, Any]:
        """Create configuration with specified domain size."""
        config = self.reference_config.copy()
        config["L"] = domain_size

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

    def _analyze_domain_effects(
        self, results: Dict[float, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze effects of domain size on results."""
        domain_sizes = sorted(results.keys())
        domain_effects = {}

        for metric in self.convergence_metrics:
            if metric in results[domain_sizes[0]]["metrics"]:
                # Extract values for this metric
                values = [
                    results[domain_size]["metrics"][metric]
                    for domain_size in domain_sizes
                ]

                # Analyze domain size dependence
                dependence = self._analyze_domain_dependence(domain_sizes, values)
                domain_effects[metric] = dependence

        # Overall domain size analysis
        overall_analysis = self._analyze_overall_domain_effects(domain_effects)

        return {
            "domain_effects": domain_effects,
            "overall_analysis": overall_analysis,
            "domain_sizes": domain_sizes,
        }

    def _analyze_domain_dependence(
        self, domain_sizes: List[float], values: List[float]
    ) -> Dict[str, Any]:
        """Analyze dependence of metric on domain size."""
        if len(values) < 2:
            return {"dependence": "insufficient_data", "slope": 0.0}

        # Compute slope of values vs domain size
        slope = np.polyfit(domain_sizes, values, 1)[0]

        # Assess dependence
        if abs(slope) < 0.01:
            dependence = "independent"
        elif abs(slope) < 0.1:
            dependence = "weak"
        elif abs(slope) < 1.0:
            dependence = "moderate"
        else:
            dependence = "strong"

        # Compute correlation safely
        try:
            correlation = np.corrcoef(domain_sizes, values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0

        return {"dependence": dependence, "slope": slope, "correlation": correlation}

    def _analyze_overall_domain_effects(
        self, domain_effects: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze overall domain size effects."""
        dependencies = [effects["dependence"] for effects in domain_effects.values()]

        # Count different types of dependence
        independent_count = dependencies.count("independent")
        weak_count = dependencies.count("weak")
        moderate_count = dependencies.count("moderate")
        strong_count = dependencies.count("strong")

        # Overall assessment
        if independent_count > len(dependencies) / 2:
            overall_dependence = "independent"
        elif weak_count > len(dependencies) / 2:
            overall_dependence = "weak"
        elif moderate_count > len(dependencies) / 2:
            overall_dependence = "moderate"
        else:
            overall_dependence = "strong"

        return {
            "overall_dependence": overall_dependence,
            "independent_count": independent_count,
            "weak_count": weak_count,
            "moderate_count": moderate_count,
            "strong_count": strong_count,
        }
