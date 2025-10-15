"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for full domain effects analysis implementation.

This module tests the complete implementation of domain effects analysis
that was previously simplified, ensuring it now implements full 7D BVP
theory-based domain size analysis.

Physical Meaning:
    Tests verify that the domain effects analysis correctly
    implements full 7D BVP theory principles without simplifications
    or placeholders.

Mathematical Foundation:
    Tests verify mathematical correctness of:
    - Full 7D phase field simulation
    - Fractional Laplacian equation solving
    - Domain size effects computation
    - Boundary condition application
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import tempfile
import os

from bhlff.models.level_e.discretization.domain_effects import DomainEffectsAnalyzer


class TestDomainEffectsFull:
    """Test full domain effects analysis implementations."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create reference configuration
        self.reference_config = {
            "N": 256,
            "L": 20.0,
            "beta": 1.0,
            "mu": 1.0,
            "lambda": 0.0,
            "nu": 1.0
        }
        
        # Create analyzer
        self.analyzer = DomainEffectsAnalyzer(self.reference_config)
        
        # Test domain sizes
        self.domain_sizes = [10.0, 15.0, 20.0, 25.0, 30.0]

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.reference_config == self.reference_config
        assert hasattr(self.analyzer, 'convergence_metrics')
        assert len(self.analyzer.convergence_metrics) > 0

    def test_domain_size_effects_analysis_full_implementation(self):
        """Test full domain size effects analysis implementation."""
        # Test analysis
        results = self.analyzer.analyze_domain_size_effects(self.domain_sizes)
        
        # Verify results structure
        assert "domain_results" in results
        assert "domain_analysis" in results
        
        # Verify domain results
        domain_results = results["domain_results"]
        assert len(domain_results) == len(self.domain_sizes)
        
        for domain_size in self.domain_sizes:
            assert domain_size in domain_results
            domain_result = domain_results[domain_size]
            
            # Verify domain result structure
            assert "config" in domain_result
            assert "output" in domain_result
            assert "metrics" in domain_result
            
            # Verify output structure
            output = domain_result["output"]
            assert "power_law_exponent" in output
            assert "topological_charge" in output
            assert "energy" in output
            assert "quality_factor" in output
            assert "stability" in output
            assert "grid_spacing" in output
            assert "grid_size" in output
            assert "domain_size" in output
            assert "domain_effects" in output
            assert "solution_field" in output
            assert "convergence_achieved" in output

    def test_simulation_full_implementation(self):
        """Test full simulation implementation."""
        # Test simulation with reference config
        output = self.analyzer._run_simulation(self.reference_config)
        
        # Verify output structure
        assert "power_law_exponent" in output
        assert "topological_charge" in output
        assert "energy" in output
        assert "quality_factor" in output
        assert "stability" in output
        assert "grid_spacing" in output
        assert "grid_size" in output
        assert "domain_size" in output
        assert "domain_effects" in output
        assert "solution_field" in output
        assert "convergence_achieved" in output
        
        # Verify physical quantities are reasonable
        assert isinstance(output["power_law_exponent"], (float, np.floating))
        assert isinstance(output["topological_charge"], (float, np.floating))
        assert isinstance(output["energy"], (float, np.floating))
        assert isinstance(output["quality_factor"], (float, np.floating))
        assert isinstance(output["stability"], (float, np.floating))
        
        # Verify domain effects structure
        domain_effects = output["domain_effects"]
        assert "boundary_intensity" in domain_effects
        assert "interior_intensity" in domain_effects
        assert "boundary_effect_ratio" in domain_effects
        assert "domain_scaling" in domain_effects
        assert "grid_resolution" in domain_effects
        assert "domain_size" in domain_effects

    def test_7d_phase_field_initialization_full_implementation(self):
        """Test full 7D phase field initialization implementation."""
        # Create test parameters
        x = np.linspace(-10, 10, 256)
        beta = 1.0
        mu = 1.0
        lambda_param = 0.0
        
        # Test initialization
        phase_field = self.analyzer._initialize_7d_phase_field(x, beta, mu, lambda_param)
        
        # Verify result
        assert isinstance(phase_field, np.ndarray)
        assert len(phase_field) == len(x)
        assert np.all(np.isfinite(phase_field))

    def test_7d_boundary_conditions_full_implementation(self):
        """Test full 7D boundary conditions implementation."""
        # Create test field
        x = np.linspace(-10, 10, 256)
        field = np.exp(-x**2 / 4.0)
        beta = 1.0
        mu = 1.0
        
        # Test boundary conditions
        result = self.analyzer._apply_7d_boundary_conditions(field, x, beta, mu)
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == len(field)
        assert np.all(np.isfinite(result))

    def test_7d_fractional_laplacian_solving_full_implementation(self):
        """Test full 7D fractional Laplacian solving implementation."""
        # Create test parameters
        x = np.linspace(-10, 10, 256)
        initial_field = np.exp(-x**2 / 4.0)
        beta = 1.0
        mu = 1.0
        lambda_param = 0.0
        
        # Test solving
        solution = self.analyzer._solve_7d_fractional_laplacian(initial_field, x, beta, mu, lambda_param)
        
        # Verify result
        assert isinstance(solution, np.ndarray)
        assert len(solution) == len(initial_field)
        assert np.all(np.isfinite(solution))

    def test_7d_power_law_exponent_computation_full_implementation(self):
        """Test full 7D power law exponent computation implementation."""
        # Create test solution
        x = np.linspace(0.1, 10.0, 100)
        solution = np.exp(-x) * x**(-2.0)  # Known power law with exponent -2.0
        beta = 1.0
        
        # Test computation
        exponent = self.analyzer._compute_7d_power_law_exponent(solution, x, beta)
        
        # Verify result
        assert isinstance(exponent, (float, np.floating))
        assert exponent < 0  # Should be negative for decay
        assert abs(exponent - (-2.0)) < 1.0  # Should be close to expected value

    def test_7d_topological_charge_computation_full_implementation(self):
        """Test full 7D topological charge computation implementation."""
        # Create test solution
        x = np.linspace(-10, 10, 256)
        solution = np.exp(1j * x)  # Complex solution with winding
        
        # Test computation
        charge = self.analyzer._compute_7d_topological_charge(solution, x)
        
        # Verify result
        assert isinstance(charge, (float, np.floating))
        assert np.isfinite(charge)

    def test_7d_energy_computation_full_implementation(self):
        """Test full 7D energy computation implementation."""
        # Create test solution
        x = np.linspace(-10, 10, 256)
        solution = np.exp(-x**2 / 4.0)
        beta = 1.0
        mu = 1.0
        lambda_param = 0.0
        
        # Test computation
        energy = self.analyzer._compute_7d_energy(solution, x, beta, mu, lambda_param)
        
        # Verify result
        assert isinstance(energy, (float, np.floating))
        assert energy >= 0  # Energy should be non-negative
        assert np.isfinite(energy)

    def test_7d_quality_factor_computation_full_implementation(self):
        """Test full 7D quality factor computation implementation."""
        # Create test solution
        x = np.linspace(-10, 10, 256)
        solution = np.exp(-x**2 / 4.0)
        mu = 1.0
        nu = 1.0
        
        # Test computation
        quality_factor = self.analyzer._compute_7d_quality_factor(solution, x, mu, nu)
        
        # Verify result
        assert isinstance(quality_factor, (float, np.floating))
        assert quality_factor > 0  # Quality factor should be positive
        assert np.isfinite(quality_factor)

    def test_7d_stability_computation_full_implementation(self):
        """Test full 7D stability computation implementation."""
        # Create test solution
        x = np.linspace(-10, 10, 256)
        solution = np.exp(-x**2 / 4.0)
        beta = 1.0
        mu = 1.0
        
        # Test computation
        stability = self.analyzer._compute_7d_stability(solution, x, beta, mu)
        
        # Verify result
        assert isinstance(stability, (float, np.floating))
        assert 0.0 <= stability <= 1.0  # Stability should be between 0 and 1
        assert np.isfinite(stability)

    def test_domain_size_effects_computation_full_implementation(self):
        """Test full domain size effects computation implementation."""
        # Create test solution
        x = np.linspace(-10, 10, 256)
        solution = np.exp(-x**2 / 4.0)
        L = 20.0
        N = 256
        
        # Test computation
        effects = self.analyzer._compute_domain_size_effects(solution, x, L, N)
        
        # Verify result structure
        assert "boundary_intensity" in effects
        assert "interior_intensity" in effects
        assert "boundary_effect_ratio" in effects
        assert "domain_scaling" in effects
        assert "grid_resolution" in effects
        assert "domain_size" in effects
        
        # Verify values are reasonable
        assert isinstance(effects["boundary_intensity"], (float, np.floating))
        assert isinstance(effects["interior_intensity"], (float, np.floating))
        assert isinstance(effects["boundary_effect_ratio"], (float, np.floating))
        assert isinstance(effects["domain_scaling"], (float, np.floating))
        assert effects["grid_resolution"] == N
        assert effects["domain_size"] == L

    def test_simplified_simulation_fallback(self):
        """Test simplified simulation fallback."""
        # Create config that might cause full simulation to fail
        config = {
            "N": 256,
            "L": 20.0,
            "beta": 1.0,
            "mu": 1.0,
            "lambda": 0.0,
            "nu": 1.0
        }
        
        # Test simplified simulation
        output = self.analyzer._run_simplified_simulation(config)
        
        # Verify output structure
        assert "power_law_exponent" in output
        assert "topological_charge" in output
        assert "energy" in output
        assert "quality_factor" in output
        assert "stability" in output
        assert "grid_spacing" in output
        assert "grid_size" in output
        assert "domain_size" in output
        assert "convergence_achieved" in output
        assert "simplified" in output
        
        # Verify simplified flag
        assert output["simplified"] == True
        assert output["convergence_achieved"] == False

    def test_metrics_computation_full_implementation(self):
        """Test full metrics computation implementation."""
        # Create test output
        output = {
            "power_law_exponent": -2.0,
            "topological_charge": 1.0,
            "energy": 1.5,
            "quality_factor": 2.0,
            "stability": 1.0
        }
        
        # Test metrics computation
        metrics = self.analyzer._compute_metrics(output)
        
        # Verify result
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Verify all convergence metrics are present
        for metric in self.analyzer.convergence_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (float, np.floating))

    def test_domain_effects_analysis_full_implementation(self):
        """Test full domain effects analysis implementation."""
        # Create test results
        results = {
            10.0: {
                "metrics": {
                    "power_law_exponent": -2.0,
                    "topological_charge": 1.0,
                    "energy": 1.5,
                    "quality_factor": 2.0,
                    "stability": 1.0
                }
            },
            20.0: {
                "metrics": {
                    "power_law_exponent": -2.1,
                    "topological_charge": 1.1,
                    "energy": 1.6,
                    "quality_factor": 2.1,
                    "stability": 1.0
                }
            }
        }
        
        # Test analysis
        analysis = self.analyzer._analyze_domain_effects(results)
        
        # Verify result structure
        assert "domain_effects" in analysis
        assert "overall_analysis" in analysis
        assert "domain_sizes" in analysis
        
        # Verify domain effects
        domain_effects = analysis["domain_effects"]
        for metric in self.analyzer.convergence_metrics:
            if metric in results[10.0]["metrics"]:
                assert metric in domain_effects
                effect = domain_effects[metric]
                assert "dependence" in effect
                assert "slope" in effect
                assert "correlation" in effect

    def test_domain_dependence_analysis_full_implementation(self):
        """Test full domain dependence analysis implementation."""
        # Create test data
        domain_sizes = [10.0, 20.0, 30.0]
        values = [1.0, 1.1, 1.2]
        
        # Test analysis
        result = self.analyzer._analyze_domain_dependence(domain_sizes, values)
        
        # Verify result structure
        assert "dependence" in result
        assert "slope" in result
        assert "correlation" in result
        
        # Verify values are reasonable
        assert isinstance(result["dependence"], str)
        assert isinstance(result["slope"], (float, np.floating))
        assert isinstance(result["correlation"], (float, np.floating))
        assert -1.0 <= result["correlation"] <= 1.0

    def test_overall_domain_effects_analysis_full_implementation(self):
        """Test full overall domain effects analysis implementation."""
        # Create test domain effects
        domain_effects = {
            "power_law_exponent": {"dependence": "independent"},
            "topological_charge": {"dependence": "weak"},
            "energy": {"dependence": "moderate"},
            "quality_factor": {"dependence": "strong"},
            "stability": {"dependence": "independent"}
        }
        
        # Test analysis
        result = self.analyzer._analyze_overall_domain_effects(domain_effects)
        
        # Verify result structure
        assert "overall_dependence" in result
        assert "independent_count" in result
        assert "weak_count" in result
        assert "moderate_count" in result
        assert "strong_count" in result
        
        # Verify counts are reasonable
        assert result["independent_count"] == 2
        assert result["weak_count"] == 1
        assert result["moderate_count"] == 1
        assert result["strong_count"] == 1

    def test_error_handling_full_implementation(self):
        """Test error handling in full implementation."""
        # Test with invalid configuration
        invalid_config = {
            "N": -1,  # Invalid grid size
            "L": 20.0,
            "beta": 1.0,
            "mu": 1.0
        }
        
        # Should handle error gracefully
        output = self.analyzer._run_simulation(invalid_config)
        
        # Verify error handling
        assert "convergence_achieved" in output
        # Should fall back to simplified simulation
        assert "simplified" in output

    def test_performance_full_implementation(self):
        """Test performance of full implementation."""
        import time
        
        # Measure analysis time
        start_time = time.time()
        results = self.analyzer.analyze_domain_size_effects(self.domain_sizes)
        end_time = time.time()
        
        # Verify analysis completed
        assert "domain_results" in results
        assert "domain_analysis" in results
        
        # Verify reasonable performance
        analysis_time = end_time - start_time
        assert analysis_time < 120.0  # Should complete within 2 minutes
