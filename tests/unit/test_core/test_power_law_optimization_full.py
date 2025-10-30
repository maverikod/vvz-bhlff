"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for full power law optimization algorithms.

This module tests the complete implementation of power law optimization
algorithms that were previously simplified, ensuring they now implement
full 7D BVP theory-based optimization.

Physical Meaning:
    Tests verify that the power law optimization algorithms correctly
    implement full 7D BVP theory principles without simplifications
    or placeholders.

Mathematical Foundation:
    Tests verify mathematical correctness of:
    - Full optimization using scipy.optimize.minimize
    - Iterative refinement with gradient-based methods
    - Parameter sensitivity analysis
    - Quality assessment metrics
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import tempfile
import os

from bhlff.core.bvp.power_law.power_law_optimization import PowerLawOptimization
from bhlff.core.bvp import BVPCore
from bhlff.core.domain import Domain


class TestPowerLawOptimizationFull:
    """Test full power law optimization implementations."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test domain
        self.domain = Domain(L=20.0, N=256, N_phi=4, N_t=8, T=1.0)

        # Create BVP core
        self.bvp_core = BVPCore(domain=self.domain)

        # Create optimizer
        self.optimizer = PowerLawOptimization(bvp_core=self.bvp_core)

        # Create test envelope
        self.test_envelope = self._create_test_envelope()

    def _create_test_envelope(self) -> np.ndarray:
        """Create test envelope for optimization."""
        # Create synthetic 7D envelope with known power law structure
        x = np.linspace(-10, 10, 256)
        y = np.linspace(-10, 10, 256)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Create envelope with power law structure
        r = np.sqrt(X**2 + Y**2)
        envelope = np.exp(-r / 5.0) * r ** (-2.0)  # Power law with exponent -2.0

        return envelope

    def test_optimization_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.bvp_core is not None
        assert self.optimizer.optimization_tolerance == 1e-6
        assert self.optimizer.max_optimization_iterations == 100

    def test_optimize_power_law_fits_full_implementation(self):
        """Test full power law optimization implementation."""
        # Test optimization with full implementation
        results = self.optimizer.optimize_power_law_fits(self.test_envelope)

        # Verify results structure
        assert "optimization_successful" in results
        assert "successful_regions" in results
        assert "total_regions" in results
        assert "success_rate" in results
        assert "average_improvement" in results
        assert "optimization_quality" in results
        assert "region_results" in results
        assert "convergence_achieved" in results

        # Verify optimization was attempted
        assert results["total_regions"] > 0
        assert 0.0 <= results["success_rate"] <= 1.0
        assert 0.0 <= results["average_improvement"] <= 10.0

    def test_region_optimization_full_implementation(self):
        """Test full region optimization implementation."""
        # Create test region
        region = {
            "mask": np.ones((256, 256), dtype=bool),
            "center": np.array([0.0, 0.0]),
            "size": 256 * 256,
            "intensity": 1.0,
        }

        # Test region optimization
        result = self.optimizer._optimize_region_fit(self.test_envelope, region)

        # Verify result structure
        assert "optimization_successful" in result
        assert "optimized_amplitude" in result
        assert "optimized_exponent" in result
        assert "initial_amplitude" in result
        assert "initial_exponent" in result
        assert "improvement" in result
        assert "fit_quality" in result
        assert "convergence_info" in result

        # Verify optimization parameters are reasonable
        assert isinstance(result["optimized_amplitude"], float)
        assert isinstance(result["optimized_exponent"], float)
        assert result["optimized_amplitude"] > 0
        assert result["optimized_exponent"] < 0  # Should be negative for decay

    def test_iterative_refinement_full_implementation(self):
        """Test full iterative refinement implementation."""
        # Create test region data
        region_data = {
            "r": np.linspace(0.1, 10.0, 100),
            "values": np.exp(-np.linspace(0.1, 10.0, 100))
            * np.linspace(0.1, 10.0, 100) ** (-2.0),
        }

        # Create initial fit
        initial_fit = {"amplitude": 1.0, "exponent": -2.0}

        # Test iterative refinement
        result = self.optimizer._iterative_refinement(region_data, initial_fit)

        # Verify result structure
        assert "refined_amplitude" in result
        assert "refined_exponent" in result
        assert "convergence_achieved" in result
        assert "iterations" in result
        assert "improvement" in result
        assert "final_objective" in result

        # Verify refinement parameters are reasonable
        assert isinstance(result["refined_amplitude"], float)
        assert isinstance(result["refined_exponent"], float)
        assert result["refined_amplitude"] > 0
        assert result["refined_exponent"] < 0  # Should be negative for decay

    def test_parameter_adjustment_full_implementation(self):
        """Test full parameter adjustment implementation."""
        # Create test fit parameters
        fit_params = {"amplitude": 1.0, "exponent": -2.0}

        # Test parameter adjustment
        result = self.optimizer._adjust_fit_parameters(fit_params)

        # Verify result structure
        assert "amplitude" in result
        assert "exponent" in result
        assert "amplitude_adjustment" in result
        assert "exponent_adjustment" in result
        assert "amplitude_sensitivity" in result
        assert "exponent_sensitivity" in result

        # Verify adjusted parameters are reasonable
        assert isinstance(result["amplitude"], float)
        assert isinstance(result["exponent"], float)
        assert result["amplitude"] > 0
        assert result["exponent"] < 0  # Should be negative for decay

    def test_optimization_quality_calculation_full_implementation(self):
        """Test full optimization quality calculation implementation."""
        # Create test optimization results
        optimized_results = [
            {
                "optimization_successful": True,
                "improvement": 0.1,
                "convergence_achieved": True,
                "fit_quality": 0.8,
            },
            {
                "optimization_successful": True,
                "improvement": 0.15,
                "convergence_achieved": True,
                "fit_quality": 0.9,
            },
            {
                "optimization_successful": False,
                "improvement": 0.0,
                "convergence_achieved": False,
                "fit_quality": 0.0,
            },
        ]

        # Test quality calculation
        result = self.optimizer._calculate_optimization_quality(optimized_results)

        # Verify result structure
        assert "average_improvement" in result
        assert "optimization_success_rate" in result
        assert "overall_quality" in result
        assert "total_improvement" in result
        assert "convergence_rate" in result
        assert "average_fit_quality" in result
        assert "successful_optimizations" in result
        assert "total_optimizations" in result

        # Verify quality metrics are reasonable
        assert 0.0 <= result["optimization_success_rate"] <= 1.0
        assert 0.0 <= result["overall_quality"] <= 1.0
        assert result["successful_optimizations"] == 2
        assert result["total_optimizations"] == 3

    def test_region_extraction_full_implementation(self):
        """Test full region extraction implementation."""
        # Test region extraction
        regions = self.optimizer._extract_optimization_regions(self.test_envelope)

        # Verify regions were extracted
        assert isinstance(regions, list)
        assert len(regions) > 0

        # Verify region structure
        for region in regions:
            assert "mask" in region
            assert "center" in region
            assert "size" in region
            assert "intensity" in region

            # Verify region properties
            assert isinstance(region["mask"], np.ndarray)
            assert isinstance(region["center"], np.ndarray)
            assert isinstance(region["size"], (int, np.integer))
            assert isinstance(region["intensity"], (float, np.floating))

    def test_region_data_extraction_full_implementation(self):
        """Test full region data extraction implementation."""
        # Create test region
        region = {
            "mask": np.ones((256, 256), dtype=bool),
            "center": np.array([0.0, 0.0]),
            "size": 256 * 256,
            "intensity": 1.0,
        }

        # Test region data extraction
        result = self.optimizer._extract_region_data(self.test_envelope, region)

        # Verify result structure
        assert "r" in result
        assert "values" in result

        # Verify data arrays
        assert isinstance(result["r"], np.ndarray)
        assert isinstance(result["values"], np.ndarray)
        assert len(result["r"]) == len(result["values"])
        assert len(result["r"]) > 0

    def test_initial_parameter_estimation_full_implementation(self):
        """Test full initial parameter estimation implementation."""
        # Create test region data
        region_data = {
            "r": np.linspace(0.1, 10.0, 100),
            "values": np.exp(-np.linspace(0.1, 10.0, 100))
            * np.linspace(0.1, 10.0, 100) ** (-2.0),
        }

        # Test initial parameter estimation
        result = self.optimizer._get_initial_parameters(region_data)

        # Verify result structure
        assert isinstance(result, np.ndarray)
        assert len(result) == 2  # amplitude and exponent

        # Verify parameters are reasonable
        assert result[0] > 0  # amplitude should be positive
        assert result[1] < 0  # exponent should be negative for decay

    def test_fit_quality_computation_full_implementation(self):
        """Test full fit quality computation implementation."""
        # Create test region data
        region_data = {
            "r": np.linspace(0.1, 10.0, 100),
            "values": np.exp(-np.linspace(0.1, 10.0, 100))
            * np.linspace(0.1, 10.0, 100) ** (-2.0),
        }

        # Create test parameters
        params = np.array([1.0, -2.0])

        # Test fit quality computation
        result = self.optimizer._compute_fit_quality(region_data, params)

        # Verify result
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0  # Quality should be between 0 and 1

    def test_gradient_computation_full_implementation(self):
        """Test full gradient computation implementation."""

        # Create test function
        def test_func(params):
            return np.sum(params**2)

        # Create test parameters
        params = np.array([1.0, -2.0])

        # Test gradient computation
        result = self.optimizer._compute_gradient(test_func, params)

        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == len(params)

        # For the test function f(x,y) = x² + y², gradient should be [2x, 2y]
        expected_gradient = 2 * params
        np.testing.assert_allclose(result, expected_gradient, rtol=1e-5)

    def test_parameter_sensitivity_computation_full_implementation(self):
        """Test full parameter sensitivity computation implementation."""
        # Test amplitude sensitivity
        amplitude_sensitivity = self.optimizer._compute_parameter_sensitivity(
            1.0, "amplitude"
        )
        assert isinstance(amplitude_sensitivity, float)
        assert 0.0 <= amplitude_sensitivity <= 1.0

        # Test exponent sensitivity
        exponent_sensitivity = self.optimizer._compute_parameter_sensitivity(
            -2.0, "exponent"
        )
        assert isinstance(exponent_sensitivity, float)
        assert 0.0 <= exponent_sensitivity <= 1.0

        # Test unknown parameter sensitivity
        unknown_sensitivity = self.optimizer._compute_parameter_sensitivity(
            1.0, "unknown"
        )
        assert isinstance(unknown_sensitivity, float)
        assert unknown_sensitivity == 0.1  # Default sensitivity

    def test_optimization_without_bvp_core(self):
        """Test optimization without BVP core (fallback mode)."""
        # Create optimizer without BVP core
        optimizer_no_core = PowerLawOptimization(bvp_core=None)

        # Test optimization
        results = optimizer_no_core.optimize_power_law_fits(self.test_envelope)

        # Verify results structure
        assert "optimization_successful" in results
        assert "total_regions" in results

        # Should still work with fallback implementation
        assert results["total_regions"] >= 0

    def test_error_handling_full_implementation(self):
        """Test error handling in full implementation."""
        # Test with invalid envelope
        invalid_envelope = np.array([])

        # Should handle error gracefully
        results = self.optimizer.optimize_power_law_fits(invalid_envelope)

        # Verify error handling
        assert "optimization_successful" in results
        assert results["optimization_successful"] == False
        assert "error" in results

    def test_performance_full_implementation(self):
        """Test performance of full implementation."""
        import time

        # Measure optimization time
        start_time = time.time()
        results = self.optimizer.optimize_power_law_fits(self.test_envelope)
        end_time = time.time()

        # Verify optimization completed
        assert "optimization_successful" in results

        # Verify reasonable performance (should complete within reasonable time)
        optimization_time = end_time - start_time
        assert optimization_time < 60.0  # Should complete within 60 seconds

    def test_convergence_criteria_full_implementation(self):
        """Test convergence criteria in full implementation."""
        # Test optimization with different tolerances
        original_tolerance = self.optimizer.optimization_tolerance

        # Test with stricter tolerance
        self.optimizer.optimization_tolerance = 1e-8
        results_strict = self.optimizer.optimize_power_law_fits(self.test_envelope)

        # Test with looser tolerance
        self.optimizer.optimization_tolerance = 1e-3
        results_loose = self.optimizer.optimize_power_law_fits(self.test_envelope)

        # Restore original tolerance
        self.optimizer.optimization_tolerance = original_tolerance

        # Verify both optimizations completed
        assert "optimization_successful" in results_strict
        assert "optimization_successful" in results_loose

        # Both should have attempted optimization
        assert results_strict["total_regions"] > 0
        assert results_loose["total_regions"] > 0
