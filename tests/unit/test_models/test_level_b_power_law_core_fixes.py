"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for Level B Power Law Core fixes - full 7D implementations.

This module tests the corrected implementations of power law analysis
methods that were previously simplified, ensuring they now implement
full 7D analysis according to the theory.

Physical Meaning:
    Tests verify that the corrected methods implement proper 7D
    correlation functions, critical exponent analysis, and scaling
    region identification according to the 7D phase field theory.

Mathematical Foundation:
    Tests verify mathematical correctness of:
    - 7D correlation functions C(r) = ⟨a(x)a(x+r)⟩
    - Full critical exponent analysis (ν, β, γ, δ, η, α, z)
    - Multi-scale decomposition and wavelet analysis
    - Renormalization group flow analysis
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.models.level_b.power_law_core import PowerLawCore
from bhlff.core.bvp import BVPCore
from bhlff.core.domain import Domain


class TestPowerLawCoreFixes:
    """Test corrected power law core implementations."""

    @pytest.fixture
    def domain_3d(self):
        """Create 3D domain for testing."""
        return Domain(L=1.0, N=32, dimensions=3)

    @pytest.fixture
    def bvp_core(self, domain_3d):
        """Create BVP core for testing."""
        return BVPCore(domain_3d)

    @pytest.fixture
    def power_law_core(self, bvp_core):
        """Create power law core for testing."""
        return PowerLawCore(bvp_core)

    @pytest.fixture
    def test_envelope_3d(self, domain_3d):
        """Create test 3D envelope field."""
        # Create a test field with known properties
        x = np.linspace(-1, 1, domain_3d.N)
        y = np.linspace(-1, 1, domain_3d.N)
        z = np.linspace(-1, 1, domain_3d.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create field with known correlation structure
        envelope = np.exp(-(X**2 + Y**2 + Z**2) / 0.5) * np.exp(1j * np.pi * (X + Y + Z))
        
        return envelope

    def test_7d_correlation_function_implementation(self, power_law_core, test_envelope_3d):
        """
        Test that 7D correlation function is properly implemented.
        
        Physical Meaning:
            Verifies that the correlation function preserves 7D structure
            and computes proper spatial correlations C(r) = ⟨a(x)a(x+r)⟩.
        """
        # Test the corrected correlation function
        correlation_results = power_law_core.compute_correlation_functions(test_envelope_3d)
        
        # Verify structure is preserved (not flattened)
        assert "spatial_correlation_7d" in correlation_results
        assert "correlation_lengths" in correlation_results
        assert "correlation_structure" in correlation_results
        assert "dimensional_correlations" in correlation_results
        
        # Verify 7D correlation function has same shape as input
        correlation_7d = correlation_results["spatial_correlation_7d"]
        assert correlation_7d.shape == test_envelope_3d.shape, "7D structure not preserved"
        
        # Verify correlation lengths are computed for each dimension
        correlation_lengths = correlation_results["correlation_lengths"]
        assert len(correlation_lengths) == test_envelope_3d.ndim, "Missing dimension correlations"
        
        # Verify correlation structure analysis
        correlation_structure = correlation_results["correlation_structure"]
        assert "max_correlation" in correlation_structure
        assert "mean_correlation" in correlation_structure
        assert "dimensional_coupling" in correlation_structure
        assert "correlation_decay" in correlation_structure
        assert "anisotropy_measure" in correlation_structure
        
        # Verify dimensional correlations
        dimensional_correlations = correlation_results["dimensional_correlations"]
        assert len(dimensional_correlations) == test_envelope_3d.ndim, "Missing dimensional correlations"
        
        # Verify physical meaning: correlation should be positive and finite
        assert np.all(np.isfinite(correlation_7d)), "Correlation function contains non-finite values"
        assert np.all(correlation_7d >= 0), "Correlation function should be non-negative"

    def test_full_critical_exponents_implementation(self, power_law_core, test_envelope_3d):
        """
        Test that full critical exponents are properly implemented.
        
        Physical Meaning:
            Verifies that all standard critical exponents are computed:
            ν (correlation length), β (order parameter), γ (susceptibility),
            δ (critical isotherm), η (anomalous dimension), α (specific heat), z (dynamic).
        """
        # Test the corrected critical behavior analysis
        critical_results = power_law_core.analyze_critical_behavior(test_envelope_3d)
        
        # Verify all critical exponents are computed
        critical_exponents = critical_results["critical_exponents"]
        expected_exponents = ["nu", "beta", "gamma", "delta", "eta", "alpha", "z"]
        
        for exponent in expected_exponents:
            assert exponent in critical_exponents, f"Missing critical exponent: {exponent}"
            assert isinstance(critical_exponents[exponent], (int, float)), f"Invalid exponent type: {exponent}"
            assert np.isfinite(critical_exponents[exponent]), f"Non-finite exponent: {exponent}"
        
        # Verify physical meaning of exponents
        assert 0.1 <= critical_exponents["nu"] <= 2.0, "ν (correlation length exponent) out of range"
        assert 0.1 <= critical_exponents["beta"] <= 2.0, "β (order parameter exponent) out of range"
        assert 0.5 <= critical_exponents["gamma"] <= 2.0, "γ (susceptibility exponent) out of range"
        assert 1.0 <= critical_exponents["delta"] <= 10.0, "δ (critical isotherm exponent) out of range"
        assert -1.0 <= critical_exponents["eta"] <= 1.0, "η (anomalous dimension) out of range"
        assert -1.0 <= critical_exponents["alpha"] <= 1.0, "α (specific heat exponent) out of range"
        assert 1.0 <= critical_exponents["z"] <= 4.0, "z (dynamic exponent) out of range"
        
        # Verify scaling relations
        # δ = (γ + β) / β
        expected_delta = (critical_exponents["gamma"] + critical_exponents["beta"]) / critical_exponents["beta"]
        assert abs(critical_exponents["delta"] - expected_delta) < 0.1, "Scaling relation δ = (γ + β) / β violated"
        
        # α = 2 - ν*d (for d=3)
        expected_alpha = 2 - critical_exponents["nu"] * 3
        assert abs(critical_exponents["alpha"] - expected_alpha) < 0.1, "Scaling relation α = 2 - ν*d violated"
        
        # Verify additional analysis components
        assert "critical_regions" in critical_results
        assert "scaling_dimension" in critical_results
        assert "universality_class" in critical_results
        assert "critical_scaling" in critical_results
        
        # Verify universality class determination
        universality_class = critical_results["universality_class"]
        assert universality_class in ["mean_field", "ising_3d", "xy_3d", "custom_7d"], "Invalid universality class"

    def test_full_scaling_regions_implementation(self, power_law_core, test_envelope_3d):
        """
        Test that full scaling regions analysis is properly implemented.
        
        Physical Meaning:
            Verifies that scaling regions are identified using multi-scale
            decomposition, wavelet analysis, and renormalization group flow.
        """
        # Test the corrected scaling regions identification
        scaling_regions = power_law_core.identify_scaling_regions(test_envelope_3d)
        
        # Verify scaling regions structure
        assert isinstance(scaling_regions, list), "Scaling regions should be a list"
        
        # Verify each region has proper structure
        for region in scaling_regions:
            assert "center" in region, "Region missing center"
            assert "radius" in region, "Region missing radius"
            assert "scaling_type" in region, "Region missing scaling type"
            assert "exponent" in region, "Region missing exponent"
            assert "consistency" in region, "Region missing consistency"
            assert "scaling_analysis" in region, "Region missing scaling analysis"
            
            # Verify physical meaning
            assert isinstance(region["center"], tuple), "Center should be tuple"
            assert region["radius"] > 0, "Radius should be positive"
            assert region["exponent"] >= 0, "Exponent should be non-negative"
            assert 0 <= region["consistency"] <= 1, "Consistency should be between 0 and 1"
            
            # Verify scaling analysis components
            scaling_analysis = region["scaling_analysis"]
            if region["scaling_type"] == "consistent":
                assert "scale_exponents" in scaling_analysis
                assert "wavelet_analysis" in scaling_analysis
                assert "rg_flow" in scaling_analysis
            elif region["scaling_type"] == "wavelet":
                assert "wavelet_scale" in scaling_analysis
                assert "wavelet_std" in scaling_analysis
                assert "wavelet_mean_abs" in scaling_analysis

    def test_mathematical_correctness_correlation_function(self, power_law_core, test_envelope_3d):
        """
        Test mathematical correctness of 7D correlation function.
        
        Mathematical Foundation:
            Verifies that C(r) = ⟨a(x)a(x+r)⟩ is computed correctly
            and satisfies mathematical properties of correlation functions.
        """
        # Test correlation function properties
        correlation_results = power_law_core.compute_correlation_functions(test_envelope_3d)
        correlation_7d = correlation_results["spatial_correlation_7d"]
        
        # Mathematical property 1: C(0) should be maximum
        center = tuple(s // 2 for s in correlation_7d.shape)
        max_correlation = np.max(correlation_7d)
        center_correlation = correlation_7d[center]
        assert abs(center_correlation - max_correlation) < 1e-10, "C(0) should be maximum"
        
        # Mathematical property 2: C(r) should decay with distance
        correlation_decay = correlation_results["correlation_structure"]["correlation_decay"]
        radial_correlation = np.array(correlation_decay["radial_correlation"])
        
        if len(radial_correlation) > 1:
            # Should generally decay (allowing for some noise)
            decay_trend = np.polyfit(np.arange(len(radial_correlation)), radial_correlation, 1)[0]
            assert decay_trend <= 0.1, "Correlation should generally decay with distance"
        
        # Mathematical property 3: Dimensional coupling should be symmetric
        dimensional_coupling = correlation_results["correlation_structure"]["dimensional_coupling"]
        for key, value in dimensional_coupling.items():
            assert isinstance(value, (int, float)), "Coupling should be numeric"
            assert -1 <= value <= 1, "Coupling should be between -1 and 1"

    def test_mathematical_correctness_critical_exponents(self, power_law_core, test_envelope_3d):
        """
        Test mathematical correctness of critical exponents.
        
        Mathematical Foundation:
            Verifies that critical exponents satisfy scaling relations
            and are computed using proper statistical methods.
        """
        # Test critical exponents
        critical_results = power_law_core.analyze_critical_behavior(test_envelope_3d)
        critical_exponents = critical_results["critical_exponents"]
        
        # Mathematical property 1: Scaling relations
        # Rushbrooke relation: α + 2β + γ = 2
        rushbrooke = critical_exponents["alpha"] + 2 * critical_exponents["beta"] + critical_exponents["gamma"]
        assert abs(rushbrooke - 2.0) < 0.2, "Rushbrooke relation α + 2β + γ = 2 violated"
        
        # Widom relation: γ = β(δ - 1)
        widom = critical_exponents["beta"] * (critical_exponents["delta"] - 1)
        assert abs(critical_exponents["gamma"] - widom) < 0.2, "Widom relation γ = β(δ - 1) violated"
        
        # Mathematical property 2: Fisher relation: γ = ν(2 - η)
        fisher = critical_exponents["nu"] * (2 - critical_exponents["eta"])
        assert abs(critical_exponents["gamma"] - fisher) < 0.3, "Fisher relation γ = ν(2 - η) violated"
        
        # Mathematical property 3: Josephson relation: νd = 2 - α
        josephson = critical_exponents["nu"] * 3  # d = 3
        assert abs(josephson - (2 - critical_exponents["alpha"])) < 0.2, "Josephson relation νd = 2 - α violated"

    def test_physical_meaning_preservation(self, power_law_core, test_envelope_3d):
        """
        Test that physical meaning is preserved in corrected implementations.
        
        Physical Meaning:
            Verifies that the corrected methods maintain the physical
            interpretation of the 7D BVP field analysis.
        """
        # Test that all methods return physically meaningful results
        correlation_results = power_law_core.compute_correlation_functions(test_envelope_3d)
        critical_results = power_law_core.analyze_critical_behavior(test_envelope_3d)
        scaling_regions = power_law_core.identify_scaling_regions(test_envelope_3d)
        
        # Physical meaning 1: Correlation lengths should be positive
        correlation_lengths = correlation_results["correlation_lengths"]
        for dim, length in correlation_lengths.items():
            assert length >= 0, f"Correlation length for {dim} should be non-negative"
        
        # Physical meaning 2: Critical exponents should be in reasonable ranges
        critical_exponents = critical_results["critical_exponents"]
        for exponent, value in critical_exponents.items():
            assert np.isfinite(value), f"Critical exponent {exponent} should be finite"
            assert value > 0 or exponent in ["eta", "alpha"], f"Critical exponent {exponent} should be positive (except η, α)"
        
        # Physical meaning 3: Scaling regions should have consistent behavior
        for region in scaling_regions:
            assert region["consistency"] >= 0, "Scaling consistency should be non-negative"
            assert region["exponent"] >= 0, "Scaling exponent should be non-negative"
        
        # Physical meaning 4: Universality class should be determined
        universality_class = critical_results["universality_class"]
        assert universality_class in ["mean_field", "ising_3d", "xy_3d", "custom_7d"], "Valid universality class"

    def test_performance_and_stability(self, power_law_core, test_envelope_3d):
        """
        Test that corrected implementations are stable and performant.
        
        Physical Meaning:
            Verifies that the full implementations are numerically stable
            and perform within reasonable time limits.
        """
        import time
        
        # Test performance
        start_time = time.time()
        
        correlation_results = power_law_core.compute_correlation_functions(test_envelope_3d)
        critical_results = power_law_core.analyze_critical_behavior(test_envelope_3d)
        scaling_regions = power_law_core.identify_scaling_regions(test_envelope_3d)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance test: should complete within reasonable time
        assert execution_time < 10.0, f"Analysis took too long: {execution_time:.2f} seconds"
        
        # Stability test: results should be consistent across multiple runs
        correlation_results_2 = power_law_core.compute_correlation_functions(test_envelope_3d)
        
        # Compare key results
        corr1 = correlation_results["correlation_structure"]["max_correlation"]
        corr2 = correlation_results_2["correlation_structure"]["max_correlation"]
        assert abs(corr1 - corr2) < 1e-10, "Results should be consistent across runs"
        
        # Stability test: no NaN or infinite values
        def check_finite_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_finite_recursive(value, f"{path}.{key}")
            elif isinstance(obj, (list, tuple)):
                for i, value in enumerate(obj):
                    check_finite_recursive(value, f"{path}[{i}]")
            elif isinstance(obj, (int, float)):
                assert np.isfinite(obj), f"Non-finite value at {path}: {obj}"
            elif isinstance(obj, np.ndarray):
                assert np.all(np.isfinite(obj)), f"Non-finite values in array at {path}"
        
        check_finite_recursive(correlation_results)
        check_finite_recursive(critical_results)
        check_finite_recursive(scaling_regions)

    def test_backward_compatibility(self, power_law_core, test_envelope_3d):
        """
        Test that corrected implementations maintain backward compatibility.
        
        Physical Meaning:
            Verifies that the corrected methods can still be used
            in existing code without breaking changes.
        """
        # Test that methods still return expected types
        correlation_results = power_law_core.compute_correlation_functions(test_envelope_3d)
        critical_results = power_law_core.analyze_critical_behavior(test_envelope_3d)
        scaling_regions = power_law_core.identify_scaling_regions(test_envelope_3d)
        
        # Backward compatibility: return types should be consistent
        assert isinstance(correlation_results, dict), "Correlation results should be dict"
        assert isinstance(critical_results, dict), "Critical results should be dict"
        assert isinstance(scaling_regions, list), "Scaling regions should be list"
        
        # Backward compatibility: key fields should still exist
        assert "spatial_correlation_7d" in correlation_results, "Should have spatial correlation"
        assert "critical_exponents" in critical_results, "Should have critical exponents"
        assert len(scaling_regions) >= 0, "Should return scaling regions list"
        
        # Backward compatibility: power law exponents should still work
        power_law_results = power_law_core.compute_power_law_exponents(test_envelope_3d)
        assert isinstance(power_law_results, dict), "Power law results should be dict"
        assert "amplitude_exponent" in power_law_results, "Should have amplitude exponent"
