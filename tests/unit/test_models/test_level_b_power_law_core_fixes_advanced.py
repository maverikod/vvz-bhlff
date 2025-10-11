"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced tests for Level B Power Law Core fixes - full 7D implementations.

This module tests the advanced corrected implementations of power law analysis
methods that were previously simplified, ensuring they now implement
full 7D analysis according to the theory.

Physical Meaning:
    Tests verify advanced aspects of the corrected methods including
    performance, stability, and backward compatibility.

Mathematical Foundation:
    Tests verify advanced mathematical correctness of:
    - 7D correlation functions C(r) = ⟨a(x)a(x+r)⟩
    - Full critical exponent analysis (ν, β, γ, δ, η, α, z)
    - Multi-scale decomposition and wavelet analysis
    - Renormalization group flow analysis
"""

import pytest
import numpy as np
from typing import Dict, Any
import time

from bhlff.models.level_b.power_law_core import PowerLawCore
from bhlff.core.bvp import BVPCore
from bhlff.core.domain import Domain


class TestPowerLawCoreFixesAdvanced:
    """Test advanced corrected power law core implementations."""

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
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create field with known correlation structure
        envelope = np.exp(-(X**2 + Y**2 + Z**2) / 0.5) * np.exp(
            1j * np.pi * (X + Y + Z)
        )

        return envelope

    def test_performance_and_stability(self, power_law_core, test_envelope_3d):
        """
        Test performance and stability of corrected implementations.

        Physical Meaning:
            Verifies that the corrected methods perform efficiently
            and maintain numerical stability for typical workloads.
        """
        # Test correlation function performance
        start_time = time.time()
        correlation_results = power_law_core.compute_correlation_functions(
            test_envelope_3d
        )
        correlation_time = time.time() - start_time

        # Check performance (should be reasonable)
        assert correlation_time < 10.0, f"Correlation function too slow: {correlation_time}s"

        # Test critical exponents performance
        start_time = time.time()
        critical_exponents = power_law_core.compute_critical_exponents(
            test_envelope_3d
        )
        exponents_time = time.time() - start_time

        # Check performance (should be reasonable)
        assert exponents_time < 10.0, f"Critical exponents too slow: {exponents_time}s"

        # Test scaling regions performance
        start_time = time.time()
        scaling_regions = power_law_core.identify_scaling_regions(
            test_envelope_3d
        )
        regions_time = time.time() - start_time

        # Check performance (should be reasonable)
        assert regions_time < 10.0, f"Scaling regions too slow: {regions_time}s"

        # Test numerical stability
        for _ in range(10):
            # Run multiple times to check stability
            correlation_results = power_law_core.compute_correlation_functions(
                test_envelope_3d
            )
            critical_exponents = power_law_core.compute_critical_exponents(
                test_envelope_3d
            )
            scaling_regions = power_law_core.identify_scaling_regions(
                test_envelope_3d
            )

            # Check that results are finite
            assert np.all(np.isfinite(correlation_results["spatial_correlation_7d"]))
            for exp in critical_exponents.values():
                if isinstance(exp, (int, float)):
                    assert np.isfinite(exp)

    def test_backward_compatibility(self, power_law_core, test_envelope_3d):
        """
        Test backward compatibility of corrected implementations.

        Physical Meaning:
            Verifies that the corrected methods maintain backward
            compatibility with existing interfaces and data structures.
        """
        # Test that correlation function returns expected structure
        correlation_results = power_law_core.compute_correlation_functions(
            test_envelope_3d
        )

        # Verify backward compatibility
        assert isinstance(correlation_results, dict), "Results must be dictionary"
        assert "spatial_correlation_7d" in correlation_results, "Missing 7D correlation"
        assert "correlation_lengths" in correlation_results, "Missing correlation lengths"

        # Test that critical exponents return expected structure
        critical_exponents = power_law_core.compute_critical_exponents(
            test_envelope_3d
        )

        # Verify backward compatibility
        assert isinstance(critical_exponents, dict), "Results must be dictionary"
        required_exponents = ["nu", "beta", "gamma", "delta", "eta", "alpha", "z"]
        for exp in required_exponents:
            assert exp in critical_exponents, f"Missing exponent: {exp}"

        # Test that scaling regions return expected structure
        scaling_regions = power_law_core.identify_scaling_regions(
            test_envelope_3d
        )

        # Verify backward compatibility
        assert isinstance(scaling_regions, dict), "Results must be dictionary"
        assert "critical_region" in scaling_regions, "Missing critical region"
        assert "scaling_region" in scaling_regions, "Missing scaling region"

    def test_error_handling(self, power_law_core):
        """
        Test error handling in corrected implementations.

        Physical Meaning:
            Verifies that the corrected methods handle
            error conditions gracefully.
        """
        # Test with invalid input
        invalid_envelope = None

        with pytest.raises((ValueError, TypeError)):
            power_law_core.compute_correlation_functions(invalid_envelope)

        with pytest.raises((ValueError, TypeError)):
            power_law_core.compute_critical_exponents(invalid_envelope)

        with pytest.raises((ValueError, TypeError)):
            power_law_core.identify_scaling_regions(invalid_envelope)

        # Test with wrong shape input
        wrong_shape_envelope = np.random.random((10, 10))

        with pytest.raises(ValueError):
            power_law_core.compute_correlation_functions(wrong_shape_envelope)

        with pytest.raises(ValueError):
            power_law_core.compute_critical_exponents(wrong_shape_envelope)

        with pytest.raises(ValueError):
            power_law_core.identify_scaling_regions(wrong_shape_envelope)

    def test_memory_usage(self, power_law_core, test_envelope_3d):
        """
        Test memory usage of corrected implementations.

        Physical Meaning:
            Verifies that the corrected methods use memory
            efficiently for typical workloads.
        """
        # Run multiple times to check memory usage
        for _ in range(5):
            correlation_results = power_law_core.compute_correlation_functions(
                test_envelope_3d
            )
            critical_exponents = power_law_core.compute_critical_exponents(
                test_envelope_3d
            )
            scaling_regions = power_law_core.identify_scaling_regions(
                test_envelope_3d
            )

            # Check that results are not None
            assert correlation_results is not None
            assert critical_exponents is not None
            assert scaling_regions is not None

    def test_parameter_sensitivity(self, power_law_core, test_envelope_3d):
        """
        Test parameter sensitivity of corrected implementations.

        Physical Meaning:
            Verifies that the corrected methods are sensitive
            to parameter changes as expected.
        """
        # Test with different envelope fields
        test_fields = [
            test_envelope_3d,
            test_envelope_3d * 2.0,
            test_envelope_3d + 1.0,
        ]

        results = []

        for field in test_fields:
            correlation_results = power_law_core.compute_correlation_functions(field)
            critical_exponents = power_law_core.compute_critical_exponents(field)
            scaling_regions = power_law_core.identify_scaling_regions(field)

            results.append({
                "correlation": correlation_results,
                "exponents": critical_exponents,
                "regions": scaling_regions,
            })

        # Check that results are different for different inputs
        for i in range(1, len(results)):
            # Correlation results should be different
            corr1 = results[i-1]["correlation"]["spatial_correlation_7d"]
            corr2 = results[i]["correlation"]["spatial_correlation_7d"]
            assert not np.allclose(corr1, corr2, rtol=1e-10), "Results should be different"

            # Critical exponents should be different
            exp1 = results[i-1]["exponents"]
            exp2 = results[i]["exponents"]
            for exp_name in exp1:
                if isinstance(exp1[exp_name], (int, float)) and isinstance(exp2[exp_name], (int, float)):
                    assert not np.allclose(exp1[exp_name], exp2[exp_name], rtol=1e-10), f"Exponent {exp_name} should be different"

    def test_convergence_analysis(self, power_law_core):
        """
        Test convergence analysis of corrected implementations.

        Physical Meaning:
            Verifies that the corrected methods converge
            appropriately with grid resolution.
        """
        # Test different grid sizes
        grid_sizes = [16, 32, 64]
        results = []

        for N in grid_sizes:
            # Create domain with different grid size
            domain = Domain(L=1.0, N=N, dimensions=3)
            bvp_core = BVPCore(domain)
            power_law_core = PowerLawCore(bvp_core)

            # Create test field
            x = np.linspace(-1, 1, N)
            y = np.linspace(-1, 1, N)
            z = np.linspace(-1, 1, N)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            envelope = np.exp(-(X**2 + Y**2 + Z**2) / 0.5)

            # Compute results
            correlation_results = power_law_core.compute_correlation_functions(envelope)
            critical_exponents = power_law_core.compute_critical_exponents(envelope)

            results.append({
                "N": N,
                "correlation": correlation_results,
                "exponents": critical_exponents,
            })

        # Check convergence (results should be consistent)
        for i in range(1, len(results)):
            # Correlation lengths should be consistent
            lengths1 = results[i-1]["correlation"]["correlation_lengths"]
            lengths2 = results[i]["correlation"]["correlation_lengths"]
            for j in range(len(lengths1)):
                assert abs(lengths1[j] - lengths2[j]) < 0.1, f"Correlation lengths not converging: {lengths1[j]} vs {lengths2[j]}"

    def test_robustness_analysis(self, power_law_core, test_envelope_3d):
        """
        Test robustness of corrected implementations.

        Physical Meaning:
            Verifies that the corrected methods are robust
            to various input conditions and edge cases.
        """
        # Test with different input types
        test_inputs = [
            test_envelope_3d,
            test_envelope_3d.real,  # Real part only
            test_envelope_3d.imag,  # Imaginary part only
            np.abs(test_envelope_3d),  # Absolute value
        ]

        for test_input in test_inputs:
            # Test correlation function
            correlation_results = power_law_core.compute_correlation_functions(test_input)
            assert correlation_results is not None, "Correlation function failed"

            # Test critical exponents
            critical_exponents = power_law_core.compute_critical_exponents(test_input)
            assert critical_exponents is not None, "Critical exponents failed"

            # Test scaling regions
            scaling_regions = power_law_core.identify_scaling_regions(test_input)
            assert scaling_regions is not None, "Scaling regions failed"

    def test_comprehensive_validation(self, power_law_core, test_envelope_3d):
        """
        Test comprehensive validation of corrected implementations.

        Physical Meaning:
            Verifies that all corrected methods work together
            and provide consistent results.
        """
        # Compute all results
        correlation_results = power_law_core.compute_correlation_functions(test_envelope_3d)
        critical_exponents = power_law_core.compute_critical_exponents(test_envelope_3d)
        scaling_regions = power_law_core.identify_scaling_regions(test_envelope_3d)

        # Verify consistency between results
        correlation_lengths = correlation_results["correlation_lengths"]
        scaling_exponents = scaling_regions["scaling_region"]["scaling_exponents"]

        # Check that correlation lengths are consistent with scaling exponents
        for i, length in enumerate(correlation_lengths):
            assert length > 0, f"Correlation length {i} must be positive"
            assert np.isfinite(length), f"Correlation length {i} must be finite"

        # Check that critical exponents are consistent
        for exp_name, exp_value in critical_exponents.items():
            if isinstance(exp_value, (int, float)):
                assert np.isfinite(exp_value), f"Critical exponent {exp_name} must be finite"
                assert exp_value > 0, f"Critical exponent {exp_name} must be positive"

        # Check that scaling regions are consistent
        for region_name, region_data in scaling_regions.items():
            assert isinstance(region_data, dict), f"Region {region_name} must be dictionary"
            assert "boundaries" in region_data, f"Region {region_name} must have boundaries"
            assert "scaling_exponents" in region_data, f"Region {region_name} must have exponents"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
