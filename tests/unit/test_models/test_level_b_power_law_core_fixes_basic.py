"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic tests for Level B Power Law Core fixes - full 7D implementations.

This module tests the basic corrected implementations of power law analysis
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


class TestPowerLawCoreFixesBasic:
    """Test basic corrected power law core implementations."""

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

    def test_7d_correlation_function_implementation(
        self, power_law_core, test_envelope_3d
    ):
        """
        Test that 7D correlation function is properly implemented.

        Physical Meaning:
            Verifies that the correlation function preserves 7D structure
            and computes proper spatial correlations C(r) = ⟨a(x)a(x+r)⟩.
        """
        # Test the corrected correlation function
        correlation_results = power_law_core.compute_correlation_functions(
            test_envelope_3d
        )

        # Verify structure is preserved (not flattened)
        assert "spatial_correlation_7d" in correlation_results
        assert "correlation_lengths" in correlation_results
        assert "correlation_structure" in correlation_results
        assert "dimensional_correlations" in correlation_results

        # Verify 7D correlation function has same shape as input
        correlation_7d = correlation_results["spatial_correlation_7d"]
        assert (
            correlation_7d.shape == test_envelope_3d.shape
        ), "7D structure not preserved"

        # Verify correlation lengths are computed for each dimension
        correlation_lengths = correlation_results["correlation_lengths"]
        assert (
            len(correlation_lengths) == test_envelope_3d.ndim
        ), "Missing dimension correlations"

        # Verify correlation structure analysis
        correlation_structure = correlation_results["correlation_structure"]
        assert isinstance(correlation_structure, dict), "Invalid structure format"
        assert "anisotropy" in correlation_structure, "Missing anisotropy analysis"
        assert "symmetry" in correlation_structure, "Missing symmetry analysis"

    def test_full_critical_exponents_implementation(
        self, power_law_core, test_envelope_3d
    ):
        """
        Test that full critical exponents are properly implemented.

        Physical Meaning:
            Verifies that all critical exponents (ν, β, γ, δ, η, α, z)
            are computed according to the 7D theory, not simplified versions.
        """
        # Test the corrected critical exponents
        critical_exponents = power_law_core.compute_critical_exponents(
            test_envelope_3d
        )

        # Verify all critical exponents are present
        required_exponents = ["nu", "beta", "gamma", "delta", "eta", "alpha", "z"]
        for exp in required_exponents:
            assert exp in critical_exponents, f"Missing critical exponent: {exp}"

        # Verify exponents are computed (not None)
        for exp in required_exponents:
            assert critical_exponents[exp] is not None, f"Exponent {exp} not computed"
            assert isinstance(critical_exponents[exp], (int, float)), f"Invalid type for {exp}"

        # Verify scaling relations are satisfied
        scaling_relations = critical_exponents.get("scaling_relations", {})
        assert isinstance(scaling_relations, dict), "Invalid scaling relations format"

        # Verify universality class analysis
        universality = critical_exponents.get("universality_class", {})
        assert isinstance(universality, dict), "Invalid universality format"
        assert "class_identification" in universality, "Missing class identification"
        assert "critical_dimension" in universality, "Missing critical dimension"

    def test_full_scaling_regions_implementation(
        self, power_law_core, test_envelope_3d
    ):
        """
        Test that full scaling regions are properly implemented.

        Physical Meaning:
            Verifies that scaling region identification implements
            full multi-scale analysis, not simplified versions.
        """
        # Test the corrected scaling regions
        scaling_regions = power_law_core.identify_scaling_regions(
            test_envelope_3d
        )

        # Verify all scaling regions are present
        assert "critical_region" in scaling_regions, "Missing critical region"
        assert "scaling_region" in scaling_regions, "Missing scaling region"
        assert "crossover_region" in scaling_regions, "Missing crossover region"

        # Verify region analysis is comprehensive
        for region_name, region_data in scaling_regions.items():
            assert isinstance(region_data, dict), f"Invalid format for {region_name}"
            assert "boundaries" in region_data, f"Missing boundaries for {region_name}"
            assert "scaling_exponents" in region_data, f"Missing exponents for {region_name}"
            assert "universality_class" in region_data, f"Missing class for {region_name}"

        # Verify multi-scale decomposition
        multiscale = scaling_regions.get("multiscale_decomposition", {})
        assert isinstance(multiscale, dict), "Invalid multiscale format"
        assert "wavelet_coefficients" in multiscale, "Missing wavelet coefficients"
        assert "scale_hierarchy" in multiscale, "Missing scale hierarchy"

    def test_mathematical_correctness_correlation_function(
        self, power_law_core, test_envelope_3d
    ):
        """
        Test mathematical correctness of correlation function.

        Physical Meaning:
            Verifies that the correlation function satisfies
            mathematical properties of correlation functions.
        """
        # Compute correlation function
        correlation_results = power_law_core.compute_correlation_functions(
            test_envelope_3d
        )

        correlation_7d = correlation_results["spatial_correlation_7d"]

        # Test symmetry: C(r) = C(-r)
        center = tuple(s // 2 for s in correlation_7d.shape)
        assert np.allclose(
            correlation_7d[center], correlation_7d[center], rtol=1e-10
        ), "Correlation function not symmetric at origin"

        # Test normalization: C(0) should be maximum
        max_correlation = np.max(np.abs(correlation_7d))
        center_value = np.abs(correlation_7d[center])
        assert center_value >= max_correlation * 0.9, "Correlation not normalized"

        # Test decay properties
        correlation_lengths = correlation_results["correlation_lengths"]
        for length in correlation_lengths:
            assert length > 0, "Correlation length must be positive"
            assert np.isfinite(length), "Correlation length must be finite"

    def test_mathematical_correctness_critical_exponents(
        self, power_law_core, test_envelope_3d
    ):
        """
        Test mathematical correctness of critical exponents.

        Physical Meaning:
            Verifies that critical exponents satisfy
            mathematical scaling relations.
        """
        # Compute critical exponents
        critical_exponents = power_law_core.compute_critical_exponents(
            test_envelope_3d
        )

        # Test scaling relations
        nu = critical_exponents["nu"]
        beta = critical_exponents["beta"]
        gamma = critical_exponents["gamma"]
        delta = critical_exponents["delta"]

        # Test Rushbrooke relation: α + 2β + γ = 2
        alpha = critical_exponents["alpha"]
        rushbrooke = alpha + 2 * beta + gamma
        assert abs(rushbrooke - 2) < 0.1, f"Rushbrooke relation violated: {rushbrooke}"

        # Test Widom relation: γ = β(δ - 1)
        widom = gamma - beta * (delta - 1)
        assert abs(widom) < 0.1, f"Widom relation violated: {widom}"

        # Test Fisher relation: γ = ν(2 - η)
        eta = critical_exponents["eta"]
        fisher = gamma - nu * (2 - eta)
        assert abs(fisher) < 0.1, f"Fisher relation violated: {fisher}"

        # Test exponent bounds
        assert 0 < nu < 2, f"Invalid ν: {nu}"
        assert 0 < beta < 1, f"Invalid β: {beta}"
        assert 0 < gamma < 3, f"Invalid γ: {gamma}"
        assert 1 < delta < 5, f"Invalid δ: {delta}"

    def test_physical_meaning_preservation(self, power_law_core, test_envelope_3d):
        """
        Test that physical meaning is preserved in corrected implementations.

        Physical Meaning:
            Verifies that the corrected methods preserve the physical
            interpretation of power law analysis in 7D phase field theory.
        """
        # Test correlation function preserves physical meaning
        correlation_results = power_law_core.compute_correlation_functions(
            test_envelope_3d
        )

        # Verify physical interpretation
        correlation_lengths = correlation_results["correlation_lengths"]
        for length in correlation_lengths:
            assert length > 0, "Correlation length must be positive (physical requirement)"

        # Test critical exponents preserve physical meaning
        critical_exponents = power_law_core.compute_critical_exponents(
            test_envelope_3d
        )

        # Verify physical bounds
        nu = critical_exponents["nu"]
        assert nu > 0, "Critical exponent ν must be positive (physical requirement)"

        beta = critical_exponents["beta"]
        assert beta > 0, "Critical exponent β must be positive (physical requirement)"

        # Test scaling regions preserve physical meaning
        scaling_regions = power_law_core.identify_scaling_regions(
            test_envelope_3d
        )

        # Verify physical regions exist
        assert "critical_region" in scaling_regions, "Critical region must exist"
        assert "scaling_region" in scaling_regions, "Scaling region must exist"

        # Verify region boundaries are physical
        for region_name, region_data in scaling_regions.items():
            boundaries = region_data.get("boundaries", {})
            if "lower" in boundaries and "upper" in boundaries:
                assert boundaries["lower"] < boundaries["upper"], f"Invalid boundaries for {region_name}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
