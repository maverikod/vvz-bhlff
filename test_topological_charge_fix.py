"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test for Topological Charge Analyzer fix.

This module tests the corrected topological charge analyzer implementation
to ensure it properly computes topological charge and analyzes defects
in BVP fields.

Physical Meaning:
    Tests that the topological charge analyzer correctly computes
    topological charge, identifies defects, and analyzes their properties
    for BVP field analysis.

Example:
    >>> python test_topological_charge_fix.py
"""

import numpy as np
import sys
import os
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from bhlff.core.domain import Domain
from bhlff.core.bvp.topological_charge_analyzer import TopologicalChargeAnalyzer
from bhlff.core.bvp.topological_defect_analyzer import TopologicalDefectAnalyzer
from bhlff.core.bvp.bvp_constants import BVPConstants


def test_topological_charge_analyzer_initialization():
    """
    Test topological charge analyzer initialization.

    Physical Meaning:
        Tests that the corrected topological charge analyzer properly
        initializes with domain and configuration parameters.
    """
    print("Testing topological charge analyzer initialization...")

    # Create test domain (minimal size to avoid hanging)
    domain = Domain(L=1.0, N=3, dimensions=7)

    # Create test configuration
    config = {
        "charge_threshold": 0.1,
        "winding_precision": 1e-6,
        "min_charge": 0.01,
        "max_charge": 10.0,
        "stability_threshold": 0.8,
    }

    # Create analyzer
    analyzer = TopologicalChargeAnalyzer(domain, config)

    # Verify analyzer has required attributes
    assert hasattr(analyzer, "domain"), "Analyzer should have domain"
    assert hasattr(analyzer, "config"), "Analyzer should have config"
    assert hasattr(
        analyzer, "charge_threshold"
    ), "Analyzer should have charge_threshold"
    assert hasattr(analyzer, "defect_analyzer"), "Analyzer should have defect_analyzer"

    # Verify parameters
    params = analyzer.get_analysis_parameters()
    assert "charge_threshold" in params, "Missing charge_threshold parameter"
    assert "winding_precision" in params, "Missing winding_precision parameter"
    assert "min_charge" in params, "Missing min_charge parameter"

    print("✓ Topological charge analyzer initialization successful")
    print(f"  Charge threshold: {params['charge_threshold']}")
    print(f"  Winding precision: {params['winding_precision']}")

    return True


def test_topological_charge_computation():
    """
    Test topological charge computation.

    Physical Meaning:
        Tests that the corrected topological charge analyzer properly
        computes topological charge for BVP fields.
    """
    print("Testing topological charge computation...")

    # Create test domain (minimal size to avoid hanging)
    domain = Domain(L=1.0, N=3, dimensions=7)

    # Create test configuration
    config = {
        "charge_threshold": 0.1,
        "winding_precision": 1e-6,
        "min_charge": 0.01,
        "max_charge": 10.0,
        "stability_threshold": 0.8,
    }

    # Create analyzer
    analyzer = TopologicalChargeAnalyzer(domain, config)

    # Create test field with known structure
    field = np.random.randn(*domain.shape) + 1j * np.random.randn(*domain.shape)

    # Add some structure to make analysis meaningful
    field[2:3, 2:3, 2:3, 1:2, 1:2, 1:2, 2:3] *= 2.0

    # Test charge computation
    try:
        print("    Computing topological charge...")
        results = analyzer.compute_topological_charge(field)
        print("    Topological charge computation completed")

        # Verify results structure
        required_keys = [
            "topological_charge",
            "charge_locations",
            "charge_stability",
            "defect_analysis",
        ]
        for key in required_keys:
            assert key in results, f"Missing result key: {key}"

        # Verify topological charge
        topological_charge = results["topological_charge"]
        assert isinstance(
            topological_charge, (int, float)
        ), "Topological charge should be numeric"
        assert np.isfinite(topological_charge), "Topological charge should be finite"

        # Verify charge locations
        charge_locations = results["charge_locations"]
        assert isinstance(charge_locations, list), "Charge locations should be list"

        # Verify charge stability
        charge_stability = results["charge_stability"]
        assert isinstance(
            charge_stability, (int, float)
        ), "Charge stability should be numeric"
        assert (
            0.0 <= charge_stability <= 1.0
        ), "Charge stability should be between 0 and 1"

        # Verify defect analysis
        defect_analysis = results["defect_analysis"]
        assert isinstance(defect_analysis, dict), "Defect analysis should be dictionary"

        print("✓ Topological charge computation successful")
        print(f"  Topological charge: {topological_charge:.4f}")
        print(f"  Charge locations: {len(charge_locations)}")
        print(f"  Charge stability: {charge_stability:.4f}")

    except Exception as e:
        print(f"✗ Topological charge computation failed: {e}")
        return False

    return True


def test_defect_analyzer_functionality():
    """
    Test topological defect analyzer functionality.

    Physical Meaning:
        Tests that the topological defect analyzer correctly
        identifies and analyzes topological defects.
    """
    print("Testing topological defect analyzer functionality...")

    # Create test domain (minimal size to avoid hanging)
    domain = Domain(L=1.0, N=3, dimensions=7)

    # Create test configuration
    config = {
        "defect_size": 2,
        "gradient_threshold": 0.5,
        "interaction_radius": 3.0,
        "min_defect_strength": 0.1,
        "max_defect_strength": 10.0,
        "stability_threshold": 0.8,
    }

    # Create analyzer
    analyzer = TopologicalDefectAnalyzer(domain, config)

    # Create test phase field with known structure
    phase = np.random.randn(*domain.shape)

    # Add some structure to make analysis meaningful
    phase[2:3, 2:3, 2:3, 1:2, 1:2, 1:2, 2:3] *= 2.0

    # Test defect finding
    try:
        print("    Finding topological defects...")
        defects = analyzer.find_topological_defects(phase)
        print(f"    Found {len(defects)} defects")

        # Verify defects structure
        assert isinstance(defects, list), "Defects should be list"

        # Test defect type analysis
        if defects:
            print("    Analyzing defect types...")
            defect_types = analyzer.analyze_defect_types(phase, defects)
            print(f"    Defect types: {defect_types}")
            assert isinstance(defect_types, list), "Defect types should be list"
            assert len(defect_types) == len(
                defects
            ), "Defect types should match defects count"

        # Test defect interaction analysis
        if len(defects) > 1:
            print("    Analyzing defect interactions...")
            # Create dummy charges for testing
            charges = [1.0] * len(defects)
            interactions = analyzer.analyze_defect_interactions(defects, charges)
            print(f"    Interaction analysis completed")

            assert isinstance(interactions, dict), "Interactions should be dictionary"
            required_interaction_keys = [
                "interaction_energy",
                "attractive_pairs",
                "repulsive_pairs",
                "interaction_strength",
            ]
            for key in required_interaction_keys:
                assert key in interactions, f"Missing interaction key: {key}"

        print("✓ Topological defect analyzer functionality successful")
        print(f"  Found defects: {len(defects)}")
        if defects:
            print(f"  Defect types: {defect_types}")

    except Exception as e:
        print(f"✗ Topological defect analyzer functionality failed: {e}")
        return False

    return True


def test_phase_structure_analysis():
    """
    Test phase structure analysis.

    Physical Meaning:
        Tests that the phase structure analysis correctly
        analyzes phase coherence and gradient properties.
    """
    print("Testing phase structure analysis...")

    # Create test domain (minimal size to avoid hanging)
    domain = Domain(L=1.0, N=3, dimensions=7)

    # Create test configuration
    config = {
        "charge_threshold": 0.1,
        "winding_precision": 1e-6,
        "min_charge": 0.01,
        "max_charge": 10.0,
        "stability_threshold": 0.8,
    }

    # Create analyzer
    analyzer = TopologicalChargeAnalyzer(domain, config)

    # Create test field
    field = np.random.randn(*domain.shape) + 1j * np.random.randn(*domain.shape)

    # Test phase structure analysis
    try:
        print("    Analyzing phase structure...")
        phase_analysis = analyzer.analyze_phase_structure(field)
        print("    Phase structure analysis completed")

        # Verify phase analysis structure
        assert isinstance(phase_analysis, dict), "Phase analysis should be dictionary"

        # Verify required keys
        required_keys = [
            "phase_coherence",
            "phase_variance",
            "gradient_mean",
            "gradient_std",
            "gradient_max",
            "high_gradient_fraction",
        ]
        for key in required_keys:
            assert key in phase_analysis, f"Missing phase analysis key: {key}"
            assert isinstance(
                phase_analysis[key], (int, float)
            ), f"Phase analysis {key} should be numeric"

        # Verify phase coherence is between -1 and 1
        phase_coherence = phase_analysis["phase_coherence"]
        assert (
            -1.0 <= phase_coherence <= 1.0
        ), "Phase coherence should be between -1 and 1"

        # Verify gradient properties are non-negative
        gradient_mean = phase_analysis["gradient_mean"]
        gradient_std = phase_analysis["gradient_std"]
        gradient_max = phase_analysis["gradient_max"]
        assert gradient_mean >= 0, "Gradient mean should be non-negative"
        assert gradient_std >= 0, "Gradient std should be non-negative"
        assert gradient_max >= 0, "Gradient max should be non-negative"

        print("✓ Phase structure analysis successful")
        print(f"  Phase coherence: {phase_coherence:.4f}")
        print(f"  Gradient mean: {gradient_mean:.4f}")
        print(
            f"  High gradient fraction: {phase_analysis['high_gradient_fraction']:.4f}"
        )

    except Exception as e:
        print(f"✗ Phase structure analysis failed: {e}")
        return False

    return True


def test_charge_stability_computation():
    """
    Test charge stability computation.

    Physical Meaning:
        Tests that the charge stability computation correctly
        evaluates the stability of topological charges.
    """
    print("Testing charge stability computation...")

    # Create test domain (minimal size to avoid hanging)
    domain = Domain(L=1.0, N=3, dimensions=7)

    # Create test configuration
    config = {
        "charge_threshold": 0.1,
        "winding_precision": 1e-6,
        "min_charge": 0.01,
        "max_charge": 10.0,
        "stability_threshold": 0.8,
    }

    # Create analyzer
    analyzer = TopologicalChargeAnalyzer(domain, config)

    # Test charge stability computation
    try:
        # Test with different charge configurations
        charges_1 = [1.0, -1.0, 0.5]
        locations_1 = [
            (1, 1, 1, 1, 1, 1, 1),
            (2, 2, 2, 2, 2, 2, 2),
            (3, 3, 3, 3, 3, 3, 3),
        ]

        stability_1 = analyzer._compute_charge_stability(charges_1, locations_1)
        assert isinstance(
            stability_1, (int, float)
        ), "Charge stability should be numeric"
        assert 0.0 <= stability_1 <= 1.0, "Charge stability should be between 0 and 1"

        # Test with empty charges
        charges_2 = []
        locations_2 = []
        stability_2 = analyzer._compute_charge_stability(charges_2, locations_2)
        assert stability_2 == 0.0, "Empty charges should have zero stability"

        print("✓ Charge stability computation successful")
        print(f"  Stability with charges: {stability_1:.4f}")
        print(f"  Stability without charges: {stability_2:.4f}")

    except Exception as e:
        print(f"✗ Charge stability computation failed: {e}")
        return False

    return True


def main():
    """
    Run all topological charge analyzer fix tests.

    Physical Meaning:
        Comprehensive testing of the corrected topological charge analyzer
        implementation to ensure it properly computes topological charge
        and analyzes defects in BVP fields.
    """
    print("=" * 60)
    print("TOPOLOGICAL CHARGE ANALYZER FIX TESTING")
    print("=" * 60)
    print()

    tests = [
        test_topological_charge_analyzer_initialization,
        test_topological_charge_computation,
        test_defect_analyzer_functionality,
        test_phase_structure_analysis,
        test_charge_stability_computation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()

    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All topological charge analyzer fix tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
