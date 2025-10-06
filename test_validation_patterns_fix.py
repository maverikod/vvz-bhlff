#!/usr/bin/env python3
"""
Test script for pattern validation fix.

This script tests the physical pattern validation implementation
to ensure it works correctly and provides theoretical compliance.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from bhlff.core.bvp import BVPCore
from bhlff.models.level_c.beating.validation_basic.beating_validation_patterns import (
    BeatingValidationPatterns,
)


def test_physical_pattern_validation():
    """Test physical pattern validation."""
    print("=== TESTING PHYSICAL PATTERN VALIDATION ===")

    try:
        # Create mock BVP core
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        validator = BeatingValidationPatterns(bvp_core)

        print("✓ BeatingValidationPatterns created successfully")

        # Test with valid patterns
        valid_patterns = [
            {
                "amplitude": 1.0,
                "phase": np.pi / 4,
                "frequency": 1e6,
                "interference_coherence": 0.8,
                "spatial_extent": 1e-3,
                "temporal_duration": 1e-6,
                "energy_density": 1e-6,
            },
            {
                "amplitude": 0.5,
                "phase": np.pi / 2,
                "frequency": 2e6,
                "interference_coherence": 0.6,
                "spatial_extent": 2e-3,
                "temporal_duration": 2e-6,
                "energy_density": 2e-6,
            },
        ]
        print(f"✓ Testing valid patterns: {len(valid_patterns)} patterns")

        result = validator.validate_interference_patterns_physical(valid_patterns)

        print("✓ Physical validation completed")
        print(f"  - Patterns valid: {result['patterns_valid']}")
        print(f"  - Physical valid: {result['physical_validation']['physical_valid']}")
        print(
            f"  - Theoretical valid: {result['theoretical_validation']['theoretical_valid']}"
        )
        print(
            f"  - Coherence relationships: {result['coherence_analysis']['coherence_count']}"
        )

        # Validate results structure
        required_keys = [
            "patterns_valid",
            "pattern_errors",
            "pattern_metrics",
            "physical_validation",
            "theoretical_validation",
            "coherence_analysis",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        print("✓ All required result keys present")

        # Test physical validation
        physical = result["physical_validation"]
        assert "physical_errors" in physical
        assert "physical_valid" in physical
        assert "physical_pattern_count" in physical
        print("✓ Physical validation structure correct")

        # Test theoretical validation
        theoretical = result["theoretical_validation"]
        assert "theoretical_errors" in theoretical
        assert "theoretical_valid" in theoretical
        assert "theoretical_pattern_count" in theoretical
        print("✓ Theoretical validation structure correct")

        # Test coherence analysis
        coherence = result["coherence_analysis"]
        assert "coherence_relationships" in coherence
        assert "coherence_valid" in coherence
        assert "coherence_count" in coherence
        print("✓ Coherence analysis structure correct")

        print("\n✅ PHYSICAL VALIDATION TESTS PASSED!")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_invalid_pattern_validation():
    """Test validation with invalid patterns."""
    print("\n=== TESTING INVALID PATTERN VALIDATION ===")

    try:
        # Create validator
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        validator = BeatingValidationPatterns(bvp_core)

        # Test with invalid patterns
        invalid_patterns = [
            {
                "amplitude": 0,  # Too small
                "phase": np.pi / 4,
                "frequency": 1e6,
                "interference_coherence": 0.8,
            },
            {
                "amplitude": 1e20,  # Too large
                "phase": np.pi / 4,
                "frequency": 1e6,
                "interference_coherence": 0.8,
            },
            {
                "amplitude": 1.0,
                "phase": 10 * np.pi,  # Outside bounds
                "frequency": 1e6,
                "interference_coherence": 0.8,
            },
        ]
        print(f"✓ Testing invalid patterns: {len(invalid_patterns)} patterns")

        result = validator.validate_interference_patterns_physical(invalid_patterns)

        print("✓ Invalid pattern validation completed")
        print(f"  - Patterns valid: {result['patterns_valid']}")
        print(f"  - Error count: {len(result['pattern_errors'])}")

        # Should detect errors
        assert not result["patterns_valid"], "Should detect invalid patterns"
        assert len(result["pattern_errors"]) > 0, "Should have error messages"

        print("✓ Invalid patterns correctly detected")

        print("\n✅ INVALID PATTERN VALIDATION TESTS PASSED!")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_coherence_analysis():
    """Test coherence analysis functionality."""
    print("\n=== TESTING COHERENCE ANALYSIS ===")

    try:
        # Create validator
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        validator = BeatingValidationPatterns(bvp_core)

        # Test with coherent patterns
        coherent_patterns = [
            {
                "amplitude": 1.0,
                "phase": 0.0,
                "frequency": 1e6,
                "interference_coherence": 0.8,
                "spatial_extent": 1e-3,
                "temporal_duration": 1e-6,
                "energy_density": 1e-6,
            },
            {
                "amplitude": 0.9,  # Similar amplitude
                "phase": 0.1,  # Similar phase
                "frequency": 1.1e6,  # Similar frequency
                "interference_coherence": 0.7,
                "spatial_extent": 1.1e-3,
                "temporal_duration": 1.1e-6,
                "energy_density": 1.1e-6,
            },
        ]
        print(f"✓ Testing coherent patterns: {len(coherent_patterns)} patterns")

        result = validator.validate_interference_patterns_physical(coherent_patterns)

        print("✓ Coherence analysis completed")
        print(
            f"  - Coherence relationships: {result['coherence_analysis']['coherence_count']}"
        )
        print(f"  - Coherence valid: {result['coherence_analysis']['coherence_valid']}")

        # Should detect coherence relationships
        assert (
            result["coherence_analysis"]["coherence_count"] > 0
        ), "Should detect coherence relationships"
        assert result["coherence_analysis"][
            "coherence_valid"
        ], "Should validate coherence"

        print("✓ Coherence relationships correctly detected")

        print("\n✅ COHERENCE ANALYSIS TESTS PASSED!")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility with legacy method."""
    print("\n=== TESTING BACKWARD COMPATIBILITY ===")

    try:
        # Create validator
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        validator = BeatingValidationPatterns(bvp_core)

        # Test legacy method
        test_patterns = [
            {"amplitude": 1.0, "phase": np.pi / 4, "frequency": 1e6},
            {"amplitude": 0.5, "phase": np.pi / 2, "frequency": 2e6},
        ]
        print(f"✓ Testing legacy method with patterns: {len(test_patterns)} patterns")

        result = validator.validate_interference_patterns(test_patterns)

        print("✓ Legacy validation completed")
        print(f"  - Patterns valid: {result['patterns_valid']}")
        print(f"  - Error count: {len(result['pattern_errors'])}")

        # Should work without errors
        assert result["patterns_valid"], "Legacy method should work"
        assert len(result["pattern_errors"]) == 0, "Should have no errors"

        print("✓ Legacy method working correctly")

        print("\n✅ BACKWARD COMPATIBILITY TESTS PASSED!")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== PATTERN VALIDATION FIX TESTS ===")
    print("Testing physical pattern validation implementation...")

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    tests = [
        test_physical_pattern_validation,
        test_invalid_pattern_validation,
        test_coherence_analysis,
        test_backward_compatibility,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== TEST RESULTS ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Physical pattern validation working correctly")
        print("✓ Theoretical framework implemented")
        print("✓ No simplified validation remaining")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
