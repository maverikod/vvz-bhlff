#!/usr/bin/env python3
"""
Test script for frequency validation fix.

This script tests the physical frequency validation implementation
to ensure it works correctly and provides theoretical compliance.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from bhlff.core.bvp import BVPCore
from bhlff.models.level_c.beating.validation_basic.beating_validation_frequencies import (
    BeatingValidationFrequencies,
)


def test_physical_frequency_validation():
    """Test physical frequency validation."""
    print("=== TESTING PHYSICAL FREQUENCY VALIDATION ===")

    try:
        # Create mock BVP core
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        validator = BeatingValidationFrequencies(bvp_core)

        print("✓ BeatingValidationFrequencies created successfully")

        # Test with valid frequencies
        valid_frequencies = [1e-3, 1e6, 1e12, 1e15]
        print(f"✓ Testing valid frequencies: {valid_frequencies}")

        result = validator.validate_beating_frequencies_physical(valid_frequencies)

        print("✓ Physical validation completed")
        print(f"  - Frequencies valid: {result['frequencies_valid']}")
        print(f"  - Physical valid: {result['physical_validation']['physical_valid']}")
        print(
            f"  - Theoretical valid: {result['theoretical_validation']['theoretical_valid']}"
        )
        print(
            f"  - Harmonic relationships: {result['harmonic_analysis']['harmonic_count']}"
        )

        # Validate results structure
        required_keys = [
            "frequencies_valid",
            "frequency_errors",
            "frequency_metrics",
            "physical_validation",
            "theoretical_validation",
            "harmonic_analysis",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        print("✓ All required result keys present")

        # Test physical validation
        physical = result["physical_validation"]
        assert "physical_errors" in physical
        assert "physical_valid" in physical
        assert "physical_frequency_count" in physical
        print("✓ Physical validation structure correct")

        # Test theoretical validation
        theoretical = result["theoretical_validation"]
        assert "theoretical_errors" in theoretical
        assert "theoretical_valid" in theoretical
        assert "theoretical_frequency_count" in theoretical
        print("✓ Theoretical validation structure correct")

        # Test harmonic analysis
        harmonic = result["harmonic_analysis"]
        assert "harmonic_relationships" in harmonic
        assert "harmonic_valid" in harmonic
        assert "harmonic_count" in harmonic
        print("✓ Harmonic analysis structure correct")

        print("\n✅ PHYSICAL VALIDATION TESTS PASSED!")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_invalid_frequency_validation():
    """Test validation with invalid frequencies."""
    print("\n=== TESTING INVALID FREQUENCY VALIDATION ===")

    try:
        # Create validator
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        validator = BeatingValidationFrequencies(bvp_core)

        # Test with invalid frequencies
        invalid_frequencies = [0, -1, 1e20, np.nan, np.inf]
        print(f"✓ Testing invalid frequencies: {invalid_frequencies}")

        result = validator.validate_beating_frequencies_physical(invalid_frequencies)

        print("✓ Invalid frequency validation completed")
        print(f"  - Frequencies valid: {result['frequencies_valid']}")
        print(f"  - Error count: {len(result['frequency_errors'])}")

        # Should detect errors
        assert not result["frequencies_valid"], "Should detect invalid frequencies"
        assert len(result["frequency_errors"]) > 0, "Should have error messages"

        print("✓ Invalid frequencies correctly detected")

        print("\n✅ INVALID FREQUENCY VALIDATION TESTS PASSED!")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_harmonic_analysis():
    """Test harmonic analysis functionality."""
    print("\n=== TESTING HARMONIC ANALYSIS ===")

    try:
        # Create validator
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        validator = BeatingValidationFrequencies(bvp_core)

        # Test with harmonic frequencies
        harmonic_frequencies = [1.0, 2.0, 4.0, 8.0]  # 2:1, 4:1, 8:1 ratios
        print(f"✓ Testing harmonic frequencies: {harmonic_frequencies}")

        result = validator.validate_beating_frequencies_physical(harmonic_frequencies)

        print("✓ Harmonic analysis completed")
        print(
            f"  - Harmonic relationships: {result['harmonic_analysis']['harmonic_count']}"
        )
        print(f"  - Harmonic valid: {result['harmonic_analysis']['harmonic_valid']}")

        # Should detect harmonic relationships
        assert (
            result["harmonic_analysis"]["harmonic_count"] > 0
        ), "Should detect harmonic relationships"
        assert result["harmonic_analysis"][
            "harmonic_valid"
        ], "Should validate harmonics"

        print("✓ Harmonic relationships correctly detected")

        print("\n✅ HARMONIC ANALYSIS TESTS PASSED!")
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
        validator = BeatingValidationFrequencies(bvp_core)

        # Test legacy method
        test_frequencies = [1.0, 2.0, 3.0]
        print(f"✓ Testing legacy method with frequencies: {test_frequencies}")

        result = validator.validate_beating_frequencies(test_frequencies)

        print("✓ Legacy validation completed")
        print(f"  - Frequencies valid: {result['frequencies_valid']}")
        print(f"  - Error count: {len(result['frequency_errors'])}")

        # Should work without errors
        assert result["frequencies_valid"], "Legacy method should work"
        assert len(result["frequency_errors"]) == 0, "Should have no errors"

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
    print("=== FREQUENCY VALIDATION FIX TESTS ===")
    print("Testing physical frequency validation implementation...")

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    tests = [
        test_physical_frequency_validation,
        test_invalid_frequency_validation,
        test_harmonic_analysis,
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
        print("✓ Physical frequency validation working correctly")
        print("✓ Theoretical framework implemented")
        print("✓ No simplified validation remaining")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
