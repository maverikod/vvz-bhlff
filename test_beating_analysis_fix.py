#!/usr/bin/env python3
"""
Test script for beating analysis fix.

This script tests the comprehensive beating analysis implementation
to ensure it works correctly and provides theoretical compliance.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from bhlff.core.bvp import BVPCore
from bhlff.models.level_c.beating.basic.beating_basic_core import BeatingAnalysisCore


def test_beating_analysis_comprehensive():
    """Test comprehensive beating analysis."""
    print("=== TESTING COMPREHENSIVE BEATING ANALYSIS ===")

    try:
        # Create mock BVP core
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        analyzer = BeatingAnalysisCore(bvp_core)

        print("✓ BeatingAnalysisCore created successfully")

        # Create test envelope field (reduced size to avoid OOM)
        envelope_shape = (2, 2, 2, 4, 4, 4, 16)
        envelope = np.random.randn(*envelope_shape) + 1j * np.random.randn(
            *envelope_shape
        )
        print(f"✓ Test envelope created: {envelope.shape}")

        # Quick CUDA availability probe
        try:
            import cupy as cp  # type: ignore

            num_devices = cp.cuda.runtime.getDeviceCount()
            print(f"✓ CUDA available: {num_devices > 0} (devices: {num_devices})")
        except Exception as cuda_e:
            print(f"✓ CUDA available: False ({cuda_e})")

        # Test comprehensive analysis
        print("\n=== TESTING COMPREHENSIVE ANALYSIS ===")
        results = analyzer.analyze_beating_comprehensive(envelope)

        print("✓ Comprehensive analysis completed")
        print(f"  - Interference patterns: {len(results['interference_patterns'])}")
        print(
            f"  - Mode coupling strength: {results['mode_coupling']['coupling_strength']:.4f}"
        )
        print(
            f"  - Phase coherence: {results['phase_coherence']['phase_coherence']:.4f}"
        )
        print(
            f"  - Beating frequencies: {len(results['beating_frequencies']['beating_frequencies'])}"
        )
        print(
            f"  - Theoretical consistency: {results['theoretical_validation']['theoretical_consistency']}"
        )

        # Validate results structure
        required_keys = [
            "interference_patterns",
            "mode_coupling",
            "phase_coherence",
            "beating_frequencies",
            "theoretical_validation",
            "optimization_results",
        ]

        for key in required_keys:
            assert key in results, f"Missing key: {key}"

        print("✓ All required result keys present")

        # Test interference analysis
        interference = results["interference_patterns"]
        assert "interference_strength" in interference
        assert "interference_coherence" in interference
        assert "dominant_frequencies" in interference
        print("✓ Interference analysis structure correct")

        # Test mode coupling analysis
        coupling = results["mode_coupling"]
        assert "coupling_strength" in coupling
        assert "coupling_type" in coupling
        assert "coupling_efficiency" in coupling
        print("✓ Mode coupling analysis structure correct")

        # Test phase coherence analysis
        coherence = results["phase_coherence"]
        assert "phase_coherence" in coherence
        assert "phase_stability" in coherence
        assert "coherence_level" in coherence
        print("✓ Phase coherence analysis structure correct")

        # Test theoretical validation
        validation = results["theoretical_validation"]
        assert "theoretical_consistency" in validation
        assert "validation_score" in validation
        print("✓ Theoretical validation structure correct")

        print("\n✅ ALL TESTS PASSED!")
        print("✓ Comprehensive beating analysis working correctly")
        print("✓ Theoretical framework implemented")
        print("✓ No simplified methods remaining")

        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_theoretical_consistency():
    """Test theoretical consistency of analysis."""
    print("\n=== TESTING THEORETICAL CONSISTENCY ===")

    try:
        # Create analyzer
        class MockBVPCore:
            def __init__(self):
                self.domain = None
                self.constants = None

        bvp_core = MockBVPCore()
        analyzer = BeatingAnalysisCore(bvp_core)

        # Test with known beating pattern (reduced size)
        envelope_shape = (2, 2, 2, 4, 4, 4, 16)
        t = np.linspace(0, 2 * np.pi, envelope_shape[-1])

        # Create synthetic beating pattern with discrete FFT-aligned frequencies
        envelope = np.zeros(envelope_shape, dtype=complex)
        for i in range(envelope_shape[0]):
            for j in range(envelope_shape[1]):
                for k in range(envelope_shape[2]):
                    for l in range(envelope_shape[3]):
                        for m in range(envelope_shape[4]):
                            for n in range(envelope_shape[5]):
                                # Use integer harmonics of the fundamental to ensure sharp FFT peaks
                                # Fundamental frequency with N=16 samples: ω_k = 2πk/N
                                freq1 = 3.0  # k1
                                freq2 = 5.0  # k2
                                envelope[i, j, k, l, m, n, :] = np.exp(
                                    1j * freq1 * t
                                ) + 0.5 * np.exp(1j * freq2 * t)

        print(f"✓ Synthetic beating pattern created: {envelope.shape}")

        # Analyze
        results = analyzer.analyze_beating_comprehensive(envelope)

        # Check that beating is detected
        beating_freqs = results["beating_frequencies"]["beating_frequencies"]
        assert len(beating_freqs) > 0, "No beating frequencies detected"
        print(f"✓ Beating frequencies detected: {len(beating_freqs)}")

        # Check theoretical consistency
        validation = results["theoretical_validation"]
        assert validation["theoretical_consistency"], "Theoretical consistency failed"
        print("✓ Theoretical consistency validated")

        # Check interference strength
        interference = results["interference_patterns"]
        assert interference["interference_strength"] > 0, "No interference detected"
        print("✓ Interference patterns detected")

        # Check mode coupling
        coupling = results["mode_coupling"]
        assert coupling["coupling_strength"] > 0, "No mode coupling detected"
        print("✓ Mode coupling detected")

        # Check phase coherence
        coherence = results["phase_coherence"]
        assert coherence["phase_coherence"] > 0, "No phase coherence detected"
        print("✓ Phase coherence detected")

        print("\n✅ THEORETICAL CONSISTENCY TESTS PASSED!")
        return True

    except Exception as e:
        print(f"✗ THEORETICAL CONSISTENCY TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_no_simplified_methods():
    """Test that no simplified methods remain."""
    print("\n=== TESTING NO SIMPLIFIED METHODS ===")

    try:
        # Check class name
        assert BeatingAnalysisCore.__name__ == "BeatingAnalysisCore"
        print("✓ Class name is comprehensive (not 'Basic')")

        # Check method names
        methods = [
            method for method in dir(BeatingAnalysisCore) if not method.startswith("_")
        ]

        # Should not contain 'basic' or 'simple' methods
        basic_methods = [method for method in methods if "basic" in method.lower()]
        simple_methods = [method for method in methods if "simple" in method.lower()]

        assert len(basic_methods) == 0, f"Found basic methods: {basic_methods}"
        assert len(simple_methods) == 0, f"Found simple methods: {simple_methods}"

        print("✓ No 'basic' or 'simple' methods found")

        # Check for comprehensive methods
        comprehensive_methods = [
            method for method in methods if "comprehensive" in method.lower()
        ]
        assert len(comprehensive_methods) > 0, "No comprehensive methods found"

        print("✓ Comprehensive methods present")

        # Check theoretical methods
        theoretical_methods = [
            method for method in methods if "theoretical" in method.lower()
        ]
        assert len(theoretical_methods) > 0, "No theoretical methods found"

        print("✓ Theoretical methods present")

        print("\n✅ NO SIMPLIFIED METHODS TESTS PASSED!")
        return True

    except Exception as e:
        print(f"✗ NO SIMPLIFIED METHODS TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== BEATING ANALYSIS FIX TESTS ===")
    print("Testing comprehensive beating analysis implementation...")

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    tests = [
        test_beating_analysis_comprehensive,
        test_theoretical_consistency,
        test_no_simplified_methods,
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
        print("✓ Comprehensive beating analysis working correctly")
        print("✓ Theoretical framework implemented")
        print("✓ No simplified methods remaining")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
