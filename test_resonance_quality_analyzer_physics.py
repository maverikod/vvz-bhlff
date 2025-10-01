#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physics validation test for Resonance Quality Analyzer.

This script tests the physical correctness of the resonance quality analyzer
implementation, including Lorentzian fitting, quality factor calculation,
and optimization algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/vasilyvz/Desktop/Инерция/7d/progs/bhlff')

from bhlff.core.bvp.resonance_quality_analyzer import ResonanceQualityAnalyzer


def test_resonance_quality_analyzer_physics():
    """Test the physical correctness of the resonance quality analyzer."""
    print("🧪 Testing Resonance Quality Analyzer Physics...")
    
    # Mock BVP constants
    class MockBVPConstants:
        def get_impedance_parameter(self, param_name):
            params = {
                "min_quality_factor": 1.0,
                "max_quality_factor": 1000.0,
                "peak_window_size": 10
            }
            return params.get(param_name, 1.0)
    
    # Initialize analyzer
    constants = MockBVPConstants()
    analyzer = ResonanceQualityAnalyzer(constants)
    
    print("✅ Resonance quality analyzer initialized successfully")
    
    # Test 1: Lorentzian fitting physics
    print("\n📊 Test 1: Lorentzian Fitting Physics")
    test_lorentzian_fitting_physics(analyzer)
    
    # Test 2: Quality factor calculation physics
    print("\n🔬 Test 2: Quality Factor Calculation Physics")
    test_quality_factor_calculation_physics(analyzer)
    
    # Test 3: Fitting quality assessment physics
    print("\n⚙️ Test 3: Fitting Quality Assessment Physics")
    test_fitting_quality_assessment_physics(analyzer)
    
    # Test 4: Fallback methods physics
    print("\n🎯 Test 4: Fallback Methods Physics")
    test_fallback_methods_physics(analyzer)
    
    print("\n✅ All physics tests passed!")


def test_lorentzian_fitting_physics(analyzer):
    """Test Lorentzian fitting physics."""
    # Create test data with known Lorentzian peak
    frequencies = np.linspace(0, 10, 100)
    
    # Create Lorentzian peak: L(f) = A * γ² / ((f - f₀)² + γ²) + offset
    f0 = 5.0  # Center frequency
    gamma = 0.5  # Half-width
    A = 10.0  # Amplitude
    offset = 1.0  # Background offset
    
    magnitude = A * gamma**2 / ((frequencies - f0)**2 + gamma**2) + offset
    
    # Add some noise
    noise = 0.1 * np.random.randn(len(magnitude))
    magnitude += noise
    
    # Find peak index
    peak_idx = np.argmax(magnitude)
    
    # Test Lorentzian fitting
    fit_results = analyzer.fit_lorentzian(frequencies, magnitude, peak_idx)
    
    # Physics checks
    assert "amplitude" in fit_results, "Should include amplitude"
    assert "center" in fit_results, "Should include center frequency"
    assert "fwhm" in fit_results, "Should include FWHM"
    assert "q_factor" in fit_results, "Should include quality factor"
    assert "fitting_quality" in fit_results, "Should include fitting quality"
    
    # Check parameter ranges
    assert fit_results["amplitude"] > 0, "Amplitude should be positive"
    assert fit_results["center"] > 0, "Center frequency should be positive"
    assert fit_results["fwhm"] > 0, "FWHM should be positive"
    assert fit_results["q_factor"] > 0, "Quality factor should be positive"
    assert 0 <= fit_results["fitting_quality"] <= 1, "Fitting quality should be between 0 and 1"
    
    # Check if fitted parameters are reasonable
    center_error = abs(fit_results["center"] - f0) / f0
    assert center_error < 0.1, f"Center frequency error too large: {center_error:.2e}"
    
    amplitude_error = abs(fit_results["amplitude"] - A) / A
    assert amplitude_error < 0.5, f"Amplitude error too large: {amplitude_error:.2e}"
    
    print(f"   Fitted center: {fit_results['center']:.3f} (expected: {f0:.3f})")
    print(f"   Fitted amplitude: {fit_results['amplitude']:.3f} (expected: {A:.3f})")
    print(f"   Fitted FWHM: {fit_results['fwhm']:.3f}")
    print(f"   Quality factor: {fit_results['q_factor']:.3f}")
    print(f"   Fitting quality: {fit_results['fitting_quality']:.3f}")


def test_quality_factor_calculation_physics(analyzer):
    """Test quality factor calculation physics."""
    # Create test data with multiple peaks
    frequencies = np.linspace(0, 20, 200)
    
    # Create multiple Lorentzian peaks
    magnitude = np.zeros_like(frequencies)
    
    # Peak 1: High Q (narrow)
    f0_1, gamma_1, A_1 = 5.0, 0.2, 8.0
    magnitude += A_1 * gamma_1**2 / ((frequencies - f0_1)**2 + gamma_1**2)
    
    # Peak 2: Low Q (wide)
    f0_2, gamma_2, A_2 = 15.0, 2.0, 6.0
    magnitude += A_2 * gamma_2**2 / ((frequencies - f0_2)**2 + gamma_2**2)
    
    # Add noise
    magnitude += 0.05 * np.random.randn(len(magnitude))
    
    # Find peak indices
    peak_indices = []
    for i in range(1, len(magnitude) - 1):
        if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
            if magnitude[i] > 0.5 * np.max(magnitude):  # Significant peaks only
                peak_indices.append(i)
    
    # Test quality factor calculation
    quality_factors = analyzer.calculate_quality_factors(frequencies, magnitude, peak_indices)
    
    # Physics checks
    assert len(quality_factors) == len(peak_indices), "Should return quality factor for each peak"
    
    for i, q_factor in enumerate(quality_factors):
        assert q_factor > 0, f"Quality factor {i} should be positive"
        assert q_factor >= 1.0, f"Quality factor {i} should be >= 1.0"
        assert q_factor <= 1000.0, f"Quality factor {i} should be <= 1000.0"
    
    # Check that high-Q peak has higher quality factor than low-Q peak
    if len(quality_factors) >= 2:
        # First peak should have higher Q (narrower)
        assert quality_factors[0] > quality_factors[1], "High-Q peak should have higher quality factor"
    
    print(f"   Found {len(peak_indices)} peaks")
    for i, q_factor in enumerate(quality_factors):
        print(f"   Peak {i+1} Q-factor: {q_factor:.3f}")


def test_fitting_quality_assessment_physics(analyzer):
    """Test fitting quality assessment physics."""
    # Create test data
    frequencies = np.linspace(0, 10, 50)
    
    # Create perfect Lorentzian (no noise)
    f0, gamma, A, offset = 5.0, 0.5, 10.0, 1.0
    magnitude = A * gamma**2 / ((frequencies - f0)**2 + gamma**2) + offset
    
    peak_idx = np.argmax(magnitude)
    
    # Test fitting quality assessment
    fit_results = analyzer.fit_lorentzian(frequencies, magnitude, peak_idx)
    
    # Physics checks
    assert "r_squared" in fit_results, "Should include R-squared"
    assert "chi_squared" in fit_results, "Should include chi-squared"
    assert "parameter_errors" in fit_results, "Should include parameter errors"
    
    # For perfect data, fitting quality should be high
    assert fit_results["r_squared"] > 0.9, "R-squared should be high for good fit"
    assert fit_results["fitting_quality"] > 0.7, "Fitting quality should be high for good fit"
    
    # Parameter errors should be reasonable
    param_errors = fit_results["parameter_errors"]
    assert len(param_errors) == 4, "Should have 4 parameter errors"
    assert all(error >= 0 for error in param_errors), "Parameter errors should be non-negative"
    
    print(f"   R-squared: {fit_results['r_squared']:.3f}")
    print(f"   Chi-squared: {fit_results['chi_squared']:.3f}")
    print(f"   Fitting quality: {fit_results['fitting_quality']:.3f}")
    print(f"   Parameter errors: {[f'{e:.3f}' for e in param_errors]}")


def test_fallback_methods_physics(analyzer):
    """Test fallback methods physics."""
    # Create test data with poor signal-to-noise ratio
    frequencies = np.linspace(0, 10, 30)
    
    # Create weak Lorentzian with high noise
    f0, gamma, A, offset = 5.0, 0.5, 2.0, 1.0
    magnitude = A * gamma**2 / ((frequencies - f0)**2 + gamma**2) + offset
    
    # Add high noise
    magnitude += 1.0 * np.random.randn(len(magnitude))
    
    peak_idx = np.argmax(magnitude)
    
    # Test fallback quality estimation
    q_factor = analyzer._fallback_quality_estimation(frequencies, magnitude, peak_idx)
    
    # Physics checks
    assert q_factor > 0, "Fallback quality factor should be positive"
    assert q_factor >= 1.0, "Fallback quality factor should be >= 1.0"
    assert q_factor <= 1000.0, "Fallback quality factor should be <= 1000.0"
    
    # Test fallback Lorentzian estimation
    fallback_results = analyzer._fallback_lorentzian_estimation(frequencies, magnitude, peak_idx)
    
    # Physics checks
    assert "amplitude" in fallback_results, "Should include amplitude"
    assert "center" in fallback_results, "Should include center frequency"
    assert "fwhm" in fallback_results, "Should include FWHM"
    assert "q_factor" in fallback_results, "Should include quality factor"
    assert "fitting_quality" in fallback_results, "Should include fitting quality"
    
    # Fallback should have lower quality
    assert fallback_results["fitting_quality"] < 0.8, "Fallback should have lower fitting quality"
    
    print(f"   Fallback Q-factor: {q_factor:.3f}")
    print(f"   Fallback center: {fallback_results['center']:.3f}")
    print(f"   Fallback FWHM: {fallback_results['fwhm']:.3f}")
    print(f"   Fallback fitting quality: {fallback_results['fitting_quality']:.3f}")


def test_optimization_physics():
    """Test optimization algorithms physics."""
    print("\n🔧 Test 5: Optimization Algorithms Physics")
    
    # Mock BVP constants
    class MockBVPConstants:
        def get_impedance_parameter(self, param_name):
            params = {
                "min_quality_factor": 1.0,
                "max_quality_factor": 1000.0,
                "peak_window_size": 15
            }
            return params.get(param_name, 1.0)
    
    # Initialize analyzer
    constants = MockBVPConstants()
    analyzer = ResonanceQualityAnalyzer(constants)
    
    # Create test data with known parameters
    frequencies = np.linspace(0, 10, 100)
    
    # Create Lorentzian with known parameters
    f0_true, gamma_true, A_true, offset_true = 5.0, 0.3, 12.0, 0.5
    magnitude = A_true * gamma_true**2 / ((frequencies - f0_true)**2 + gamma_true**2) + offset_true
    
    # Add moderate noise
    magnitude += 0.2 * np.random.randn(len(magnitude))
    
    peak_idx = np.argmax(magnitude)
    
    # Test optimization with different methods
    fit_results = analyzer.fit_lorentzian(frequencies, magnitude, peak_idx)
    
    # Physics checks
    assert fit_results["fitting_quality"] > 0.5, "Optimization should achieve reasonable quality"
    
    # Check parameter recovery
    center_error = abs(fit_results["center"] - f0_true) / f0_true
    amplitude_error = abs(fit_results["amplitude"] - A_true) / A_true
    
    assert center_error < 0.2, f"Center frequency recovery error too large: {center_error:.2e}"
    assert amplitude_error < 0.5, f"Amplitude recovery error too large: {amplitude_error:.2e}"
    
    print(f"   True center: {f0_true:.3f}, Fitted: {fit_results['center']:.3f}")
    print(f"   True amplitude: {A_true:.3f}, Fitted: {fit_results['amplitude']:.3f}")
    print(f"   Optimization quality: {fit_results['fitting_quality']:.3f}")


if __name__ == "__main__":
    try:
        test_resonance_quality_analyzer_physics()
        test_optimization_physics()
        print("\n🎉 All physics tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Physics test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
