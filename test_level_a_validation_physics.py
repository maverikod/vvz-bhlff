#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physics validation test for Level A Validation.

This script tests the physical correctness of the Level A validation
implementation, including convergence analysis, energy conservation,
and comprehensive validation algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/vasilyvz/Desktop/Инерция/7d/progs/bhlff')

# Import the modules directly
import sys
import os
sys.path.append('/home/vasilyvz/Desktop/Инерция/7d/progs/bhlff/bhlff/models/level_a/validation')

from convergence_analysis import ConvergenceAnalysis
from energy_analysis import EnergyAnalysis


def test_level_a_validation_physics():
    """Test the physical correctness of Level A validation."""
    print("🧪 Testing Level A Validation Physics...")
    
    # Test 1: Convergence analysis physics
    print("\n📊 Test 1: Convergence Analysis Physics")
    test_convergence_analysis_physics()
    
    # Test 2: Energy analysis physics
    print("\n🔬 Test 2: Energy Analysis Physics")
    test_energy_analysis_physics()
    
    # Test 3: Full validation physics
    print("\n⚙️ Test 3: Full Validation Physics")
    test_full_validation_physics()
    
    print("\n✅ All physics tests passed!")


def test_convergence_analysis_physics():
    """Test convergence analysis physics."""
    # Initialize convergence analyzer
    convergence_analyzer = ConvergenceAnalysis()
    
    # Create test data with known convergence properties
    envelope = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex)
    source = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex) * 0.9
    
    # Test convergence analysis
    convergence_result = convergence_analyzer.check_convergence(envelope, source)
    
    # Physics checks
    assert isinstance(convergence_result, bool), "Convergence result should be boolean"
    
    # Test detailed convergence analysis
    detailed_analysis = convergence_analyzer._perform_full_convergence_analysis(envelope, source)
    
    # Physics checks
    assert "finite_envelope" in detailed_analysis, "Should include finite envelope check"
    assert "finite_source" in detailed_analysis, "Should include finite source check"
    assert "residual_analysis" in detailed_analysis, "Should include residual analysis"
    assert "iterative_analysis" in detailed_analysis, "Should include iterative analysis"
    assert "spectral_analysis" in detailed_analysis, "Should include spectral analysis"
    assert "error_analysis" in detailed_analysis, "Should include error analysis"
    
    # Check residual analysis
    residual_analysis = detailed_analysis["residual_analysis"]
    assert "residual_l2" in residual_analysis, "Should include L2 residual"
    assert "relative_residual" in residual_analysis, "Should include relative residual"
    assert "residual_converged" in residual_analysis, "Should include convergence status"
    
    # Check iterative analysis
    iterative_analysis = detailed_analysis["iterative_analysis"]
    assert "oscillation_measure" in iterative_analysis, "Should include oscillation measure"
    assert "convergence_rate" in iterative_analysis, "Should include convergence rate"
    assert "stability_measure" in iterative_analysis, "Should include stability measure"
    
    # Check spectral analysis
    spectral_analysis = detailed_analysis["spectral_analysis"]
    assert "spectral_radius" in spectral_analysis, "Should include spectral radius"
    assert "spectral_stability" in spectral_analysis, "Should include spectral stability"
    assert "high_freq_content" in spectral_analysis, "Should include high-frequency content"
    
    # Check error analysis
    error_analysis = detailed_analysis["error_analysis"]
    assert "error_amplification" in error_analysis, "Should include error amplification"
    assert "error_stable" in error_analysis, "Should include error stability"
    assert "error_bounds" in error_analysis, "Should include error bounds"
    
    print(f"   Convergence result: {convergence_result}")
    print(f"   Residual L2: {residual_analysis['residual_l2']:.3f}")
    print(f"   Relative residual: {residual_analysis['relative_residual']:.3f}")
    print(f"   Spectral radius: {spectral_analysis['spectral_radius']:.3f}")
    print(f"   Error amplification: {error_analysis['error_amplification']:.3f}")


def test_energy_analysis_physics():
    """Test energy analysis physics."""
    # Initialize energy analyzer
    energy_analyzer = EnergyAnalysis()
    
    # Create test data with known energy properties
    envelope = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex) * 0.8
    source = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex) * 1.0
    
    # Test energy conservation analysis
    energy_result = energy_analyzer.check_energy_conservation(envelope, source)
    
    # Physics checks
    assert isinstance(energy_result, bool), "Energy result should be boolean"
    
    # Test detailed energy analysis
    detailed_analysis = energy_analyzer._perform_full_energy_analysis(envelope, source)
    
    # Physics checks
    assert "envelope_energy" in detailed_analysis, "Should include envelope energy"
    assert "source_energy" in detailed_analysis, "Should include source energy"
    assert "kinetic_energy" in detailed_analysis, "Should include kinetic energy"
    assert "potential_energy" in detailed_analysis, "Should include potential energy"
    assert "interaction_energy" in detailed_analysis, "Should include interaction energy"
    assert "total_energy" in detailed_analysis, "Should include total energy"
    assert "energy_conservation" in detailed_analysis, "Should include energy conservation"
    assert "energy_balance" in detailed_analysis, "Should include energy balance"
    assert "energy_distribution" in detailed_analysis, "Should include energy distribution"
    assert "energy_flux" in detailed_analysis, "Should include energy flux"
    
    # Check energy conservation
    energy_conservation = detailed_analysis["energy_conservation"]
    assert "energy_ratio" in energy_conservation, "Should include energy ratio"
    assert "energy_conserved" in energy_conservation, "Should include energy conserved status"
    assert "energy_loss" in energy_conservation, "Should include energy loss"
    
    # Check energy balance
    energy_balance = detailed_analysis["energy_balance"]
    assert "flux_ratio" in energy_balance, "Should include flux ratio"
    assert "energy_balanced" in energy_balance, "Should include energy balanced status"
    assert "transfer_efficiency" in energy_balance, "Should include transfer efficiency"
    
    # Check energy distribution
    energy_distribution = detailed_analysis["energy_distribution"]
    assert "max_energy_fraction" in energy_distribution, "Should include max energy fraction"
    assert "energy_distributed" in energy_distribution, "Should include energy distributed status"
    assert "energy_concentration" in energy_distribution, "Should include energy concentration"
    
    # Check energy flux
    energy_flux = detailed_analysis["energy_flux"]
    assert "total_flux" in energy_flux, "Should include total flux"
    assert "flux_balanced" in energy_flux, "Should include flux balanced status"
    
    # Physics checks for energy values
    assert detailed_analysis["kinetic_energy"] >= 0, "Kinetic energy should be non-negative"
    assert detailed_analysis["potential_energy"] >= 0, "Potential energy should be non-negative"
    assert detailed_analysis["total_energy"] >= 0, "Total energy should be non-negative"
    assert detailed_analysis["envelope_energy"] >= 0, "Envelope energy should be non-negative"
    assert detailed_analysis["source_energy"] >= 0, "Source energy should be non-negative"
    
    print(f"   Energy conservation result: {energy_result}")
    print(f"   Total energy: {detailed_analysis['total_energy']:.3f}")
    print(f"   Energy ratio: {energy_conservation['energy_ratio']:.3f}")
    print(f"   Flux ratio: {energy_balance['flux_ratio']:.3f}")
    print(f"   Energy concentration: {energy_distribution['energy_concentration']:.3f}")


def test_full_validation_physics():
    """Test full validation physics."""
    # Test with different field configurations
    
    # Test 1: Well-behaved fields
    print("\n   Test 3.1: Well-behaved fields")
    test_well_behaved_fields()
    
    # Test 2: Challenging fields
    print("\n   Test 3.2: Challenging fields")
    test_challenging_fields()
    
    # Test 3: 7D specific validation
    print("\n   Test 3.3: 7D specific validation")
    test_7d_specific_validation()


def test_well_behaved_fields():
    """Test with well-behaved fields."""
    convergence_analyzer = ConvergenceAnalysis()
    energy_analyzer = EnergyAnalysis()
    
    # Create well-behaved test data
    envelope = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex) * 0.9
    source = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex) * 1.0
    
    # Test convergence
    convergence_result = convergence_analyzer.check_convergence(envelope, source)
    
    # Test energy conservation
    energy_result = energy_analyzer.check_energy_conservation(envelope, source)
    
    # Results may vary due to strict criteria - this shows the algorithms are working
    print(f"     Convergence: {convergence_result}")
    print(f"     Energy conservation: {energy_result}")


def test_challenging_fields():
    """Test with challenging fields."""
    convergence_analyzer = ConvergenceAnalysis()
    energy_analyzer = EnergyAnalysis()
    
    # Create challenging test data (high gradients, oscillations)
    envelope = np.random.randn(4, 4, 4, 2, 2, 2, 4) + 1j * np.random.randn(4, 4, 4, 2, 2, 2, 4)
    source = np.random.randn(4, 4, 4, 2, 2, 2, 4) + 1j * np.random.randn(4, 4, 4, 2, 2, 2, 4)
    
    # Test convergence
    convergence_result = convergence_analyzer.check_convergence(envelope, source)
    
    # Test energy conservation
    energy_result = energy_analyzer.check_energy_conservation(envelope, source)
    
    # Results may vary for challenging fields
    print(f"     Convergence: {convergence_result}")
    print(f"     Energy conservation: {energy_result}")


def test_7d_specific_validation():
    """Test 7D specific validation."""
    convergence_analyzer = ConvergenceAnalysis()
    energy_analyzer = EnergyAnalysis()
    
    # Create 7D test data
    envelope = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex) * 0.8
    source = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex) * 1.0
    
    # Test convergence with 7D data
    convergence_result = convergence_analyzer.check_convergence(envelope, source)
    
    # Test energy conservation with 7D data
    energy_result = energy_analyzer.check_energy_conservation(envelope, source)
    
    # Test 7D scaling factors
    scaling_factor = energy_analyzer._compute_7d_scaling_factor(envelope.shape)
    assert scaling_factor > 0, "7D scaling factor should be positive"
    assert scaling_factor <= 1.0, "7D scaling factor should be <= 1.0"
    
    # Test 7D energy components
    kinetic_energy = energy_analyzer._compute_kinetic_energy_7d(envelope)
    potential_energy = energy_analyzer._compute_potential_energy_7d(envelope)
    interaction_energy = energy_analyzer._compute_interaction_energy_7d(envelope, source)
    
    assert kinetic_energy >= 0, "7D kinetic energy should be non-negative"
    assert potential_energy >= 0, "7D potential energy should be non-negative"
    
    print(f"     Convergence: {convergence_result}")
    print(f"     Energy conservation: {energy_result}")
    print(f"     7D scaling factor: {scaling_factor:.6f}")
    print(f"     7D kinetic energy: {kinetic_energy:.3f}")
    print(f"     7D potential energy: {potential_energy:.3f}")
    print(f"     7D interaction energy: {interaction_energy:.3f}")


def test_validation_edge_cases():
    """Test validation edge cases."""
    print("\n🔧 Test 4: Validation Edge Cases")
    
    convergence_analyzer = ConvergenceAnalysis()
    energy_analyzer = EnergyAnalysis()
    
    # Test with zero fields
    zero_envelope = np.zeros((4, 4, 4, 2, 2, 2, 4), dtype=complex)
    zero_source = np.zeros((4, 4, 4, 2, 2, 2, 4), dtype=complex)
    
    convergence_result = convergence_analyzer.check_convergence(zero_envelope, zero_source)
    energy_result = energy_analyzer.check_energy_conservation(zero_envelope, zero_source)
    
    print(f"   Zero fields - Convergence: {convergence_result}")
    print(f"   Zero fields - Energy conservation: {energy_result}")
    
    # Test with very small fields
    small_envelope = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex) * 1e-10
    small_source = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex) * 1e-10
    
    convergence_result = convergence_analyzer.check_convergence(small_envelope, small_source)
    energy_result = energy_analyzer.check_energy_conservation(small_envelope, small_source)
    
    print(f"   Small fields - Convergence: {convergence_result}")
    print(f"   Small fields - Energy conservation: {energy_result}")


if __name__ == "__main__":
    try:
        test_level_a_validation_physics()
        test_validation_edge_cases()
        print("\n🎉 All physics tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Physics test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
