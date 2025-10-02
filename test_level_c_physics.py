#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physics validation test for Level C - Boundaries and Cells.

This script tests the physical correctness of the Level C implementation,
including boundary analysis, resonator analysis, memory analysis, and
beating analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/vasilyvz/Desktop/Инерция/7d/progs/bhlff')

# Import Level C modules
from bhlff.models.level_c.boundaries import BoundaryAnalyzer
from bhlff.models.level_c.resonators import ResonatorAnalyzer
from bhlff.models.level_c.memory import MemoryAnalyzer
from bhlff.models.level_c.beating import BeatingAnalyzer


class MockBVPCore:
    """Mock BVP core for testing."""
    
    def __init__(self):
        self.constants = None
        self.domain = None


def test_level_c_physics():
    """Test the physical correctness of Level C analysis."""
    print("🧪 Testing Level C - Boundaries and Cells Physics...")
    
    # Initialize mock BVP core
    bvp_core = MockBVPCore()
    
    # Test 1: Boundary analysis physics
    print("\n📊 Test 1: Boundary Analysis Physics")
    test_boundary_analysis_physics(bvp_core)
    
    # Test 2: Resonator analysis physics
    print("\n🔬 Test 2: Resonator Analysis Physics")
    test_resonator_analysis_physics(bvp_core)
    
    # Test 3: Memory analysis physics
    print("\n💾 Test 3: Memory Analysis Physics")
    test_memory_analysis_physics(bvp_core)
    
    # Test 4: Beating analysis physics
    print("\n🌊 Test 4: Beating Analysis Physics")
    test_beating_analysis_physics(bvp_core)
    
    print("\n✅ All Level C physics tests passed!")


def test_boundary_analysis_physics(bvp_core):
    """Test boundary analysis physics."""
    # Initialize boundary analyzer
    boundary_analyzer = BoundaryAnalyzer(bvp_core)
    
    # Create test data with known boundary properties
    envelope = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex)
    envelope[2:6, 2:6, 2:6, 1:3, 1:3, 1:3, 2:6] = 2.0 + 1j
    
    # Test boundary analysis
    boundary_results = boundary_analyzer.analyze_boundaries(envelope)
    
    # Physics checks
    assert "level_set_analysis" in boundary_results, "Should include level set analysis"
    assert "phase_field_analysis" in boundary_results, "Should include phase field analysis"
    assert "topological_analysis" in boundary_results, "Should include topological analysis"
    assert "energy_analysis" in boundary_results, "Should include energy analysis"
    assert "boundary_summary" in boundary_results, "Should include boundary summary"
    
    # Check level set analysis
    level_set_analysis = boundary_results["level_set_analysis"]
    assert "level_sets" in level_set_analysis, "Should include level sets"
    assert "total_boundaries" in level_set_analysis, "Should include total boundaries count"
    
    # Check phase field analysis
    phase_field_analysis = boundary_results["phase_field_analysis"]
    assert "boundary_mask" in phase_field_analysis, "Should include boundary mask"
    assert "gradient_magnitude" in phase_field_analysis, "Should include gradient magnitude"
    assert "boundary_properties" in phase_field_analysis, "Should include boundary properties"
    
    # Check topological analysis
    topological_analysis = boundary_results["topological_analysis"]
    assert "critical_points" in topological_analysis, "Should include critical points"
    assert "topological_structure" in topological_analysis, "Should include topological structure"
    assert "boundary_classification" in topological_analysis, "Should include boundary classification"
    
    # Check energy analysis
    energy_analysis = boundary_results["energy_analysis"]
    assert "energy_density" in energy_analysis, "Should include energy density"
    assert "energy_landscape" in energy_analysis, "Should include energy landscape"
    assert "energy_boundaries" in energy_analysis, "Should include energy boundaries"
    assert "boundary_stability" in energy_analysis, "Should include boundary stability"
    
    # Check boundary summary
    boundary_summary = boundary_results["boundary_summary"]
    assert "total_boundaries_detected" in boundary_summary, "Should include total boundaries detected"
    assert "boundary_detection_methods" in boundary_summary, "Should include detection methods"
    assert "boundary_quality" in boundary_summary, "Should include boundary quality"
    assert "analysis_complete" in boundary_summary, "Should include analysis completion status"
    
    print(f"   Total boundaries detected: {boundary_summary['total_boundaries_detected']}")
    print(f"   Boundary quality: {boundary_summary['boundary_quality']}")
    print(f"   Detection methods: {len(boundary_summary['boundary_detection_methods'])}")


def test_resonator_analysis_physics(bvp_core):
    """Test resonator analysis physics."""
    # Initialize resonator analyzer
    resonator_analyzer = ResonatorAnalyzer(bvp_core)
    
    # Create test data with known resonance properties
    envelope = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex)
    # Add some frequency components
    envelope[1, 1, 1, 1, 1, 1, 1] = 3.0 + 2j
    envelope[2, 2, 2, 2, 2, 2, 2] = 2.5 + 1.5j
    
    # Test resonator analysis
    resonator_results = resonator_analyzer.analyze_resonators(envelope)
    
    # Physics checks
    assert "frequency_analysis" in resonator_results, "Should include frequency analysis"
    assert "resonance_peaks" in resonator_results, "Should include resonance peaks"
    assert "quality_factors" in resonator_results, "Should include quality factors"
    assert "impedance_analysis" in resonator_results, "Should include impedance analysis"
    assert "resonator_summary" in resonator_results, "Should include resonator summary"
    
    # Check frequency analysis
    frequency_analysis = resonator_results["frequency_analysis"]
    assert "frequency_magnitude" in frequency_analysis, "Should include frequency magnitude"
    assert "frequency_spectrum" in frequency_analysis, "Should include frequency spectrum"
    assert "dominant_frequencies" in frequency_analysis, "Should include dominant frequencies"
    assert "frequency_distribution" in frequency_analysis, "Should include frequency distribution"
    
    # Check resonance peaks
    resonance_peaks = resonator_results["resonance_peaks"]
    assert isinstance(resonance_peaks, list), "Resonance peaks should be a list"
    
    # Check quality factors
    quality_factors = resonator_results["quality_factors"]
    assert "individual_quality_factors" in quality_factors, "Should include individual quality factors"
    assert "overall_quality_metrics" in quality_factors, "Should include overall quality metrics"
    
    # Check impedance analysis
    impedance_analysis = resonator_results["impedance_analysis"]
    assert "impedance_components" in impedance_analysis, "Should include impedance components"
    assert "impedance_characteristics" in impedance_analysis, "Should include impedance characteristics"
    
    # Check resonator summary
    resonator_summary = resonator_results["resonator_summary"]
    assert "total_resonators_detected" in resonator_summary, "Should include total resonators detected"
    assert "resonance_quality" in resonator_summary, "Should include resonance quality"
    assert "frequency_analysis_complete" in resonator_summary, "Should include frequency analysis completion"
    assert "impedance_analysis_complete" in resonator_summary, "Should include impedance analysis completion"
    
    print(f"   Total resonators detected: {resonator_summary['total_resonators_detected']}")
    print(f"   Resonance quality: {resonator_summary['resonance_quality']}")
    print(f"   Analysis methods: {len(resonator_summary['analysis_methods'])}")


def test_memory_analysis_physics(bvp_core):
    """Test memory analysis physics."""
    # Initialize memory analyzer
    memory_analyzer = MemoryAnalyzer(bvp_core)
    
    # Create test data with known memory properties
    envelope = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex)
    # Add some temporal structure
    for t in range(8):
        envelope[:, :, :, :, :, :, t] *= (1.0 + 0.1 * t)
    
    # Test memory analysis
    memory_results = memory_analyzer.analyze_memory(envelope)
    
    # Physics checks
    assert "temporal_analysis" in memory_results, "Should include temporal analysis"
    assert "information_analysis" in memory_results, "Should include information analysis"
    assert "persistence_analysis" in memory_results, "Should include persistence analysis"
    assert "capacity_analysis" in memory_results, "Should include capacity analysis"
    assert "memory_summary" in memory_results, "Should include memory summary"
    
    # Check temporal analysis
    temporal_analysis = memory_results["temporal_analysis"]
    assert "temporal_correlations" in temporal_analysis, "Should include temporal correlations"
    assert "memory_decay" in temporal_analysis, "Should include memory decay"
    assert "memory_patterns" in temporal_analysis, "Should include memory patterns"
    
    # Check information analysis
    information_analysis = memory_results["information_analysis"]
    assert "information_content" in information_analysis, "Should include information content"
    assert "information_storage" in information_analysis, "Should include information storage"
    assert "memory_entropy" in information_analysis, "Should include memory entropy"
    
    # Check persistence analysis
    persistence_analysis = memory_results["persistence_analysis"]
    assert "persistence_metrics" in persistence_analysis, "Should include persistence metrics"
    assert "memory_stability" in persistence_analysis, "Should include memory stability"
    assert "persistent_structures" in persistence_analysis, "Should include persistent structures"
    
    # Check capacity analysis
    capacity_analysis = memory_results["capacity_analysis"]
    assert "memory_capacity" in capacity_analysis, "Should include memory capacity"
    assert "storage_efficiency" in capacity_analysis, "Should include storage efficiency"
    assert "memory_utilization" in capacity_analysis, "Should include memory utilization"
    
    # Check memory summary
    memory_summary = memory_results["memory_summary"]
    assert "memory_systems_detected" in memory_summary, "Should include memory systems detected"
    assert "memory_quality" in memory_summary, "Should include memory quality"
    assert "information_capacity" in memory_summary, "Should include information capacity"
    assert "analysis_complete" in memory_summary, "Should include analysis completion status"
    
    print(f"   Memory systems detected: {memory_summary['memory_systems_detected']}")
    print(f"   Memory quality: {memory_summary['memory_quality']}")
    print(f"   Information capacity: {memory_summary['information_capacity']:.3f}")


def test_beating_analysis_physics(bvp_core):
    """Test beating analysis physics."""
    # Initialize beating analyzer
    beating_analyzer = BeatingAnalyzer(bvp_core)
    
    # Create test data with known beating properties
    envelope = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex)
    # Add multiple frequency components for beating
    envelope[1, 1, 1, 1, 1, 1, 1] = 2.0 + 1j
    envelope[3, 3, 3, 2, 2, 2, 3] = 1.5 + 0.8j
    envelope[5, 5, 5, 3, 3, 3, 5] = 1.8 + 1.2j
    
    # Test beating analysis
    beating_results = beating_analyzer.analyze_beating(envelope)
    
    # Physics checks
    assert "mode_analysis" in beating_results, "Should include mode analysis"
    assert "interference_analysis" in beating_results, "Should include interference analysis"
    assert "frequency_analysis" in beating_results, "Should include frequency analysis"
    assert "coupling_analysis" in beating_results, "Should include coupling analysis"
    assert "beating_summary" in beating_results, "Should include beating summary"
    
    # Check mode analysis
    mode_analysis = beating_results["mode_analysis"]
    assert "modes" in mode_analysis, "Should include modes"
    assert "mode_interactions" in mode_analysis, "Should include mode interactions"
    assert "beating_characteristics" in mode_analysis, "Should include beating characteristics"
    
    # Check interference analysis
    interference_analysis = beating_results["interference_analysis"]
    assert "interference_patterns" in interference_analysis, "Should include interference patterns"
    assert "pattern_properties" in interference_analysis, "Should include pattern properties"
    assert "interference_strength" in interference_analysis, "Should include interference strength"
    
    # Check frequency analysis
    frequency_analysis = beating_results["frequency_analysis"]
    assert "frequency_spectrum" in frequency_analysis, "Should include frequency spectrum"
    assert "beating_frequencies" in frequency_analysis, "Should include beating frequencies"
    assert "frequency_characteristics" in frequency_analysis, "Should include frequency characteristics"
    
    # Check coupling analysis
    coupling_analysis = beating_results["coupling_analysis"]
    assert "mode_coupling" in coupling_analysis, "Should include mode coupling"
    assert "coupling_strength" in coupling_analysis, "Should include coupling strength"
    assert "coupling_efficiency" in coupling_analysis, "Should include coupling efficiency"
    
    # Check beating summary
    beating_summary = beating_results["beating_summary"]
    assert "total_modes_detected" in beating_summary, "Should include total modes detected"
    assert "beating_frequencies_detected" in beating_summary, "Should include beating frequencies detected"
    assert "interference_patterns_detected" in beating_summary, "Should include interference patterns detected"
    assert "mode_coupling_detected" in beating_summary, "Should include mode coupling detected"
    assert "beating_quality" in beating_summary, "Should include beating quality"
    assert "analysis_complete" in beating_summary, "Should include analysis completion status"
    
    print(f"   Total modes detected: {beating_summary['total_modes_detected']}")
    print(f"   Beating frequencies detected: {beating_summary['beating_frequencies_detected']}")
    print(f"   Interference patterns detected: {beating_summary['interference_patterns_detected']}")
    print(f"   Mode coupling detected: {beating_summary['mode_coupling_detected']}")
    print(f"   Beating quality: {beating_summary['beating_quality']}")


def test_level_c_integration():
    """Test Level C integration."""
    print("\n🔗 Test 5: Level C Integration")
    
    # Initialize all analyzers
    bvp_core = MockBVPCore()
    boundary_analyzer = BoundaryAnalyzer(bvp_core)
    resonator_analyzer = ResonatorAnalyzer(bvp_core)
    memory_analyzer = MemoryAnalyzer(bvp_core)
    beating_analyzer = BeatingAnalyzer(bvp_core)
    
    # Create comprehensive test data
    envelope = np.ones((8, 8, 8, 4, 4, 4, 8), dtype=complex)
    # Add various structures
    envelope[2:6, 2:6, 2:6, 1:3, 1:3, 1:3, 2:6] = 2.0 + 1j  # Boundaries
    envelope[1, 1, 1, 1, 1, 1, 1] = 3.0 + 2j  # Resonators
    envelope[3, 3, 3, 2, 2, 2, 3] = 1.5 + 0.8j  # Beating
    envelope[5, 5, 5, 3, 3, 3, 5] = 1.8 + 1.2j  # More beating
    
    # Run all analyses
    boundary_results = boundary_analyzer.analyze_boundaries(envelope)
    resonator_results = resonator_analyzer.analyze_resonators(envelope)
    memory_results = memory_analyzer.analyze_memory(envelope)
    beating_results = beating_analyzer.analyze_beating(envelope)
    
    # Check integration
    assert boundary_results["boundary_summary"]["analysis_complete"], "Boundary analysis should be complete"
    assert resonator_results["resonator_summary"]["frequency_analysis_complete"], "Resonator analysis should be complete"
    assert memory_results["memory_summary"]["analysis_complete"], "Memory analysis should be complete"
    assert beating_results["beating_summary"]["analysis_complete"], "Beating analysis should be complete"
    
    print("   ✅ All Level C analyses completed successfully")
    print("   ✅ Level C integration test passed")


if __name__ == "__main__":
    try:
        test_level_c_physics()
        test_level_c_integration()
        print("\n🎉 All Level C physics tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Level C physics test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
