"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Example usage of Level B fundamental properties analysis.

This example demonstrates how to use the Level B analysis tools to
validate fundamental properties of the phase field in homogeneous medium.

Theoretical Background:
    Level B analysis validates the fundamental behavior of the phase field
    governed by the Riesz operator L_β = μ(-Δ)^β + λ, including power law
    tails, absence of nodes, topological stability, and zone separation.

Example:
    >>> python examples/level_b_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_core import BVPCore
from bhlff.core.bvp.parameters import BVPParameters
from bhlff.core.sources.point_source import PointSource
from bhlff.models.level_b import (
    LevelBPowerLawAnalyzer, 
    LevelBNodeAnalyzer, 
    LevelBZoneAnalyzer,
    LevelBVisualizer
)
from bhlff.tests.unit.test_level_b.test_fundamental_properties import LevelBFundamentalPropertiesTests


def run_level_b_analysis_example():
    """
    Run comprehensive Level B analysis example.
    
    Physical Meaning:
        Demonstrates the complete Level B analysis workflow,
        showing how to validate fundamental properties of the
        phase field in homogeneous medium.
    """
    print("="*60)
    print("LEVEL B FUNDAMENTAL PROPERTIES ANALYSIS EXAMPLE")
    print("="*60)
    
    # 1. Setup domain and parameters
    print("\n1. Setting up domain and parameters...")
    domain = Domain(L=10.0, N=256, N_phi=4, N_t=8, T=1.0)
    bvp_params = BVPParameters(mu=1.0, beta=1.0, lambda_param=0.0)
    bvp_core = BVPCore(domain, bvp_params)
    
    print(f"Domain shape: {domain.shape}")
    print(f"BVP parameters: μ={bvp_params.mu}, β={bvp_params.beta}, λ={bvp_params.lambda_param}")
    
    # 2. Create point source
    print("\n2. Creating point source...")
    center = [5.0, 5.0, 5.0]
    source = PointSource(center=center, amplitude=1.0)
    source_field = source.create_field(domain)
    
    print(f"Source center: {center}")
    print(f"Source field shape: {source_field.shape}")
    
    # 3. Solve BVP equation
    print("\n3. Solving BVP equation...")
    solution = bvp_core.solve_envelope(source_field)
    print(f"Solution shape: {solution.shape}")
    print(f"Solution amplitude range: [{np.min(np.abs(solution)):.3e}, {np.max(np.abs(solution)):.3e}]")
    
    # 4. Initialize analyzers
    print("\n4. Initializing analyzers...")
    power_law_analyzer = LevelBPowerLawAnalyzer()
    node_analyzer = LevelBNodeAnalyzer()
    zone_analyzer = LevelBZoneAnalyzer()
    visualizer = LevelBVisualizer()
    
    # 5. Run individual analyses
    print("\n5. Running individual analyses...")
    
    # Power law analysis
    print("   - Power law analysis...")
    power_law_result = power_law_analyzer.analyze_power_law_tail(solution, 1.0, center)
    print(f"     Slope: {power_law_result['slope']:.3f} (theoretical: {power_law_result['theoretical_slope']:.3f})")
    print(f"     R²: {power_law_result['r_squared']:.3f}")
    print(f"     Passed: {power_law_result['passed']}")
    
    # Node analysis
    print("   - Node analysis...")
    node_result = node_analyzer.check_spherical_nodes(solution, center)
    print(f"     Sign changes: {node_result['sign_changes']}")
    print(f"     Zeros found: {len(node_result['zeros'])}")
    print(f"     Monotonic: {node_result['is_monotonic']}")
    print(f"     Passed: {node_result['passed']}")
    
    # Topological charge
    print("   - Topological charge...")
    charge_result = node_analyzer.compute_topological_charge(solution, center)
    print(f"     Charge: {charge_result['charge']:.3f}")
    print(f"     Integer charge: {charge_result['integer_charge']}")
    print(f"     Error: {charge_result['error']:.3f}")
    print(f"     Passed: {charge_result['passed']}")
    
    # Zone separation
    print("   - Zone separation...")
    thresholds = {'N_core': 3.0, 'S_core': 1.0, 'N_tail': 0.3, 'S_tail': 0.3}
    zone_result = zone_analyzer.separate_zones(solution, center, thresholds)
    print(f"     Core radius: {zone_result['r_core']:.3f}")
    print(f"     Tail radius: {zone_result['r_tail']:.3f}")
    print(f"     Core fraction: {zone_result['zone_stats']['core']['volume_fraction']:.3f}")
    print(f"     Quality score: {zone_result['quality_metrics']['overall_score']:.3f}")
    
    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    output_dir = "level_b_analysis_output"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Individual visualizations
    power_law_analyzer.visualize_power_law_analysis(
        power_law_result, 
        f"{output_dir}/power_law_analysis.png"
    )
    
    node_analyzer.visualize_node_analysis(
        node_result,
        f"{output_dir}/node_analysis.png"
    )
    
    zone_analyzer.visualize_zone_analysis(
        zone_result,
        f"{output_dir}/zone_analysis.png"
    )
    
    # 3D visualization
    visualizer.create_3d_visualization(
        solution, center,
        f"{output_dir}/3d_field_visualization.png"
    )
    
    print(f"   Visualizations saved to {output_dir}/")
    
    # 7. Run comprehensive test suite
    print("\n7. Running comprehensive test suite...")
    test_suite = LevelBFundamentalPropertiesTests()
    test_results = test_suite.run_all_tests()
    
    # Create comprehensive report
    print("\n8. Creating comprehensive report...")
    visualizer.create_comprehensive_report(
        test_results,
        f"{output_dir}/comprehensive_report"
    )
    
    # 8. Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result.get('passed', False))
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Success rate: {success_rate:.1%}")
    
    print("\nIndividual test results:")
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result.get('passed', False) else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not result.get('passed', False) and 'error' in result:
            print(f"    Error: {result['error']}")
    
    print(f"\nAll visualizations and results saved to: {output_dir}/")
    print("="*60)


def run_parameter_variation_example():
    """
    Run parameter variation analysis example.
    
    Physical Meaning:
        Demonstrates how to analyze the sensitivity of Level B
        properties to different parameters, validating the
        theoretical predictions across parameter space.
    """
    print("\n" + "="*60)
    print("PARAMETER VARIATION ANALYSIS")
    print("="*60)
    
    # Setup base parameters
    domain = Domain(L=10.0, N=128, N_phi=4, N_t=8, T=1.0)
    center = [5.0, 5.0, 5.0]
    source = PointSource(center=center, amplitude=1.0)
    
    # Test different beta values
    beta_values = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.4]
    power_law_analyzer = LevelBPowerLawAnalyzer()
    
    print("Testing power law behavior for different β values:")
    print("β\tSlope\tTheoretical\tError\tR²\tPassed")
    print("-" * 50)
    
    for beta in beta_values:
        try:
            bvp_params = BVPParameters(mu=1.0, beta=beta, lambda_param=0.0)
            bvp_core = BVPCore(domain, bvp_params)
            
            source_field = source.create_field(domain)
            solution = bvp_core.solve_envelope(source_field)
            
            result = power_law_analyzer.analyze_power_law_tail(solution, beta, center)
            
            print(f"{beta:.1f}\t{result['slope']:.3f}\t{result['theoretical_slope']:.3f}\t"
                  f"{result['relative_error']:.1%}\t{result['r_squared']:.3f}\t{result['passed']}")
            
        except Exception as e:
            print(f"{beta:.1f}\tERROR: {str(e)}")
    
    print("\nParameter variation analysis completed!")


if __name__ == "__main__":
    # Run main example
    run_level_b_analysis_example()
    
    # Run parameter variation example
    run_parameter_variation_example()
    
    print("\n🎉 Level B analysis example completed successfully!")
    print("Check the 'level_b_analysis_output' directory for all results and visualizations.")
