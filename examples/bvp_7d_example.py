"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Example usage of the complete 7D BVP framework.

This example demonstrates how to use the full 7D BVP framework including:
- 7D space-time domain setup
- 7D envelope equation solving
- All 9 BVP postulates validation
- Complete BVP workflow

Physical Meaning:
    Demonstrates the complete BVP workflow in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ,
    showing how all components work together to solve the BVP envelope equation
    and validate the field properties.

Example:
    >>> python bvp_7d_example.py
"""

import numpy as np
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bhlff.core.domain.domain_7d import Domain7D, SpatialConfig, PhaseConfig, TemporalConfig
from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_core import BVPCore


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_7d_domain(config: dict) -> Domain7D:
    """Create 7D space-time domain from configuration."""
    spatial_config = SpatialConfig(**config['domain_7d']['spatial'])
    phase_config = PhaseConfig(**config['domain_7d']['phase'])
    temporal_config = TemporalConfig(**config['domain_7d']['temporal'])
    
    return Domain7D(spatial_config, phase_config, temporal_config)


def create_3d_domain(config: dict) -> Domain:
    """Create 3D domain for compatibility."""
    spatial = config['domain_7d']['spatial']
    return Domain(
        L=spatial['L_x'],
        N=spatial['N_x'],
        dimensions=7
    )


def create_source_7d(domain_7d: Domain7D) -> np.ndarray:
    """
    Create 7D source term for BVP equation.
    
    Physical Meaning:
        Creates a source term s(x,φ,t) that represents external excitations
        or initial conditions in 7D space-time.
    """
    # Get 7D shape
    full_shape = domain_7d.get_full_7d_shape()
    
    # Create simple 7D source array
    source_7d = np.ones(full_shape, dtype=complex) * 0.1
    
    return source_7d


def run_bvp_7d_example():
    """Run complete 7D BVP example."""
    print("=== 7D BVP Framework Example ===")
    print()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'bvp_7d_config.json')
    config = load_config(config_path)
    print("✓ Configuration loaded")
    
    # Create 7D domain
    domain_7d = create_7d_domain(config)
    print(f"✓ 7D domain created: {domain_7d}")
    
    # Create 3D domain for compatibility
    domain_3d = create_3d_domain(config)
    print(f"✓ 3D domain created: {domain_3d}")
    
    # Create BVP core with 7D support
    bvp_core = BVPCore(domain_3d, config['bvp_7d_core'], domain_7d)
    print("✓ BVP core initialized with 7D support")
    
    # Create 7D source
    source_7d = create_source_7d(domain_7d)
    print(f"✓ 7D source created with shape: {source_7d.shape}")
    
    # Solve 7D envelope equation
    print("\n--- Solving 7D Envelope Equation ---")
    try:
        envelope_7d = bvp_core.solve_envelope_7d(source_7d)
        print(f"✓ 7D envelope equation solved")
        print(f"  Envelope shape: {envelope_7d.shape}")
        print(f"  Envelope magnitude range: [{np.min(np.abs(envelope_7d)):.2e}, {np.max(np.abs(envelope_7d)):.2e}]")
    except Exception as e:
        print(f"✗ Error solving 7D envelope equation: {e}")
        return
    
    # Validate all 9 BVP postulates
    print("\n--- Validating BVP Postulates ---")
    try:
        postulate_results = bvp_core.validate_postulates_7d(envelope_7d)
        print(f"✓ All postulates validated")
        print(f"  Overall satisfied: {postulate_results['overall_satisfied']}")
        print(f"  Satisfied postulates: {postulate_results['satisfaction_count']}/{postulate_results['total_postulates']}")
        
        # Print individual postulate results
        print("\n  Individual postulate results:")
        for name, result in postulate_results['postulate_results'].items():
            status = "✓" if result.get('postulate_satisfied', False) else "✗"
            print(f"    {status} {name}: {result.get('postulate_satisfied', False)}")
            
    except Exception as e:
        print(f"✗ Error validating postulates: {e}")
        return
    
    # Demonstrate phase operations
    print("\n--- Phase Operations ---")
    try:
        phase_ops = bvp_core.get_phase_operations()
        phase_components = phase_ops.get_phase_components()
        total_phase = phase_ops.get_total_phase()
        phase_coherence = phase_ops.compute_phase_coherence()
        
        print(f"✓ Phase operations completed")
        print(f"  Number of phase components: {len(phase_components)}")
        print(f"  Total phase shape: {total_phase.shape}")
        print(f"  Phase coherence: {phase_coherence:.3f}")
        
    except Exception as e:
        print(f"✗ Error in phase operations: {e}")
    
    # Demonstrate parameter access
    print("\n--- Parameter Access ---")
    try:
        param_access = bvp_core.get_parameter_access()
        carrier_freq = param_access.get_carrier_frequency()
        envelope_params = param_access.get_envelope_parameters()
        quench_thresholds = param_access.get_quench_thresholds()
        
        print(f"✓ Parameter access completed")
        print(f"  Carrier frequency: {carrier_freq:.2e}")
        print(f"  Envelope parameters: {envelope_params}")
        print(f"  Quench thresholds: {quench_thresholds}")
        
    except Exception as e:
        print(f"✗ Error in parameter access: {e}")
    
    # Demonstrate 7D domain operations
    print("\n--- 7D Domain Operations ---")
    try:
        spatial_coords = domain_7d.get_spatial_coordinates()
        phase_coords = domain_7d.get_phase_coordinates()
        temporal_coords = domain_7d.get_temporal_coordinates()
        metric_tensor = domain_7d.get_metric_tensor()
        volume_element = domain_7d.compute_7d_volume_element()
        
        print(f"✓ 7D domain operations completed")
        print(f"  Spatial coordinates shape: {[c.shape for c in spatial_coords]}")
        print(f"  Phase coordinates shape: {[c.shape for c in phase_coords]}")
        print(f"  Temporal coordinates shape: {temporal_coords.shape}")
        print(f"  Metric tensor shape: {metric_tensor.shape}")
        print(f"  7D volume element: {volume_element:.2e}")
        
    except Exception as e:
        print(f"✗ Error in 7D domain operations: {e}")
    
    print("\n=== 7D BVP Example Completed Successfully ===")


if __name__ == "__main__":
    run_bvp_7d_example()
