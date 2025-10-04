"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level D example: Multimode superposition and field projections.

This example demonstrates the usage of Level D models for analyzing
multimode superposition patterns, field projections onto different
interaction windows, and phase streamline analysis.

Physical Meaning:
    Level D represents the multimode superposition and field projection level
    where all observed particles emerge as envelope functions of a
    high-frequency carrier field through different frequency-amplitude
    windows corresponding to electromagnetic, strong, and weak interactions.

Mathematical Foundation:
    - Multimode superposition: a(x,t) = Σ_m A_m(T) φ_m(x) e^(-iω_m t)
    - Field projections: P_EM[a], P_STRONG[a], P_WEAK[a] for different
      frequency windows
    - Phase streamlines: Analysis of ∇φ flow patterns around defects

Example:
    >>> python examples/level_d_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import json
import os

from bhlff.models.level_d import LevelDModels
from bhlff.core.domain import Domain


def create_test_field(domain: Domain) -> np.ndarray:
    """
    Create test field for Level D analysis.
    
    Physical Meaning:
        Creates a test field with multiple frequency components
        to demonstrate multimode superposition analysis.
        
    Args:
        domain (Domain): Computational domain
        
    Returns:
        np.ndarray: Test field
    """
    # Create a simple test field for 7D
    field = np.random.randn(*domain.shape) + 1j * np.random.randn(*domain.shape)
    return field


def run_mode_superposition_analysis(models: LevelDModels, field: np.ndarray) -> Dict[str, Any]:
    """
    Run mode superposition analysis (D1).
    
    Physical Meaning:
        Tests the stability of the phase field frame when
        adding new modes, ensuring topological robustness.
        
    Args:
        models (LevelDModels): Level D models
        field (np.ndarray): Base field
        
    Returns:
        Dict: Analysis results
    """
    print("Running mode superposition analysis (D1)...")
    
    # Define new modes to add
    new_modes = [
        {
            'frequency': 1.5,
            'amplitude': 0.3,
            'phase': 0.0,
            'spatial_mode': 'bvp_envelope_modulation'
        },
        {
            'frequency': 2.5,
            'amplitude': 0.2,
            'phase': 0.0,
            'spatial_mode': 'bvp_envelope_modulation'
        }
    ]
    
    # Analyze mode superposition
    results = models.analyze_mode_superposition(field, new_modes)
    
    print(f"Jaccard index: {results['jaccard_index']:.3f}")
    print(f"Frame stability: {'PASSED' if results['passed'] else 'FAILED'}")
    
    return results


def run_field_projection_analysis(models: LevelDModels, field: np.ndarray) -> Dict[str, Any]:
    """
    Run field projection analysis (D2).
    
    Physical Meaning:
        Separates the unified phase field into different
        interaction regimes based on frequency and amplitude
        characteristics.
        
    Args:
        models (LevelDModels): Level D models
        field (np.ndarray): Input field
        
    Returns:
        Dict: Analysis results
    """
    print("Running field projection analysis (D2)...")
    
    # Define window parameters
    window_params = {
        'em': {
            'frequency_range': [0.1, 1.0],
            'amplitude_threshold': 0.1,
            'filter_type': 'bandpass'
        },
        'strong': {
            'frequency_range': [1.0, 10.0],
            'q_threshold': 100,
            'filter_type': 'high_q'
        },
        'weak': {
            'frequency_range': [0.01, 0.1],
            'q_threshold': 10,
            'filter_type': 'chiral'
        }
    }
    
    # Project fields
    results = models.project_field_windows(field, window_params)
    
    # Analyze signatures
    signatures = results['signatures']
    
    print("Field projection results:")
    for field_type, signature in signatures.items():
        print(f"  {field_type.upper()}:")
        print(f"    Field norm: {signature['field_norm']:.3f}")
        print(f"    Localization: {signature['localization']:.3f}")
        print(f"    Anisotropy: {signature['anisotropy']:.3f}")
    
    return results


def run_streamline_analysis(models: LevelDModels, field: np.ndarray) -> Dict[str, Any]:
    """
    Run phase streamline analysis (D3).
    
    Physical Meaning:
        Computes streamlines of the phase gradient field,
        revealing the topological structure of phase flow
        around defects and singularities.
        
    Args:
        models (LevelDModels): Level D models
        field (np.ndarray): Input field
        
    Returns:
        Dict: Analysis results
    """
    print("Running phase streamline analysis (D3)...")
    
    # Define center point
    center = (5.0, 5.0, 5.0)
    
    # Trace streamlines
    results = models.trace_phase_streamlines(field, center)
    
    # Analyze topology
    topology = results['topology']
    
    print("Streamline analysis results:")
    print(f"  Number of streamlines: {topology['streamline_density']}")
    print(f"  Topology class: {topology['topology_class']}")
    print(f"  Stability index: {topology['stability_index']:.3f}")
    
    return results


def visualize_results(results: Dict[str, Any], output_dir: str = "output"):
    """
    Visualize analysis results.
    
    Physical Meaning:
        Creates visualizations of the analysis results to
        understand the field structure and dynamics.
        
    Args:
        results (Dict): Analysis results
        output_dir (str): Output directory
    """
    print("Creating visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize field projections
    if 'projections' in results:
        projections = results['projections']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Field Projections Analysis')
        
        # Original field
        axes[0, 0].imshow(projections['em_projection'][:, :, 16], cmap='viridis')
        axes[0, 0].set_title('EM Projection')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        
        axes[0, 1].imshow(projections['strong_projection'][:, :, 16], cmap='plasma')
        axes[0, 1].set_title('Strong Projection')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        
        axes[1, 0].imshow(projections['weak_projection'][:, :, 16], cmap='coolwarm')
        axes[1, 0].set_title('Weak Projection')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        
        # Phase field
        if 'streamlines' in results:
            phase = results['streamlines']['phase']
            axes[1, 1].imshow(phase[:, :, 16], cmap='hsv')
            axes[1, 1].set_title('Phase Field')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'field_projections.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def save_results(results: Dict[str, Any], output_dir: str = "output"):
    """
    Save analysis results to files.
    
    Physical Meaning:
        Saves the analysis results to JSON files for
        further analysis and documentation.
        
    Args:
        results (Dict): Analysis results
        output_dir (str): Output directory
    """
    print("Saving results...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mode superposition analysis
    if 'superposition' in results:
        with open(os.path.join(output_dir, 'mode_superposition_analysis.json'), 'w') as f:
            json.dump(results['superposition'], f, indent=2)
    
    # Save field projection analysis
    if 'projections' in results:
        with open(os.path.join(output_dir, 'field_projection_analysis.json'), 'w') as f:
            json.dump(results['projections'], f, indent=2)
    
    # Save streamline analysis
    if 'streamlines' in results:
        with open(os.path.join(output_dir, 'phase_streamline_analysis.json'), 'w') as f:
            json.dump(results['streamlines'], f, indent=2)
    
    print(f"Results saved to {output_dir}/")


def main():
    """Main function for Level D example."""
    print("Level D Example: Multimode Superposition and Field Projections")
    print("=" * 60)
    
    # Create domain
    domain = Domain(L=10.0, N=64, dimensions=7, N_phi=32, N_t=64, T=1.0)
    print(f"Domain: L={domain.L}, N={domain.N}, dimensions={domain.dimensions}")
    
    # Create parameters
    parameters = {
        'jaccard_threshold': 0.8,
        'frequency_tolerance': 0.05,
        'mode_threshold': 0.1,
        'num_streamlines': 100,
        'integration_steps': 1000,
        'step_size': 0.01
    }
    
    # Initialize Level D models
    models = LevelDModels(domain, parameters)
    print("Level D models initialized")
    
    # Create test field
    field = create_test_field(domain)
    print(f"Test field created: shape={field.shape}")
    
    # Run comprehensive analysis
    print("\nRunning comprehensive multimode field analysis...")
    results = models.analyze_multimode_field(field)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("=" * 30)
    
    if 'superposition' in results:
        superposition = results['superposition']
        print(f"Mode superposition analysis:")
        print(f"  Jaccard index: {superposition.get('jaccard_index', 'N/A'):.3f}")
        print(f"  Frame stability: {'PASSED' if superposition.get('passed', False) else 'FAILED'}")
    
    if 'projections' in results:
        projections = results['projections']
        print(f"Field projections:")
        print(f"  EM projection norm: {np.linalg.norm(projections['em_projection']):.3f}")
        print(f"  Strong projection norm: {np.linalg.norm(projections['strong_projection']):.3f}")
        print(f"  Weak projection norm: {np.linalg.norm(projections['weak_projection']):.3f}")
    
    if 'streamlines' in results:
        streamlines = results['streamlines']
        topology = streamlines['topology']
        print(f"Phase streamlines:")
        print(f"  Number of streamlines: {topology.get('streamline_density', 'N/A')}")
        print(f"  Topology class: {topology.get('topology_class', 'N/A')}")
        print(f"  Stability index: {topology.get('stability_index', 'N/A'):.3f}")
    
    # Save results
    save_results(results)
    
    # Create visualizations
    visualize_results(results)
    
    print("\nLevel D example completed successfully!")


if __name__ == "__main__":
    main()
