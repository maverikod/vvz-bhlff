"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Example usage of level G models for cosmological and astrophysical applications.

This example demonstrates how to use the level G models for studying
cosmological evolution, large-scale structure formation, astrophysical
objects, and gravitational effects in the 7D phase field theory.

Physical Meaning:
    Demonstrates the application of 7D phase field theory to cosmological
    and astrophysical problems, including universe evolution, structure
    formation, and particle physics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import level G models
from bhlff.models.level_g import (
    CosmologicalModel, 
    AstrophysicalObjectModel,
    GravitationalEffectsModel,
    LargeScaleStructureModel,
    ParticleInversion,
    ParticleValidation
)


def example_cosmological_evolution():
    """
    Example of cosmological evolution.
    
    Physical Meaning:
        Demonstrates the evolution of phase field in expanding universe,
        including structure formation and cosmological parameters.
    """
    print("=== Cosmological Evolution Example ===")
    
    # Set up initial conditions
    initial_conditions = {
        'type': 'gaussian_fluctuations',
        'domain_size': 1000.0,  # Mpc
        'resolution': 128,
        'seed': 42
    }
    
    # Set up cosmological parameters
    cosmology_params = {
        'time_start': 0.0,
        'time_end': 1.0,  # Gyr (shortened for example)
        'dt': 0.01,
        'c_phi': 1e10,  # Phase velocity
        'H0': 70.0,  # Hubble constant km/s/Mpc
        'omega_m': 0.3,  # Matter density
        'omega_lambda': 0.7,  # Dark energy density
        'domain_size': 1000.0,
        'resolution': 128
    }
    
    # Create cosmological model
    cosmology = CosmologicalModel(initial_conditions, cosmology_params)
    
    # Evolve universe
    print("Evolving universe...")
    evolution_results = cosmology.evolve_universe()
    
    # Analyze results
    print(f"Evolution time: {evolution_results['time'][-1]:.2f} Gyr")
    print(f"Final scale factor: {evolution_results['scale_factor'][-1]:.2f}")
    print(f"Number of time steps: {len(evolution_results['time'])}")
    
    # Analyze structure formation
    structure_analysis = cosmology.analyze_structure_formation()
    print(f"Structure growth rate: {structure_analysis.get('structure_growth_rate', 0):.2e}")
    
    # Compute cosmological parameters
    cosmological_params = cosmology.compute_cosmological_parameters()
    print(f"Current Hubble parameter: {cosmological_params.get('current_hubble_parameter', 0):.2f}")
    print(f"Phase velocity: {cosmological_params.get('phase_velocity', 0):.2e}")
    
    return evolution_results


def example_astrophysical_objects():
    """
    Example of astrophysical objects.
    
    Physical Meaning:
        Demonstrates the representation of astrophysical objects as
        phase field configurations with specific topological properties.
    """
    print("\n=== Astrophysical Objects Example ===")
    
    # Create star model
    stellar_params = {
        'mass': 1.0,  # Solar masses
        'radius': 1.0,  # Solar radii
        'temperature': 5778.0,  # K
        'phase_amplitude': 1.0,
        'grid_size': 64,
        'domain_size': 10.0
    }
    
    star = AstrophysicalObjectModel('star', stellar_params)
    star_properties = star.analyze_phase_properties()
    star_observables = star.compute_observable_properties()
    
    print(f"Star - Topological charge: {star_properties['topological_charge']}")
    print(f"Star - Phase amplitude: {star_properties['phase_amplitude']:.2e}")
    print(f"Star - Total mass: {star_observables['total_mass']:.2f} M☉")
    
    # Create galaxy model
    galactic_params = {
        'mass': 1e11,  # Solar masses
        'radius': 10.0,  # kpc
        'spiral_arms': 2,
        'bulge_ratio': 0.3,
        'grid_size': 64,
        'domain_size': 50.0
    }
    
    galaxy = AstrophysicalObjectModel('galaxy', galactic_params)
    galaxy_properties = galaxy.analyze_phase_properties()
    galaxy_observables = galaxy.compute_observable_properties()
    
    print(f"Galaxy - Topological charge: {galaxy_properties['topological_charge']}")
    print(f"Galaxy - Phase amplitude: {galaxy_properties['phase_amplitude']:.2e}")
    print(f"Galaxy - Total mass: {galaxy_observables['total_mass']:.2e} M☉")
    
    # Create black hole model
    bh_params = {
        'mass': 10.0,  # Solar masses
        'spin': 0.5,
        'schwarzschild_radius': 1.0,
        'alpha': 1.0,
        'grid_size': 64,
        'domain_size': 20.0
    }
    
    black_hole = AstrophysicalObjectModel('black_hole', bh_params)
    bh_properties = black_hole.analyze_phase_properties()
    bh_observables = black_hole.compute_observable_properties()
    
    print(f"Black hole - Topological charge: {bh_properties['topological_charge']}")
    print(f"Black hole - Phase amplitude: {bh_properties['phase_amplitude']:.2e}")
    print(f"Black hole - Total mass: {bh_observables['total_mass']:.2f} M☉")
    
    return {
        'star': (star_properties, star_observables),
        'galaxy': (galaxy_properties, galaxy_observables),
        'black_hole': (bh_properties, bh_observables)
    }


def example_gravitational_effects():
    """
    Example of gravitational effects.
    
    Physical Meaning:
        Demonstrates the connection between phase field and gravity,
        including spacetime curvature and gravitational waves.
    """
    print("\n=== Gravitational Effects Example ===")
    
    # Create mock system
    class MockSystem:
        def __init__(self):
            self.phase_field = np.random.normal(0, 0.1, (64, 64, 64))
    
    system = MockSystem()
    
    # Set up gravitational parameters
    gravity_params = {
        'G': 6.67430e-11,  # Gravitational constant
        'c': 299792458.0,  # Speed of light
        'phase_gravity_coupling': 1.0,
        'dimensions': 4,
        'resolution': 64,
        'domain_size': 100.0
    }
    
    # Create gravitational effects model
    gravity = GravitationalEffectsModel(system, gravity_params)
    
    # Compute spacetime metric
    print("Computing spacetime metric...")
    metric = gravity.compute_spacetime_metric()
    print(f"Metric tensor shape: {metric.shape}")
    
    # Analyze spacetime curvature
    print("Analyzing spacetime curvature...")
    curvature_analysis = gravity.analyze_spacetime_curvature()
    print(f"Curvature analysis keys: {list(curvature_analysis.keys())}")
    
    # Compute gravitational waves
    print("Computing gravitational waves...")
    gw_analysis = gravity.compute_gravitational_waves()
    print(f"Gravitational wave analysis keys: {list(gw_analysis.keys())}")
    
    return {
        'metric': metric,
        'curvature': curvature_analysis,
        'gravitational_waves': gw_analysis
    }


def example_large_scale_structure():
    """
    Example of large-scale structure formation.
    
    Physical Meaning:
        Demonstrates the formation of large-scale structure in the
        universe through phase field evolution and gravitational effects.
    """
    print("\n=== Large-Scale Structure Example ===")
    
    # Create initial fluctuations
    resolution = 64
    initial_fluctuations = np.random.normal(0, 0.1, (resolution, resolution, resolution))
    
    # Set up evolution parameters
    evolution_params = {
        'time_start': 0.0,
        'time_end': 1.0,  # Gyr
        'dt': 0.01,
        'domain_size': 1000.0,  # Mpc
        'resolution': resolution,
        'structure_analysis': True,
        'cosmology': {
            'G': 6.67430e-11,
            'rho_m': 2.7e-27,
            'H0': 70.0
        }
    }
    
    # Create large-scale structure model
    structure = LargeScaleStructureModel(initial_fluctuations, evolution_params)
    
    # Evolve structure
    print("Evolving large-scale structure...")
    evolution_results = structure.evolve_structure()
    
    print(f"Evolution time: {evolution_results['time'][-1]:.2f} Gyr")
    print(f"Number of time steps: {len(evolution_results['time'])}")
    
    # Analyze galaxy formation
    galaxy_analysis = structure.analyze_galaxy_formation()
    print(f"Galaxy formation analysis keys: {list(galaxy_analysis.keys())}")
    
    return evolution_results


def example_particle_inversion():
    """
    Example of particle parameter inversion.
    
    Physical Meaning:
        Demonstrates the inversion of model parameters from
        observable particle properties.
    """
    print("\n=== Particle Inversion Example ===")
    
    # Set up observables for electron
    electron_observables = {
        'tail': 1.0,
        'jr': 0.5,
        'Achi': 0.3,
        'peaks': 0,  # Electron should have no peaks
        'mobility': 0.8,
        'Meff': 1.0
    }
    
    # Set up priors
    electron_priors = {
        'beta': [0.6, 1.4],
        'layers_count': [0, 1],
        'eta': [0.0, 0.05],
        'gamma': [0.0, 0.2],
        'tau': [0.5, 1.5],
        'q': [1, -1]
    }
    
    # Set up loss weights
    loss_weights = {
        'tail': 1.0,
        'jr': 1.0,
        'Achi': 0.5,
        'peaks': 0.5,
        'mobility': 0.5,
        'Meff': 1.0
    }
    
    # Set up optimization parameters
    optimization_params = {
        'max_iterations': 100,  # Small for example
        'tolerance': 1e-6,
        'learning_rate': 0.01
    }
    
    # Create particle inversion
    inversion = ParticleInversion(
        electron_observables, 
        electron_priors, 
        loss_weights, 
        optimization_params
    )
    
    # Invert parameters
    print("Inverting electron parameters...")
    inversion_results = inversion.invert_parameters()
    
    print(f"Optimized parameters: {inversion_results['optimized_parameters']}")
    print(f"Final loss: {inversion_results['final_loss']:.2e}")
    
    # Validate parameters
    validation_criteria = {
        'chi_squared_threshold': 0.05,
        'confidence_level': 0.95,
        'parameter_tolerance': 0.01
    }
    
    experimental_data = {
        'mass': 9.10938356e-31,
        'charge': -1.602176634e-19,
        'magnetic_moment': -9.2847647043e-24
    }
    
    validation = ParticleValidation(
        inversion_results, 
        validation_criteria, 
        experimental_data
    )
    
    print("Validating electron parameters...")
    validation_results = validation.validate_parameters()
    
    print(f"Parameter validation: {validation_results['parameter_validation']}")
    print(f"Overall validation: {validation_results['overall_validation']}")
    
    return {
        'inversion': inversion_results,
        'validation': validation_results
    }


def example_visualization():
    """
    Example of visualization.
    
    Physical Meaning:
        Demonstrates visualization of cosmological and astrophysical
        results from the 7D phase field theory.
    """
    print("\n=== Visualization Example ===")
    
    # Create simple phase field for visualization
    x = np.linspace(-5, 5, 64)
    y = np.linspace(-5, 5, 64)
    X, Y = np.meshgrid(x, y)
    
    # Create phase field with spiral structure
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    phase_field = np.exp(-r**2/4) * np.cos(2*theta + r)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot phase field
    im1 = axes[0].imshow(phase_field, extent=[-5, 5, -5, 5], origin='lower')
    axes[0].set_title('Phase Field Configuration')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot phase field magnitude
    phase_magnitude = np.abs(phase_field)
    im2 = axes[1].imshow(phase_magnitude, extent=[-5, 5, -5, 5], origin='lower')
    axes[1].set_title('Phase Field Magnitude')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('level_g_phase_field_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'level_g_phase_field_example.png'")


def main():
    """
    Main example function.
    
    Physical Meaning:
        Demonstrates the complete workflow of level G models for
        cosmological and astrophysical applications.
    """
    print("Level G Models Example")
    print("=====================")
    print("Demonstrating cosmological and astrophysical applications")
    print("of the 7D phase field theory.")
    
    try:
        # Run examples
        cosmology_results = example_cosmological_evolution()
        astrophysics_results = example_astrophysical_objects()
        gravity_results = example_gravitational_effects()
        structure_results = example_large_scale_structure()
        particle_results = example_particle_inversion()
        
        # Create visualization
        example_visualization()
        
        print("\n=== Summary ===")
        print("All examples completed successfully!")
        print(f"Cosmological evolution: {len(cosmology_results['time'])} time steps")
        print(f"Astrophysical objects: {len(astrophysics_results)} object types")
        print(f"Gravitational effects: {len(gravity_results)} analysis types")
        print(f"Large-scale structure: {len(structure_results['time'])} time steps")
        print(f"Particle inversion: {len(particle_results)} result types")
        
    except Exception as e:
        print(f"Error in example: {e}")
        raise


if __name__ == "__main__":
    main()
