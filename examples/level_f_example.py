"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Example usage of Level F models for collective effects and multi-particle interactions.

This example demonstrates how to use the Level F models to study
collective effects in multi-particle systems, including collective
excitations, phase transitions, and nonlinear effects.

Physical Meaning:
    This example shows how to:
    1. Create multi-particle systems with topological defects
    2. Study collective excitations and their dispersion relations
    3. Analyze phase transitions and critical points
    4. Investigate nonlinear effects and soliton solutions

Example:
    >>> python level_f_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from bhlff.core.domain import Domain
from bhlff.models.level_f import (
    MultiParticleSystem,
    CollectiveExcitations,
    PhaseTransitions,
    NonlinearEffects,
)
from bhlff.models.level_f.multi_particle import Particle


def create_example_system():
    """
    Create example multi-particle system.

    Physical Meaning:
        Creates a system with two oppositely charged particles
        to study collective effects and interactions.
    """
    # Create domain
    domain = Domain(L=20.0, N=128, dimensions=3)

    # Create particles
    particles = [
        Particle(position=np.array([5.0, 10.0, 10.0]), charge=1, phase=0.0),
        Particle(position=np.array([15.0, 10.0, 10.0]), charge=-1, phase=np.pi),
    ]

    # Create system
    system = MultiParticleSystem(
        domain, particles, interaction_range=5.0, interaction_strength=1.0
    )

    return system


def study_collective_excitations(system):
    """
    Study collective excitations in the system.

    Physical Meaning:
        Analyzes collective excitations by applying external
        fields and studying the system response.
    """
    print("Studying collective excitations...")

    # Create excitation parameters
    excitation_params = {
        "frequency_range": [0.1, 10.0],
        "amplitude": 0.1,
        "type": "harmonic",
        "duration": 100.0,
    }

    # Create excitations model
    excitations = CollectiveExcitations(system, excitation_params)

    # Create external field
    external_field = np.random.randn(*system.domain.shape) * 0.1

    # Excite system
    response = excitations.excite_system(external_field)
    print(f"System response shape: {response.shape}")

    # Analyze response
    analysis = excitations.analyze_response(response)
    print(f"Found {len(analysis['peaks']['frequencies'])} spectral peaks")

    # Compute dispersion relations
    dispersion = excitations.compute_dispersion_relations()
    print(f"Dispersion relation computed for {len(dispersion['k_values'])} k values")

    return excitations, analysis, dispersion


def study_phase_transitions(system):
    """
    Study phase transitions in the system.

    Physical Meaning:
        Analyzes phase transitions by varying system parameters
        and monitoring order parameters.
    """
    print("Studying phase transitions...")

    # Create transitions model
    transitions = PhaseTransitions(system)

    # Perform parameter sweep
    parameter = "interaction_strength"
    values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

    result = transitions.parameter_sweep(parameter, values)
    print(f"Parameter sweep completed for {len(values)} values")

    # Compute order parameters
    order_params = transitions.compute_order_parameters()
    print(f"Order parameters: {order_params}")

    # Identify critical points
    critical_points = transitions.identify_critical_points(result["phase_diagram"])
    print(f"Found {len(critical_points)} critical points")

    return transitions, result, critical_points


def study_nonlinear_effects(system):
    """
    Study nonlinear effects in the system.

    Physical Meaning:
        Analyzes nonlinear effects by adding nonlinear
        interactions and studying soliton solutions.
    """
    print("Studying nonlinear effects...")

    # Create nonlinear parameters
    nonlinear_params = {
        "strength": 1.0,
        "order": 3,
        "type": "cubic",
        "coupling": "local",
    }

    # Create nonlinear effects model
    nonlinear = NonlinearEffects(system, nonlinear_params)

    # Add nonlinear interactions
    nonlinear.add_nonlinear_interactions(nonlinear_params)

    # Find nonlinear modes
    modes = nonlinear.find_nonlinear_modes()
    print(f"Found {len(modes['nonlinear_frequencies'])} nonlinear modes")

    # Find soliton solutions
    solitons = nonlinear.find_soliton_solutions()
    print(f"Found {len(solitons['solitons'])} soliton solutions")

    # Check stability
    stability = nonlinear.check_nonlinear_stability()
    print(f"System stability: {stability['linear_stability']['is_stable']}")

    return nonlinear, modes, solitons, stability


def visualize_results(system, excitations, transitions, nonlinear):
    """
    Visualize results from Level F analysis.

    Physical Meaning:
        Creates visualizations to show the results of
        collective effects analysis.
    """
    print("Creating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Level F: Collective Effects Analysis", fontsize=16)

    # Plot 1: Effective potential
    potential = system.compute_effective_potential()
    im1 = axes[0, 0].imshow(
        potential[:, :, potential.shape[2] // 2], cmap="viridis", origin="lower"
    )
    axes[0, 0].set_title("Effective Potential")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot 2: Collective modes
    modes = system.find_collective_modes()
    axes[0, 1].bar(range(len(modes["frequencies"])), modes["frequencies"])
    axes[0, 1].set_title("Collective Mode Frequencies")
    axes[0, 1].set_xlabel("Mode Index")
    axes[0, 1].set_ylabel("Frequency")

    # Plot 3: Order parameters
    order_params = transitions.compute_order_parameters()
    param_names = list(order_params.keys())
    param_values = list(order_params.values())
    axes[1, 0].bar(param_names, param_values)
    axes[1, 0].set_title("Order Parameters")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Plot 4: Soliton profiles
    solitons = nonlinear.find_soliton_solutions()
    if solitons["profiles"]:
        for i, profile in enumerate(solitons["profiles"][:3]):  # Show first 3 profiles
            axes[1, 1].plot(profile, label=f"Soliton {i+1}")
        axes[1, 1].set_title("Soliton Profiles")
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Amplitude")
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("level_f_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Visualization saved as 'level_f_analysis.png'")


def main():
    """
    Main function to demonstrate Level F models.

    Physical Meaning:
        Demonstrates the complete workflow for studying
        collective effects in multi-particle systems.
    """
    print("Level F: Collective Effects and Multi-Particle Interactions")
    print("=" * 60)

    # Create example system
    system = create_example_system()
    print(f"Created system with {len(system.particles)} particles")

    # Study collective excitations
    excitations, excitation_analysis, dispersion = study_collective_excitations(system)

    # Study phase transitions
    transitions, transition_result, critical_points = study_phase_transitions(system)

    # Study nonlinear effects
    nonlinear, nonlinear_modes, solitons, stability = study_nonlinear_effects(system)

    # Visualize results
    visualize_results(system, excitations, transitions, nonlinear)

    # Print summary
    print("\nSummary:")
    print(f"- System has {len(system.particles)} particles")
    print(
        f"- Found {len(excitation_analysis['peaks']['frequencies'])} collective modes"
    )
    print(f"- Identified {len(critical_points)} critical points")
    print(f"- Found {len(solitons['solitons'])} soliton solutions")
    print(f"- System stability: {stability['linear_stability']['is_stable']}")

    print("\nLevel F analysis completed successfully!")


if __name__ == "__main__":
    main()
