"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Example usage of Level C analysis modules.

This example demonstrates how to use the Level C analysis modules
for boundary analysis, resonator chains, quench memory, and mode beating.

Physical Meaning:
    Demonstrates the Level C analysis capabilities:
    - C1: Single wall boundary effects and resonance mode analysis
    - C2: Resonator chain analysis with ABCD model validation
    - C3: Quench memory and pinning effects analysis
    - C4: Mode beating and drift velocity analysis

Mathematical Foundation:
    Shows practical usage of:
    - Boundary analysis: Y(ω) = I(ω)/V(ω), A(r) = (1/4π) ∫_S(r) |a(x)|² dS
    - ABCD model: T_total = ∏ T_ℓ, det(T_total - I) = 0
    - Memory analysis: Γ_memory[a] = -γ ∫_0^t K(t-τ) a(τ) dτ
    - Beating analysis: v_cell^pred = Δω / |k₂ - k₁|

Example:
    >>> python examples/level_c_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from bhlff.models.level_c import (
    ABCDModel,
    ResonatorLayer,
    SystemMode,
    MemoryParameters,
    QuenchEvent,
    DualModeSource,
    BeatingPattern,
)


def example_abcd_model():
    """
    Example of ABCD model usage.

    Physical Meaning:
        Demonstrates how to use the ABCD model for analyzing
        resonator chains and finding system resonance modes.
    """
    print("=== ABCD Model Example ===")

    # Create resonator layers
    resonators = [
        ResonatorLayer(radius=1.0, thickness=0.1, contrast=0.3),
        ResonatorLayer(radius=2.0, thickness=0.1, contrast=0.5),
        ResonatorLayer(radius=3.0, thickness=0.1, contrast=0.7),
    ]

    # Create ABCD model
    model = ABCDModel(resonators)

    # Compute transmission matrix at specific frequency
    frequency = 1.0
    T = model.compute_transmission_matrix(frequency)
    print(f"Transmission matrix at ω = {frequency}:")
    print(T)

    # Compute system admittance
    admittance = model.compute_system_admittance(frequency)
    print(f"System admittance at ω = {frequency}: {admittance}")

    # Find resonance conditions
    frequency_range = (0.1, 3.0)
    resonances = model.find_resonance_conditions(frequency_range)
    print(f"Resonance frequencies in range {frequency_range}: {resonances}")

    # Find system modes
    modes = model.find_system_modes(frequency_range)
    print(f"Found {len(modes)} system modes:")
    for i, mode in enumerate(modes):
        print(f"  Mode {i}: ω = {mode.frequency:.3f}, Q = {mode.quality_factor:.3f}")

    return model, modes


def example_memory_parameters():
    """
    Example of memory parameters usage.

    Physical Meaning:
        Demonstrates how to create and use memory parameters
        for quench memory analysis.
    """
    print("\n=== Memory Parameters Example ===")

    # Create memory parameters
    spatial_distribution = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    memory = MemoryParameters(
        gamma=0.6, tau=2.0, spatial_distribution=spatial_distribution
    )

    print(f"Memory strength γ = {memory.gamma}")
    print(f"Relaxation time τ = {memory.tau}")
    print(f"Spatial distribution: {memory.spatial_distribution}")

    # Create quench event
    event = QuenchEvent(
        location=np.array([1.0, 2.0, 3.0]),
        time=5.0,
        intensity=0.8,
        threshold_type="amplitude",
    )

    print(f"Quench event at location {event.location}, time {event.time}")
    print(f"Intensity: {event.intensity}, threshold type: {event.threshold_type}")

    return memory, event


def example_dual_mode_source():
    """
    Example of dual-mode source usage.

    Physical Meaning:
        Demonstrates how to create dual-mode sources for
        mode beating analysis.
    """
    print("\n=== Dual-Mode Source Example ===")

    # Create dual-mode source
    profile1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    profile2 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    source = DualModeSource(
        frequency_1=0.9,
        frequency_2=1.1,
        amplitude_1=0.8,
        amplitude_2=1.2,
        profile_1=profile1,
        profile_2=profile2,
    )

    print(f"Frequency 1: {source.frequency_1}, Amplitude 1: {source.amplitude_1}")
    print(f"Frequency 2: {source.frequency_2}, Amplitude 2: {source.amplitude_2}")
    print(f"Beating frequency: {abs(source.frequency_2 - source.frequency_1)}")

    return source


def example_beating_pattern():
    """
    Example of beating pattern usage.

    Physical Meaning:
        Demonstrates how to create and analyze beating patterns
        for mode beating analysis.
    """
    print("\n=== Beating Pattern Example ===")

    # Create beating pattern
    amplitude_modulation = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1])
    phase_evolution = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.3, 0.4, 0.5]),
        np.array([0.5, 0.6, 0.7]),
    ]

    pattern = BeatingPattern(
        beating_frequency=0.1,
        amplitude_modulation=amplitude_modulation,
        phase_evolution=phase_evolution,
        temporal_coherence=0.9,
    )

    print(f"Beating frequency: {pattern.beating_frequency}")
    print(f"Amplitude modulation shape: {pattern.amplitude_modulation.shape}")
    print(f"Phase evolution steps: {len(pattern.phase_evolution)}")
    print(f"Temporal coherence: {pattern.temporal_coherence}")

    return pattern


def example_mathematical_operations():
    """
    Example of mathematical operations.

    Physical Meaning:
        Demonstrates basic mathematical operations used in
        Level C analysis.
    """
    print("\n=== Mathematical Operations Example ===")

    # Matrix operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = A @ B
    det_A = np.linalg.det(A)

    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Matrix C = A @ B:\n{C}")
    print(f"Determinant of A: {det_A}")

    # Complex operations
    z1 = complex(1, 2)
    z2 = complex(3, 4)
    z_sum = z1 + z2
    z_product = z1 * z2
    z_magnitude = abs(z1)

    print(f"Complex z1: {z1}")
    print(f"Complex z2: {z2}")
    print(f"Sum: {z_sum}")
    print(f"Product: {z_product}")
    print(f"Magnitude of z1: {z_magnitude}")

    # Numerical stability
    small_number = 1e-12
    result = 1.0 / small_number
    print(f"Division by small number: {result}")
    print(f"Is finite: {np.isfinite(result)}")


def main():
    """
    Main function to run all examples.

    Physical Meaning:
        Runs all Level C analysis examples to demonstrate
        the capabilities of the framework.
    """
    print("Level C Analysis Examples")
    print("=" * 50)

    try:
        # Run examples
        model, modes = example_abcd_model()
        memory, event = example_memory_parameters()
        source = example_dual_mode_source()
        pattern = example_beating_pattern()
        example_mathematical_operations()

        print("\n=== Summary ===")
        print(f"ABCD model created with {len(model.resonators)} resonators")
        print(f"Found {len(modes)} system modes")
        print(f"Memory parameters: γ = {memory.gamma}, τ = {memory.tau}")
        print(f"Dual-mode source: ω₁ = {source.frequency_1}, ω₂ = {source.frequency_2}")
        print(f"Beating pattern: frequency = {pattern.beating_frequency}")

        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
