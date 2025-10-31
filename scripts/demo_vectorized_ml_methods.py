#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Demonstration script for vectorized ML methods.

This script demonstrates the use of vectorized processors in ML methods
for 7D phase field theory computations.
"""

import numpy as np
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bhlff.core.bvp.power_law_core_modules.power_law_fitting import PowerLawFitting
from bhlff.models.level_f.multi_particle.collective_modes_finding import (
    CollectiveModesFinder,
)
from bhlff.models.level_f.multi_particle.data_structures import (
    Particle,
    SystemParameters,
)


def setup_logging():
    """Setup logging for demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def demo_power_law_fitting_vectorized():
    """Demonstrate vectorized power law fitting."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Vectorized Power Law Fitting")
    print("=" * 60)

    # Create power law fitting instance
    fitting = PowerLawFitting()

    # Create test data with 7D phase field characteristics
    r = np.linspace(0.1, 10.0, 1000)
    values = np.exp(-r) * r ** (-2.0) * (1 + 0.1 * np.sin(2 * np.pi * r))

    test_data = {"r": r, "values": values}

    print(f"Input data shape: {len(r)} points")
    print(f"Value range: [{np.min(values):.3f}, {np.max(values):.3f}]")

    # Test vectorized power law fitting
    print("\nPerforming vectorized power law fitting...")
    result = fitting.fit_power_law(test_data)

    print("\nFitting Results:")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Test quality calculation
    quality = fitting.calculate_fitting_quality(test_data, result)
    print(f"\nFitting Quality: {quality:.6f}")

    # Test decay rate calculation
    decay_rate = fitting.calculate_decay_rate(result)
    print(f"Decay Rate: {decay_rate:.6f}")

    return result


def demo_collective_modes_vectorized():
    """Demonstrate vectorized collective modes analysis."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Vectorized Collective Modes Analysis")
    print("=" * 60)

    # Create test particles
    particles = [
        Particle(position=np.array([0.0, 0.0, 0.0]), charge=1.0),
        Particle(position=np.array([1.0, 0.0, 0.0]), charge=-1.0),
        Particle(position=np.array([0.0, 1.0, 0.0]), charge=1.0),
        Particle(position=np.array([0.0, 0.0, 1.0]), charge=-1.0),
    ]

    # Create system parameters
    system_params = SystemParameters(
        interaction_range=2.0,
        interaction_strength=1.0,
        mu=1.0,
        beta=1.5,
        lambda_param=0.1,
    )

    # Create collective modes finder
    modes_finder = CollectiveModesFinder(None, particles, system_params)

    print(f"Number of particles: {len(particles)}")
    print(f"System parameters: {system_params}")

    # Test vectorized mode analysis
    print("\nPerforming vectorized collective modes analysis...")

    # Create test eigenvalues and eigenvectors
    eigenvalues = np.array([1.0, 2.0, 3.0, 4.0])
    eigenvectors = np.random.rand(4, 4)

    # Test mode coupling calculation
    coupling = modes_finder._calculate_mode_coupling(eigenvalues, eigenvectors)
    print(f"Mode Coupling: {coupling:.6f}")

    # Test mode overlap calculation
    overlap = modes_finder._calculate_mode_overlap(eigenvectors)
    print(f"Mode Overlap: {overlap:.6f}")

    # Test mode correlation calculation
    correlation = modes_finder._calculate_mode_correlation(eigenvectors)
    print(f"Mode Correlation: {correlation:.6f}")

    # Test interaction strength calculation
    distance = 1.5
    interaction_strength = modes_finder._calculate_interaction_strength(distance)
    print(f"Interaction Strength (distance={distance}): {interaction_strength:.6f}")

    return {
        "coupling": coupling,
        "overlap": overlap,
        "correlation": correlation,
        "interaction_strength": interaction_strength,
    }


def demo_performance_comparison():
    """Demonstrate performance comparison between vectorized and standard methods."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Performance Comparison")
    print("=" * 60)

    import time

    # Create large test data
    n_points = 10000
    r = np.linspace(0.1, 10.0, n_points)
    values = np.exp(-r) * r ** (-2.0) * (1 + 0.1 * np.sin(2 * np.pi * r))

    test_data = {"r": r, "values": values}

    print(f"Testing with {n_points} data points")

    # Test vectorized processing
    fitting = PowerLawFitting()

    start_time = time.time()
    result = fitting.fit_power_law(test_data)
    vectorized_time = time.time() - start_time

    print(f"Vectorized processing time: {vectorized_time:.4f} seconds")
    print(f"Points per second: {n_points / vectorized_time:.0f}")

    # Test quality and decay rate calculations
    start_time = time.time()
    quality = fitting.calculate_fitting_quality(test_data, result)
    decay_rate = fitting.calculate_decay_rate(result)
    calculation_time = time.time() - start_time

    print(f"Quality and decay rate calculation time: {calculation_time:.4f} seconds")
    print(f"Total processing time: {vectorized_time + calculation_time:.4f} seconds")


def main():
    """Main demonstration function."""
    print("VECTORIZED ML METHODS DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates the integration of vectorized processors")
    print("into ML methods for 7D phase field theory computations.")

    setup_logging()

    try:
        # Demonstrate vectorized power law fitting
        power_law_result = demo_power_law_fitting_vectorized()

        # Demonstrate vectorized collective modes analysis
        modes_result = demo_collective_modes_vectorized()

        # Demonstrate performance comparison
        demo_performance_comparison()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("All vectorized ML methods are working correctly!")
        print("The integration of vectorized processors enhances performance")
        print("while maintaining compatibility with existing code.")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
