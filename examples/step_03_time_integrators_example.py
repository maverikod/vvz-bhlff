"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Step 03 Example: Time integrators for dynamic 7D BVP.

This example demonstrates the use of time integrators for solving
dynamic phase field equations in 7D space-time with CUDA acceleration,
vectorization, and block processing.
"""

import numpy as np
from pathlib import Path

from bhlff.core.domain import Domain, Parameters
from bhlff.core.time import (
    CrankNicolsonIntegrator,
    BVPEnvelopeIntegrator,
    AdaptiveIntegrator,
)


def example_crank_nicolson_integrator():
    """
    Example: Crank-Nicolson integrator for 7D phase field dynamics.

    Physical Meaning:
        Demonstrates second-order accurate temporal integration with
        unconditional stability for dynamic phase field equations.
    """
    print("=" * 60)
    print("Example: Crank-Nicolson Integrator")
    print("=" * 60)

    # Create domain
    domain = Domain(L=1.0, N=8, N_phi=4, N_t=16, T=1.0, dimensions=7)
    print(f"Domain shape: {domain.shape}")

    # Create parameters
    parameters = Parameters(
        mu=1.0,
        beta=1.0,
        lambda_param=0.1,
        nu=1.0,
        precision="float64",
    )

    # Create integrator
    integrator = CrankNicolsonIntegrator(domain, parameters)
    print(f"Integrator: {integrator}")

    # Create initial field
    initial_field = np.random.random(domain.shape).astype(np.complex128)
    print(f"Initial field shape: {initial_field.shape}")

    # Create time steps
    time_steps = np.linspace(0.0, domain.T, domain.N_t)
    print(f"Time steps: {len(time_steps)} points from {time_steps[0]:.3f} to {time_steps[-1]:.3f}")

    # Create source field (zero for this example)
    source_field = np.zeros(
        (len(time_steps),) + domain.shape, dtype=np.complex128
    )

    # Perform integration
    print("\nPerforming integration...")
    result = integrator.integrate(initial_field, source_field, time_steps)
    print(f"Integration completed. Result shape: {result.shape}")

    # Analyze results
    initial_magnitude = np.abs(result[0]).mean()
    final_magnitude = np.abs(result[-1]).mean()
    print(f"\nResults:")
    print(f"  Initial field magnitude: {initial_magnitude:.6f}")
    print(f"  Final field magnitude: {final_magnitude:.6f}")
    print(f"  Decay factor: {final_magnitude / initial_magnitude:.6f}")

    return result


def example_bvp_envelope_integrator():
    """
    Example: BVP envelope integrator for envelope modulation.

    Physical Meaning:
        Demonstrates BVP envelope integration where all observed "modes"
        are envelope modulations and beatings of the Base High-Frequency Field.
    """
    print("\n" + "=" * 60)
    print("Example: BVP Envelope Integrator")
    print("=" * 60)

    # Create domain
    domain = Domain(L=1.0, N=8, N_phi=4, N_t=16, T=1.0, dimensions=7)
    print(f"Domain shape: {domain.shape}")

    # Create parameters with BVP-specific parameters
    parameters = Parameters(
        mu=1.0,
        beta=1.0,
        lambda_param=0.1,
        nu=1.0,
        precision="float64",
    )
    # Add BVP-specific parameters
    parameters.kappa_0 = 1.0
    parameters.kappa_2 = 0.1
    parameters.chi_prime = 1.0
    parameters.chi_double_prime = 0.1

    # Create integrator
    integrator = BVPEnvelopeIntegrator(domain, parameters)
    print(f"Integrator: {integrator}")

    # Create initial field
    initial_field = np.random.random(domain.shape).astype(np.complex128)

    # Create time steps
    time_steps = np.linspace(0.0, domain.T, domain.N_t)

    # Create source field
    source_field = np.zeros(
        (len(time_steps),) + domain.shape, dtype=np.complex128
    )

    # Perform integration
    print("\nPerforming BVP envelope integration...")
    result = integrator.integrate(initial_field, source_field, time_steps)
    print(f"Integration completed. Result shape: {result.shape}")

    # Analyze results
    initial_magnitude = np.abs(result[0]).mean()
    final_magnitude = np.abs(result[-1]).mean()
    print(f"\nResults:")
    print(f"  Initial envelope magnitude: {initial_magnitude:.6f}")
    print(f"  Final envelope magnitude: {final_magnitude:.6f}")

    return result


def example_adaptive_integrator():
    """
    Example: Adaptive integrator with error control.

    Physical Meaning:
        Demonstrates adaptive time stepping with automatic error control
        for optimal accuracy and performance.
    """
    print("\n" + "=" * 60)
    print("Example: Adaptive Integrator")
    print("=" * 60)

    # Create domain
    domain = Domain(L=1.0, N=8, N_phi=4, N_t=16, T=1.0, dimensions=7)
    print(f"Domain shape: {domain.shape}")

    # Create parameters
    parameters = Parameters(
        mu=1.0,
        beta=1.0,
        lambda_param=0.1,
        nu=1.0,
        precision="float64",
    )

    # Create adaptive integrator with error control
    integrator = AdaptiveIntegrator(
        domain,
        parameters,
        tolerance=1e-8,
        safety_factor=0.9,
        min_dt=1e-6,
        max_dt=1e-2,
    )
    print(f"Integrator: {integrator}")
    print(f"  Tolerance: {integrator._tolerance:.2e}")
    print(f"  Time step range: [{integrator._min_dt:.2e}, {integrator._max_dt:.2e}]")

    # Create initial field
    initial_field = np.random.random(domain.shape).astype(np.complex128)

    # Create time steps (adaptive integrator will adjust internally)
    time_steps = np.linspace(0.0, domain.T, domain.N_t)

    # Create source field
    source_field = np.zeros(
        (len(time_steps),) + domain.shape, dtype=np.complex128
    )

    # Perform integration
    print("\nPerforming adaptive integration...")
    result = integrator.integrate(initial_field, source_field, time_steps)
    print(f"Integration completed. Result shape: {result.shape}")

    # Get current time step (may have been adjusted)
    current_dt = integrator.get_current_time_step()
    print(f"\nResults:")
    print(f"  Current time step: {current_dt:.6f}")

    return result


def example_memory_efficient_integration():
    """
    Example: Memory-efficient integration for large fields.

    Physical Meaning:
        Demonstrates memory-efficient integration with automatic
        memory management and periodic cleanup for large 7D fields.
    """
    print("\n" + "=" * 60)
    print("Example: Memory-Efficient Integration")
    print("=" * 60)

    # Create larger domain
    domain = Domain(L=1.0, N=12, N_phi=6, N_t=20, T=1.0, dimensions=7)
    field_size_mb = np.prod(domain.shape) * np.dtype(np.complex128).itemsize / (1024**2)
    print(f"Domain shape: {domain.shape}")
    print(f"Field size: {field_size_mb:.2f} MB")

    # Create parameters
    parameters = Parameters(
        mu=1.0,
        beta=1.0,
        lambda_param=0.1,
        nu=1.0,
        precision="float64",
    )

    # Create integrator
    integrator = CrankNicolsonIntegrator(domain, parameters)

    # Create initial field
    initial_field = np.random.random(domain.shape).astype(np.complex128)

    # Create time steps
    time_steps = np.linspace(0.0, domain.T, domain.N_t)

    # Create source field
    source_field = np.zeros(
        (len(time_steps),) + domain.shape, dtype=np.complex128
    )

    # Perform integration (memory-efficient processing will be used automatically)
    print("\nPerforming memory-efficient integration...")
    result = integrator.integrate(initial_field, source_field, time_steps)
    print(f"Integration completed. Result shape: {result.shape}")

    return result


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Step 03: Time Integrators Examples")
    print("=" * 60)
    print("\nThis example demonstrates time integrators for 7D BVP framework")
    print("with CUDA acceleration, vectorization, and block processing.\n")

    try:
        # Example 1: Crank-Nicolson integrator
        example_crank_nicolson_integrator()

        # Example 2: BVP envelope integrator
        example_bvp_envelope_integrator()

        # Example 3: Adaptive integrator
        example_adaptive_integrator()

        # Example 4: Memory-efficient integration
        example_memory_efficient_integration()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
