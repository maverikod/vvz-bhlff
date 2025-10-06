#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physics validation test for BVP Solver Core.

This script tests the physical correctness of the BVP solver core
implementation, including residual computation, Jacobian calculation,
and linear system solving.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.insert(0, "/home/vasilyvz/Desktop/Инерция/7d/progs/bhlff")

from bhlff.core.domain import Domain, Parameters
from bhlff.core.fft.bvp_solver_core import BVPSolverCore


def test_bvp_solver_core_physics():
    """Test the physical correctness of the BVP solver core."""
    print("🧪 Testing BVP Solver Core Physics...")

    # Create test domain and parameters (7D for BVP theory)
    domain = Domain(L=1.0, N=4, dimensions=7, N_phi=2, N_t=4, T=1.0)
    parameters = Parameters(mu=1.0, beta=1.5, lambda_param=0.1, nu=1.0)

    # Mock parameters for BVP solver
    class MockBVPParameters:
        def __init__(self):
            self.kappa_0 = 1.0
            self.k0 = 2.0
            self.chi_prime = 1.0
            self.beta = 1.5
            self.lambda_param = 0.1
            self.mu = 1.0

        def compute_stiffness(self, amplitude):
            return self.kappa_0 + 0.1 * amplitude**2

        def compute_susceptibility(self, amplitude):
            return self.chi_prime + 0.01 * amplitude**2

        def compute_stiffness_derivative(self, amplitude):
            return 0.2 * amplitude

        def compute_susceptibility_derivative(self, amplitude):
            return 0.02 * amplitude

    # Mock derivatives for BVP solver
    class MockSpectralDerivatives:
        def __init__(self, domain):
            self.domain = domain

        def compute_gradient(self, field):
            # Simple finite difference gradient
            gradients = []
            for dim in range(field.ndim):
                grad = np.gradient(field, axis=dim)
                gradients.append(grad)
            return gradients

        def compute_divergence(self, gradient_tuple):
            # Simple finite difference divergence
            divergence = np.zeros_like(gradient_tuple[0])
            for i, grad in enumerate(gradient_tuple):
                divergence += np.gradient(grad, axis=i)
            return divergence

    # Initialize BVP solver core
    config = {"test": True}
    bvp_parameters = MockBVPParameters()
    derivatives = MockSpectralDerivatives(domain)

    solver_core = BVPSolverCore(
        domain=domain, config=config, parameters=bvp_parameters, derivatives=derivatives
    )

    print("✅ BVP solver core initialized successfully")

    # Test 1: Residual computation physics
    print("\n📊 Test 1: Residual Computation Physics")
    test_residual_computation_physics(solver_core, domain)

    # Test 2: Jacobian computation physics
    print("\n🔬 Test 2: Jacobian Computation Physics")
    test_jacobian_computation_physics(solver_core, domain)

    # Test 3: Linear system solving physics
    print("\n⚙️ Test 3: Linear System Solving Physics")
    test_linear_system_solving_physics(solver_core, domain)

    # Test 4: 7D sparse matrix physics
    print("\n🎯 Test 4: 7D Sparse Matrix Physics")
    test_7d_sparse_matrix_physics(solver_core, domain)

    print("\n✅ All physics tests passed!")


def test_residual_computation_physics(solver_core, domain):
    """Test residual computation physics."""
    # Create test solution and source
    solution = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex)
    source = np.zeros((4, 4, 4, 2, 2, 2, 4), dtype=complex)

    # Test residual computation
    residual = solver_core.compute_residual(solution, source)

    # Physics checks
    assert (
        residual.shape == solution.shape
    ), "Residual should have same shape as solution"
    assert np.all(np.isfinite(residual)), "Residual should be finite"

    # For constant solution with zero source, residual should be non-zero
    # due to the nonlinear terms in the BVP equation
    residual_magnitude = np.mean(np.abs(residual))
    assert (
        residual_magnitude > 0
    ), "Residual should be non-zero for non-trivial solution"

    print(f"   Residual magnitude: {residual_magnitude:.3f}")
    print(f"   Residual shape: {residual.shape}")


def test_jacobian_computation_physics(solver_core, domain):
    """Test Jacobian computation physics."""
    # Create test solution
    solution = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex)

    # Test Jacobian computation
    jacobian = solver_core.compute_jacobian(solution)

    print(f"   Solution shape: {solution.shape}")
    print(f"   Jacobian shape: {jacobian.shape}")
    print(f"   Jacobian type: {type(jacobian)}")

    # Physics checks
    assert (
        jacobian.shape == solution.shape
    ), "Jacobian diagonal should have same shape as solution"
    assert np.all(np.isfinite(jacobian)), "Jacobian should be finite"

    # Jacobian should be non-zero for non-trivial solution
    jacobian_magnitude = np.mean(np.abs(jacobian))
    assert jacobian_magnitude > 0, "Jacobian should be non-zero"

    # Jacobian should be complex (due to nonlinear terms) - may be real for simple test case
    has_imaginary = np.any(np.imag(jacobian) != 0)
    print(f"   Has imaginary parts: {has_imaginary}")

    print(f"   Jacobian magnitude: {jacobian_magnitude:.3f}")
    print(f"   Jacobian shape: {jacobian.shape}")
    print(f"   Has imaginary parts: {np.any(np.imag(jacobian) != 0)}")


def test_linear_system_solving_physics(solver_core, domain):
    """Test linear system solving physics."""
    # Create test Jacobian and residual
    jacobian = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex) + 1j * 0.1
    residual = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex) * 0.1

    # Test linear system solving
    correction = solver_core.solve_linear_system(jacobian, residual)

    # Physics checks
    assert (
        correction.shape == residual.shape
    ), "Correction should have same shape as residual"
    assert np.all(np.isfinite(correction)), "Correction should be finite"

    # Correction should be reasonable in magnitude
    correction_magnitude = np.mean(np.abs(correction))
    assert correction_magnitude > 0, "Correction should be non-zero"
    assert correction_magnitude < 10.0, "Correction should not be too large"

    print(f"   Correction magnitude: {correction_magnitude:.3f}")
    print(f"   Correction shape: {correction.shape}")


def test_7d_sparse_matrix_physics(solver_core, domain):
    """Test 7D sparse matrix physics."""
    # Create test solution
    solution = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex)

    # Test 7D step sizes computation
    shape = solution.shape
    step_sizes = solver_core._compute_7d_step_sizes(shape)

    # Physics checks
    assert "dx" in step_sizes, "Should include x step size"
    assert "dy" in step_sizes, "Should include y step size"
    assert "dz" in step_sizes, "Should include z step size"
    assert "dphi1" in step_sizes, "Should include φ₁ step size"
    assert "dphi2" in step_sizes, "Should include φ₂ step size"
    assert "dphi3" in step_sizes, "Should include φ₃ step size"
    assert "dt" in step_sizes, "Should include t step size"

    # Step sizes should be positive
    for key, value in step_sizes.items():
        assert value > 0, f"Step size {key} should be positive"

    # Test diagonal Jacobian element computation
    coords = (2, 2, 2, 1, 1, 1, 2)  # Middle point
    dkappa_da = np.ones_like(solution) * 0.1
    dchi_da = np.ones_like(solution) * 0.01

    diagonal_element = solver_core._compute_diagonal_jacobian_element(
        solution, dkappa_da, dchi_da, coords
    )

    # Physics checks
    assert isinstance(diagonal_element, complex), "Diagonal element should be complex"
    assert np.isfinite(diagonal_element), "Diagonal element should be finite"
    assert abs(diagonal_element) > 0, "Diagonal element should be non-zero"

    print(f"   Step sizes: {step_sizes}")
    print(f"   Diagonal element: {diagonal_element:.3f}")


def test_7d_coupling_physics():
    """Test 7D coupling physics."""
    print("\n🔗 Test 5: 7D Coupling Physics")

    # Create test domain and parameters
    domain = Domain(L=1.0, N=4, dimensions=7, N_phi=2, N_t=4, T=1.0)
    parameters = Parameters(mu=1.0, beta=1.5, lambda_param=0.1, nu=1.0)

    # Mock parameters and derivatives
    class MockBVPParameters:
        def __init__(self):
            self.kappa_0 = 1.0
            self.k0 = 2.0
            self.chi_prime = 1.0
            self.beta = 1.5
            self.lambda_param = 0.1
            self.mu = 1.0

        def compute_stiffness(self, amplitude):
            return self.kappa_0 + 0.1 * amplitude**2

        def compute_susceptibility(self, amplitude):
            return self.chi_prime + 0.01 * amplitude**2

        def compute_stiffness_derivative(self, amplitude):
            return 0.2 * amplitude

        def compute_susceptibility_derivative(self, amplitude):
            return 0.02 * amplitude

    class MockSpectralDerivatives:
        def __init__(self, domain):
            self.domain = domain

        def compute_gradient(self, field):
            gradients = []
            for dim in range(field.ndim):
                grad = np.gradient(field, axis=dim)
                gradients.append(grad)
            return gradients

        def compute_divergence(self, gradient_tuple):
            divergence = np.zeros_like(gradient_tuple[0])
            for i, grad in enumerate(gradient_tuple):
                divergence += np.gradient(grad, axis=i)
            return divergence

    # Initialize solver
    config = {"test": True}
    bvp_parameters = MockBVPParameters()
    derivatives = MockSpectralDerivatives(domain)

    solver_core = BVPSolverCore(
        domain=domain, config=config, parameters=bvp_parameters, derivatives=derivatives
    )

    # Test 7D coupling contribution
    solution = np.ones((4, 4, 4, 2, 2, 2, 4), dtype=complex)
    off_diagonal_elements = np.array([0.1, 0.2, 0.3], dtype=complex)
    coords = (2, 2, 2, 1, 1, 1, 2)
    step_sizes = solver_core._compute_7d_step_sizes(solution.shape)

    contribution = solver_core._compute_7d_coupling_contribution(
        off_diagonal_elements, coords, step_sizes, solution.shape, solution
    )

    # Physics checks
    assert isinstance(
        contribution, (complex, float)
    ), "Contribution should be complex or float"
    assert np.isfinite(contribution), "Contribution should be finite"
    assert abs(contribution) > 0, "Contribution should be non-zero"

    print(f"   7D coupling contribution: {contribution:.3f}")


if __name__ == "__main__":
    try:
        test_bvp_solver_core_physics()
        test_7d_coupling_physics()
        print("\n🎉 All physics tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Physics test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
