"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive unit tests for abstract solver module.

This module provides comprehensive unit tests for the abstract solver module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain, Parameters
from bhlff.solvers.base.abstract_solver import AbstractSolver


class ConcreteSolver(AbstractSolver):
    """Concrete implementation of AbstractSolver for testing."""

    def __init__(self, domain: Domain, parameters: Parameters):
        """Initialize concrete solver."""
        super().__init__(domain, parameters)
        self._initialized = True

    def solve(self, source: np.ndarray) -> np.ndarray:
        """Solve the phase field equation."""
        self.validate_input(source, "source")
        # Simple solution: return source scaled by domain volume
        volume = (
            np.prod(self.domain.shape)
            * self.domain.dx**3
            * self.domain.dphi**3
            * self.domain.dt
        )
        return source * volume

    def solve_time_evolution(
        self,
        initial_field: np.ndarray,
        source: np.ndarray,
        time_steps: int,
        dt: float,
    ) -> np.ndarray:
        """Solve time evolution of the phase field."""
        self.validate_input(initial_field, "initial_field")
        self.validate_input(source, "source")

        # Simple time evolution: linear growth
        result = np.zeros((time_steps,) + initial_field.shape)
        result[0] = initial_field

        for t in range(1, time_steps):
            result[t] = result[t - 1] + source * dt

        return result


class TestAbstractSolver:
    """Comprehensive tests for AbstractSolver class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(L=1.0, N=8, dimensions=7, N_phi=4, N_t=8, T=1.0)

    @pytest.fixture
    def parameters(self):
        """Create parameters for testing."""
        return Parameters(mu=1.0, beta=1.5, lambda_param=0.1, nu=1.0)

    @pytest.fixture
    def solver(self, domain, parameters):
        """Create concrete solver for testing."""
        return ConcreteSolver(domain, parameters)

    def test_solver_initialization(self, solver, domain, parameters):
        """Test solver initialization."""
        assert solver.domain == domain
        assert solver.parameters == parameters
        assert solver._initialized == True

    def test_solver_solve(self, solver):
        """Test solver solve method."""
        # Create test source
        source = np.random.random(solver.domain.shape)

        # Solve
        solution = solver.solve(source)

        assert solution.shape == source.shape
        assert isinstance(solution, np.ndarray)
        assert np.all(solution > 0)  # Should be positive due to scaling

    def test_solver_solve_time_evolution(self, solver):
        """Test solver time evolution method."""
        # Create test fields
        initial_field = np.random.random(solver.domain.shape)
        source = np.random.random(solver.domain.shape)
        time_steps = 10
        dt = 0.1

        # Solve time evolution
        evolution = solver.solve_time_evolution(initial_field, source, time_steps, dt)

        assert evolution.shape == (time_steps,) + initial_field.shape
        assert isinstance(evolution, np.ndarray)
        assert np.allclose(evolution[0], initial_field)

    def test_solver_validate_input(self, solver):
        """Test input validation."""
        # Valid input
        valid_field = np.random.random(solver.domain.shape)
        solver.validate_input(valid_field, "test_field")

        # Invalid input
        invalid_field = np.random.random((4, 4, 4))
        with pytest.raises(ValueError):
            solver.validate_input(invalid_field, "test_field")

    def test_solver_compute_residual(self, solver):
        """Test residual computation."""
        # Create test field and source
        field = np.random.random(solver.domain.shape)
        source = np.random.random(solver.domain.shape)

        # Compute residual
        residual = solver.compute_residual(field, source)

        assert residual.shape == field.shape
        assert isinstance(residual, np.ndarray)

    def test_solver_get_energy(self, solver):
        """Test energy computation."""
        # Create test field
        field = np.random.random(solver.domain.shape)

        # Compute energy
        energy = solver.get_energy(field)

        assert isinstance(energy, float)
        assert energy >= 0  # Energy should be non-negative

    def test_solver_is_initialized(self, solver):
        """Test initialization status."""
        assert solver.is_initialized() == True

    def test_solver_repr(self, solver):
        """Test solver string representation."""
        repr_str = repr(solver)
        assert "ConcreteSolver" in repr_str
        assert "domain=" in repr_str
        assert "parameters=" in repr_str

    def test_solver_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        domain = Domain(L=1.0, N=8, dimensions=7)
        parameters = Parameters(mu=1.0, beta=1.5, lambda_param=0.1, nu=1.0)

        # Create abstract solver (should not be possible, but test the methods)
        solver = AbstractSolver(domain, parameters)

        with pytest.raises(NotImplementedError):
            solver.solve(np.random.random(domain.shape))

        with pytest.raises(NotImplementedError):
            solver.solve_time_evolution(
                np.random.random(domain.shape), np.random.random(domain.shape), 10, 0.1
            )

    def test_solver_residual_physics(self, solver):
        """Test residual computation physics."""
        # Create field that should give zero residual for specific source
        field = np.ones(solver.domain.shape)
        source = np.zeros(solver.domain.shape)

        # Compute residual
        residual = solver.compute_residual(field, source)

        # Residual should be non-zero for non-zero field and zero source
        assert not np.allclose(residual, 0.0)

    def test_solver_energy_physics(self, solver):
        """Test energy computation physics."""
        # Create test field
        field = np.random.random(solver.domain.shape)

        # Compute energy
        energy = solver.get_energy(field)

        # Energy should be positive for non-zero field
        assert energy > 0

        # Energy should be zero for zero field
        zero_field = np.zeros(solver.domain.shape)
        zero_energy = solver.get_energy(zero_field)
        assert zero_energy == 0.0

    def test_solver_spectral_coefficients(self, solver):
        """Test spectral coefficients computation."""
        # Create test wave vector magnitude
        k_magnitude = np.array([0.0, 1.0, 2.0])

        # Get spectral coefficients
        coeffs = solver.parameters.get_spectral_coefficients(k_magnitude)

        assert isinstance(coeffs, np.ndarray)
        assert coeffs.shape == k_magnitude.shape
        assert np.all(coeffs > 0)  # Should be positive

    def test_solver_domain_properties(self, solver):
        """Test domain properties access."""
        assert solver.domain.L == 1.0
        assert solver.domain.N == 8
        assert solver.domain.dimensions == 7
        assert solver.domain.shape == (8, 8, 8, 4, 4, 4, 8)

    def test_solver_parameters_properties(self, solver):
        """Test parameters properties access."""
        assert solver.parameters.mu == 1.0
        assert solver.parameters.beta == 1.5
        assert solver.parameters.lambda_param == 0.1
        assert solver.parameters.nu == 1.0

    def test_solver_fft_operations(self, solver):
        """Test FFT operations in residual computation."""
        # Create test field
        field = np.random.random(solver.domain.shape)
        source = np.random.random(solver.domain.shape)

        # Compute residual (uses FFT internally)
        residual = solver.compute_residual(field, source)

        # Should not raise any errors
        assert isinstance(residual, np.ndarray)
        assert residual.shape == field.shape

    def test_solver_energy_conservation(self, solver):
        """Test energy conservation properties."""
        # Create test field
        field = np.random.random(solver.domain.shape)

        # Compute energy
        energy = solver.get_energy(field)

        # Energy should be conserved (same for same field)
        energy2 = solver.get_energy(field)
        assert energy == energy2

    def test_solver_time_evolution_properties(self, solver):
        """Test time evolution properties."""
        # Create test fields
        initial_field = np.random.random(solver.domain.shape)
        source = np.random.random(solver.domain.shape)
        time_steps = 5
        dt = 0.1

        # Solve time evolution
        evolution = solver.solve_time_evolution(initial_field, source, time_steps, dt)

        # Check properties
        assert evolution.shape[0] == time_steps
        assert np.allclose(evolution[0], initial_field)

        # Check that evolution is monotonic (for positive source)
        if np.all(source > 0):
            for t in range(1, time_steps):
                assert np.all(evolution[t] >= evolution[t - 1])

    def test_solver_error_handling(self, solver):
        """Test error handling."""
        # Test with wrong input shapes
        with pytest.raises(ValueError):
            solver.solve(np.random.random((4, 4, 4)))

        with pytest.raises(ValueError):
            solver.solve_time_evolution(
                np.random.random((4, 4, 4)),
                np.random.random(solver.domain.shape),
                10,
                0.1,
            )

        with pytest.raises(ValueError):
            solver.solve_time_evolution(
                np.random.random(solver.domain.shape),
                np.random.random((4, 4, 4)),
                10,
                0.1,
            )

    def test_solver_numerical_stability(self, solver):
        """Test numerical stability."""
        # Create test with very small values
        small_field = np.full(solver.domain.shape, 1e-10)
        small_source = np.full(solver.domain.shape, 1e-10)

        # Should not raise errors
        solution = solver.solve(small_source)
        residual = solver.compute_residual(small_field, small_source)
        energy = solver.get_energy(small_field)

        assert isinstance(solution, np.ndarray)
        assert isinstance(residual, np.ndarray)
        assert isinstance(energy, float)

    def test_solver_large_values(self, solver):
        """Test with large values."""
        # Create test with large values
        large_field = np.full(solver.domain.shape, 1e10)
        large_source = np.full(solver.domain.shape, 1e10)

        # Should not raise errors
        solution = solver.solve(large_source)
        residual = solver.compute_residual(large_field, large_source)
        energy = solver.get_energy(large_field)

        assert isinstance(solution, np.ndarray)
        assert isinstance(residual, np.ndarray)
        assert isinstance(energy, float)
        assert not np.isnan(energy)
        assert not np.isinf(energy)
