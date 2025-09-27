"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

3D FFT solver implementation.

This module implements the 3D FFT solver for the 7D phase field theory,
providing efficient spectral methods for 3D problems.

Physical Meaning:
    3D FFT solver implements spectral methods for solving phase field
    equations in 3D space, providing efficient computation of fractional
    operators and related equations.

Mathematical Foundation:
    Implements 3D spectral methods including FFT-based solvers for the
    fractional Riesz operator and related equations in 3D frequency space.

Example:
    >>> solver = FFTSolver3D(domain, config)
    >>> solution = solver.solve(source)
"""

import numpy as np
from typing import Dict, Any

from ...core.domain import Domain
from ...core.fft import FFTBackend, SpectralOperations


class FFTSolver3D:
    """
    3D FFT solver for phase field equations.

    Physical Meaning:
        Implements spectral methods for solving phase field equations
        in 3D space, providing efficient computation of fractional
        operators and related equations.

    Mathematical Foundation:
        3D FFT solver implements spectral methods for solving:
        L_β a = s(x) in 3D frequency space using FFT operations.

    Attributes:
        domain (Domain): Computational domain (must be 3D).
        config (Dict[str, Any]): Solver configuration.
        fft_backend (FFTBackend): FFT backend for operations.
        spectral_ops (SpectralOperations): Spectral operations.
        _spectral_coeffs (np.ndarray): Pre-computed spectral coefficients.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize 3D FFT solver.

        Physical Meaning:
            Sets up the 3D FFT solver with computational domain and
            configuration parameters for spectral solution methods.

        Args:
            domain (Domain): Computational domain (must be 3D).
            config (Dict[str, Any]): Solver configuration parameters.

        Raises:
            ValueError: If domain is not 3D.
        """
        if domain.dimensions != 3:
            raise ValueError("FFTSolver3D requires 3D domain")

        self.domain = domain
        self.config = config
        self.fft_backend = FFTBackend(domain, config.get("fft_config", {}))
        self.spectral_ops = SpectralOperations(domain, self.fft_backend)
        self._spectral_coeffs: np.ndarray
        self._setup_spectral_coefficients()

    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for 3D solver.

        Physical Meaning:
            Pre-computes the spectral representation of the operator
            for efficient 3D spectral solution methods.

        Mathematical Foundation:
            Computes spectral coefficients for 3D fractional operator
            in frequency space.
        """
        # Get 3D frequency arrays
        kx, ky, kz = self.fft_backend.get_frequency_arrays()
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

        # Get operator parameters
        mu = self.config.get("mu", 1.0)
        beta = self.config.get("beta", 1.0)
        lambda_param = self.config.get("lambda", 0.0)

        # Compute spectral coefficients for fractional operator
        self._spectral_coeffs = mu * (k_magnitude ** (2 * beta)) + lambda_param

        # Handle k=0 mode
        if lambda_param == 0:
            self._spectral_coeffs[0, 0, 0] = 1.0  # Avoid division by zero

    def solve(self, source: np.ndarray) -> np.ndarray:
        """
        Solve 3D phase field equation.

        Physical Meaning:
            Solves the 3D phase field equation using spectral methods,
            computing the field configuration that satisfies the equation.

        Mathematical Foundation:
            Solves L_β a = s in 3D spectral space:
            â(k) = ŝ(k) / (μ|k|^(2β) + λ)
            where k is the 3D wave vector and |k| is its magnitude.

        Args:
            source (np.ndarray): Source term s(x) in 3D real space.

        Returns:
            np.ndarray: Solution field a(x) in 3D real space.

        Raises:
            ValueError: If source shape is incompatible with 3D domain.
        """
        if source.shape != self.domain.shape:
            raise ValueError(
                f"Source shape {source.shape} incompatible with "
                f"3D domain shape {self.domain.shape}"
            )

        # Transform source to spectral space
        source_spectral = self.fft_backend.fft(source)

        # Apply spectral operator
        solution_spectral = source_spectral / self._spectral_coeffs

        # Transform back to real space
        solution = self.fft_backend.ifft(solution_spectral)

        return solution.real

    def solve_with_boundary_conditions(
        self, source: np.ndarray, boundary_conditions: str = "periodic"
    ) -> np.ndarray:
        """
        Solve with specified boundary conditions.

        Physical Meaning:
            Solves the 3D phase field equation with specified boundary
            conditions, handling different types of boundary constraints.

        Mathematical Foundation:
            Applies boundary conditions in spectral space by modifying
            the spectral coefficients appropriately.

        Args:
            source (np.ndarray): Source term s(x) in 3D real space.
            boundary_conditions (str): Type of boundary conditions.

        Returns:
            np.ndarray: Solution field a(x) with boundary conditions.

        Raises:
            ValueError: If boundary_conditions is not supported.
        """
        if boundary_conditions not in ["periodic", "dirichlet", "neumann"]:
            raise ValueError(f"Unsupported boundary conditions: {boundary_conditions}")

        # Get base solution
        solution = self.solve(source)

        # Apply boundary conditions
        if boundary_conditions == "dirichlet":
            solution = self._apply_dirichlet_boundary(solution)
        elif boundary_conditions == "neumann":
            solution = self._apply_neumann_boundary(solution)
        # Periodic boundary conditions are handled naturally by FFT

        return solution

    def _apply_dirichlet_boundary(self, field: np.ndarray) -> np.ndarray:
        """
        Apply Dirichlet boundary conditions.

        Physical Meaning:
            Applies Dirichlet boundary conditions (fixed values) to the
            field at the domain boundaries.

        Mathematical Foundation:
            Sets field values to zero at domain boundaries:
            a(x_boundary) = 0

        Args:
            field (np.ndarray): Field to apply boundary conditions to.

        Returns:
            np.ndarray: Field with Dirichlet boundary conditions.
        """
        field_with_bc = field.copy()

        # Set boundaries to zero
        field_with_bc[0, :, :] = 0.0
        field_with_bc[-1, :, :] = 0.0
        field_with_bc[:, 0, :] = 0.0
        field_with_bc[:, -1, :] = 0.0
        field_with_bc[:, :, 0] = 0.0
        field_with_bc[:, :, -1] = 0.0

        return field_with_bc

    def _apply_neumann_boundary(self, field: np.ndarray) -> np.ndarray:
        """
        Apply Neumann boundary conditions.

        Physical Meaning:
            Applies Neumann boundary conditions (fixed gradients) to the
            field at the domain boundaries.

        Mathematical Foundation:
            Sets field gradients to zero at domain boundaries:
            ∂a/∂n(x_boundary) = 0

        Args:
            field (np.ndarray): Field to apply boundary conditions to.

        Returns:
            np.ndarray: Field with Neumann boundary conditions.
        """
        field_with_bc = field.copy()

        # Apply Neumann boundary conditions by copying adjacent values
        field_with_bc[0, :, :] = field_with_bc[1, :, :]
        field_with_bc[-1, :, :] = field_with_bc[-2, :, :]
        field_with_bc[:, 0, :] = field_with_bc[:, 1, :]
        field_with_bc[:, -1, :] = field_with_bc[:, -2, :]
        field_with_bc[:, :, 0] = field_with_bc[:, :, 1]
        field_with_bc[:, :, -1] = field_with_bc[:, :, -2]

        return field_with_bc

    def compute_spectral_derivative(
        self, field: np.ndarray, order: int = 1, axis: int = 0
    ) -> np.ndarray:
        """
        Compute spectral derivative in 3D.

        Physical Meaning:
            Computes the spectral derivative of the field in 3D using
            high-accuracy spectral methods.

        Mathematical Foundation:
            Spectral derivative: ∂^n/∂x^n a(x) = IFFT((ik)^n * FFT(a(x)))
            where k is the frequency and n is the derivative order.

        Args:
            field (np.ndarray): Input field a(x).
            order (int): Derivative order (default: 1).
            axis (int): Axis along which to compute derivative (default: 0).

        Returns:
            np.ndarray: Spectral derivative of the field.
        """
        return self.spectral_ops.spectral_derivative(field, order, axis)

    def compute_spectral_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute spectral Laplacian in 3D.

        Physical Meaning:
            Computes the spectral Laplacian of the field in 3D using
            high-accuracy spectral methods.

        Mathematical Foundation:
            Spectral Laplacian: Δa(x) = IFFT(-|k|² * FFT(a(x)))
            where |k|² is the squared magnitude of the 3D wave vector.

        Args:
            field (np.ndarray): Input field a(x).

        Returns:
            np.ndarray: Spectral Laplacian of the field.
        """
        return self.spectral_ops.spectral_laplacian(field)

    def get_spectral_coefficients(self) -> np.ndarray:
        """
        Get spectral coefficients of the operator.

        Physical Meaning:
            Returns the pre-computed spectral coefficients for the
            3D fractional operator.

        Returns:
            np.ndarray: Spectral coefficients.
        """
        return self._spectral_coeffs.copy()

    def get_fft_backend(self) -> FFTBackend:
        """
        Get the FFT backend.

        Physical Meaning:
            Returns the FFT backend used for spectral operations.

        Returns:
            FFTBackend: FFT backend.
        """
        return self.fft_backend

    def get_spectral_operations(self) -> SpectralOperations:
        """
        Get the spectral operations.

        Physical Meaning:
            Returns the spectral operations object for advanced
            spectral computations.

        Returns:
            SpectralOperations: Spectral operations.
        """
        return self.spectral_ops

    def __repr__(self) -> str:
        """String representation of the 3D FFT solver."""
        return (
            f"FFTSolver3D(domain={self.domain}, "
            f"mu={self.config.get('mu', 1.0)}, "
            f"beta={self.config.get('beta', 1.0)})"
        )
