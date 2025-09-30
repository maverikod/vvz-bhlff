"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

3D FFT solver boundary conditions implementation.

This module provides boundary conditions functionality for the 3D FFT solver
in the 7D phase field theory.

Physical Meaning:
    3D FFT solver boundary conditions handle the application of various
    boundary conditions including Dirichlet and Neumann conditions for
    spectral solution methods.

Mathematical Foundation:
    Implements boundary condition application for spectral methods including
    Dirichlet (fixed values) and Neumann (fixed derivatives) conditions.

Example:
    >>> boundary_handler = FFTSolver3DBoundary(domain)
    >>> field_with_bc = boundary_handler.apply_dirichlet_boundary(field)
"""

import numpy as np
from typing import Dict, Any

from ...core.domain import Domain


class FFTSolver3DBoundary:
    """
    3D FFT solver boundary conditions handler.

    Physical Meaning:
        Handles the application of various boundary conditions for the
        3D FFT solver including Dirichlet and Neumann conditions.

    Mathematical Foundation:
        Implements boundary condition application for spectral methods:
        - Dirichlet: u|∂Ω = g(x)
        - Neumann: ∂u/∂n|∂Ω = h(x)

    Attributes:
        domain (Domain): Computational domain.
    """

    def __init__(self, domain: Domain) -> None:
        """
        Initialize 3D FFT solver boundary conditions handler.

        Physical Meaning:
            Sets up the boundary conditions handler for the 3D FFT solver
            with the computational domain.

        Args:
            domain (Domain): Computational domain.
        """
        self.domain = domain

    def apply_dirichlet_boundary(
        self, field: np.ndarray, boundary_values: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Apply Dirichlet boundary conditions.

        Physical Meaning:
            Applies Dirichlet boundary conditions (fixed values) to the
            field at the domain boundaries.

        Mathematical Foundation:
            Dirichlet boundary condition: u|∂Ω = g(x)
            where g(x) specifies the values at the boundary.

        Args:
            field (np.ndarray): Field to apply boundary conditions to.
            boundary_values (Dict[str, float]): Boundary values for each face.
                Keys: 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
                Values: boundary values

        Returns:
            np.ndarray: Field with Dirichlet boundary conditions applied.
        """
        if boundary_values is None:
            boundary_values = {
                "x_min": 0.0,
                "x_max": 0.0,
                "y_min": 0.0,
                "y_max": 0.0,
                "z_min": 0.0,
                "z_max": 0.0,
            }

        field_with_bc = field.copy()

        # Apply boundary conditions on each face
        # x = 0 face
        if "x_min" in boundary_values:
            field_with_bc[0, :, :] = boundary_values["x_min"]

        # x = L face
        if "x_max" in boundary_values:
            field_with_bc[-1, :, :] = boundary_values["x_max"]

        # y = 0 face
        if "y_min" in boundary_values:
            field_with_bc[:, 0, :] = boundary_values["y_min"]

        # y = L face
        if "y_max" in boundary_values:
            field_with_bc[:, -1, :] = boundary_values["y_max"]

        # z = 0 face
        if "z_min" in boundary_values:
            field_with_bc[:, :, 0] = boundary_values["z_min"]

        # z = L face
        if "z_max" in boundary_values:
            field_with_bc[:, :, -1] = boundary_values["z_max"]

        return field_with_bc

    def apply_neumann_boundary(
        self, field: np.ndarray, boundary_derivatives: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Apply Neumann boundary conditions.

        Physical Meaning:
            Applies Neumann boundary conditions (fixed derivatives) to the
            field at the domain boundaries.

        Mathematical Foundation:
            Neumann boundary condition: ∂u/∂n|∂Ω = h(x)
            where h(x) specifies the normal derivatives at the boundary.

        Args:
            field (np.ndarray): Field to apply boundary conditions to.
            boundary_derivatives (Dict[str, float]): Normal derivatives for each face.
                Keys: 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
                Values: normal derivatives

        Returns:
            np.ndarray: Field with Neumann boundary conditions applied.
        """
        if boundary_derivatives is None:
            boundary_derivatives = {
                "x_min": 0.0,
                "x_max": 0.0,
                "y_min": 0.0,
                "y_max": 0.0,
                "z_min": 0.0,
                "z_max": 0.0,
            }

        field_with_bc = field.copy()
        dx = self.domain.dx

        # Apply boundary conditions on each face
        # x = 0 face (∂u/∂x = h)
        if "x_min" in boundary_derivatives:
            field_with_bc[0, :, :] = (
                field_with_bc[1, :, :] - dx * boundary_derivatives["x_min"]
            )

        # x = L face (∂u/∂x = h)
        if "x_max" in boundary_derivatives:
            field_with_bc[-1, :, :] = (
                field_with_bc[-2, :, :] + dx * boundary_derivatives["x_max"]
            )

        # y = 0 face (∂u/∂y = h)
        if "y_min" in boundary_derivatives:
            field_with_bc[:, 0, :] = (
                field_with_bc[:, 1, :] - dx * boundary_derivatives["y_min"]
            )

        # y = L face (∂u/∂y = h)
        if "y_max" in boundary_derivatives:
            field_with_bc[:, -1, :] = (
                field_with_bc[:, -2, :] + dx * boundary_derivatives["y_max"]
            )

        # z = 0 face (∂u/∂z = h)
        if "z_min" in boundary_derivatives:
            field_with_bc[:, :, 0] = (
                field_with_bc[:, :, 1] - dx * boundary_derivatives["z_min"]
            )

        # z = L face (∂u/∂z = h)
        if "z_max" in boundary_derivatives:
            field_with_bc[:, :, -1] = (
                field_with_bc[:, :, -2] + dx * boundary_derivatives["z_max"]
            )

        return field_with_bc

    def apply_periodic_boundary(self, field: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary conditions.

        Physical Meaning:
            Applies periodic boundary conditions to the field, ensuring
            continuity across domain boundaries.

        Mathematical Foundation:
            Periodic boundary condition: u(x + L) = u(x)
            where L is the domain size.

        Args:
            field (np.ndarray): Field to apply boundary conditions to.

        Returns:
            np.ndarray: Field with periodic boundary conditions applied.
        """
        field_with_bc = field.copy()

        # Periodic boundary conditions are naturally satisfied in spectral methods
        # when using FFT, but we can enforce them explicitly if needed

        # Ensure continuity at boundaries
        # x-direction
        field_with_bc[0, :, :] = field_with_bc[-1, :, :]
        field_with_bc[-1, :, :] = field_with_bc[0, :, :]

        # y-direction
        field_with_bc[:, 0, :] = field_with_bc[:, -1, :]
        field_with_bc[:, -1, :] = field_with_bc[:, 0, :]

        # z-direction
        field_with_bc[:, :, 0] = field_with_bc[:, :, -1]
        field_with_bc[:, :, -1] = field_with_bc[:, :, 0]

        return field_with_bc

    def get_boundary_info(self) -> Dict[str, Any]:
        """
        Get boundary condition information.

        Physical Meaning:
            Returns information about the boundary conditions setup
            for the 3D FFT solver.

        Returns:
            Dict[str, Any]: Boundary condition information.
        """
        return {
            "domain_shape": self.domain.shape,
            "domain_size": self.domain.L,
            "grid_spacing": self.domain.dx,
            "boundary_types": ["dirichlet", "neumann", "periodic"],
            "supported_conditions": {
                "dirichlet": "Fixed values at boundaries",
                "neumann": "Fixed normal derivatives at boundaries",
                "periodic": "Periodic continuity across boundaries",
            },
        }
