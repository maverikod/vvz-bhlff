"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Boundary condition handler for 3D FFT solver.

This module implements boundary condition handling for the 3D FFT solver,
including Dirichlet, Neumann, and periodic boundary conditions.

Physical Meaning:
    Applies boundary conditions to solutions of the 3D spectral equation,
    ensuring proper field behavior at domain boundaries.

Mathematical Foundation:
    Implements boundary conditions:
    - Dirichlet: u|∂Ω = g(x)
    - Neumann: ∂u/∂n|∂Ω = h(x)
    - Periodic: u(x + L) = u(x)

Example:
    >>> handler = BoundaryHandler(domain)
    >>> solution = handler.apply_dirichlet_boundary(solution, values)
"""

import numpy as np
from typing import Dict, Any

from ....core.domain import Domain


class BoundaryHandler:
    """
    Boundary condition handler for 3D FFT solver.

    Physical Meaning:
        Applies boundary conditions to solutions of the 3D spectral equation,
        ensuring proper field behavior at domain boundaries.

    Mathematical Foundation:
        Implements boundary conditions:
        - Dirichlet: u|∂Ω = g(x)
        - Neumann: ∂u/∂n|∂Ω = h(x)
        - Periodic: u(x + L) = u(x)

    Attributes:
        domain (Domain): 3D computational domain.
    """

    def __init__(self, domain: Domain) -> None:
        """
        Initialize boundary condition handler.

        Physical Meaning:
            Sets up the boundary condition handler with the
            3D computational domain.

        Args:
            domain (Domain): 3D computational domain.
        """
        self.domain = domain

    def apply_dirichlet_boundary(
        self, solution: np.ndarray, boundary_values: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply Dirichlet boundary conditions.

        Physical Meaning:
            Applies Dirichlet boundary conditions u|∂Ω = g(x) to the solution,
            setting field values at domain boundaries.

        Mathematical Foundation:
            Dirichlet boundary condition: u|∂Ω = g(x) where g(x) is
            the prescribed boundary values.

        Args:
            solution (np.ndarray): Solution field to apply boundary conditions to.
            boundary_values (Dict[str, Any]): Boundary condition values.

        Returns:
            np.ndarray: Solution with Dirichlet boundary conditions applied.
        """
        solution_with_bc = solution.copy()

        # Apply boundary values at domain boundaries
        if "x_min" in boundary_values:
            solution_with_bc[0, :, :] = boundary_values["x_min"]
        if "x_max" in boundary_values:
            solution_with_bc[-1, :, :] = boundary_values["x_max"]
        if "y_min" in boundary_values:
            solution_with_bc[:, 0, :] = boundary_values["y_min"]
        if "y_max" in boundary_values:
            solution_with_bc[:, -1, :] = boundary_values["y_max"]
        if "z_min" in boundary_values:
            solution_with_bc[:, :, 0] = boundary_values["z_min"]
        if "z_max" in boundary_values:
            solution_with_bc[:, :, -1] = boundary_values["z_max"]

        return solution_with_bc

    def apply_neumann_boundary(
        self, solution: np.ndarray, boundary_values: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply Neumann boundary conditions.

        Physical Meaning:
            Applies Neumann boundary conditions ∂u/∂n|∂Ω = h(x) to the solution,
            setting normal derivatives at domain boundaries.

        Mathematical Foundation:
            Neumann boundary condition: ∂u/∂n|∂Ω = h(x) where h(x) is
            the prescribed normal derivative values.

        Args:
            solution (np.ndarray): Solution field to apply boundary conditions to.
            boundary_values (Dict[str, Any]): Boundary condition values.

        Returns:
            np.ndarray: Solution with Neumann boundary conditions applied.
        """
        solution_with_bc = solution.copy()

        # Apply normal derivative boundary values
        dx = self.domain.dx

        if "x_min_derivative" in boundary_values:
            # Forward difference at x=0
            solution_with_bc[0, :, :] = (
                solution_with_bc[1, :, :] - dx * boundary_values["x_min_derivative"]
            )
        if "x_max_derivative" in boundary_values:
            # Backward difference at x=L
            solution_with_bc[-1, :, :] = (
                solution_with_bc[-2, :, :] + dx * boundary_values["x_max_derivative"]
            )
        if "y_min_derivative" in boundary_values:
            # Forward difference at y=0
            solution_with_bc[:, 0, :] = (
                solution_with_bc[:, 1, :] - dx * boundary_values["y_min_derivative"]
            )
        if "y_max_derivative" in boundary_values:
            # Backward difference at y=L
            solution_with_bc[:, -1, :] = (
                solution_with_bc[:, -2, :] + dx * boundary_values["y_max_derivative"]
            )
        if "z_min_derivative" in boundary_values:
            # Forward difference at z=0
            solution_with_bc[:, :, 0] = (
                solution_with_bc[:, :, 1] - dx * boundary_values["z_min_derivative"]
            )
        if "z_max_derivative" in boundary_values:
            # Backward difference at z=L
            solution_with_bc[:, :, -1] = (
                solution_with_bc[:, :, -2] + dx * boundary_values["z_max_derivative"]
            )

        return solution_with_bc

    def apply_periodic_boundary(self, solution: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary conditions.

        Physical Meaning:
            Applies periodic boundary conditions u(x + L) = u(x) to the solution,
            ensuring field periodicity across domain boundaries.

        Mathematical Foundation:
            Periodic boundary condition: u(x + L) = u(x) where L is
            the domain size in each direction.

        Args:
            solution (np.ndarray): Solution field to apply boundary conditions to.

        Returns:
            np.ndarray: Solution with periodic boundary conditions applied.
        """
        # For periodic boundary conditions, the FFT naturally handles periodicity
        # This method is mainly for consistency and potential post-processing
        return solution

    def __repr__(self) -> str:
        """String representation of boundary handler."""
        return f"BoundaryHandler(domain={self.domain})"
