"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

3D FFT solver core implementation.

This module implements the core 3D FFT solver for the 7D phase field theory,
providing efficient spectral methods for 3D problems.

Physical Meaning:
    3D FFT solver implements spectral methods for solving BVP envelope
    equations in 3D space, providing efficient computation of fractional
    operators and BVP envelope modulations.

Mathematical Foundation:
    Implements 3D spectral methods including FFT-based solvers for the
    BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x) in 3D frequency space.

Example:
    >>> solver = FFTSolver3D(domain, config)
    >>> solution = solver.solve(source)
"""

import numpy as np
from typing import Dict, Any, Optional

from ...core.domain import Domain
from ...core.fft import FFTBackend, SpectralOperations
from ...core.bvp import BVPCore
from .fft_solver_3d_boundary import FFTSolver3DBoundary
from .fft_solver_3d_bvp import FFTSolver3DBVP


class FFTSolver3D:
    """
    3D FFT solver for BVP envelope equations.

    Physical Meaning:
        Implements spectral methods for solving BVP envelope equations
        in 3D space, providing efficient computation of fractional
        operators and BVP envelope modulations.

    Mathematical Foundation:
        3D FFT solver implements spectral methods for solving:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x) in 3D frequency space using FFT operations.

    Attributes:
        domain (Domain): Computational domain (must be 3D).
        config (Dict[str, Any]): Solver configuration.
        fft_backend (FFTBackend): FFT backend for operations.
        spectral_ops (SpectralOperations): Spectral operations.
        _spectral_coeffs (np.ndarray): Pre-computed spectral coefficients.
        boundary_handler (FFTSolver3DBoundary): Boundary conditions handler.
        bvp_handler (FFTSolver3DBVP): BVP integration handler.
    """

    def __init__(
        self, domain: Domain, config: Dict[str, Any], bvp_core: Optional[BVPCore] = None
    ) -> None:
        """
        Initialize 3D FFT solver with BVP framework integration.

        Physical Meaning:
            Sets up the 3D FFT solver with computational domain and
            configuration parameters for spectral solution methods,
            with optional BVP framework integration.

        Args:
            domain (Domain): Computational domain (must be 3D).
            config (Dict[str, Any]): Solver configuration parameters.
            bvp_core (Optional[BVPCore]): BVP framework integration.

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

        # Initialize component handlers
        self.boundary_handler = FFTSolver3DBoundary(domain)
        self.bvp_handler = FFTSolver3DBVP(domain, bvp_core, config)

        self._setup_spectral_coefficients()

    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for 3D FFT solver.

        Physical Meaning:
            Pre-computes spectral coefficients for efficient solution
            of the 3D spectral equations.

        Mathematical Foundation:
            Computes spectral coefficients for the 3D spectral operator
            including wave vector arrays and spectral weights.
        """
        # Compute wave vectors
        kx = np.fft.fftfreq(self.domain.N, self.domain.dx)
        ky = np.fft.fftfreq(self.domain.N, self.domain.dx)
        kz = np.fft.fftfreq(self.domain.N, self.domain.dx)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k_squared = KX**2 + KY**2 + KZ**2

        # Spectral coefficients for 3D solver
        # This represents the spectral operator for the 3D equation
        self._spectral_coeffs = k_squared + 1.0  # Add constant term for stability

        # Handle k=0 mode
        self._spectral_coeffs[0, 0, 0] = 1.0

    def solve(self, source: np.ndarray) -> np.ndarray:
        """
        Solve 3D spectral equation.

        Physical Meaning:
            Solves the 3D spectral equation for the given source term
            using FFT-based spectral methods.

        Mathematical Foundation:
            Solves the 3D spectral equation in frequency space:
            â(k) = ŝ(k) / spectral_coeffs(k)
            where k is the 3D wave vector.

        Args:
            source (np.ndarray): Source term s(x) in real space.

        Returns:
            np.ndarray: Solution field a(x) in real space.

        Raises:
            ValueError: If source has incompatible shape with domain.
        """
        if source.shape != self.domain.shape:
            raise ValueError(
                f"Source shape {source.shape} incompatible with domain shape {self.domain.shape}"
            )

        # Transform to spectral space
        source_spectral = np.fft.fftn(source)

        # Apply spectral operator
        solution_spectral = source_spectral / self._spectral_coeffs

        # Transform back to real space
        solution = np.fft.ifftn(solution_spectral)

        return solution.real

    def solve_with_boundary_conditions(
        self,
        source: np.ndarray,
        boundary_type: str = "dirichlet",
        boundary_values: Dict[str, Any] = None,
    ) -> np.ndarray:
        """
        Solve with boundary conditions.

        Physical Meaning:
            Solves the 3D spectral equation with specified boundary
            conditions applied.

        Mathematical Foundation:
            Solves the 3D spectral equation with boundary conditions:
            - Dirichlet: u|∂Ω = g(x)
            - Neumann: ∂u/∂n|∂Ω = h(x)
            - Periodic: u(x + L) = u(x)

        Args:
            source (np.ndarray): Source term s(x).
            boundary_type (str): Type of boundary condition.
            boundary_values (Dict[str, Any]): Boundary condition values.

        Returns:
            np.ndarray: Solution with boundary conditions applied.
        """
        # Solve without boundary conditions first
        solution = self.solve(source)

        # Apply boundary conditions
        if boundary_type == "dirichlet":
            solution = self.boundary_handler.apply_dirichlet_boundary(
                solution, boundary_values
            )
        elif boundary_type == "neumann":
            solution = self.boundary_handler.apply_neumann_boundary(
                solution, boundary_values
            )
        elif boundary_type == "periodic":
            solution = self.boundary_handler.apply_periodic_boundary(solution)

        return solution

    def compute_spectral_derivative(
        self, field: np.ndarray, order: int = 1, axis: int = 0
    ) -> np.ndarray:
        """
        Compute spectral derivative.

        Physical Meaning:
            Computes the spectral derivative of the field using FFT
            methods for efficient computation.

        Mathematical Foundation:
            Computes spectral derivative: ∂ⁿu/∂xⁿ = IFFT((ik)ⁿ * FFT(u))
            where k is the wave vector and n is the derivative order.

        Args:
            field (np.ndarray): Field to differentiate.
            order (int): Derivative order.
            axis (int): Axis along which to differentiate.

        Returns:
            np.ndarray: Spectral derivative.
        """
        return self.spectral_ops.compute_derivative(field, order, axis)

    def compute_spectral_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute spectral Laplacian.

        Physical Meaning:
            Computes the spectral Laplacian of the field using FFT
            methods for efficient computation.

        Mathematical Foundation:
            Computes spectral Laplacian: ∇²u = IFFT(-k² * FFT(u))
            where k² is the squared wave vector magnitude.

        Args:
            field (np.ndarray): Field to compute Laplacian of.

        Returns:
            np.ndarray: Spectral Laplacian.
        """
        return self.spectral_ops.compute_laplacian(field)

    def get_spectral_coefficients(self) -> np.ndarray:
        """
        Get spectral coefficients.

        Physical Meaning:
            Returns the pre-computed spectral coefficients used
            in the 3D FFT solver.

        Returns:
            np.ndarray: Spectral coefficients.
        """
        return self._spectral_coeffs

    def get_fft_backend(self) -> FFTBackend:
        """
        Get FFT backend.

        Physical Meaning:
            Returns the FFT backend used by the 3D FFT solver.

        Returns:
            FFTBackend: FFT backend.
        """
        return self.fft_backend

    def get_spectral_operations(self) -> SpectralOperations:
        """
        Get spectral operations.

        Physical Meaning:
            Returns the spectral operations handler used by the
            3D FFT solver.

        Returns:
            SpectralOperations: Spectral operations handler.
        """
        return self.spectral_ops

    def solve_bvp_envelope(self, source: np.ndarray) -> np.ndarray:
        """
        Solve BVP envelope equation.

        Physical Meaning:
            Solves the BVP envelope equation for the given source term
            using BVP framework integration.

        Args:
            source (np.ndarray): Source term for BVP equation.

        Returns:
            np.ndarray: BVP envelope solution.
        """
        return self.bvp_handler.solve_bvp_envelope(source)

    def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quenches in BVP envelope.

        Physical Meaning:
            Detects quench events in the BVP envelope using BVP
            framework integration.

        Args:
            envelope (np.ndarray): BVP envelope to analyze.

        Returns:
            Dict[str, Any]: Quench detection results.
        """
        return self.bvp_handler.detect_quenches(envelope)

    def compute_bvp_impedance(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Compute BVP impedance.

        Physical Meaning:
            Computes the BVP impedance for the given envelope
            using BVP framework integration.

        Args:
            envelope (np.ndarray): BVP envelope configuration.

        Returns:
            Dict[str, Any]: BVP impedance analysis results.
        """
        return self.bvp_handler.compute_bvp_impedance(envelope)

    def get_bvp_core(self) -> Optional[BVPCore]:
        """
        Get BVP core.

        Physical Meaning:
            Returns the BVP framework integration core.

        Returns:
            Optional[BVPCore]: BVP core if available.
        """
        return self.bvp_handler.get_bvp_core()

    def set_bvp_core(self, bvp_core: BVPCore) -> None:
        """
        Set BVP core.

        Physical Meaning:
            Sets the BVP framework integration core.

        Args:
            bvp_core (BVPCore): BVP core to set.
        """
        self.bvp_handler.set_bvp_core(bvp_core)

    def __repr__(self) -> str:
        """String representation of the 3D FFT solver."""
        return (
            f"FFTSolver3D(domain={self.domain}, "
            f"bvp_core={'available' if self.get_bvp_core() is not None else 'none'})"
        )
