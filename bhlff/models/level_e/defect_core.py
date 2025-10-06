"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core defect model implementation for Level E experiments in 7D phase field theory.

This module implements the base DefectModel class and core functionality
for topological defects representing localized distortions in the phase field.

Theoretical Background:
    Topological defects are singularities in the phase field that carry
    non-trivial winding numbers and create localized distortions in the
    field configuration. They represent fundamental structures in the
    7D theory with rich dynamics and interactions.

Mathematical Foundation:
    Implements defects with topological charge q ∈ ℤ where
    ∮∇φ·dl = 2πq around the defect core. The dynamics follows the
    Thiele equation: ẋ = -∇U_eff + G × ẋ + D ẋ.

Example:
    >>> defect = DefectModel(domain, physics_params)
    >>> field = defect.create_defect(position, charge)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod


class DefectModel(ABC):
    """
    Base class for topological defect models.

    Physical Meaning:
        Represents topological defects in the phase field that carry
        non-trivial winding numbers and create localized distortions
        in the field configuration.

    Mathematical Foundation:
        Implements defects with topological charge q ∈ ℤ where
        ∮∇φ·dl = 2πq around the defect core.
    """

    def __init__(self, domain: "Domain", physics_params: Dict[str, Any]):
        """
        Initialize defect model.

        Physical Meaning:
            Sets up the computational framework for topological defect
            calculations in the 7D phase field theory, including
            fractional Laplacian operators and interaction potentials.

        Args:
            domain: Computational domain with grid information
            physics_params: Physical parameters including β, μ, λ
        """
        self.domain = domain
        self.params = physics_params
        self._setup_defect_operators()

    def _setup_defect_operators(self) -> None:
        """
        Setup operators for defect calculations.
        
        Physical Meaning:
            Initializes the mathematical operators required for
            defect dynamics, including fractional Laplacian and
            interaction potentials.
        """
        # Setup fractional Laplacian for defect dynamics
        self._setup_fractional_laplacian()

        # Setup interaction potential
        self._setup_interaction_potential()

    def _setup_fractional_laplacian(self) -> None:
        """
        Setup fractional Laplacian operator.
        
        Physical Meaning:
            Computes the spectral representation of the fractional
            Laplacian operator μ(-Δ)^β required for defect dynamics
            in the 7D phase field theory.
        """
        mu = self.params.get("mu", 1.0)
        beta = self.params.get("beta", 1.0)

        # Compute wave vectors
        kx = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        ky = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        kz = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

        # Fractional Laplacian in spectral space
        self._frac_laplacian_spectral = mu * (k_magnitude ** (2 * beta))

    def _setup_interaction_potential(self) -> None:
        """
        Setup interaction potential between defects.
        
        Physical Meaning:
            Initializes the interaction potential between topological
            defects, which governs their mutual forces and dynamics.
            The potential is based on Green functions and depends on
            the defect charges and separations.
        """
        self.interaction_strength = self.params.get("interaction_strength", 1.0)
        self.interaction_range = self.params.get("interaction_range", 1.0)
        self.cutoff_radius = self.params.get("cutoff_radius", 0.1)
        
        # Setup Green function parameters for defect interactions
        self.green_function_prefactor = self.interaction_strength / (4 * np.pi)
        self.screening_length = self.interaction_range

    def create_defect(self, position: np.ndarray, charge: int) -> np.ndarray:
        """
        Create topological defect at specified position.

        Physical Meaning:
            Generates a field configuration with topological defect
            of specified charge at the given position, creating
            localized phase winding.

        Mathematical Foundation:
            Constructs field with phase φ = q·arctan2(y-y₀, x-x₀)
            around position (x₀, y₀) with charge q.

        Args:
            position: 3D position of defect center
            charge: Topological charge q ∈ ℤ

        Returns:
            Complex field configuration with defect
        """
        # Create coordinate grids
        x = np.linspace(0, self.domain.L, self.domain.N, endpoint=False)
        y = np.linspace(0, self.domain.L, self.domain.N, endpoint=False)
        z = np.linspace(0, self.domain.L, self.domain.N, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Compute distances from defect center
        dx = X - position[0]
        dy = Y - position[1]
        dz = Z - position[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Create amplitude profile
        amplitude = self._create_amplitude_profile(r, charge)

        # Create phase profile
        phase = charge * np.arctan2(dy, dx)

        # Combine amplitude and phase
        field = amplitude * np.exp(1j * phase)

        return field

    def _create_amplitude_profile(self, r: np.ndarray, charge: int) -> np.ndarray:
        """
        Create amplitude profile for defect.

        Physical Meaning:
            Generates the amplitude profile that determines the
            spatial extent and shape of the topological defect.
            The profile ensures smooth transition from defect core
            to the surrounding field.

        Mathematical Foundation:
            Uses tanh profile: A(r) = tanh(r/ξ) where ξ is the
            coherence length.

        Args:
            r: Distance from defect center
            charge: Topological charge

        Returns:
            Amplitude profile
        """
        coherence_length = self.params.get("coherence_length", 0.5)
        core_radius = self.params.get("core_radius", 0.1)

        # Tanh profile for smooth amplitude transition
        amplitude = np.tanh(r / coherence_length)

        # Ensure zero amplitude at core for non-zero charge
        if charge != 0:
            amplitude = np.where(r < core_radius, 0.0, amplitude)

        return amplitude

    def compute_defect_charge(self, field: np.ndarray, center: np.ndarray) -> float:
        """
        Compute topological charge of defect.

        Physical Meaning:
            Calculates the topological charge by integrating the
            phase gradient around a closed loop surrounding the
            defect core.

        Mathematical Foundation:
            q = (1/2π)∮∇φ·dl where the integral is taken around
            a closed loop enclosing the defect.

        Args:
            field: Complex field configuration
            center: Approximate center of defect

        Returns:
            Topological charge q ∈ ℤ
        """
        # Extract phase
        phase = np.angle(field)

        # Define integration radius
        integration_radius = self.params.get("integration_radius", 2.0)
        N = self.domain.N
        L = self.domain.L

        # Create circular integration path
        theta = np.linspace(0, 2*np.pi, 100)
        x_path = center[0] + integration_radius * np.cos(theta)
        y_path = center[1] + integration_radius * np.sin(theta)

        # Interpolate phase along path
        phase_path = self._interpolate_phase_along_path(phase, x_path, y_path)

        # Compute phase gradient
        dphase = np.diff(phase_path)
        dphase = np.append(dphase, phase_path[0] - phase_path[-1])

        # Handle phase jumps
        dphase = np.where(dphase > np.pi, dphase - 2*np.pi, dphase)
        dphase = np.where(dphase < -np.pi, dphase + 2*np.pi, dphase)

        # Integrate to get charge
        charge = np.sum(dphase) / (2 * np.pi)

        return np.round(charge)

    def _interpolate_phase_along_path(self, phase: np.ndarray, x_path: np.ndarray, y_path: np.ndarray) -> np.ndarray:
        """
        Interpolate phase along integration path.
        
        Physical Meaning:
            Computes the phase values along a circular path around
            the defect for topological charge calculation.
        """
        N = self.domain.N
        L = self.domain.L
        
        # Convert to grid indices
        i_path = np.round(x_path * N / L).astype(int) % N
        j_path = np.round(y_path * N / L).astype(int) % N
        
        # Extract phase values
        phase_values = phase[i_path, j_path]
        
        return phase_values
