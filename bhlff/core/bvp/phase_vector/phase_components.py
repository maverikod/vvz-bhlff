"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Phase components management for U(1)³ phase vector structure.

This module handles the three U(1) phase components Θ_a (a=1..3)
that form the fundamental phase structure of the BVP field.

Physical Meaning:
    Manages the three independent U(1) phase components that
    represent different phase degrees of freedom in the BVP field.
    Each component corresponds to a different U(1) symmetry group.

Mathematical Foundation:
    The three components Θ₁, Θ₂, Θ₃ are independent U(1) phase
    degrees of freedom that combine to form the total phase:
    Θ_total = Σ_a Θ_a + coupling_terms

Example:
    >>> components = PhaseComponents(domain, config)
    >>> theta_components = components.get_components()
    >>> total_phase = components.get_total_phase()
"""

import numpy as np
from typing import Dict, Any, List
import logging

from bhlff.core.domain import Domain

# CUDA optimization
try:
    import cupy as cp

    CUDA_AVAILABLE = True
    logging.info("CUDA support enabled with CuPy")
except ImportError:
    CUDA_AVAILABLE = False
    logging.warning("CUDA not available, falling back to CPU")


class PhaseComponents:
    """
    Management of three U(1) phase components Θ_a (a=1..3).

    Physical Meaning:
        Handles the three independent U(1) phase components that
        form the U(1)³ structure of the BVP field.

    Mathematical Foundation:
        Manages Θ₁, Θ₂, Θ₃ as independent U(1) phase degrees
        of freedom with weak hierarchical coupling.

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Phase components configuration.
        theta_components (List[np.ndarray]): Three phase components Θ_a.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize phase components manager.

        Physical Meaning:
            Sets up the three U(1) phase components Θ_a (a=1..3)
            with proper spatial distribution and frequencies.

        Args:
            domain (Domain): Computational domain.
            config (Dict[str, Any]): Phase components configuration including:
                - phase_amplitudes: Amplitudes for each phase component
                - phase_frequencies: Frequencies for each phase component
        """
        self.domain = domain
        self.config = config
        self.theta_components = []

        # CUDA optimization setup
        self.cuda_available = CUDA_AVAILABLE
        self.use_cuda = config.get("use_cuda", True) and self.cuda_available
        self.logger = logging.getLogger(__name__)

        if self.use_cuda:
            self.logger.info("PhaseComponents: CUDA optimization enabled")
        else:
            self.logger.info("PhaseComponents: Using CPU computation")

        self._setup_phase_components()

    def _setup_phase_components(self) -> None:
        """
        Setup the three U(1) phase components Θ_a (a=1..3).

        Physical Meaning:
            Initializes the three independent U(1) phase components
            that form the U(1)³ structure of the BVP field.
        """
        # Get phase configuration
        phase_config = self.config.get("phase_components", {})

        for a in range(3):  # Three U(1) components
            # Initialize phase component Θ_a
            theta_a = np.zeros(self.domain.shape, dtype=complex)

            # Set amplitude and frequency for this component
            amplitude = phase_config.get(f"amplitude_{a+1}", 1.0)
            frequency = phase_config.get(f"frequency_{a+1}", 1.0)

            # Create spatial phase distribution for 7D structure
            if self.domain.dimensions == 7:
                # 7D structure: ℝ³ₓ × 𝕋³_φ × ℝₜ
                # Use domain shape for proper 7D structure
                theta_a = np.zeros(self.domain.shape, dtype=complex)

                # Create simple 7D phase distribution
                # For testing purposes, create a simple phase pattern
                indices = np.indices(self.domain.shape)
                phase_sum = np.sum(indices, axis=0)
                theta_a = amplitude * np.exp(
                    1j * frequency * phase_sum / np.max(phase_sum)
                )
            elif self.domain.dimensions == 1:
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                theta_a = amplitude * np.exp(1j * frequency * x)
            elif self.domain.dimensions == 2:
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                X, Y = np.meshgrid(x, y, indexing="ij")
                theta_a = amplitude * np.exp(1j * frequency * (X + Y))
            else:  # 3D
                x = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                y = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                z = np.linspace(-self.domain.L / 2, self.domain.L / 2, self.domain.N)
                X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
                theta_a = amplitude * np.exp(1j * frequency * (X + Y + Z))

            self.theta_components.append(theta_a)

    def get_components(self) -> List[np.ndarray]:
        """
        Get the three U(1) phase components Θ_a (a=1..3).

        Physical Meaning:
            Returns the three independent U(1) phase components
            that form the U(1)³ structure.

        Returns:
            List[np.ndarray]: List of three phase components Θ_a.
        """
        return self.theta_components.copy()

    def get_total_phase(self, coupling_matrix: np.ndarray = None) -> np.ndarray:
        """
        Get the total phase from U(1)³ structure.

        Physical Meaning:
            Computes the total phase by combining the three
            U(1) components with proper coupling.

        Mathematical Foundation:
            Θ_total = Σ_a Θ_a + Σ_{a,b} g_{ab} Θ_a Θ_b
            where g_{ab} are the coupling coefficients.

        Args:
            coupling_matrix (np.ndarray, optional): Coupling matrix for components.

        Returns:
            np.ndarray: Total phase field.
        """
        # Start with sum of individual components
        total_phase = np.zeros_like(self.theta_components[0])

        for theta_a in self.theta_components:
            total_phase += theta_a

        # Add coupling terms if provided
        if coupling_matrix is not None:
            for i, theta_i in enumerate(self.theta_components):
                for j, theta_j in enumerate(self.theta_components):
                    if i != j:
                        coupling_strength = coupling_matrix[i, j]
                        total_phase += coupling_strength * theta_i * theta_j

        return total_phase

    def update_components(self, envelope: np.ndarray) -> None:
        """
        Update phase components from solved envelope.

        Physical Meaning:
            Updates the three U(1) phase components Θ_a (a=1..3)
            from the solved BVP envelope field.

        Mathematical Foundation:
            Extracts phase components from the envelope solution
            and updates the U(1)³ phase structure.

        Args:
            envelope (np.ndarray): Solved BVP envelope in 7D space-time.
        """
        # If envelope is a vector field, extract components
        if envelope.ndim > self.domain.dimensions:
            # Envelope has additional dimensions for phase components
            for a in range(3):
                if a < envelope.shape[-1]:  # Check if component exists
                    self.theta_components[a] = envelope[..., a]
        else:
            # Single envelope field - distribute to components
            for a in range(3):
                # Each component gets a phase-shifted version
                phase_shift = 2 * np.pi * a / 3
                self.theta_components[a] = envelope * np.exp(1j * phase_shift)

    def compute_phase_coherence(self) -> np.ndarray:
        """
        Compute phase coherence measure.

        Physical Meaning:
            Computes a measure of phase coherence across the
            U(1)³ structure, indicating the degree of
            synchronization between the three phase components.

        Mathematical Foundation:
            Coherence = |Σ_a exp(iΘ_a)| / 3
            where the magnitude indicates coherence strength.

        Returns:
            np.ndarray: Phase coherence measure.
        """
        # Sum of complex exponentials
        coherence_sum = np.zeros_like(self.theta_components[0])
        coherence_sum = self._to_gpu(coherence_sum)

        for theta_a in self.theta_components:
            theta_a_gpu = self._to_gpu(theta_a)
            coherence_sum += self._cuda_exp(1j * self._cuda_angle(theta_a_gpu))

        # Normalize by number of components
        coherence = self._cuda_abs(coherence_sum) / 3.0

        return self._to_cpu(coherence)

    def _to_gpu(self, array: np.ndarray) -> "cp.ndarray":
        """
        Convert numpy array to GPU array.

        Physical Meaning:
            Transfers array to GPU memory for CUDA computation.

        Args:
            array (np.ndarray): Input array.

        Returns:
            cp.ndarray: GPU array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.asarray(array)
        return array

    def _to_cpu(self, array) -> np.ndarray:
        """
        Convert GPU array to numpy array.

        Physical Meaning:
            Transfers array from GPU memory to CPU memory.

        Args:
            array: Input array (GPU or CPU).

        Returns:
            np.ndarray: CPU array.
        """
        if self.use_cuda and CUDA_AVAILABLE and hasattr(array, "get"):
            return array.get()
        return array

    def _cuda_exp(self, array) -> "cp.ndarray":
        """
        Compute exponential using CUDA.

        Physical Meaning:
            Computes exponential using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Exponential array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.exp(array)
        return np.exp(array)

    def _cuda_abs(self, array) -> "cp.ndarray":
        """
        Compute absolute value using CUDA.

        Physical Meaning:
            Computes absolute value using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Absolute value array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.abs(array)
        return np.abs(array)

    def _cuda_angle(self, array) -> "cp.ndarray":
        """
        Compute angle using CUDA.

        Physical Meaning:
            Computes angle using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Angle array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.angle(array)
        return np.angle(array)

    def _cuda_mean(self, array, axis=None) -> "cp.ndarray":
        """
        Compute mean using CUDA.

        Physical Meaning:
            Computes mean using CUDA for optimal performance.

        Args:
            array: Input array.
            axis: Axis along which to compute mean.

        Returns:
            cp.ndarray: Mean array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.mean(array, axis=axis)
        return np.mean(array, axis=axis)

    def __repr__(self) -> str:
        """String representation of phase components."""
        cuda_status = "CUDA" if self.use_cuda else "CPU"
        return f"PhaseComponents(domain={self.domain}, num_components={len(self.theta_components)}, compute={cuda_status})"
