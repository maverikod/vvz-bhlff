"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Multi-particle potential analysis module.

This module implements potential analysis functionality for multi-particle systems
in Level F of 7D phase field theory.

Physical Meaning:
    Computes effective potentials for multi-particle systems
    including single-particle, pair-wise, and higher-order interactions.

Example:
    >>> potential_analyzer = MultiParticlePotentialAnalyzer(domain, particles)
    >>> potential = potential_analyzer.compute_effective_potential()
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from bhlff.core.bvp import BVPCore
from .multi_particle.data_structures import Particle, SystemParameters


class MultiParticlePotentialAnalyzer:
    """
    Multi-particle potential analyzer for Level F.

    Physical Meaning:
        Computes effective potentials for multi-particle systems
        including single-particle, pair-wise, and higher-order interactions.

    Mathematical Foundation:
        Implements effective potential calculation:
        - Effective potential: U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ
    """

    def __init__(
        self,
        domain,
        particles: List[Particle],
        interaction_range: float = 2.0,
        params: Dict[str, Any] = None,
    ):
        """
        Initialize multi-particle potential analyzer.

        Physical Meaning:
            Sets up the potential analysis system with
            appropriate parameters and methods.

        Args:
            domain: Domain parameters.
            particles (List[Particle]): List of particles.
            interaction_range (float): Interaction range parameter.
            params (Dict[str, Any]): Additional parameters for step resonator model.
        """
        self.domain = domain
        self.particles = particles
        self.interaction_range = float(interaction_range)
        self.params = params or {}
        self.logger = logging.getLogger(__name__)

        # Vectorization buffers
        self._positions = (
            np.asarray([p.position for p in self.particles], dtype=float)
            if self.particles
            else np.zeros((0, 3), dtype=float)
        )
        self._charges = (
            np.asarray([float(p.charge) for p in self.particles], dtype=float)
            if self.particles
            else np.zeros((0,), dtype=float)
        )

        # Precompute adjacency for pair/three-body uniform contributions
        self._close_pairs: List[tuple[int, int]] = []
        self._close_triples: List[tuple[int, int, int]] = []
        try:
            if self._positions.size:
                diffs = self._positions[:, None, :] - self._positions[None, :, :]
                d2 = np.sum(diffs * diffs, axis=-1)
                mask_pairs = d2 < (self.interaction_range**2)
                n = mask_pairs.shape[0]
                for i in range(n):
                    for j in range(i + 1, n):
                        if mask_pairs[i, j]:
                            self._close_pairs.append((i, j))
                for i in range(n):
                    for j in range(i + 1, n):
                        if not mask_pairs[i, j]:
                            continue
                        for k in range(j + 1, n):
                            if mask_pairs[i, k] and mask_pairs[j, k]:
                                self._close_triples.append((i, j, k))
        except Exception:
            self._close_pairs = []
            self._close_triples = []

        # Setup interaction matrices (kept for compatibility)
        self._setup_interaction_matrices()

    def compute_effective_potential(self, *_: Any) -> np.ndarray:
        """
        Compute effective potential.

        Physical Meaning:
            Computes effective potential for multi-particle system
            including all interaction terms.

        Mathematical Foundation:
            Effective potential: U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ

        Returns:
            np.ndarray: Effective potential field.
        """
        self.logger.info("Computing effective potential (vectorized, block-processed)")

        # Spatial grid shape (3D)
        N = int(self.domain.N)
        spatial_shape = (N, N, N)
        result = np.zeros(spatial_shape, dtype=np.float64)

        if self._positions.size == 0:
            return result

        # Coordinate arrays
        x = np.linspace(0.0, float(getattr(self.domain, "L", N)), N)
        y = np.linspace(0.0, float(getattr(self.domain, "L", N)), N)
        z = np.linspace(0.0, float(getattr(self.domain, "L", N)), N)

        # Memory-aware block size for CPU (aim ~ few hundred MB per block)
        cpu_block_size = int(self.params.get("cpu_block_size", 64))
        cpu_block_size = max(8, min(cpu_block_size, N))

        # Particle sub-batching to limit peak memory
        n_particles = self._positions.shape[0]
        strength = float(self.params.get("interaction_strength", 1.0))
        r_cut = self.interaction_range
        rc2 = r_cut * r_cut

        # Iterate spatial blocks
        for i0 in range(0, N, cpu_block_size):
            i1 = min(N, i0 + cpu_block_size)
            for j0 in range(0, N, cpu_block_size):
                j1 = min(N, j0 + cpu_block_size)
                for k0 in range(0, N, cpu_block_size):
                    k1 = min(N, k0 + cpu_block_size)
                    Xb, Yb, Zb = np.meshgrid(
                        x[i0:i1], y[j0:j1], z[k0:k1], indexing="ij"
                    )
                    block = np.zeros(Xb.shape, dtype=np.float64)

                    # Estimate sub-batch size: target ~600 MB cap per sub-batch
                    elements = Xb.size
                    bytes_per_elem = 8
                    est_per_particle = (
                        elements * bytes_per_elem * 6
                    )  # dx,dy,dz,r2,mask,accum
                    budget = int(600 * (1024**2))  # ~600 MB
                    if est_per_particle <= 0:
                        subbatch = min(n_particles, 64)
                    else:
                        subbatch = max(1, min(n_particles, budget // est_per_particle))
                        subbatch = min(subbatch, 128)

                    for p0 in range(0, n_particles, subbatch):
                        p1 = min(n_particles, p0 + subbatch)
                        px = self._positions[p0:p1, 0]
                        py = self._positions[p0:p1, 1]
                        pz = self._positions[p0:p1, 2]
                        cq = self._charges[p0:p1]

                        dx = Xb[..., None] - px[None, None, None, :]
                        dy = Yb[..., None] - py[None, None, None, :]
                        dz = Zb[..., None] - pz[None, None, None, :]
                        r2 = dx * dx + dy * dy + dz * dz
                        mask = r2 < rc2
                        contrib = mask.astype(np.float64) * cq[None, None, None, :]
                        block += strength * contrib.sum(axis=-1)

                    # Uniform pair and three-body contributions (precomputed adjacency)
                    if self._close_pairs:
                        block += strength * len(self._close_pairs)
                    if self._close_triples:
                        block += strength * len(self._close_triples)

                    result[i0:i1, j0:j1, k0:k1] = block

        self.logger.info("Effective potential computed (CPU vectorized)")
        return result

    # Methods expected by tests for patching
    def compute_single_particle_potential(
        self, particle: Particle, particles: List[Particle]
    ) -> Any:
        return self._compute_single_particle_potential(particle)

    def compute_pair_interaction(self, particle1: Particle, particle2: Particle) -> Any:
        return self._compute_pair_interaction(particle1, particle2)

    def compute_three_body_interaction(
        self, p1: Particle, p2: Particle, p3: Particle
    ) -> Any:
        return self._compute_three_body_interaction(p1, p2, p3)

    def _setup_interaction_matrices(self) -> None:
        """
        Setup interaction matrices.

        Physical Meaning:
            Sets up interaction matrices for multi-particle system
            to enable efficient potential calculations.
        """
        # Setup interaction matrices
        # In practice, this would involve proper matrix setup
        self.interaction_matrix = np.zeros((len(self.particles), len(self.particles)))

        # Calculate interaction strengths
        for i, particle1 in enumerate(self.particles):
            for j, particle2 in enumerate(self.particles):
                if i != j:
                    distance = np.linalg.norm(particle1.position - particle2.position)
                    interaction_strength = self._calculate_interaction_strength(
                        distance
                    )
                    self.interaction_matrix[i, j] = interaction_strength

    def _compute_single_particle_potential(self, particle: Particle) -> np.ndarray:
        """
        Compute single-particle potential.

        Physical Meaning:
            Computes potential contribution from single particle
            in the multi-particle system.

        Args:
            particle (Particle): Particle object.

        Returns:
            np.ndarray: Single-particle potential field.
        """
        # Create coordinate arrays
        x = np.linspace(0, self.domain.L, self.domain.N)
        y = np.linspace(0, self.domain.L, self.domain.N)
        z = np.linspace(0, self.domain.L, self.domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Calculate distance from particle
        distance = np.sqrt(
            (X - particle.position[0]) ** 2
            + (Y - particle.position[1]) ** 2
            + (Z - particle.position[2]) ** 2
        )

        # Compute potential using step resonator model
        potential = particle.charge * self._step_interaction_potential(distance)

        return potential

    def _compute_pair_interaction(
        self, particle1: Particle, particle2: Particle
    ) -> np.ndarray:
        """
        Compute pair interaction potential.

        Physical Meaning:
            Computes potential contribution from pair interaction
            between two particles.

        Args:
            particle1 (Particle): First particle.
            particle2 (Particle): Second particle.

        Returns:
            np.ndarray: Pair interaction potential field.
        """
        # Calculate distance between particles
        distance = np.linalg.norm(particle1.position - particle2.position)

        # Compute interaction strength
        interaction_strength = self._calculate_interaction_strength(distance)

        # Create potential field
        spatial_shape = (int(self.domain.N), int(self.domain.N), int(self.domain.N))
        potential = np.zeros(spatial_shape)

        # Add interaction contribution
        if distance < self.interaction_range:
            potential += interaction_strength * np.ones(self.domain.shape)

        return potential

    def _compute_higher_order_interactions(self) -> np.ndarray:
        """
        Compute higher-order interactions.

        Physical Meaning:
            Computes potential contribution from higher-order
            interactions in the multi-particle system.

        Returns:
            np.ndarray: Higher-order interaction potential field.
        """
        # Initialize potential
        spatial_shape = (int(self.domain.N), int(self.domain.N), int(self.domain.N))
        potential = np.zeros(spatial_shape)

        # Compute three-body interactions
        for i, particle1 in enumerate(self.particles):
            for j, particle2 in enumerate(self.particles[i + 1 :], i + 1):
                for k, particle3 in enumerate(self.particles[j + 1 :], j + 1):
                    three_body_potential = self._compute_three_body_interaction(
                        particle1, particle2, particle3
                    )
                    potential += three_body_potential

        return potential

    def _compute_three_body_interaction(
        self, particle1: Particle, particle2: Particle, particle3: Particle
    ) -> np.ndarray:
        """
        Compute three-body interaction potential.

        Physical Meaning:
            Computes potential contribution from three-body
            interaction between three particles.

        Args:
            particle1 (Particle): First particle.
            particle2 (Particle): Second particle.
            particle3 (Particle): Third particle.

        Returns:
            np.ndarray: Three-body interaction potential field.
        """
        # Calculate distances
        distance_12 = np.linalg.norm(particle1.position - particle2.position)
        distance_13 = np.linalg.norm(particle1.position - particle3.position)
        distance_23 = np.linalg.norm(particle2.position - particle3.position)

        # Compute three-body interaction strength
        interaction_strength = self._calculate_three_body_strength(
            distance_12, distance_13, distance_23
        )

        # Create potential field
        potential = np.zeros(self.domain.shape)

        # Add three-body interaction contribution
        if (
            distance_12 < self.interaction_range
            and distance_13 < self.interaction_range
            and distance_23 < self.interaction_range
        ):
            potential += interaction_strength * np.ones(self.domain.shape)

        return potential

    def _calculate_interaction_strength(self, distance: float) -> float:
        """
        Calculate interaction strength.

        Physical Meaning:
            Calculates interaction strength between particles
            based on distance.

        Args:
            distance (float): Distance between particles.

        Returns:
            float: Interaction strength.
        """
        # Step resonator interaction strength calculation
        # Based on 7D BVP theory principles
        return self._step_interaction_potential(distance)

    def _calculate_three_body_strength(
        self, distance_12: float, distance_13: float, distance_23: float
    ) -> float:
        """
        Calculate three-body interaction strength.

        Physical Meaning:
            Calculates three-body interaction strength
            based on particle distances.

        Args:
            distance_12 (float): Distance between particles 1 and 2.
            distance_13 (float): Distance between particles 1 and 3.
            distance_23 (float): Distance between particles 2 and 3.

        Returns:
            float: Three-body interaction strength.
        """
        # Step resonator three-body interaction strength calculation
        # Based on 7D BVP theory principles
        if (
            distance_12 < self.interaction_range
            and distance_13 < self.interaction_range
            and distance_23 < self.interaction_range
        ):
            return self._step_three_body_interaction_potential(
                distance_12, distance_13, distance_23
            )
        else:
            return 0.0

    def _step_interaction_potential(self, distance):
        """
        Step function interaction potential.

        Physical Meaning:
            Implements step resonator model for particle interactions instead of
            exponential decay. This follows 7D BVP theory principles where
            energy exchange occurs through semi-transparent boundaries.

        Mathematical Foundation:
            V(r) = V₀ * Θ(r_cutoff - r) where Θ is the Heaviside step function
            and r_cutoff is the cutoff distance for the interaction.

        Args:
            distance (float): Distance between particles

        Returns:
            float: Step function interaction potential
        """
        # Step resonator parameters
        interaction_strength = self.params.get("interaction_strength", 1.0)

        # Support scalar or ndarray distance: return array mask or float
        mask = distance < self.interaction_range
        return interaction_strength * (
            mask.astype(float) if hasattr(mask, "astype") else float(mask)
        )

    def _step_three_body_interaction_potential(
        self, distance_12: float, distance_13: float, distance_23: float
    ) -> float:
        """
        Step function three-body interaction potential.

        Physical Meaning:
            Implements step resonator model for three-body interactions instead of
            exponential decay. This follows 7D BVP theory principles where
            energy exchange occurs through semi-transparent boundaries.

        Mathematical Foundation:
            V(r₁₂,r₁₃,r₂₃) = V₀ * Θ(r_cutoff - r_avg) where Θ is the Heaviside step function
            and r_avg is the average distance between particles.

        Args:
            distance_12 (float): Distance between particles 1 and 2
            distance_13 (float): Distance between particles 1 and 3
            distance_23 (float): Distance between particles 2 and 3

        Returns:
            float: Step function three-body interaction potential
        """
        # Step resonator parameters
        interaction_strength = self.params.get("interaction_strength", 1.0)

        # Average distance for three-body interaction
        avg_distance = (distance_12 + distance_13 + distance_23) / 3.0

        # Step function three-body interaction: 1.0 below cutoff, 0.0 above
        return interaction_strength if avg_distance < self.interaction_range else 0.0
