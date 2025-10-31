"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA-accelerated multi-particle potential analyzer for Level F.

This module implements a GPU-optimized computation of effective potentials
for multi-particle systems with strict memory-aware block processing that
targets 80% of available GPU memory. It leverages CuPy for vectorized
operations and integrates with the 7D block processing tools.

Physical Meaning:
    Computes the effective potential for systems of multiple topological
    defects interacting via a step-resonator model. The potential includes
    single-particle, pair-wise, and higher-order (three-body) terms.

Mathematical Foundation:
    U_eff = \\sum_i U_i + \\sum_{i<j} U_{ij} + \\sum_{i<j<k} U_{ijk}
    where the interactions follow a step potential with cutoff r_cutoff.

Example:
    >>> analyzer = MultiParticlePotentialAnalyzerCUDA(
    ...     domain,
    ...     particles,
    ...     interaction_range=5.0,
    ...     params={"interaction_strength": 1.0},
    ... )
    >>> potential = analyzer.compute_effective_potential()
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import logging
import numpy as np

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CUDA_AVAILABLE = False

from bhlff.core.domain.cuda_block_processor import CUDABlockProcessor
from bhlff.utils.cuda_utils import CUDABackend, get_optimal_backend
from ..multi_particle.data_structures import Particle, SystemParameters


class MultiParticlePotentialAnalyzerCUDA:
    """
    CUDA-accelerated potential analyzer for multi-particle systems.

    Physical Meaning:
        Computes the effective potential of a multi-particle system on GPU
        using vectorized CuPy operations with block processing sized to use
        approximately 80% of free GPU memory without out-of-memory errors.

    Mathematical Foundation:
        Implements step-resonator interactions for single, pair, and three-body
        terms and aggregates their contributions over the computational domain.

    Attributes:
        domain: Computational domain with attributes `L`, `N`, `shape`.
        particles: List of `Particle` objects describing the system.
        interaction_range: Cutoff radius r_cutoff for step interactions.
        params: Additional parameters, including `interaction_strength`.
        system_params: Optional `SystemParameters`
            for consistency with CPU path.
    """

    def __init__(
        self,
        domain: Any,
        particles: List[Particle],
        interaction_range: float = 2.0,
        params: Optional[Dict[str, Any]] = None,
        system_params: Optional[SystemParameters] = None,
    ) -> None:
        """
        Initialize the CUDA analyzer.

        Physical Meaning:
            Sets up GPU backend and block processor for memory-safe computation
            of the effective potential across the full domain.

        Args:
            domain: Computational domain instance.
            particles (List[Particle]):
                Particles participating in interactions.
            interaction_range (float): Cutoff for step interactions.
            params (Optional[Dict[str, Any]]): Additional parameters.
            system_params (Optional[SystemParameters]):
                System-level parameters.
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA not available for MultiParticlePotentialAnalyzerCUDA"
            )

        self.logger = logging.getLogger(__name__)
        self.domain = domain
        self.particles = particles
        self.interaction_range = float(interaction_range)
        self.params: Dict[str, Any] = params or {}
        self.system_params = system_params or SystemParameters()

        # Select optimal backend (will be CUDA when available)
        backend = get_optimal_backend()
        if not isinstance(backend, CUDABackend):
            raise RuntimeError(
                "CUDA backend not selected; cannot construct CUDA analyzer"
            )
        self.backend = backend

        # Determine optimal block size targeting ~80% free GPU memory
        self.block_size = self._compute_optimal_block_size_7d()

        # Initialize CUDA block processor with computed block size
        self.block_processor = CUDABlockProcessor(
            domain, block_size=self.block_size
        )

    def compute_effective_potential(self) -> np.ndarray:
        """
        Compute the effective potential field on GPU with block processing.

        Physical Meaning:
            Aggregates contributions from single, pair, and three-body
            interactions across the entire domain using memory-aware
            CUDA block processing.

        Returns:
            np.ndarray: Effective potential on CPU memory with domain.shape.
        """
        # Preallocate on CPU; assemble per block to reduce GPU pressure
        result = np.zeros(self.domain.shape, dtype=np.float64)

        # Prepare particle data as CuPy arrays for vectorized math
        particle_positions = [
            cp.asarray(p.position, dtype=cp.float64) for p in self.particles
        ]
        particle_charges = [
            float(p.charge) for p in self.particles
        ]

        # Grid coordinates (per-block will slice views to minimize allocations)
        x = cp.linspace(0.0, float(self.domain.L), int(self.domain.N))
        y = cp.linspace(0.0, float(self.domain.L), int(self.domain.N))
        z = cp.linspace(0.0, float(self.domain.L), int(self.domain.N))

        # Iterate CUDA blocks by indices; slice coordinates to each block
        block_id = 0
        for _, block_info in self.block_processor.iterate_blocks_cuda():
            # Compute block slices
            start = block_info.start_indices
            end = block_info.end_indices
            slices = tuple(slice(s, e) for s, e in zip(start, end))

            # Coordinate sub-grids for this block
            Xb, Yb, Zb = cp.meshgrid(
                x[slices[0]], y[slices[1]], z[slices[2]], indexing="ij"
            )

            # Accumulate block potential on GPU
            block_potential = cp.zeros(Xb.shape, dtype=cp.float64)

            # Single-particle contributions:
            # step potential around each particle
            strength = float(self.params.get("interaction_strength", 1.0))
            r_cut = self.interaction_range
            rc2 = r_cut * r_cut

            for p_pos, p_q in zip(particle_positions, particle_charges):
                dx = Xb - p_pos[0]
                dy = Yb - p_pos[1]
                dz = Zb - p_pos[2]
                r2 = dx * dx + dy * dy + dz * dz
                mask = r2 <= rc2
                # Step potential:
                # V0 below cutoff, zero above, weighted by charge
                block_potential += strength * p_q * mask.astype(cp.float64)

            # Pair-wise uniform contributions if pair within cutoff
            n = len(self.particles)
            for i in range(n):
                for j in range(i + 1, n):
                    d = cp.linalg.norm(
                        particle_positions[i]
                        - particle_positions[j]
                    )
                    if float(d) < r_cut:
                        block_potential += strength

            # Three-body uniform contributions when all three
            # mutually within cutoff
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        dij = cp.linalg.norm(
                            particle_positions[i]
                            - particle_positions[j]
                        )
                        dik = cp.linalg.norm(
                            particle_positions[i]
                            - particle_positions[k]
                        )
                        djk = cp.linalg.norm(
                            particle_positions[j] - particle_positions[k]
                        )
                        if (
                            float(dij) < r_cut
                            and float(dik) < r_cut
                            and float(djk) < r_cut
                        ):
                            block_potential += strength

            # Bring block result to CPU and insert
            result[slices[0], slices[1], slices[2]] = cp.asnumpy(
                block_potential
            )

            # Periodic cleanup to respect memory budget
            if block_id % 8 == 0:
                cp.get_default_memory_pool().free_all_blocks()
            block_id += 1

        # Final cleanup of GPU memory pools
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        return result

    def _compute_optimal_block_size_7d(self) -> int:
        """
        Compute block size per dimension using ~80% of free GPU memory.

        Physical Meaning:
            Ensures that all intermediate arrays within a block fit
            into GPU memory with safety margin while maximizing throughput.

        Returns:
            int: Block size per spatial dimension (clamped to domain dims).
        """
        mem = self.backend.get_memory_info()
        free_bytes = int(mem.get("free_memory", 0))
        usable = int(free_bytes * 0.8)  # target 80%

        # Memory model per element: we keep several temporaries
        # on GPU simultaneously.
        # For single-particle contributions we use Xb, Yb, Zb (3 arrays),
        # r2 (1), mask (1), block_potential (1) → ~6 arrays; additional
        # temporaries for pair/three-body are tiny. Use a conservative
        # factor of 8 arrays of float64 (8 bytes) per element.
        arrays_per_element = 8
        bytes_per_element = 8  # float64
        budget_per_element = arrays_per_element * bytes_per_element

        # 7D domain here effectively uses 3D spatial for potential;
        # other dims are not used.
        # We compute block size for 3D spatial grid.
        max_elements = usable // budget_per_element
        if max_elements <= 0:
            return 4

        # Compute cubic block side for 3D
        side = int(max_elements ** (1.0 / 3.0))

        # Clamp to domain dimensions and reasonable bounds
        side = max(4, min(side, 256))
        side = min(
            side,
            int(self.domain.shape[0]),
            int(self.domain.shape[1]),
            int(self.domain.shape[2]),
        )

        return max(4, side)
