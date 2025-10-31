"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Multi-particle system implementation for Level F collective effects.

This module provides a facade for multi-particle system functionality
for Level F models in 7D phase field theory, ensuring proper functionality
of all multi-particle analysis components.

Theoretical Background:
    Multi-particle systems in 7D phase field theory are described by
    effective potentials that include single-particle, pair-wise, and
    higher-order interactions:
    U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ + ...

    Collective modes arise from the diagonalization of the dynamics matrix
    E⁻¹K, where E is the energy matrix and K is the stiffness matrix.

Example:
    >>> particles = [Particle(position=[5,10,10], charge=1, phase=0),
    ...              Particle(position=[15,10,10], charge=-1, phase=π)]
    >>> system = MultiParticleSystem(domain, particles)
    >>> potential = system.compute_effective_potential()
    >>> modes = system.find_collective_modes()
"""

import numpy as np
from typing import List, Dict, Any, Optional, cast
import os
from ..base.abstract_model import AbstractModel
from .multi_particle.data_structures import Particle, SystemParameters
from .multi_particle_potential import MultiParticlePotentialAnalyzer
from .multi_particle_modes import MultiParticleModesAnalyzer
from .multi_particle_analysis import MultiParticleSystemAnalyzer


class MultiParticleSystem(AbstractModel):
    """
    Multi-particle system for Level F collective effects.

    Physical Meaning:
        Studies collective effects in systems with multiple
        topological defects, including effective potential
        calculations and collective mode analysis.

    Mathematical Foundation:
        Implements multi-particle system analysis:
        - Effective potential: U_eff = Σᵢ Uᵢ + Σᵢ<ⱼ Uᵢⱼ + Σᵢ<ⱼ<ₖ Uᵢⱼₖ
        - Collective modes: diagonalization of M⁻¹K
        - Correlation functions: G(x,t) = ⟨ψ*(x,t)ψ(0,0)⟩
    """

    def __init__(
        self,
        domain: Any,
        particles: List[Particle],
        interaction_range: float = 2.0,
        system_params: Optional[SystemParameters] = None,
        use_cuda: bool = True,
    ):
        """
        Initialize multi-particle system.

        Physical Meaning:
            Sets up the multi-particle system with particles and
            interaction parameters for collective effects analysis.

        Args:
            domain: Domain parameters.
            particles (List[Particle]): List of particles.
            interaction_range (float): Interaction range parameter.
            system_params (Optional[SystemParameters]): System parameters.
        """
        super().__init__(domain)
        self.domain = domain
        self.particles = particles
        self.interaction_range = interaction_range
        self.system_params = system_params or SystemParameters()
        self._use_cuda = bool(use_cuda)

        # Initialize analysis components
        self._potential_analyzer = MultiParticlePotentialAnalyzer(
            domain, particles, interaction_range
        )
        self._modes_analyzer = MultiParticleModesAnalyzer(
            domain, particles, interaction_range
        )
        self._system_analyzer = MultiParticleSystemAnalyzer(
            domain, particles, interaction_range
        )

        # Optional CUDA analyzer (lazy import to respect environments
        # without CUDA)
        self._potential_analyzer_cuda = None
        if self._use_cuda:
            try:
                import cupy as _cp  # noqa: F401
                from .cuda.multi_particle_potential_cuda import (
                    MultiParticlePotentialAnalyzerCUDA,
                )

                # Optional CUDA tuning via environment (minimal impact path)
                cuda_params: Dict[str, Any] = {}
                dev = os.getenv("BHLFF_DEVICE_ID")
                prec = os.getenv("BHLFF_PRECISION")
                memf = os.getenv("BHLFF_MEMORY_FRACTION")
                if dev is not None:
                    try:
                        cuda_params["device_id"] = int(dev)
                    except Exception:
                        pass
                if prec is not None:
                    cuda_params["precision"] = prec
                if memf is not None:
                    try:
                        cuda_params["memory_fraction"] = float(memf)
                    except Exception:
                        pass

                self._potential_analyzer_cuda = MultiParticlePotentialAnalyzerCUDA(
                    domain,
                    particles,
                    interaction_range=interaction_range,
                    params=cuda_params,
                    system_params=self.system_params,
                )
            except Exception:
                self._potential_analyzer_cuda = None

    def compute_effective_potential(self) -> np.ndarray:
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
        if self._potential_analyzer_cuda is not None:
            return self._potential_analyzer_cuda.compute_effective_potential()
        return self._potential_analyzer.compute_effective_potential()

    def find_collective_modes(self) -> Dict[str, Any]:
        """
        Find collective modes.

        Physical Meaning:
            Finds collective modes in multi-particle system
            through diagonalization of dynamics matrix.

        Mathematical Foundation:
            Collective modes: diagonalization of E⁻¹K
            where E is the energy matrix and K is the stiffness matrix.

        Returns:
            Dict[str, Any]: Collective modes analysis results.
        """
        return cast(Dict[str, Any], self._modes_analyzer.find_collective_modes())

    def compute_correlation_function(
        self, field: np.ndarray, time_points: np.ndarray
    ) -> np.ndarray:
        """
        Compute correlation function.

        Physical Meaning:
            Computes correlation function for multi-particle system
            to analyze collective behavior.

        Mathematical Foundation:
            Correlation function: G(x,t) = ⟨ψ*(x,t)ψ(0,0)⟩

        Args:
            field (np.ndarray): Field configuration.
            time_points (np.ndarray): Time points for correlation.

        Returns:
            np.ndarray: Correlation function.
        """
        return self._modes_analyzer.compute_correlation_function(field, time_points)

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze multi-particle system properties.

        Physical Meaning:
            Provides a concise analysis bundle for the multi-particle system,
            including effective potential and collective modes. If a field and
            time grid are provided in `data`, computes correlation function.

        Args:
            data (Any): Optional dict with keys 'field' (np.ndarray) and
                'time_points' (np.ndarray) to compute correlations.

        Returns:
            Dict[str, Any]: Combined analysis results.
        """
        self.log_analysis_start("multi_particle_system")

        results: Dict[str, Any] = {}
        # Effective potential (CUDA path used automatically if available)
        results["effective_potential"] = self.compute_effective_potential()

        # Collective modes
        results["collective_modes"] = self.find_collective_modes()

        # Optional correlation function
        if isinstance(data, dict) and "field" in data and "time_points" in data:
            try:
                results["correlation_function"] = self.compute_correlation_function(
                    data["field"], data["time_points"]
                )
            except Exception as exc:
                results["correlation_error"] = str(exc)

        self.log_analysis_complete("multi_particle_system", results)
        return results

    # --- Internal utilities ---
    def _compute_dynamics_matrix(self) -> np.ndarray:
        """
        Construct a simple symmetric positive-definite dynamics matrix.

        Physical Meaning:
            Provides a minimal stiffness-like matrix for linear response
            in excitation analysis. Diagonal terms represent self-stiffness;
            off-diagonal couplings are small for close pairs.

        Returns:
            np.ndarray: (n_particles, n_particles) SPD matrix.
        """
        n = len(self.particles)
        K = np.eye(n, dtype=float) * 2.0
        if n <= 1:
            return K
        for i in range(n):
            for j in range(i + 1, n):
                d = float(
                    np.linalg.norm(
                        self.particles[i].position - self.particles[j].position
                    )
                )
                if d < self.interaction_range:
                    K[i, j] = K[j, i] = 0.1
        return K

    def analyze_system_properties(self) -> Dict[str, Any]:
        """
        Analyze system properties.

        Physical Meaning:
            Analyzes system properties including energy, stability,
            and optimization for multi-particle system.

        Mathematical Foundation:
            Analyzes system properties through:
            - System energy: E = ∫ U_eff(x) dx
            - System stability: analysis of collective modes
            - System optimization: parameter optimization

        Returns:
            Dict[str, Any]: System properties analysis results.
        """
        return cast(Dict[str, Any], self._system_analyzer.analyze_system_properties())
