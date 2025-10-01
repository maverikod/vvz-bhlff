"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP core pipeline.

This module provides comprehensive integration tests for the BVP core
pipeline, ensuring physical consistency and theoretical correctness
of the core BVP components.

Physical Meaning:
    Tests validate the BVP core pipeline:
    - 7D space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    - BVP envelope equation solution
    - Energy conservation throughout pipeline
    - Physical consistency across all components

Mathematical Foundation:
    Validates the core 7D BVP theory:
    - Envelope equation: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    - Energy conservation: ∂E/∂t + ∇·S = 0

Example:
    >>> pytest tests/integration/test_bvp_core_pipeline_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_core.bvp_core import BVPCore
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


class TestBVPCorePipelinePhysics:
    """BVP core pipeline physical validation tests."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for complete pipeline testing."""
        return Domain(
            L=2.0,  # Larger domain for better physics
            N=64,   # Higher resolution
            dimensions=3,
            N_phi=32,  # More phase points
            N_t=128,   # More time points
            T=2.0      # Longer evolution
        )

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for complete pipeline testing."""
        config = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
            }
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def bvp_core(self, domain_7d, bvp_constants):
        """Create BVP core for complete pipeline testing."""
        return BVPCore(domain_7d, bvp_constants)

    def test_complete_bvp_pipeline_physics(self, domain_7d, bvp_core):
        """
        Test complete BVP pipeline physics.
        
        Physical Meaning:
            Validates the complete BVP pipeline from source to solution,
            ensuring physical consistency and theoretical correctness
            throughout the entire process.
            
        Mathematical Foundation:
            Tests the complete 7D BVP theory implementation:
            1. Source generation
            2. Envelope equation solution
            3. Energy conservation
            4. Physical consistency
        """
        # Step 1: Generate physical source
        source = self._generate_physical_source(domain_7d)
        
        # Step 2: Solve BVP envelope equation
        envelope_solution = bvp_core.solve_envelope(source)
        
        # Step 3: Compute physical quantities
        physical_quantities = self._compute_physical_quantities(envelope_solution, domain_7d)
        
        # Step 4: Validate energy conservation
        energy_conservation = self._validate_energy_conservation(envelope_solution, source, domain_7d)
        
        # Physical validation 1: Envelope solution should be physically meaningful
        assert np.all(np.isfinite(envelope_solution)), "Envelope solution contains non-finite values"
        assert np.all(np.abs(envelope_solution) < 1e6), "Envelope solution too large"
        
        # Physical validation 2: Energy should be conserved
        assert energy_conservation['conserved'], f"Energy not conserved: {energy_conservation['error']}"
        
        # Physical validation 3: Physical quantities should be reasonable
        assert physical_quantities['total_energy'] > 0, "Negative total energy"
        assert 0 <= physical_quantities['phase_coherence'] <= 1, "Phase coherence out of range"

    def test_bvp_energy_conservation_pipeline(self, domain_7d, bvp_core):
        """
        Test energy conservation throughout BVP pipeline.
        
        Physical Meaning:
            Validates that energy is conserved throughout the entire
            BVP pipeline, ensuring fundamental conservation laws.
            
        Mathematical Foundation:
            Tests energy conservation: ∂E/∂t + ∇·S = 0
            throughout the pipeline.
        """
        # Create time-evolving source
        source_evolution = self._generate_time_evolving_source(domain_7d)
        
        # Solve envelope evolution
        envelope_evolution = []
        for t in range(domain_7d.N_t):
            envelope = bvp_core.solve_envelope(source_evolution[t])
            envelope_evolution.append(envelope)
        
        # Compute energy evolution
        energy_evolution = []
        for envelope in envelope_evolution:
            energy = self._compute_total_energy(envelope, domain_7d)
            energy_evolution.append(energy)
        
        # Physical validation 1: Energy should be conserved
        initial_energy = energy_evolution[0]
        final_energy = energy_evolution[-1]
        energy_conservation_error = abs(final_energy - initial_energy) / initial_energy
        
        assert energy_conservation_error < 1e-2, \
            f"Energy not conserved in pipeline: error = {energy_conservation_error}"
        
        # Physical validation 2: Energy should be positive
        for energy in energy_evolution:
            assert energy > 0, f"Negative energy in pipeline: {energy}"
        
        # Physical validation 3: Energy should be bounded
        max_energy = max(energy_evolution)
        min_energy = min(energy_evolution)
        assert max_energy / min_energy < 100, "Energy varies too much in pipeline"

    def _generate_physical_source(self, domain: Domain) -> np.ndarray:
        """Generate a physical source for testing."""
        source = np.zeros(domain.shape)
        
        # Create localized source in center
        center = domain.N // 2
        source[center-2:center+3, center-2:center+3, center-2:center+3, 
               :, :, :, :] = 1.0
        
        return source

    def _generate_time_evolving_source(self, domain: Domain) -> List[np.ndarray]:
        """Generate time-evolving source for energy conservation test."""
        sources = []
        for t in range(domain.N_t):
            source = np.zeros(domain.shape)
            # Moving source
            center = domain.N // 2 + int(5 * np.sin(2 * np.pi * t / domain.N_t))
            if 0 <= center < domain.N:
                source[center, center, center, :, :, :, t] = 1.0
            sources.append(source)
        return sources

    def _compute_physical_quantities(self, envelope: np.ndarray, domain: Domain) -> Dict[str, Any]:
        """Compute physical quantities from envelope."""
        # Compute total energy
        total_energy = self._compute_total_energy(envelope, domain)
        
        # Compute gradient magnitude
        gradient_magnitude = self._compute_gradient_magnitude(envelope, domain)
        
        # Compute phase coherence
        phase_coherence = self._compute_phase_coherence(envelope, domain)
        
        return {
            'total_energy': total_energy,
            'gradient_magnitude': gradient_magnitude,
            'phase_coherence': phase_coherence
        }

    def _validate_energy_conservation(self, envelope: np.ndarray, source: np.ndarray, 
                                    domain: Domain) -> Dict[str, Any]:
        """Validate energy conservation."""
        # Compute total energy
        total_energy = self._compute_total_energy(envelope, domain)
        
        # Compute source energy
        source_energy = np.sum(np.abs(source)**2)
        
        # Energy should be conserved (within numerical precision)
        energy_error = abs(total_energy - source_energy) / source_energy
        
        return {
            'conserved': energy_error < 1e-2,
            'error': energy_error,
            'total_energy': total_energy,
            'source_energy': source_energy
        }

    def _compute_total_energy(self, envelope: np.ndarray, domain: Domain) -> float:
        """Compute total energy of the envelope."""
        # Compute gradient energy
        grad_x = np.gradient(envelope, axis=0)
        grad_y = np.gradient(envelope, axis=1)
        grad_z = np.gradient(envelope, axis=2)
        
        gradient_energy = np.sum(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Compute potential energy
        potential_energy = np.sum(envelope**2)
        
        return gradient_energy + potential_energy

    def _compute_gradient_magnitude(self, envelope: np.ndarray, domain: Domain) -> np.ndarray:
        """Compute gradient magnitude."""
        grad_x = np.gradient(envelope, axis=0)
        grad_y = np.gradient(envelope, axis=1)
        grad_z = np.gradient(envelope, axis=2)
        
        return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    def _compute_phase_coherence(self, envelope: np.ndarray, domain: Domain) -> float:
        """Compute phase coherence."""
        # Compute phase
        phase = np.angle(envelope)
        
        # Compute phase coherence (simplified)
        phase_variance = np.var(phase)
        coherence = np.exp(-phase_variance)
        
        return coherence
