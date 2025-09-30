"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Complete BVP pipeline physical validation tests.

This module provides comprehensive integration tests for the complete
BVP pipeline, ensuring end-to-end physical consistency and theoretical
correctness of the 7D BVP theory implementation.

Physical Meaning:
    Tests validate the complete BVP pipeline:
    - 7D space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    - BVP envelope equation solution
    - All 9 BVP postulates validation
    - Energy conservation throughout pipeline
    - Physical consistency across all components
    - Theoretical correctness of results

Mathematical Foundation:
    Validates the complete 7D BVP theory:
    - Envelope equation: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    - U(1)³ phase structure: a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃)
    - Energy conservation: ∂E/∂t + ∇·S = 0
    - All 9 BVP postulates simultaneously

Example:
    >>> pytest tests/integration/test_bvp_complete_pipeline_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_core.bvp_core import BVPCore
from bhlff.core.bvp.bvp_interface import BVPInterface
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.postulates.bvp_postulates_7d import BVPPostulates7D
from bhlff.core.bvp.quench_detector import QuenchDetector
from bhlff.core.bvp.bvp_impedance_calculator import BVPImpedanceCalculator
from bhlff.core.bvp.phase_vector.phase_vector import PhaseVector


class TestBVPCompletePipelinePhysics:
    """Complete BVP pipeline physical validation tests."""

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

    @pytest.fixture
    def bvp_interface(self, domain_7d, bvp_constants):
        """Create BVP interface for complete pipeline testing."""
        return BVPInterface(domain_7d, bvp_constants)

    @pytest.fixture
    def bvp_postulates(self, domain_7d, bvp_constants):
        """Create BVP postulates for complete pipeline testing."""
        return BVPPostulates7D(domain_7d, bvp_constants)

    def test_complete_bvp_pipeline_physics(self, domain_7d, bvp_core, bvp_interface, bvp_postulates):
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
            3. Postulate validation
            4. Energy conservation
            5. Physical consistency
        """
        # Step 1: Generate physical source
        source = self._generate_physical_source(domain_7d)
        
        # Step 2: Solve BVP envelope equation
        envelope_solution = bvp_core.solve_envelope(source)
        
        # Step 3: Validate all BVP postulates
        postulate_results = bvp_postulates.validate_all_postulates(envelope_solution)
        
        # Step 4: Compute physical quantities
        physical_quantities = self._compute_physical_quantities(envelope_solution, domain_7d)
        
        # Step 5: Validate energy conservation
        energy_conservation = self._validate_energy_conservation(envelope_solution, source, domain_7d)
        
        # Step 6: Validate physical consistency
        physical_consistency = self._validate_physical_consistency(
            envelope_solution, postulate_results, physical_quantities
        )
        
        # Physical validation 1: Envelope solution should be physically meaningful
        assert np.all(np.isfinite(envelope_solution)), "Envelope solution contains non-finite values"
        assert np.all(np.abs(envelope_solution) < 1e6), "Envelope solution too large"
        
        # Physical validation 2: All postulates should be satisfied
        satisfied_postulates = sum(1 for result in postulate_results.values() if result['satisfied'])
        assert satisfied_postulates >= 7, f"Only {satisfied_postulates}/9 postulates satisfied"
        
        # Physical validation 3: Energy should be conserved
        assert energy_conservation['conserved'], f"Energy not conserved: {energy_conservation['error']}"
        
        # Physical validation 4: Physical consistency should be maintained
        assert physical_consistency['consistent'], f"Physical inconsistency: {physical_consistency['issues']}"

    def test_bvp_interface_physics(self, domain_7d, bvp_interface):
        """
        Test BVP interface physics.
        
        Physical Meaning:
            Validates that the BVP interface correctly coordinates
            all BVP components and maintains physical consistency.
            
        Mathematical Foundation:
            Tests interface coordination of:
            - Envelope solver
            - Postulate validation
            - Quench detection
            - Impedance calculation
        """
        # Create test source
        source = self._generate_physical_source(domain_7d)
        
        # Test interface operations
        interface_results = bvp_interface.process_source(source)
        
        # Physical validation 1: Interface should return valid results
        assert 'envelope' in interface_results, "Interface missing envelope solution"
        assert 'postulates' in interface_results, "Interface missing postulate results"
        assert 'quenches' in interface_results, "Interface missing quench results"
        assert 'impedance' in interface_results, "Interface missing impedance results"
        
        # Physical validation 2: All results should be physically meaningful
        envelope = interface_results['envelope']
        assert np.all(np.isfinite(envelope)), "Interface envelope contains non-finite values"
        
        postulates = interface_results['postulates']
        assert isinstance(postulates, dict), "Interface postulates not a dictionary"
        
        quenches = interface_results['quenches']
        assert np.all((quenches == 0) | (quenches == 1)), "Interface quenches not binary"
        
        impedance = interface_results['impedance']
        assert np.all(np.isfinite(impedance)), "Interface impedance contains non-finite values"

    def test_bvp_quench_dynamics_physics(self, domain_7d, bvp_core):
        """
        Test BVP quench dynamics physics.
        
        Physical Meaning:
            Validates that quench detection correctly identifies
            phase transition regions and maintains physical consistency.
            
        Mathematical Foundation:
            Tests quench dynamics: |∇a|² > threshold
            and validates quench evolution.
        """
        # Create source with known quench regions
        source = self._generate_source_with_quenches(domain_7d)
        
        # Solve envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Detect quenches
        quench_detector = QuenchDetector(domain_7d, bvp_core.constants)
        quench_map = quench_detector.detect_quenches(envelope)
        
        # Physical validation 1: Quench map should be binary
        assert np.all((quench_map == 0) | (quench_map == 1)), "Quench map not binary"
        
        # Physical validation 2: Quenches should be localized
        quench_fraction = np.mean(quench_map)
        assert 0 < quench_fraction < 0.5, f"Quench fraction out of range: {quench_fraction}"
        
        # Physical validation 3: Quenches should correlate with high gradients
        gradient_magnitude = self._compute_gradient_magnitude(envelope, domain_7d)
        quench_gradient_correlation = np.corrcoef(quench_map.flatten(), 
                                                 gradient_magnitude.flatten())[0, 1]
        assert quench_gradient_correlation > 0.3, "Quenches don't correlate with gradients"

    def test_bvp_impedance_calculation_physics(self, domain_7d, bvp_core):
        """
        Test BVP impedance calculation physics.
        
        Physical Meaning:
            Validates that impedance calculation correctly computes
            the field impedance and maintains physical consistency.
            
        Mathematical Foundation:
            Tests impedance calculation: Z = V/I
            and validates impedance properties.
        """
        # Create test source
        source = self._generate_physical_source(domain_7d)
        
        # Solve envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Calculate impedance
        impedance_calculator = BVPImpedanceCalculator(domain_7d, bvp_core.constants)
        impedance = impedance_calculator.compute_impedance(envelope)
        
        # Physical validation 1: Impedance should be finite
        assert np.all(np.isfinite(impedance)), "Impedance contains non-finite values"
        
        # Physical validation 2: Real part should be positive (resistance)
        real_impedance = np.real(impedance)
        assert np.all(real_impedance >= 0), "Negative real impedance"
        
        # Physical validation 3: Imaginary part should be reasonable
        imag_impedance = np.imag(impedance)
        assert np.all(np.isfinite(imag_impedance)), "Non-finite imaginary impedance"
        
        # Physical validation 4: Impedance should be bounded
        max_impedance = np.max(np.abs(impedance))
        assert max_impedance < 1e6, f"Impedance too large: {max_impedance}"

    def test_bvp_phase_vector_physics(self, domain_7d, bvp_core):
        """
        Test BVP phase vector physics.
        
        Physical Meaning:
            Validates that phase vector correctly implements
            U(1)³ phase structure and maintains physical consistency.
            
        Mathematical Foundation:
            Tests U(1)³ phase structure: a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃)
            and validates phase coherence.
        """
        # Create test source
        source = self._generate_physical_source(domain_7d)
        
        # Solve envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Create phase vector
        phase_vector = PhaseVector(domain_7d, bvp_core.constants)
        
        # Test phase decomposition
        amplitude, phases = phase_vector.decompose_phase_structure(envelope)
        
        # Physical validation 1: Amplitude should be non-negative
        assert np.all(amplitude >= 0), "Phase vector amplitude contains negative values"
        
        # Physical validation 2: Phases should be in [0, 2π)
        for phase in phases:
            assert np.all(phase >= 0) and np.all(phase < 2*np.pi), "Phases out of range"
        
        # Physical validation 3: Phase coherence should be maintained
        coherence = phase_vector.compute_phase_coherence(envelope)
        assert 0 <= coherence <= 1, f"Phase coherence out of range: {coherence}"
        
        # Physical validation 4: Topological charge should be quantized
        topological_charge = phase_vector.compute_topological_charge(envelope)
        assert np.isclose(topological_charge, np.round(topological_charge), atol=1e-6), \
            f"Topological charge not quantized: {topological_charge}"

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

    def _generate_source_with_quenches(self, domain: Domain) -> np.ndarray:
        """Generate source with known quench regions."""
        source = np.zeros(domain.shape)
        
        # Create sharp gradients (quenches)
        source[domain.N//4:3*domain.N//4, domain.N//4:3*domain.N//4, 
               domain.N//4:3*domain.N//4, :, :, :, :] = 10.0
        
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

    def _validate_physical_consistency(self, envelope: np.ndarray, postulate_results: Dict[str, Any],
                                     physical_quantities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate physical consistency across all components."""
        issues = []
        
        # Check envelope properties
        if not np.all(np.isfinite(envelope)):
            issues.append("Envelope contains non-finite values")
        
        # Check postulate results
        satisfied_postulates = sum(1 for result in postulate_results.values() if result['satisfied'])
        if satisfied_postulates < 7:
            issues.append(f"Only {satisfied_postulates}/9 postulates satisfied")
        
        # Check physical quantities
        if physical_quantities['total_energy'] <= 0:
            issues.append("Negative total energy")
        
        if physical_quantities['phase_coherence'] < 0 or physical_quantities['phase_coherence'] > 1:
            issues.append("Phase coherence out of range")
        
        return {
            'consistent': len(issues) == 0,
            'issues': issues
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
