"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP postulates.

This module provides comprehensive physical validation tests for all 9 BVP
postulates, ensuring they correctly implement the theoretical foundations
of the Base High-Frequency Field in 7D space-time.

Physical Meaning:
    Tests validate that each BVP postulate correctly implements its
    specific physical property:
    1. Carrier Primacy - high-frequency carrier dominance
    2. Scale Separation - separation between carrier and envelope
    3. BVP Rigidity - field stability and coherence
    4. U(1)³ Phase Structure - phase coherence and topology
    5. Quenches - phase transition dynamics
    6. Tail Resonatorness - resonance properties
    7. Transition Zone - nonlinear interface behavior
    8. Core Renormalization - renormalization effects
    9. Power Balance - energy conservation

Mathematical Foundation:
    Each postulate implements specific mathematical conditions that
    must be satisfied for physical consistency of the BVP theory.

Example:
    >>> pytest tests/unit/test_core/test_bvp_postulates_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.postulates.carrier_primacy_postulate import BVPPostulate1_CarrierPrimacy
from bhlff.core.bvp.postulates.scale_separation_postulate import BVPPostulate2_ScaleSeparation
from bhlff.core.bvp.postulates.bvp_rigidity_postulate import BVPPostulate3_BVPRigidity
from bhlff.core.bvp.postulates.u1_phase_structure_postulate import BVPPostulate4_U1PhaseStructure
from bhlff.core.bvp.postulates.quenches_postulate import BVPPostulate5_Quenches
from bhlff.core.bvp.postulates.tail_resonatorness_postulate import BVPPostulate6_TailResonatorness
from bhlff.core.bvp.postulates.transition_zone_postulate import BVPPostulate7_TransitionZone
from bhlff.core.bvp.postulates.core_renormalization_postulate import BVPPostulate8_CoreRenormalization
from bhlff.core.bvp.postulates.power_balance.power_balance_postulate import PowerBalancePostulate


class TestBVPPostulatesPhysics:
    """Physical validation tests for BVP postulates."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for postulate testing."""
        return Domain(
            L=1.0,
            N=8,
            dimensions=3,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for postulate testing."""
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
    def test_envelope(self, domain_7d):
        """Create test envelope for postulate validation."""
        envelope = np.zeros(domain_7d.shape)
        
        # Create envelope with known properties
        center = domain_7d.N // 2
        envelope[center-4:center+5, center-4:center+5, center-4:center+5,
                :, :, :, :] = 1.0
        
        # Add phase structure
        phi1 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
        phi2 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
        phi3 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
        
        PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing='ij')
        phase_factor = np.exp(1j * (PHI1 + PHI2 + PHI3))
        
        envelope = envelope * phase_factor
        
        return envelope

    def test_carrier_primacy_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Carrier Primacy Postulate physics.
        
        Physical Meaning:
            Validates that the high-frequency carrier dominates the field
            structure, ensuring the BVP is truly a high-frequency field
            with envelope modulation.
            
        Mathematical Foundation:
            Tests that |a_carrier| >> |a_envelope| where a_carrier is the
            high-frequency component and a_envelope is the slow modulation.
        """
        postulate = BVPPostulate1_CarrierPrimacy(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "Carrier Primacy postulate not satisfied"
        
        # Physical validation 2: Carrier dominance ratio should be > 1
        carrier_dominance = result['carrier_dominance_ratio']
        assert carrier_dominance > 1.0, f"Carrier not dominant: ratio = {carrier_dominance}"
        
        # Physical validation 3: Carrier frequency should be high
        carrier_frequency = result['carrier_frequency']
        assert carrier_frequency > 1.0, f"Carrier frequency too low: {carrier_frequency}"
        
        # Physical validation 4: Envelope should be smooth
        envelope_smoothness = result['envelope_smoothness']
        assert envelope_smoothness > 0.5, f"Envelope not smooth: {envelope_smoothness}"

    def test_scale_separation_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Scale Separation Postulate physics.
        
        Physical Meaning:
            Validates that there is clear separation between the carrier
            scale (high-frequency) and envelope scale (low-frequency),
            ensuring the BVP approximation is valid.
            
        Mathematical Foundation:
            Tests that λ_carrier << λ_envelope where λ are characteristic
            wavelengths of carrier and envelope components.
        """
        postulate = BVPPostulate2_ScaleSeparation(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "Scale Separation postulate not satisfied"
        
        # Physical validation 2: Scale separation ratio should be > 1
        scale_ratio = result['scale_separation_ratio']
        assert scale_ratio > 1.0, f"Scale separation insufficient: ratio = {scale_ratio}"
        
        # Physical validation 3: Carrier wavelength should be small
        carrier_wavelength = result['carrier_wavelength']
        assert carrier_wavelength < 1.0, f"Carrier wavelength too large: {carrier_wavelength}"
        
        # Physical validation 4: Envelope wavelength should be large
        envelope_wavelength = result['envelope_wavelength']
        assert envelope_wavelength > 1.0, f"Envelope wavelength too small: {envelope_wavelength}"

    def test_bvp_rigidity_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test BVP Rigidity Postulate physics.
        
        Physical Meaning:
            Validates that the BVP field maintains its structure and
            coherence under perturbations, ensuring field stability.
            
        Mathematical Foundation:
            Tests that the field remains coherent under small perturbations
            and maintains its topological properties.
        """
        postulate = BVPPostulate3_BVPRigidity(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "BVP Rigidity postulate not satisfied"
        
        # Physical validation 2: Rigidity coefficient should be > 0
        rigidity_coefficient = result['rigidity_coefficient']
        assert rigidity_coefficient > 0, f"Negative rigidity: {rigidity_coefficient}"
        
        # Physical validation 3: Coherence should be maintained
        coherence = result['coherence']
        assert coherence > 0.5, f"Low coherence: {coherence}"
        
        # Physical validation 4: Stability should be positive
        stability = result['stability']
        assert stability > 0, f"Unstable field: {stability}"

    def test_u1_phase_structure_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test U(1)³ Phase Structure Postulate physics.
        
        Physical Meaning:
            Validates that the field has proper U(1)³ phase structure
            with correct phase coherence and topological properties.
            
        Mathematical Foundation:
            Tests that a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃) with proper phase
            coherence and quantized topological charge.
        """
        postulate = BVPPostulate4_U1PhaseStructure(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "U(1)³ Phase Structure postulate not satisfied"
        
        # Physical validation 2: Phase coherence should be high
        phase_coherence = result['phase_coherence']
        assert phase_coherence > 0.7, f"Low phase coherence: {phase_coherence}"
        
        # Physical validation 3: Topological charge should be quantized
        topological_charge = result['topological_charge']
        assert np.isclose(topological_charge, np.round(topological_charge), atol=1e-6), \
            f"Topological charge not quantized: {topological_charge}"
        
        # Physical validation 4: Phase winding should be consistent
        phase_winding = result['phase_winding']
        assert np.all(np.isfinite(phase_winding)), "Phase winding contains non-finite values"

    def test_quenches_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Quenches Postulate physics.
        
        Physical Meaning:
            Validates that quench detection correctly identifies phase
            transition regions where the field gradient exceeds threshold.
            
        Mathematical Foundation:
            Tests quench condition: |∇a|² > threshold and validates
            quench dynamics and memory effects.
        """
        postulate = BVPPostulate5_Quenches(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "Quenches postulate not satisfied"
        
        # Physical validation 2: Quench fraction should be reasonable
        quench_fraction = result['quench_fraction']
        assert 0 <= quench_fraction <= 1, f"Quench fraction out of range: {quench_fraction}"
        
        # Physical validation 3: Quench intensity should be positive
        quench_intensity = result['quench_intensity']
        assert quench_intensity >= 0, f"Negative quench intensity: {quench_intensity}"
        
        # Physical validation 4: Memory effects should be present
        memory_effects = result['memory_effects']
        assert memory_effects >= 0, f"Negative memory effects: {memory_effects}"

    def test_tail_resonatorness_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Tail Resonatorness Postulate physics.
        
        Physical Meaning:
            Validates that the field tail exhibits resonator properties
            with proper resonance frequencies and quality factors.
            
        Mathematical Foundation:
            Tests resonance condition: ω = ω₀ ± Δω with quality factor Q
            and validates resonance properties.
        """
        postulate = BVPPostulate6_TailResonatorness(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "Tail Resonatorness postulate not satisfied"
        
        # Physical validation 2: Quality factor should be > 1
        quality_factor = result['quality_factor']
        assert quality_factor > 1.0, f"Low quality factor: {quality_factor}"
        
        # Physical validation 3: Resonance frequency should be positive
        resonance_frequency = result['resonance_frequency']
        assert resonance_frequency > 0, f"Negative resonance frequency: {resonance_frequency}"
        
        # Physical validation 4: Resonance width should be reasonable
        resonance_width = result['resonance_width']
        assert resonance_width > 0, f"Zero resonance width: {resonance_width}"

    def test_transition_zone_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Transition Zone Postulate physics.
        
        Physical Meaning:
            Validates that the transition zone between different field
            regions exhibits proper nonlinear interface behavior.
            
        Mathematical Foundation:
            Tests nonlinear interface equations and validates transition
            zone dynamics.
        """
        postulate = BVPPostulate7_TransitionZone(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "Transition Zone postulate not satisfied"
        
        # Physical validation 2: Interface thickness should be positive
        interface_thickness = result['interface_thickness']
        assert interface_thickness > 0, f"Zero interface thickness: {interface_thickness}"
        
        # Physical validation 3: Transition energy should be positive
        transition_energy = result['transition_energy']
        assert transition_energy >= 0, f"Negative transition energy: {transition_energy}"
        
        # Physical validation 4: Nonlinear effects should be present
        nonlinear_strength = result['nonlinear_strength']
        assert nonlinear_strength >= 0, f"Negative nonlinear strength: {nonlinear_strength}"

    def test_core_renormalization_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Core Renormalization Postulate physics.
        
        Physical Meaning:
            Validates that renormalization effects in the field core
            are properly accounted for, ensuring physical consistency.
            
        Mathematical Foundation:
            Tests renormalization group flow and validates renormalized
            parameters.
        """
        postulate = BVPPostulate8_CoreRenormalization(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "Core Renormalization postulate not satisfied"
        
        # Physical validation 2: Renormalization flow should be convergent
        convergence = result['convergence']
        assert convergence > 0, f"Non-convergent renormalization: {convergence}"
        
        # Physical validation 3: Renormalized parameters should be finite
        renormalized_params = result['renormalized_parameters']
        assert np.all(np.isfinite(renormalized_params)), "Non-finite renormalized parameters"
        
        # Physical validation 4: Beta function should be well-defined
        beta_function = result['beta_function']
        assert np.all(np.isfinite(beta_function)), "Non-finite beta function"

    def test_power_balance_postulate_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test Power Balance Postulate physics.
        
        Physical Meaning:
            Validates that energy is conserved in the BVP system,
            ensuring the fundamental conservation law is satisfied.
            
        Mathematical Foundation:
            Tests energy conservation: ∂E/∂t + ∇·S = 0 and validates
            power balance at boundaries.
        """
        postulate = PowerBalancePostulate(domain_7d, bvp_constants)
        
        # Apply postulate
        result = postulate.apply(test_envelope)
        
        # Physical validation 1: Postulate should be satisfied
        assert result['satisfied'], "Power Balance postulate not satisfied"
        
        # Physical validation 2: Energy should be conserved
        energy_conservation_error = result['energy_conservation_error']
        assert energy_conservation_error < 1e-3, \
            f"Energy not conserved: error = {energy_conservation_error}"
        
        # Physical validation 3: Power flux should be balanced
        power_flux_balance = result['power_flux_balance']
        assert power_flux_balance > 0.9, f"Power flux not balanced: {power_flux_balance}"
        
        # Physical validation 4: Total energy should be positive
        total_energy = result['total_energy']
        assert total_energy > 0, f"Negative total energy: {total_energy}"

    def test_all_postulates_integration_physics(self, domain_7d, bvp_constants, test_envelope):
        """
        Test integration of all BVP postulates.
        
        Physical Meaning:
            Validates that all 9 BVP postulates work together to ensure
            complete physical consistency of the BVP theory.
            
        Mathematical Foundation:
            Tests that all postulates are satisfied simultaneously,
            ensuring the complete BVP framework is physically consistent.
        """
        # Create all postulates
        postulates = [
            BVPPostulate1_CarrierPrimacy(domain_7d, bvp_constants),
            BVPPostulate2_ScaleSeparation(domain_7d, bvp_constants),
            BVPPostulate3_BVPRigidity(domain_7d, bvp_constants),
            BVPPostulate4_U1PhaseStructure(domain_7d, bvp_constants),
            BVPPostulate5_Quenches(domain_7d, bvp_constants),
            BVPPostulate6_TailResonatorness(domain_7d, bvp_constants),
            BVPPostulate7_TransitionZone(domain_7d, bvp_constants),
            BVPPostulate8_CoreRenormalization(domain_7d, bvp_constants),
            PowerBalancePostulate(domain_7d, bvp_constants)
        ]
        
        # Apply all postulates
        results = []
        for postulate in postulates:
            result = postulate.apply(test_envelope)
            results.append(result)
        
        # Physical validation 1: All postulates should be satisfied
        satisfied_count = sum(1 for result in results if result['satisfied'])
        assert satisfied_count == len(postulates), \
            f"Only {satisfied_count}/{len(postulates)} postulates satisfied"
        
        # Physical validation 2: Overall consistency should be high
        overall_consistency = satisfied_count / len(postulates)
        assert overall_consistency > 0.8, f"Low overall consistency: {overall_consistency}"
        
        # Physical validation 3: No contradictory results
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results):
                if i != j:
                    # Check for contradictions in key parameters
                    if 'energy' in result1 and 'energy' in result2:
                        energy_ratio = result1['energy'] / result2['energy']
                        assert 0.1 < energy_ratio < 10.0, \
                            f"Contradictory energy values: {result1['energy']} vs {result2['energy']}"
        
        # Physical validation 4: Physical parameters should be consistent
        self._validate_physical_consistency(results)

    def _validate_physical_consistency(self, results: List[Dict[str, Any]]) -> None:
        """Validate physical consistency across all postulate results."""
        # Extract key physical parameters
        energies = [r.get('energy', 0) for r in results if 'energy' in r]
        frequencies = [r.get('frequency', 0) for r in results if 'frequency' in r]
        coherence_values = [r.get('coherence', 0) for r in results if 'coherence' in r]
        
        # Check energy consistency
        if energies:
            energy_variance = np.var(energies) / np.mean(energies)**2
            assert energy_variance < 0.1, f"High energy variance: {energy_variance}"
        
        # Check frequency consistency
        if frequencies:
            frequency_variance = np.var(frequencies) / np.mean(frequencies)**2
            assert frequency_variance < 0.1, f"High frequency variance: {frequency_variance}"
        
        # Check coherence consistency
        if coherence_values:
            coherence_variance = np.var(coherence_values)
            assert coherence_variance < 0.1, f"High coherence variance: {coherence_variance}"
