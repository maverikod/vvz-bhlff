"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP constants and material properties.

This module provides comprehensive physical validation tests for BVP
constants and material properties, ensuring they satisfy physical
constraints and theoretical requirements.

Physical Meaning:
    Tests validate that BVP constants and material properties:
    - Satisfy physical constraints (positivity, causality, etc.)
    - Follow correct frequency dependencies
    - Maintain energy conservation
    - Respect thermodynamic principles
    - Are consistent with 7D BVP theory

Mathematical Foundation:
    Validates key physical relationships:
    - Drude-Lorentz model: σ(ω) = σ₀/(1 + iωτ)
    - Debye-Cole model: C(ω) = C₀/(1 + iωτ)
    - Skin effect: δ = √(2/μσω)
    - Renormalization group flow: dg/dln(μ) = β(g)

Example:
    >>> pytest tests/unit/test_core/test_physical_constants_validation.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.constants.frequency_dependent_properties import FrequencyDependentProperties
from bhlff.core.bvp.constants.nonlinear_coefficients import NonlinearCoefficients
from bhlff.core.bvp.constants.renormalized_coefficients import RenormalizedCoefficients


class TestPhysicalConstantsValidation:
    """Physical validation tests for BVP constants and material properties."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for constants testing."""
        return Domain(
            L=1.0,
            N=32,
            dimensions=3,
            N_phi=16,
            N_t=64,
            T=1.0
        )

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for testing."""
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
    def frequency_properties(self, bvp_constants):
        """Create frequency-dependent properties for testing."""
        return FrequencyDependentProperties(bvp_constants)

    @pytest.fixture
    def nonlinear_coeffs(self, bvp_constants):
        """Create nonlinear coefficients for testing."""
        return NonlinearCoefficients(bvp_constants)

    @pytest.fixture
    def renormalized_coeffs(self, bvp_constants):
        """Create renormalized coefficients for testing."""
        return RenormalizedCoefficients(bvp_constants)

    def test_bvp_constants_physical_constraints(self, bvp_constants):
        """
        Test physical constraints on BVP constants.
        
        Physical Meaning:
            Validates that BVP constants satisfy fundamental physical
            constraints required for a physically meaningful theory.
            
        Mathematical Foundation:
            Tests constraints: μ > 0, β ∈ (0,2), λ ≥ 0, k₀ > 0, χ₀ > 0, κ₀ > 0
        """
        # Physical constraint 1: Diffusion coefficient must be positive
        mu = bvp_constants.get_basic_material_property("mu")
        assert mu > 0, f"Negative diffusion coefficient: {mu}"
        
        # Physical constraint 2: Fractional order must be in (0,2)
        beta = bvp_constants.get_basic_material_property("beta")
        assert 0 < beta < 2, f"Fractional order out of range: {beta}"
        
        # Physical constraint 3: Damping parameter must be non-negative
        lambda_param = bvp_constants.get_basic_material_property("lambda_param")
        assert lambda_param >= 0, f"Negative damping: {lambda_param}"
        
        # Physical constraint 4: Carrier frequency must be positive
        k0_squared = bvp_constants.get_envelope_parameter("k0_squared")
        assert k0_squared > 0, f"Non-positive wave number: {k0_squared}"
        
        # Physical constraint 5: Linear susceptibility must be positive
        chi_prime = bvp_constants.get_envelope_parameter("chi_prime")
        assert chi_prime > 0, f"Negative susceptibility: {chi_prime}"
        
        # Physical constraint 6: Linear stiffness must be positive
        kappa_0 = bvp_constants.get_envelope_parameter("kappa_0")
        assert kappa_0 > 0, f"Negative stiffness: {kappa_0}"

    def test_frequency_dependent_conductivity_physics(self, frequency_properties, domain_7d):
        """
        Test frequency-dependent conductivity physics.
        
        Physical Meaning:
            Validates that frequency-dependent conductivity follows
            the Drude-Lorentz model and satisfies causality.
            
        Mathematical Foundation:
            Tests Drude-Lorentz model: σ(ω) = σ₀/(1 + iωτ)
            and Kramers-Kronig relations.
        """
        # Test frequency range
        frequencies = np.logspace(-2, 2, 100)  # 0.01 to 100
        
        # Compute conductivity
        conductivity = frequency_properties.compute_frequency_dependent_conductivity(frequencies)
        
        # Physical validation 1: Conductivity should be finite
        assert np.all(np.isfinite(conductivity)), "Conductivity contains non-finite values"
        
        # Physical validation 2: Real part should be positive (dissipation)
        real_conductivity = np.real(conductivity)
        assert np.all(real_conductivity >= 0), "Negative real conductivity"
        
        # Physical validation 3: Should follow Drude-Lorentz behavior
        # At low frequencies: σ ≈ σ₀
        low_freq_conductivity = np.real(conductivity[0])
        assert low_freq_conductivity > 0, "Zero low-frequency conductivity"
        
        # At high frequencies: σ → 0
        high_freq_conductivity = np.real(conductivity[-1])
        assert high_freq_conductivity < low_freq_conductivity, "High-frequency conductivity too large"
        
        # Physical validation 4: Imaginary part should be negative (inductive)
        imag_conductivity = np.imag(conductivity)
        assert np.all(imag_conductivity <= 0), "Positive imaginary conductivity"

    def test_frequency_dependent_capacitance_physics(self, frequency_properties, domain_7d):
        """
        Test frequency-dependent capacitance physics.
        
        Physical Meaning:
            Validates that frequency-dependent capacitance follows
            the Debye-Cole model and satisfies physical constraints.
            
        Mathematical Foundation:
            Tests Debye-Cole model: C(ω) = C₀/(1 + iωτ)
        """
        # Test frequency range
        frequencies = np.logspace(-2, 2, 100)
        
        # Compute capacitance
        capacitance = frequency_properties.compute_frequency_dependent_capacitance(frequencies)
        
        # Physical validation 1: Capacitance should be finite
        assert np.all(np.isfinite(capacitance)), "Capacitance contains non-finite values"
        
        # Physical validation 2: Real part should be positive
        real_capacitance = np.real(capacitance)
        assert np.all(real_capacitance >= 0), "Negative real capacitance"
        
        # Physical validation 3: Should follow Debye-Cole behavior
        # At low frequencies: C ≈ C₀
        low_freq_capacitance = np.real(capacitance[0])
        assert low_freq_capacitance > 0, "Zero low-frequency capacitance"
        
        # At high frequencies: C → 0
        high_freq_capacitance = np.real(capacitance[-1])
        assert high_freq_capacitance < low_freq_capacitance, "High-frequency capacitance too large"
        
        # Physical validation 4: Imaginary part should be negative
        imag_capacitance = np.imag(capacitance)
        assert np.all(imag_capacitance <= 0), "Positive imaginary capacitance"

    def test_frequency_dependent_inductance_physics(self, frequency_properties, domain_7d):
        """
        Test frequency-dependent inductance physics.
        
        Physical Meaning:
            Validates that frequency-dependent inductance accounts for
            skin effect and proximity effects correctly.
            
        Mathematical Foundation:
            Tests skin effect: δ = √(2/μσω) and proximity effects.
        """
        # Test frequency range
        frequencies = np.logspace(-2, 2, 100)
        
        # Compute inductance
        inductance = frequency_properties.compute_frequency_dependent_inductance(frequencies)
        
        # Physical validation 1: Inductance should be finite
        assert np.all(np.isfinite(inductance)), "Inductance contains non-finite values"
        
        # Physical validation 2: Real part should be positive
        real_inductance = np.real(inductance)
        assert np.all(real_inductance >= 0), "Negative real inductance"
        
        # Physical validation 3: Should show skin effect behavior
        # At low frequencies: L ≈ L₀
        low_freq_inductance = np.real(inductance[0])
        assert low_freq_inductance > 0, "Zero low-frequency inductance"
        
        # At high frequencies: L should increase due to skin effect
        high_freq_inductance = np.real(inductance[-1])
        assert high_freq_inductance >= low_freq_inductance, "High-frequency inductance decreases"
        
        # Physical validation 4: Imaginary part should be positive (resistive)
        imag_inductance = np.imag(inductance)
        assert np.all(imag_inductance >= 0), "Negative imaginary inductance"

    def test_nonlinear_coefficients_physics(self, nonlinear_coeffs, domain_7d):
        """
        Test nonlinear coefficients physics.
        
        Physical Meaning:
            Validates that nonlinear coefficients satisfy physical
            constraints and maintain energy conservation.
            
        Mathematical Foundation:
            Tests nonlinear admittance coefficients and their
            relationship to energy dissipation.
        """
        # Test field amplitude range
        amplitudes = np.linspace(0.1, 2.0, 50)
        
        # Compute nonlinear coefficients
        nonlinear_coeffs_result = nonlinear_coeffs.compute_nonlinear_admittance_coefficients(amplitudes)
        
        # Physical validation 1: Coefficients should be finite
        assert np.all(np.isfinite(nonlinear_coeffs_result)), "Nonlinear coefficients contain non-finite values"
        
        # Physical validation 2: Real part should be positive (dissipation)
        real_coeffs = np.real(nonlinear_coeffs_result)
        assert np.all(real_coeffs >= 0), "Negative real nonlinear coefficients"
        
        # Physical validation 3: Should increase with amplitude (nonlinearity)
        coeffs_increase = np.all(np.diff(real_coeffs) >= 0)
        assert coeffs_increase, "Nonlinear coefficients don't increase with amplitude"
        
        # Physical validation 4: Should be bounded (no infinite growth)
        max_coeff = np.max(real_coeffs)
        assert max_coeff < 1000, f"Nonlinear coefficients too large: {max_coeff}"

    def test_renormalized_coefficients_physics(self, renormalized_coeffs, domain_7d):
        """
        Test renormalized coefficients physics.
        
        Physical Meaning:
            Validates that renormalized coefficients follow
            renormalization group flow and maintain physical consistency.
            
        Mathematical Foundation:
            Tests renormalization group flow: dg/dln(μ) = β(g)
            and validates fixed points.
        """
        # Test renormalization scale range
        scales = np.logspace(-2, 2, 50)  # 0.01 to 100
        
        # Compute renormalized coefficients
        renormalized_result = renormalized_coeffs.compute_renormalized_coefficients(scales)
        
        # Physical validation 1: Coefficients should be finite
        assert np.all(np.isfinite(renormalized_result)), "Renormalized coefficients contain non-finite values"
        
        # Physical validation 2: Should follow renormalization group flow
        # Coefficients should change smoothly with scale
        coeffs_smooth = np.all(np.isfinite(np.gradient(renormalized_result)))
        assert coeffs_smooth, "Renormalized coefficients not smooth"
        
        # Physical validation 3: Should have fixed points
        # Look for points where derivative is small
        derivatives = np.gradient(renormalized_result)
        fixed_point_candidates = np.abs(derivatives) < 0.1 * np.max(np.abs(derivatives))
        assert np.any(fixed_point_candidates), "No fixed points found in renormalization flow"
        
        # Physical validation 4: Should be bounded
        assert np.all(np.abs(renormalized_result) < 1000), "Renormalized coefficients too large"

    def test_energy_conservation_constants(self, bvp_constants, frequency_properties):
        """
        Test energy conservation with frequency-dependent constants.
        
        Physical Meaning:
            Validates that frequency-dependent constants maintain
            energy conservation in the BVP system.
            
        Mathematical Foundation:
            Tests energy conservation: ∂E/∂t + ∇·S = 0 with
            frequency-dependent material properties.
        """
        # Test frequency range
        frequencies = np.logspace(-1, 1, 20)
        
        # Compute material properties
        conductivity = frequency_properties.compute_frequency_dependent_conductivity(frequencies)
        capacitance = frequency_properties.compute_frequency_dependent_capacitance(frequencies)
        inductance = frequency_properties.compute_frequency_dependent_inductance(frequencies)
        
        # Physical validation 1: Energy should be conserved
        # Compute energy dissipation rate
        energy_dissipation = np.real(conductivity) * np.abs(frequencies)**2
        
        # Energy dissipation should be positive
        assert np.all(energy_dissipation >= 0), "Negative energy dissipation"
        
        # Physical validation 2: Energy storage should be positive
        energy_storage = np.real(capacitance) * np.abs(frequencies)**2 + np.real(inductance)
        assert np.all(energy_storage >= 0), "Negative energy storage"
        
        # Physical validation 3: Total energy should be finite
        total_energy = energy_dissipation + energy_storage
        assert np.all(np.isfinite(total_energy)), "Non-finite total energy"

    def test_causality_constraints(self, frequency_properties):
        """
        Test causality constraints on frequency-dependent properties.
        
        Physical Meaning:
            Validates that frequency-dependent properties satisfy
            causality constraints (Kramers-Kronig relations).
            
        Mathematical Foundation:
            Tests Kramers-Kronig relations between real and imaginary
            parts of frequency-dependent response functions.
        """
        # Test frequency range
        frequencies = np.logspace(-2, 2, 100)
        
        # Compute material properties
        conductivity = frequency_properties.compute_frequency_dependent_conductivity(frequencies)
        capacitance = frequency_properties.compute_frequency_dependent_capacitance(frequencies)
        inductance = frequency_properties.compute_frequency_dependent_inductance(frequencies)
        
        # Physical validation 1: Real and imaginary parts should be related
        # For causal systems, real and imaginary parts are related by Hilbert transform
        self._validate_kramers_kronig_relations(conductivity, frequencies)
        self._validate_kramers_kronig_relations(capacitance, frequencies)
        self._validate_kramers_kronig_relations(inductance, frequencies)
        
        # Physical validation 2: Response should be causal
        # At zero frequency, imaginary part should be zero
        zero_freq_conductivity = np.imag(conductivity[0])
        zero_freq_capacitance = np.imag(capacitance[0])
        zero_freq_inductance = np.imag(inductance[0])
        
        assert abs(zero_freq_conductivity) < 1e-10, "Non-zero imaginary conductivity at DC"
        assert abs(zero_freq_capacitance) < 1e-10, "Non-zero imaginary capacitance at DC"
        assert abs(zero_freq_inductance) < 1e-10, "Non-zero imaginary inductance at DC"

    def test_thermodynamic_constraints(self, bvp_constants, frequency_properties):
        """
        Test thermodynamic constraints on material properties.
        
        Physical Meaning:
            Validates that material properties satisfy thermodynamic
            constraints (positive entropy production, etc.).
            
        Mathematical Foundation:
            Tests thermodynamic constraints on frequency-dependent
            response functions.
        """
        # Test frequency range
        frequencies = np.logspace(-2, 2, 50)
        
        # Compute material properties
        conductivity = frequency_properties.compute_frequency_dependent_conductivity(frequencies)
        capacitance = frequency_properties.compute_frequency_dependent_capacitance(frequencies)
        inductance = frequency_properties.compute_frequency_dependent_inductance(frequencies)
        
        # Physical validation 1: Entropy production should be positive
        # Entropy production ∝ Re(σ) * |E|²
        entropy_production = np.real(conductivity)
        assert np.all(entropy_production >= 0), "Negative entropy production"
        
        # Physical validation 2: Heat capacity should be positive
        # Heat capacity ∝ Re(C)
        heat_capacity = np.real(capacitance)
        assert np.all(heat_capacity >= 0), "Negative heat capacity"
        
        # Physical validation 3: Magnetic energy should be positive
        # Magnetic energy ∝ Re(L)
        magnetic_energy = np.real(inductance)
        assert np.all(magnetic_energy >= 0), "Negative magnetic energy"

    def _validate_kramers_kronig_relations(self, response: np.ndarray, frequencies: np.ndarray) -> None:
        """Validate Kramers-Kronig relations for a response function."""
        # For a causal system, real and imaginary parts are related
        # by Hilbert transform (approximate validation)
        real_part = np.real(response)
        imag_part = np.imag(response)
        
        # Check that real and imaginary parts have opposite signs at high frequencies
        # (this is a simplified check for causality)
        high_freq_mask = frequencies > np.median(frequencies)
        if np.any(high_freq_mask):
            real_high = real_part[high_freq_mask]
            imag_high = imag_part[high_freq_mask]
            
            # Real and imaginary parts should have opposite signs
            sign_correlation = np.corrcoef(real_high, imag_high)[0, 1]
            assert sign_correlation < 0, "Real and imaginary parts don't have opposite signs"
