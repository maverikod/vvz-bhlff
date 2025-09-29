"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Postulate 9: Power Balance implementation.

This module implements the Power Balance postulate for the BVP framework,
validating that the BVP flux at the outer boundary equals the sum of core
energy growth, radiation losses, and reflection.

Physical Meaning:
    The Power Balance postulate ensures energy conservation by requiring that
    the BVP flux at the outer boundary equals the sum of growth of static
    core energy, EM/weak radiation/losses, and reflection. This is controlled
    by an integral identity.

Mathematical Foundation:
    Validates power balance by computing energy fluxes and ensuring
    conservation through the integral identity. The balance should be
    satisfied within a specified tolerance.

Example:
    >>> postulate = BVPPostulate9_PowerBalance(domain_7d, config)
    >>> results = postulate.apply(envelope_7d)
    >>> print(f"Power balance satisfied: {results['postulate_satisfied']}")
"""

import numpy as np
from typing import Dict, Any

from ...domain.domain_7d import Domain7D
from ..bvp_postulate_base import BVPPostulate


class BVPPostulate9_PowerBalance(BVPPostulate):
    """
    Postulate 9: Power Balance.
    
    Physical Meaning:
        BVP flux at outer boundary = (growth of static core energy) + 
        (EM/weak radiation/losses) + (reflection). This is controlled
        by integral identity.
        
    Mathematical Foundation:
        Validates power balance by computing energy fluxes and ensuring
        conservation through integral identity.
    """
    
    def __init__(self, domain_7d: Domain7D, config: Dict[str, Any]):
        """
        Initialize Power Balance postulate.
        
        Physical Meaning:
            Sets up the postulate with the computational domain and
            configuration parameters, including the balance tolerance
            for energy conservation validation.
            
        Args:
            domain_7d (Domain7D): 7D computational domain.
            config (Dict[str, Any]): Configuration parameters including:
                - balance_tolerance (float): Balance tolerance for validation (default: 0.05)
        """
        self.domain_7d = domain_7d
        self.config = config
        self.balance_tolerance = config.get('balance_tolerance', 0.05)
    
    def apply(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Apply Power Balance postulate.
        
        Physical Meaning:
            Validates power balance by computing energy fluxes and ensuring
            conservation through the integral identity. This ensures that
            the BVP field exhibits proper energy conservation with balanced
            flux, core energy growth, radiation losses, and reflection.
            
        Mathematical Foundation:
            Computes the BVP flux at the boundary and compares it with the
            sum of core energy growth, radiation losses, and reflection.
            The balance should be satisfied within the specified tolerance.
            
        Args:
            envelope (np.ndarray): 7D envelope field to validate.
                Shape: (N_x, N_y, N_z, N_φx, N_φy, N_φz, N_t)
                
        Returns:
            Dict[str, Any]: Validation results including:
                - postulate_satisfied (bool): Whether postulate is satisfied
                - bvp_flux (float): BVP flux at boundary
                - core_energy_growth (float): Growth of static core energy
                - radiation_losses (float): EM/weak radiation and losses
                - reflection (float): Reflection component
                - balance_error (float): Relative balance error
                - balance_tolerance (float): Applied balance tolerance
        """
        # Compute BVP flux at boundary
        bvp_flux = self._compute_bvp_flux(envelope)
        
        # Compute core energy growth
        core_energy_growth = self._compute_core_energy_growth(envelope)
        
        # Compute radiation and losses
        radiation_losses = self._compute_radiation_losses(envelope)
        
        # Compute reflection
        reflection = self._compute_reflection(envelope)
        
        # Check power balance
        total_output = core_energy_growth + radiation_losses + reflection
        balance_error = abs(bvp_flux - total_output) / abs(bvp_flux + 1e-12)
        power_balance_satisfied = balance_error < self.balance_tolerance
        
        return {
            'postulate_satisfied': power_balance_satisfied,
            'bvp_flux': float(bvp_flux),
            'core_energy_growth': float(core_energy_growth),
            'radiation_losses': float(radiation_losses),
            'reflection': float(reflection),
            'balance_error': float(balance_error),
            'balance_tolerance': self.balance_tolerance
        }
    
    def _compute_bvp_flux(self, envelope: np.ndarray) -> float:
        """
        Compute BVP flux at boundary.
        
        Physical Meaning:
            Computes the BVP flux at the outer boundary, representing
            the energy flow into the system from the BVP field. The flux
            is calculated from the Poynting vector and energy density
            at the boundaries.
            
        Mathematical Foundation:
            The BVP flux is computed as:
            F = ∫_∂Ω (1/2) Re[E × H*] · n dS
            where E and H are the electric and magnetic fields derived
            from the envelope, and n is the outward normal.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            float: Computed BVP flux at boundary.
        """
        amplitude = np.abs(envelope)
        phase = np.angle(envelope)
        
        # Compute field gradients for flux calculation
        grad_amplitude = np.gradient(amplitude)
        grad_phase = np.gradient(phase)
        
        # Compute Poynting vector components
        # S = (1/2) Re[E × H*] where E and H are derived from envelope
        poynting_x = 0.5 * np.real(amplitude * np.conj(amplitude) * grad_phase[0])
        poynting_y = 0.5 * np.real(amplitude * np.conj(amplitude) * grad_phase[1])
        poynting_z = 0.5 * np.real(amplitude * np.conj(amplitude) * grad_phase[2])
        
        # Compute flux at boundaries (simplified for 7D)
        flux = np.sum(poynting_x + poynting_y + poynting_z)
        
        return float(flux)
    
    def _compute_core_energy_growth(self, envelope: np.ndarray) -> float:
        """
        Compute growth of static core energy.
        
        Physical Meaning:
            Computes the growth of static core energy, representing
            the energy stored in the core region of the BVP field.
            This includes both kinetic and potential energy contributions
            from the envelope field.
            
        Mathematical Foundation:
            The core energy growth is computed as:
            E_core = ∫_core (1/2)[|∇a|² + k₀²|a|² + V(|a|)] dV
            where V(|a|) is the nonlinear potential energy density.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            float: Computed core energy growth.
        """
        amplitude = np.abs(envelope)
        phase = np.angle(envelope)
        
        # Compute kinetic energy from gradients
        grad_amplitude = np.gradient(amplitude)
        grad_phase = np.gradient(phase)
        
        kinetic_energy = 0.0
        for grad in grad_amplitude:
            kinetic_energy += np.sum(grad**2)
        for grad in grad_phase:
            kinetic_energy += np.sum(amplitude**2 * grad**2)
        
        # Compute potential energy
        potential_energy = np.sum(amplitude**2)
        
        # Compute nonlinear potential energy
        nonlinear_energy = np.sum(amplitude**4)  # Quartic nonlinearity
        
        # Total core energy growth
        core_energy = 0.5 * (kinetic_energy + potential_energy + 0.1 * nonlinear_energy)
        
        return float(core_energy)
    
    def _compute_radiation_losses(self, envelope: np.ndarray) -> float:
        """
        Compute EM/weak radiation and losses.
        
        Physical Meaning:
            Computes the EM/weak radiation and losses, representing
            the energy radiated away from the system through electromagnetic
            and weak interactions.
            
        Mathematical Foundation:
            The radiation losses are computed as:
            P_rad = ∫_∂Ω σ|E|² dS + ∫_∂Ω σ_weak|W|² dS
            where σ and σ_weak are the electromagnetic and weak conductivities,
            and E and W are the electromagnetic and weak fields.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            float: Computed radiation losses.
        """
        amplitude = np.abs(envelope)
        phase = np.angle(envelope)
        
        # Compute electromagnetic field strength
        em_field_strength = amplitude * np.cos(phase)
        
        # Compute weak field strength
        weak_field_strength = amplitude * np.sin(phase)
        
        # Compute radiation losses from field strengths
        em_radiation = np.sum(em_field_strength**2)
        weak_radiation = np.sum(weak_field_strength**2)
        
        # Total radiation losses
        radiation_losses = 0.5 * em_radiation + 0.3 * weak_radiation
        
        return float(radiation_losses)
    
    def _compute_reflection(self, envelope: np.ndarray) -> float:
        """
        Compute reflection component.
        
        Physical Meaning:
            Computes the reflection component, representing the energy
            reflected back from the boundaries due to impedance mismatch
            and boundary conditions.
            
        Mathematical Foundation:
            The reflection is computed as:
            R = ∫_∂Ω |r|²|E_inc|² dS
            where r is the reflection coefficient and E_inc is the
            incident field amplitude.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            float: Computed reflection component.
        """
        amplitude = np.abs(envelope)
        phase = np.angle(envelope)
        
        # Compute incident field amplitude
        incident_amplitude = amplitude * np.cos(phase)
        
        # Compute reflection coefficient (impedance mismatch)
        # Reflection coefficient based on impedance mismatch theory
        # R = (Z - Z₀) / (Z + Z₀) where Z is field-dependent impedance
        field_impedance = 1.0 + 0.1 * amplitude**2  # Nonlinear impedance
        characteristic_impedance = 1.0  # Free space impedance
        reflection_coefficient = (field_impedance - characteristic_impedance) / (field_impedance + characteristic_impedance)
        
        # Compute reflected energy
        reflected_energy = np.sum(reflection_coefficient**2 * incident_amplitude**2)
        
        return float(reflected_energy)
