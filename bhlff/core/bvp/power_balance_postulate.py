"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Power Balance Postulate implementation for BVP framework.

This module implements Postulate 9 of the BVP framework, which states that
BVP flux at external boundary equals the sum of growth of static core energy,
EM/weak radiation/losses, and reflection, controlled by integral identity.

Theoretical Background:
    Power balance is maintained at the external boundary through proper
    accounting of energy flows. The integral identity ensures conservation
    of energy in the BVP system.

Example:
    >>> postulate = PowerBalancePostulate(domain, constants)
    >>> results = postulate.apply(envelope)
"""

import numpy as np
from typing import Dict, Any
from ..domain.domain import Domain
from .bvp_constants import BVPConstants
from .bvp_postulate_base import BVPPostulate


class PowerBalancePostulate(BVPPostulate):
    """
    Postulate 9: Power Balance.
    
    Physical Meaning:
        BVP flux at external boundary = (growth of static core energy) +
        (EM/weak radiation/losses) + (reflection). This is controlled
        by integral identity.
    """
    
    def __init__(self, domain: Domain, constants: BVPConstants):
        """
        Initialize power balance postulate.
        
        Physical Meaning:
            Sets up the postulate for analyzing power balance
            at external boundaries.
            
        Args:
            domain (Domain): Computational domain for analysis.
            constants (BVPConstants): BVP physical constants.
        """
        self.domain = domain
        self.constants = constants
        self.power_balance_tolerance = constants.get_quench_parameter("power_balance_tolerance", 0.05)
        self.flux_threshold = constants.get_quench_parameter("flux_threshold", 0.1)
    
    def apply(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Apply power balance postulate.
        
        Physical Meaning:
            Verifies that power balance is maintained at the external
            boundary with proper accounting of energy flows.
            
        Mathematical Foundation:
            Checks integral identity: BVP flux = core energy growth +
            radiation/losses + reflection.
            
        Args:
            envelope (np.ndarray): BVP envelope to analyze.
            
        Returns:
            Dict[str, Any]: Results including power balance components,
                flux analysis, and balance validation.
        """
        # Compute BVP flux at external boundary
        bvp_flux = self._compute_bvp_flux(envelope)
        
        # Compute core energy growth
        core_energy_growth = self._compute_core_energy_growth(envelope)
        
        # Compute radiation and losses
        radiation_losses = self._compute_radiation_losses(envelope)
        
        # Compute reflection
        reflection = self._compute_reflection(envelope)
        
        # Analyze power balance
        power_balance = self._analyze_power_balance(
            bvp_flux, core_energy_growth, radiation_losses, reflection
        )
        
        # Validate power balance
        is_balanced = self._validate_power_balance(power_balance)
        
        return {
            "bvp_flux": bvp_flux,
            "core_energy_growth": core_energy_growth,
            "radiation_losses": radiation_losses,
            "reflection": reflection,
            "power_balance": power_balance,
            "is_balanced": is_balanced,
            "postulate_satisfied": is_balanced
        }
    
    def _compute_bvp_flux(self, envelope: np.ndarray) -> float:
        """
        Compute BVP flux at external boundary.
        
        Physical Meaning:
            Calculates energy flux across external boundaries
            from amplitude gradients.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            float: BVP flux at boundary.
        """
        amplitude = np.abs(envelope)
        gradient = np.gradient(amplitude, self.domain.dx, axis=0)
        
        # Flux is proportional to gradient at boundary
        boundary_flux = np.mean(np.abs(gradient[0, ...])) + np.mean(np.abs(gradient[-1, ...]))
        boundary_flux += np.mean(np.abs(gradient[:, 0, ...])) + np.mean(np.abs(gradient[:, -1, ...]))
        boundary_flux += np.mean(np.abs(gradient[:, :, 0, ...])) + np.mean(np.abs(gradient[:, :, -1, ...]))
        
        return boundary_flux / 6.0  # Average over 6 faces
    
    def _compute_core_energy_growth(self, envelope: np.ndarray) -> float:
        """
        Compute growth of static core energy.
        
        Physical Meaning:
            Calculates rate of energy growth in the core region
            from envelope dynamics.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            float: Core energy growth rate.
        """
        amplitude = np.abs(envelope)
        
        # Core energy is proportional to amplitude squared
        core_energy = np.sum(amplitude**2)
        
        # Growth rate (simplified - in practice would use time derivative)
        growth_rate = np.mean(amplitude) * np.std(amplitude)
        
        return growth_rate
    
    def _compute_radiation_losses(self, envelope: np.ndarray) -> float:
        """
        Compute EM/weak radiation and losses.
        
        Physical Meaning:
            Calculates energy losses due to electromagnetic and
            weak radiation from the envelope using full field theory.
            
        Mathematical Foundation:
            Radiation losses include:
            - EM radiation: P_EM = σ_EM * |A|² * ω² / (8π²c²)
            - Weak radiation: P_weak = σ_weak * |A|⁴ * ω⁴ / (16π⁴c⁴)
            - Total: P_total = P_EM + P_weak
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            float: Radiation losses.
        """
        amplitude = np.abs(envelope)
        carrier_frequency = self.constants.get_physical_parameter("carrier_frequency")
        
        # Get material properties for radiation calculations
        em_conductivity = self.constants.get_material_property("em_conductivity")
        weak_conductivity = self.constants.get_material_property("weak_conductivity")
        speed_of_light = self.constants.get_physical_constant("speed_of_light")
        
        # Compute EM radiation losses using full electromagnetic theory
        # P_EM = σ_EM * |A|² * ω² / (8π²c²)
        omega = 2 * np.pi * carrier_frequency
        em_radiation_losses = (
            em_conductivity * 
            np.mean(amplitude**2) * 
            omega**2 / (8 * np.pi**2 * speed_of_light**2)
        )
        
        # Compute weak radiation losses using weak interaction theory
        # P_weak = σ_weak * |A|⁴ * ω⁴ / (16π⁴c⁴)
        weak_radiation_losses = (
            weak_conductivity * 
            np.mean(amplitude**4) * 
            omega**4 / (16 * np.pi**4 * speed_of_light**4)
        )
        
        # Total radiation losses
        total_radiation_losses = em_radiation_losses + weak_radiation_losses
        
        return total_radiation_losses
    
    def _compute_reflection(self, envelope: np.ndarray) -> float:
        """
        Compute reflection at boundaries.
        
        Physical Meaning:
            Calculates energy reflection at boundaries due to
            impedance mismatch using full electromagnetic theory.
            
        Mathematical Foundation:
            Reflection coefficient: R = |(Z_L - Z_0)/(Z_L + Z_0)|²
            where Z_L is load impedance and Z_0 is characteristic impedance.
            Reflected power: P_reflected = R * P_incident
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            float: Reflected energy.
        """
        amplitude = np.abs(envelope)
        
        # Compute impedance mismatch from envelope properties
        # Characteristic impedance Z_0 from material properties
        vacuum_permeability = self.constants.get_physical_constant("vacuum_permeability")
        vacuum_permittivity = self.constants.get_physical_constant("vacuum_permittivity")
        z0_characteristic = np.sqrt(vacuum_permeability / vacuum_permittivity)
        
        # Load impedance Z_L from envelope admittance
        # Z_L = 1/Y where Y is admittance from envelope
        envelope_admittance = self._compute_envelope_admittance(envelope)
        zl_load = 1.0 / (envelope_admittance + 1e-12)  # Avoid division by zero
        
        # Compute reflection coefficient using full electromagnetic theory
        # R = |(Z_L - Z_0)/(Z_L + Z_0)|²
        reflection_coefficient = np.abs((zl_load - z0_characteristic) / (zl_load + z0_characteristic))**2
        
        # Compute incident power from envelope amplitude
        incident_power = np.mean(amplitude**2)
        
        # Reflected power
        reflected_power = reflection_coefficient * incident_power
        
        return reflected_power
    
    def _compute_envelope_admittance(self, envelope: np.ndarray) -> float:
        """
        Compute envelope admittance from field properties.
        
        Physical Meaning:
            Calculates admittance from envelope gradient and amplitude
            using transmission line theory.
            
        Mathematical Foundation:
            Y = (1/Z) * (∇A/A) where A is envelope amplitude.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            float: Envelope admittance.
        """
        amplitude = np.abs(envelope)
        
        # Compute spatial gradient of amplitude
        gradient = np.gradient(amplitude, self.domain.dx, axis=0)
        gradient_magnitude = np.abs(gradient)
        
        # Compute admittance from gradient-to-amplitude ratio
        # Y = (1/Z) * (∇A/A) where Z is characteristic impedance
        vacuum_permeability = self.constants.get_physical_constant("vacuum_permeability")
        vacuum_permittivity = self.constants.get_physical_constant("vacuum_permittivity")
        z_characteristic = np.sqrt(vacuum_permeability / vacuum_permittivity)
        
        # Average admittance over the domain
        admittance = np.mean(gradient_magnitude / (amplitude + 1e-12)) / z_characteristic
        
        return admittance
    
    def _analyze_power_balance(self, bvp_flux: float, core_energy_growth: float, 
                             radiation_losses: float, reflection: float) -> Dict[str, Any]:
        """
        Analyze power balance components.
        
        Physical Meaning:
            Computes power balance ratio and error to verify
            energy conservation.
            
        Mathematical Foundation:
            Balance ratio = BVP_flux / (core_growth + radiation + reflection)
            
        Args:
            bvp_flux (float): BVP flux at boundary.
            core_energy_growth (float): Core energy growth rate.
            radiation_losses (float): Radiation losses.
            reflection (float): Reflected energy.
            
        Returns:
            Dict[str, Any]: Power balance analysis.
        """
        total_output = core_energy_growth + radiation_losses + reflection
        balance_ratio = bvp_flux / (total_output + 1e-12)
        balance_error = abs(balance_ratio - 1.0)
        
        return {
            "total_input": bvp_flux,
            "total_output": total_output,
            "balance_ratio": balance_ratio,
            "balance_error": balance_error,
            "components": {
                "core_energy_growth": core_energy_growth,
                "radiation_losses": radiation_losses,
                "reflection": reflection
            }
        }
    
    def _validate_power_balance(self, power_balance: Dict[str, Any]) -> bool:
        """
        Validate that power balance is maintained.
        
        Physical Meaning:
            Checks if power balance error is within acceptable
            tolerance for energy conservation.
            
        Args:
            power_balance (Dict[str, Any]): Power balance analysis.
            
        Returns:
            bool: True if power balance is maintained.
        """
        balance_error = power_balance["balance_error"]
        return balance_error < self.power_balance_tolerance
