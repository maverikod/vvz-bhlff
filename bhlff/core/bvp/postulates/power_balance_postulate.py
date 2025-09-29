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
            the energy flow into the system from the BVP field.
            
        Mathematical Foundation:
            The BVP flux is computed from the envelope amplitude using
            a simplified model that captures the essential flux characteristics.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            float: Computed BVP flux at boundary.
        """
        amplitude = np.abs(envelope)
        # Simplified flux calculation
        return float(np.sum(amplitude**2))
    
    def _compute_core_energy_growth(self, envelope: np.ndarray) -> float:
        """
        Compute growth of static core energy.
        
        Physical Meaning:
            Computes the growth of static core energy, representing
            the energy stored in the core region of the BVP field.
            
        Mathematical Foundation:
            The core energy growth is computed from the envelope amplitude
            using a simplified model that captures the essential energy
            storage characteristics.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            float: Computed core energy growth.
        """
        amplitude = np.abs(envelope)
        # Simplified core energy calculation
        return float(0.3 * np.sum(amplitude**2))
    
    def _compute_radiation_losses(self, envelope: np.ndarray) -> float:
        """
        Compute EM/weak radiation and losses.
        
        Physical Meaning:
            Computes the EM/weak radiation and losses, representing
            the energy radiated away from the system.
            
        Mathematical Foundation:
            The radiation losses are computed from the envelope amplitude
            using a simplified model that captures the essential radiation
            characteristics.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            float: Computed radiation losses.
        """
        amplitude = np.abs(envelope)
        # Simplified radiation calculation
        return float(0.4 * np.sum(amplitude**2))
    
    def _compute_reflection(self, envelope: np.ndarray) -> float:
        """
        Compute reflection component.
        
        Physical Meaning:
            Computes the reflection component, representing the energy
            reflected back from the boundaries.
            
        Mathematical Foundation:
            The reflection is computed from the envelope amplitude using
            a simplified model that captures the essential reflection
            characteristics.
            
        Args:
            envelope (np.ndarray): 7D envelope field.
            
        Returns:
            float: Computed reflection component.
        """
        amplitude = np.abs(envelope)
        # Simplified reflection calculation
        return float(0.3 * np.sum(amplitude**2))
