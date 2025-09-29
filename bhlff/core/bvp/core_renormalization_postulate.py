"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core Renormalization Postulate implementation for BVP framework.

This module implements Postulate 8 of the BVP framework, which states that
the core is a minimum of ω₀-averaged energy where BVP "renormalizes"
core coefficients c_i^eff(|A|,|∇A|) and sets boundary "pressure/stiffness".

Theoretical Background:
    The core represents an energy minimum with renormalized coefficients
    that depend on envelope amplitude and gradient. This renormalization
    is controlled by BVP field dynamics and sets boundary conditions.

Example:
    >>> postulate = CoreRenormalizationPostulate(domain, constants)
    >>> results = postulate.apply(envelope)
"""

import numpy as np
from typing import Dict, Any, List
from ..domain.domain import Domain
from .bvp_constants import BVPConstants
from .bvp_postulate_base import BVPPostulate


class CoreRenormalizationPostulate(BVPPostulate):
    """
    Postulate 8: Core - Averaged Minimum.
    
    Physical Meaning:
        Core is minimum of ω₀-averaged energy: BVP "renormalizes"
        core coefficients c_i^eff(|A|,|∇A|) and sets boundary
        "pressure/stiffness".
    """
    
    def __init__(self, domain: Domain, constants: BVPConstants):
        """
        Initialize core renormalization postulate.
        
        Physical Meaning:
            Sets up the postulate for analyzing core energy minimization
            and coefficient renormalization.
            
        Args:
            domain (Domain): Computational domain for analysis.
            constants (BVPConstants): BVP physical constants.
        """
        self.domain = domain
        self.constants = constants
        self.renormalization_threshold = constants.get_quench_parameter("renormalization_threshold", 0.1)
        self.core_radius = constants.get_physical_parameter("core_radius", 1.0)
    
    def apply(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Apply core renormalization postulate.
        
        Physical Meaning:
            Verifies that the core represents a minimum of
            ω₀-averaged energy with renormalized coefficients
            and proper boundary conditions.
            
        Mathematical Foundation:
            Analyzes core energy minimization and coefficient
            renormalization c_i^eff(|A|,|∇A|) from BVP envelope.
            
        Args:
            envelope (np.ndarray): BVP envelope to analyze.
            
        Returns:
            Dict[str, Any]: Results including renormalized coefficients,
                core energy, and boundary conditions.
        """
        # Identify core region
        core_region = self._identify_core_region(envelope)
        
        # Compute renormalized coefficients
        renormalized_coefficients = self._compute_renormalized_coefficients(envelope, core_region)
        
        # Analyze core energy minimization
        energy_analysis = self._analyze_core_energy_minimization(envelope, core_region)
        
        # Compute boundary pressure/stiffness
        boundary_conditions = self._compute_boundary_conditions(envelope, core_region)
        
        # Validate core renormalization
        is_renormalized = self._validate_core_renormalization(
            renormalized_coefficients, energy_analysis
        )
        
        return {
            "core_region": core_region,
            "renormalized_coefficients": renormalized_coefficients,
            "energy_analysis": energy_analysis,
            "boundary_conditions": boundary_conditions,
            "is_renormalized": is_renormalized,
            "postulate_satisfied": is_renormalized
        }
    
    def _identify_core_region(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Identify the core region of the envelope.
        
        Physical Meaning:
            Finds the central region where envelope amplitude is highest
            and defines core boundaries based on amplitude decay.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            Dict[str, Any]: Core region parameters.
        """
        amplitude = np.abs(envelope)
        
        # Find center of mass
        center = self._find_center_of_mass(amplitude)
        
        # Define core radius based on amplitude decay
        core_radius = self._compute_core_radius(amplitude, center)
        
        # Create core mask
        core_mask = self._create_core_mask(amplitude, center, core_radius)
        
        return {
            "center": center,
            "radius": core_radius,
            "mask": core_mask,
            "volume": np.sum(core_mask)
        }
    
    def _find_center_of_mass(self, amplitude: np.ndarray) -> List[float]:
        """
        Find center of mass of the amplitude distribution.
        
        Physical Meaning:
            Computes center of mass as weighted average of coordinates
            with amplitude as weight.
            
        Args:
            amplitude (np.ndarray): Envelope amplitude.
            
        Returns:
            List[float]: Center of mass coordinates.
        """
        center = []
        for axis in range(amplitude.ndim):
            axis_center = np.sum(amplitude * np.arange(amplitude.shape[axis])) / np.sum(amplitude)
            center.append(axis_center)
        return center
    
    def _compute_core_radius(self, amplitude: np.ndarray, center: List[float]) -> float:
        """
        Compute effective core radius.
        
        Physical Meaning:
            Finds radius where amplitude drops to 1/e of maximum,
            defining effective core boundary.
            
        Args:
            amplitude (np.ndarray): Envelope amplitude.
            center (List[float]): Center coordinates.
            
        Returns:
            float: Effective core radius.
        """
        # Find radius where amplitude drops to 1/e of maximum
        max_amplitude = np.max(amplitude)
        threshold = max_amplitude / np.e
        
        # Find distance from center where amplitude drops below threshold
        distances = self._compute_distances_from_center(amplitude, center)
        core_radius = np.min(distances[amplitude < threshold])
        
        return core_radius
    
    def _compute_distances_from_center(self, amplitude: np.ndarray, center: List[float]) -> np.ndarray:
        """
        Compute distances from center for each point.
        
        Physical Meaning:
            Calculates Euclidean distance from center for each
            point in the domain.
            
        Args:
            amplitude (np.ndarray): Envelope amplitude.
            center (List[float]): Center coordinates.
            
        Returns:
            np.ndarray: Distance field.
        """
        # Create coordinate arrays
        coords = np.meshgrid(*[np.arange(amplitude.shape[i]) for i in range(amplitude.ndim)], indexing='ij')
        
        # Compute distances
        distances = np.zeros_like(amplitude)
        for i, coord in enumerate(coords):
            distances += (coord - center[i])**2
        distances = np.sqrt(distances)
        
        return distances
    
    def _create_core_mask(self, amplitude: np.ndarray, center: List[float], radius: float) -> np.ndarray:
        """
        Create mask for core region.
        
        Physical Meaning:
            Creates boolean mask identifying points within core radius.
            
        Args:
            amplitude (np.ndarray): Envelope amplitude.
            center (List[float]): Center coordinates.
            radius (float): Core radius.
            
        Returns:
            np.ndarray: Core region mask.
        """
        distances = self._compute_distances_from_center(amplitude, center)
        return distances <= radius
    
    def _compute_renormalized_coefficients(self, envelope: np.ndarray, core_region: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute renormalized coefficients c_i^eff(|A|,|∇A|).
        
        Physical Meaning:
            Calculates effective coefficients that depend on envelope
            amplitude and gradient, representing BVP renormalization.
            
        Mathematical Foundation:
            c_i^eff = c_i + α_i|A|² + β_i|∇A|²/ω₀²
            
        Args:
            envelope (np.ndarray): BVP envelope.
            core_region (Dict[str, Any]): Core region parameters.
            
        Returns:
            Dict[str, float]: Renormalized coefficients.
        """
        amplitude = np.abs(envelope)
        gradient = np.gradient(amplitude, self.domain.dx, axis=0)
        gradient_magnitude = np.abs(gradient)
        
        core_mask = core_region["mask"]
        core_amplitude = amplitude[core_mask]
        core_gradient = gradient_magnitude[core_mask]
        
        # Renormalized coefficients depend on envelope amplitude and gradient
        # c_i^eff = c_i + α_i|A|² + β_i|∇A|²/ω₀²
        alpha_2 = self.constants.get_envelope_parameter("renormalization_alpha_2", 0.1)
        alpha_4 = self.constants.get_envelope_parameter("renormalization_alpha_4", 0.01)
        alpha_6 = self.constants.get_envelope_parameter("renormalization_alpha_6", 0.001)
        
        beta_2 = self.constants.get_envelope_parameter("renormalization_beta_2", 0.1)
        beta_4 = self.constants.get_envelope_parameter("renormalization_beta_4", 0.01)
        beta_6 = self.constants.get_envelope_parameter("renormalization_beta_6", 0.001)
        
        omega_0 = self.constants.get_physical_parameter("carrier_frequency")
        
        # Compute effective coefficients
        c2_eff = 1.0 + alpha_2 * np.mean(core_amplitude**2) + beta_2 * np.mean(core_gradient**2) / omega_0**2
        c4_eff = 1.0 + alpha_4 * np.mean(core_amplitude**4) + beta_4 * np.mean(core_gradient**4) / omega_0**4
        c6_eff = 1.0 + alpha_6 * np.mean(core_amplitude**6) + beta_6 * np.mean(core_gradient**6) / omega_0**6
        
        return {
            "c2_eff": c2_eff,
            "c4_eff": c4_eff,
            "c6_eff": c6_eff,
            "renormalization_strength": np.mean([alpha_2, alpha_4, alpha_6])
        }
    
    def _analyze_core_energy_minimization(self, envelope: np.ndarray, core_region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze core energy minimization.
        
        Physical Meaning:
            Computes energy components and checks if core is at
            energy minimum.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            core_region (Dict[str, Any]): Core region parameters.
            
        Returns:
            Dict[str, Any]: Energy analysis results.
        """
        amplitude = np.abs(envelope)
        core_mask = core_region["mask"]
        core_amplitude = amplitude[core_mask]
        
        # Compute energy components
        potential_energy = np.sum(core_amplitude**2)
        gradient_energy = np.sum(np.gradient(core_amplitude, self.domain.dx, axis=0)**2)
        total_energy = potential_energy + gradient_energy
        
        # Check if core is at energy minimum
        energy_gradient = np.gradient(total_energy)
        is_minimum = np.allclose(energy_gradient, 0, atol=1e-6)
        
        return {
            "potential_energy": potential_energy,
            "gradient_energy": gradient_energy,
            "total_energy": total_energy,
            "is_minimum": is_minimum,
            "energy_gradient": energy_gradient
        }
    
    def _compute_boundary_conditions(self, envelope: np.ndarray, core_region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute boundary pressure/stiffness conditions.
        
        Physical Meaning:
            Calculates boundary pressure and stiffness from amplitude
            gradients and second derivatives at core boundary.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            core_region (Dict[str, Any]): Core region parameters.
            
        Returns:
            Dict[str, Any]: Boundary conditions.
        """
        amplitude = np.abs(envelope)
        core_mask = core_region["mask"]
        
        # Compute boundary pressure (gradient of amplitude at core boundary)
        boundary_pressure = self._compute_boundary_pressure(amplitude, core_mask)
        
        # Compute boundary stiffness (second derivative at boundary)
        boundary_stiffness = self._compute_boundary_stiffness(amplitude, core_mask)
        
        return {
            "boundary_pressure": boundary_pressure,
            "boundary_stiffness": boundary_stiffness,
            "boundary_stability": boundary_stiffness > 0
        }
    
    def _compute_boundary_pressure(self, amplitude: np.ndarray, core_mask: np.ndarray) -> float:
        """
        Compute boundary pressure from amplitude gradient.
        
        Physical Meaning:
            Calculates pressure at core boundary from amplitude
            gradient magnitude.
            
        Args:
            amplitude (np.ndarray): Envelope amplitude.
            core_mask (np.ndarray): Core region mask.
            
        Returns:
            float: Boundary pressure.
        """
        gradient = np.gradient(amplitude, self.domain.dx, axis=0)
        gradient_magnitude = np.abs(gradient)
        
        # Pressure at core boundary
        boundary_gradient = gradient_magnitude[~core_mask]
        return np.mean(boundary_gradient)
    
    def _compute_boundary_stiffness(self, amplitude: np.ndarray, core_mask: np.ndarray) -> float:
        """
        Compute boundary stiffness from second derivative.
        
        Physical Meaning:
            Calculates stiffness at core boundary from second
            derivative of amplitude.
            
        Args:
            amplitude (np.ndarray): Envelope amplitude.
            core_mask (np.ndarray): Core region mask.
            
        Returns:
            float: Boundary stiffness.
        """
        second_derivative = np.gradient(np.gradient(amplitude, self.domain.dx, axis=0), self.domain.dx, axis=0)
        
        # Stiffness at core boundary
        boundary_stiffness = second_derivative[~core_mask]
        return np.mean(boundary_stiffness)
    
    def _validate_core_renormalization(self, renormalized_coefficients: Dict[str, float], 
                                     energy_analysis: Dict[str, Any]) -> bool:
        """
        Validate that core is properly renormalized.
        
        Physical Meaning:
            Checks renormalization strength and energy minimization
            to confirm proper core renormalization.
            
        Args:
            renormalized_coefficients (Dict[str, float]): Renormalized coefficients.
            energy_analysis (Dict[str, Any]): Energy analysis results.
            
        Returns:
            bool: True if core is properly renormalized.
        """
        # Check renormalization strength
        renormalization_strength = renormalized_coefficients["renormalization_strength"]
        
        # Check energy minimization
        is_minimum = energy_analysis["is_minimum"]
        
        return (renormalization_strength > self.renormalization_threshold and is_minimum)
