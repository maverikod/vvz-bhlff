"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Multi-soliton physical properties computation.

This module implements physical properties computation for multi-soliton
solutions using 7D BVP theory.

Physical Meaning:
    Implements physical properties computation including energy calculations,
    stability metrics, phase coherence, and 7D BVP specific properties
    for multi-soliton systems.

Example:
    >>> properties = MultiSolitonPhysicalProperties(system, nonlinear_params)
    >>> props = properties.compute_two_soliton_physical_properties(amp1, width1, pos1, amp2, width2, pos2, solution)
"""

import numpy as np
from typing import Dict, Any
import logging

from .base import SolitonAnalysisBase


class MultiSolitonPhysicalProperties(SolitonAnalysisBase):
    """
    Multi-soliton physical properties computation.
    
    Physical Meaning:
        Implements physical properties computation including energy calculations,
        stability metrics, phase coherence, and 7D BVP specific properties
        for multi-soliton systems.
        
    Mathematical Foundation:
        Computes comprehensive physical properties using 7D BVP theory
        including energy conservation, phase coherence, and stability analysis.
    """
    
    def __init__(self, system, nonlinear_params: Dict[str, Any]):
        """Initialize multi-soliton physical properties."""
        super().__init__(system, nonlinear_params)
        self.logger = logging.getLogger(__name__)
    
    def compute_two_soliton_physical_properties(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive two-soliton physical properties.
        
        Physical Meaning:
            Computes all relevant physical properties of the two-soliton
            system including individual energies, interaction energy,
            stability metrics, and 7D BVP specific properties.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            solution (Dict[str, Any]): Two-soliton solution.
            
        Returns:
            Dict[str, Any]: Complete physical properties.
        """
        try:
            # Compute individual soliton energies
            energy1 = self._compute_individual_soliton_energy(solution['soliton_1_profile'], solution['spatial_grid'])
            energy2 = self._compute_individual_soliton_energy(solution['soliton_2_profile'], solution['spatial_grid'])
            
            # Compute interaction energy
            interaction_energy = self._compute_interaction_energy(amp1, width1, pos1, amp2, width2, pos2)
            
            # Compute stability metrics
            stability_metric = self._compute_two_soliton_stability(solution)
            
            # Compute phase coherence
            phase_coherence = self._compute_two_soliton_phase_coherence(solution)
            
            # Compute 7D BVP specific properties
            bvp_properties = self._compute_two_soliton_7d_bvp_properties(solution, amp1, width1, amp2, width2)
            
            return {
                "individual_energies": [energy1, energy2],
                "interaction_energy": interaction_energy,
                "total_energy": energy1 + energy2 + interaction_energy,
                "stability_metric": stability_metric,
                "phase_coherence": phase_coherence,
                "7d_bvp_properties": bvp_properties,
                "energy_ratio": interaction_energy / (energy1 + energy2) if (energy1 + energy2) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Two-soliton physical properties computation failed: {e}")
            return {}
    
    def compute_three_soliton_physical_properties(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, amp3: float, width3: float, pos3: float, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive three-soliton physical properties.
        
        Physical Meaning:
            Computes all relevant physical properties of the three-soliton
            system including individual energies, pairwise interactions,
            three-body interactions, stability metrics, and 7D BVP specific properties.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            amp3, width3, pos3 (float): Third soliton parameters.
            solution (Dict[str, Any]): Three-soliton solution.
            
        Returns:
            Dict[str, Any]: Complete physical properties.
        """
        try:
            # Compute individual soliton energies
            energy1 = self._compute_individual_soliton_energy(solution['soliton_1_profile'], solution['spatial_grid'])
            energy2 = self._compute_individual_soliton_energy(solution['soliton_2_profile'], solution['spatial_grid'])
            energy3 = self._compute_individual_soliton_energy(solution['soliton_3_profile'], solution['spatial_grid'])
            
            # Compute pairwise interaction energies
            interaction_12 = self._compute_interaction_energy(amp1, width1, pos1, amp2, width2, pos2)
            interaction_13 = self._compute_interaction_energy(amp1, width1, pos1, amp3, width3, pos3)
            interaction_23 = self._compute_interaction_energy(amp2, width2, pos2, amp3, width3, pos3)
            
            # Compute three-body interaction energy
            three_body_energy = self._compute_three_body_interaction_energy(amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3)
            
            # Compute stability metrics
            stability_metric = self._compute_three_soliton_stability(solution)
            
            # Compute phase coherence
            phase_coherence = self._compute_three_soliton_phase_coherence(solution)
            
            # Compute 7D BVP specific properties
            bvp_properties = self._compute_three_soliton_7d_bvp_properties(solution, amp1, width1, amp2, width2, amp3, width3)
            
            return {
                "individual_energies": [energy1, energy2, energy3],
                "pairwise_interactions": [interaction_12, interaction_13, interaction_23],
                "three_body_interaction": three_body_energy,
                "total_energy": energy1 + energy2 + energy3 + interaction_12 + interaction_13 + interaction_23 + three_body_energy,
                "stability_metric": stability_metric,
                "phase_coherence": phase_coherence,
                "7d_bvp_properties": bvp_properties,
                "interaction_ratio": (interaction_12 + interaction_13 + interaction_23 + three_body_energy) / (energy1 + energy2 + energy3) if (energy1 + energy2 + energy3) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Three-soliton physical properties computation failed: {e}")
            return {}
    
    def _compute_individual_soliton_energy(self, profile: np.ndarray, x: np.ndarray) -> float:
        """Compute energy of individual soliton."""
        try:
            # Kinetic energy
            kinetic_energy = 0.5 * np.trapz(np.gradient(profile) ** 2, x)
            
            # Potential energy
            potential_energy = 0.5 * self.lambda_param * np.trapz(profile ** 2, x)
            
            return kinetic_energy + potential_energy
            
        except Exception as e:
            self.logger.error(f"Individual soliton energy computation failed: {e}")
            return 0.0
    
    def _compute_interaction_energy(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float) -> float:
        """Compute interaction energy between solitons."""
        try:
            distance = abs(pos2 - pos1)
            interaction_range = width1 + width2
            
            # Step resonator interaction energy
            if distance < interaction_range:
                return self.interaction_strength * amp1 * amp2
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Interaction energy computation failed: {e}")
            return 0.0
    
    def _compute_three_body_interaction_energy(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, amp3: float, width3: float, pos3: float) -> float:
        """Compute three-body interaction energy."""
        try:
            # Compute distances between all solitons
            distance_12 = abs(pos2 - pos1)
            distance_13 = abs(pos3 - pos1)
            distance_23 = abs(pos3 - pos2)
            
            # Compute interaction range for three-body interaction
            interaction_range = width1 + width2 + width3
            
            # Three-body interaction using step resonator theory
            total_distance = distance_12 + distance_13 + distance_23
            if total_distance < interaction_range:
                return self.three_body_strength * amp1 * amp2 * amp3
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Three-body interaction energy computation failed: {e}")
            return 0.0
    
    def _compute_two_soliton_stability(self, solution: Dict[str, Any]) -> float:
        """Compute two-soliton stability metric."""
        try:
            total_profile = solution['total_profile']
            x = solution['spatial_grid']
            
            # Compute energy distribution
            energy_density = 0.5 * (np.gradient(total_profile) ** 2 + self.lambda_param * total_profile ** 2)
            
            # Compute stability as energy localization
            peak_energy = np.max(energy_density)
            total_energy = np.trapz(energy_density, x)
            
            if total_energy > 0:
                return peak_energy / total_energy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Two-soliton stability computation failed: {e}")
            return 0.0
    
    def _compute_three_soliton_stability(self, solution: Dict[str, Any]) -> float:
        """Compute three-soliton stability metric."""
        try:
            total_profile = solution['total_profile']
            x = solution['spatial_grid']
            
            # Compute energy distribution
            energy_density = 0.5 * (np.gradient(total_profile) ** 2 + self.lambda_param * total_profile ** 2)
            
            # Compute stability as energy localization
            peak_energy = np.max(energy_density)
            total_energy = np.trapz(energy_density, x)
            
            if total_energy > 0:
                return peak_energy / total_energy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Three-soliton stability computation failed: {e}")
            return 0.0
    
    def _compute_two_soliton_phase_coherence(self, solution: Dict[str, Any]) -> float:
        """Compute two-soliton phase coherence."""
        try:
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            
            # Compute phase fields
            phase1 = np.arctan2(profile1, np.gradient(profile1))
            phase2 = np.arctan2(profile2, np.gradient(profile2))
            
            # Compute phase coherence as correlation
            if len(phase1) == len(phase2):
                correlation = np.corrcoef(phase1, phase2)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Two-soliton phase coherence computation failed: {e}")
            return 0.0
    
    def _compute_three_soliton_phase_coherence(self, solution: Dict[str, Any]) -> float:
        """Compute three-soliton phase coherence."""
        try:
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            profile3 = solution['soliton_3_profile']
            
            # Compute phase fields
            phase1 = np.arctan2(profile1, np.gradient(profile1))
            phase2 = np.arctan2(profile2, np.gradient(profile2))
            phase3 = np.arctan2(profile3, np.gradient(profile3))
            
            # Compute phase coherence as average correlation
            if len(phase1) == len(phase2) == len(phase3):
                corr_12 = np.corrcoef(phase1, phase2)[0, 1]
                corr_13 = np.corrcoef(phase1, phase3)[0, 1]
                corr_23 = np.corrcoef(phase2, phase3)[0, 1]
                
                correlations = [corr_12, corr_13, corr_23]
                valid_correlations = [c for c in correlations if not np.isnan(c)]
                
                if valid_correlations:
                    return np.mean(np.abs(valid_correlations))
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Three-soliton phase coherence computation failed: {e}")
            return 0.0
    
    def _compute_two_soliton_7d_bvp_properties(self, solution: Dict[str, Any], amp1: float, width1: float, amp2: float, width2: float) -> Dict[str, Any]:
        """Compute 7D BVP specific properties for two-soliton system."""
        try:
            total_profile = solution['total_profile']
            x = solution['spatial_grid']
            
            # Compute fractional Laplacian contribution
            fractional_contribution = self._compute_fractional_laplacian_contribution(total_profile, x)
            
            # Compute step resonator efficiency
            step_efficiency = self._compute_two_soliton_step_efficiency(solution, width1, width2)
            
            # Compute interaction efficiency
            interaction_efficiency = self._compute_interaction_efficiency(solution)
            
            return {
                "fractional_laplacian_contribution": fractional_contribution,
                "step_resonator_efficiency": step_efficiency,
                "interaction_efficiency": interaction_efficiency,
                "7d_phase_space_properties": self._compute_7d_phase_space_properties(total_profile, x)
            }
            
        except Exception as e:
            self.logger.error(f"Two-soliton 7D BVP properties computation failed: {e}")
            return {}
    
    def _compute_three_soliton_7d_bvp_properties(self, solution: Dict[str, Any], amp1: float, width1: float, amp2: float, width2: float, amp3: float, width3: float) -> Dict[str, Any]:
        """Compute 7D BVP specific properties for three-soliton system."""
        try:
            total_profile = solution['total_profile']
            x = solution['spatial_grid']
            
            # Compute fractional Laplacian contribution
            fractional_contribution = self._compute_fractional_laplacian_contribution(total_profile, x)
            
            # Compute step resonator efficiency
            step_efficiency = self._compute_three_soliton_step_efficiency(solution, width1, width2, width3)
            
            # Compute multi-body interaction efficiency
            interaction_efficiency = self._compute_three_soliton_interaction_efficiency(solution)
            
            return {
                "fractional_laplacian_contribution": fractional_contribution,
                "step_resonator_efficiency": step_efficiency,
                "multi_body_interaction_efficiency": interaction_efficiency,
                "7d_phase_space_properties": self._compute_7d_phase_space_properties(total_profile, x)
            }
            
        except Exception as e:
            self.logger.error(f"Three-soliton 7D BVP properties computation failed: {e}")
            return {}
    
    def _compute_fractional_laplacian_contribution(self, profile: np.ndarray, x: np.ndarray) -> float:
        """Compute fractional Laplacian contribution."""
        try:
            # Compute fractional Laplacian using FFT
            dx = x[1] - x[0] if len(x) > 1 else 1.0
            profile_fft = np.fft.fft(profile)
            k = np.fft.fftfreq(len(x), dx) * 2 * np.pi
            k_magnitude = np.abs(k)
            k_magnitude[0] = 1e-10  # Avoid division by zero
            
            fractional_spectrum = (k_magnitude ** (2 * self.beta)) * profile_fft
            fractional_laplacian = np.real(np.fft.ifft(fractional_spectrum))
            
            # Compute contribution
            total_energy = np.trapz(profile ** 2, x)
            frac_energy = np.trapz(profile * fractional_laplacian, x)
            
            if total_energy > 0:
                return abs(frac_energy) / total_energy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Fractional Laplacian contribution computation failed: {e}")
            return 0.0
    
    def _compute_two_soliton_step_efficiency(self, solution: Dict[str, Any], width1: float, width2: float) -> float:
        """Compute step resonator efficiency for two-soliton system."""
        try:
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            x = solution['spatial_grid']
            
            # Compute step resonator profiles
            step1 = self._step_resonator_profile(x, solution.get('soliton_1_position', 0.0), width1)
            step2 = self._step_resonator_profile(x, solution.get('soliton_2_position', 0.0), width2)
            
            # Compute efficiency as overlap
            overlap1 = np.trapz(profile1 * step1, x)
            overlap2 = np.trapz(profile2 * step2, x)
            total1 = np.trapz(np.abs(profile1), x)
            total2 = np.trapz(np.abs(profile2), x)
            
            efficiency1 = overlap1 / total1 if total1 > 0 else 0.0
            efficiency2 = overlap2 / total2 if total2 > 0 else 0.0
            
            return (efficiency1 + efficiency2) / 2.0
            
        except Exception as e:
            self.logger.error(f"Two-soliton step efficiency computation failed: {e}")
            return 0.0
    
    def _compute_three_soliton_step_efficiency(self, solution: Dict[str, Any], width1: float, width2: float, width3: float) -> float:
        """Compute step resonator efficiency for three-soliton system."""
        try:
            profile1 = solution['soliton_1_profile']
            profile2 = solution['soliton_2_profile']
            profile3 = solution['soliton_3_profile']
            x = solution['spatial_grid']
            
            # Compute step resonator profiles
            step1 = self._step_resonator_profile(x, solution.get('soliton_1_position', 0.0), width1)
            step2 = self._step_resonator_profile(x, solution.get('soliton_2_position', 0.0), width2)
            step3 = self._step_resonator_profile(x, solution.get('soliton_3_position', 0.0), width3)
            
            # Compute efficiency as overlap for each soliton
            overlap1 = np.trapz(profile1 * step1, x)
            overlap2 = np.trapz(profile2 * step2, x)
            overlap3 = np.trapz(profile3 * step3, x)
            
            total1 = np.trapz(np.abs(profile1), x)
            total2 = np.trapz(np.abs(profile2), x)
            total3 = np.trapz(np.abs(profile3), x)
            
            efficiency1 = overlap1 / total1 if total1 > 0 else 0.0
            efficiency2 = overlap2 / total2 if total2 > 0 else 0.0
            efficiency3 = overlap3 / total3 if total3 > 0 else 0.0
            
            return (efficiency1 + efficiency2 + efficiency3) / 3.0
            
        except Exception as e:
            self.logger.error(f"Three-soliton step efficiency computation failed: {e}")
            return 0.0
    
    def _compute_interaction_efficiency(self, solution: Dict[str, Any]) -> float:
        """Compute interaction efficiency."""
        try:
            overlap_integral = solution.get('overlap_integral', 0.0)
            total_mass = solution.get('total_mass', 1.0)
            
            if total_mass > 0:
                return overlap_integral / total_mass
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Interaction efficiency computation failed: {e}")
            return 0.0
    
    def _compute_three_soliton_interaction_efficiency(self, solution: Dict[str, Any]) -> float:
        """Compute three-soliton interaction efficiency."""
        try:
            overlap_integrals = solution.get('overlap_integrals', [])
            total_mass = solution.get('total_mass', 1.0)
            
            if total_mass > 0 and len(overlap_integrals) >= 3:
                # Compute average interaction efficiency
                total_overlap = sum(overlap_integrals)
                return total_overlap / total_mass
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Three-soliton interaction efficiency computation failed: {e}")
            return 0.0
    
    def _compute_7d_phase_space_properties(self, profile: np.ndarray, x: np.ndarray) -> Dict[str, float]:
        """Compute 7D phase space properties."""
        try:
            # Compute momentum space representation
            profile_fft = np.fft.fft(profile)
            k = np.fft.fftfreq(len(x), x[1] - x[0]) * 2 * np.pi
            
            # Compute phase space volume
            phase_space_volume = np.trapz(np.abs(profile_fft) ** 2, k)
            
            # Compute phase space entropy
            prob_dist = np.abs(profile_fft) ** 2
            prob_dist = prob_dist / np.sum(prob_dist)  # Normalize
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
            
            return {
                "phase_space_volume": phase_space_volume,
                "phase_space_entropy": entropy,
                "spectral_width": np.std(k * np.abs(profile_fft))
            }
            
        except Exception as e:
            self.logger.error(f"7D phase space properties computation failed: {e}")
            return {}
    
    def _step_resonator_profile(self, x: np.ndarray, position: float, width: float) -> np.ndarray:
        """Step resonator profile using 7D BVP theory."""
        try:
            distance = np.abs(x - position)
            return np.where(distance < width, 1.0, 0.0)
        except Exception as e:
            self.logger.error(f"Step resonator profile computation failed: {e}")
            return np.zeros_like(x)
