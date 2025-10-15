"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Soliton interaction analysis and stability.

This module implements comprehensive soliton interaction analysis
including stability, binding energy, and collective properties.

Physical Meaning:
    Analyzes soliton-soliton interactions in 7D phase field theory,
    including pairwise and multi-body interactions, stability
    criteria, and binding properties.

Example:
    >>> analyzer = SolitonInteractionAnalyzer(system, nonlinear_params)
    >>> analysis = analyzer.analyze_interactions(multi_solitons)
"""

import numpy as np
from typing import Dict, Any, List
import logging

from .base import SolitonAnalysisBase


class SolitonInteractionAnalyzer(SolitonAnalysisBase):
    """
    Soliton interaction analyzer and stability assessor.
    
    Physical Meaning:
        Analyzes soliton-soliton interactions including stability,
        binding energy, and collective properties in 7D phase field theory.
        
    Mathematical Foundation:
        Computes interaction energies, stability criteria, and
        binding properties for multi-soliton systems.
    """
    
    def __init__(self, system, nonlinear_params: Dict[str, Any]):
        """Initialize soliton interaction analyzer."""
        super().__init__(system, nonlinear_params)
        self.logger = logging.getLogger(__name__)
    
    def analyze_interactions(self, multi_solitons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze interactions between multiple solitons.
        
        Physical Meaning:
            Analyzes the collective interaction properties of multiple
            solitons, including stability, binding, and coherence.
            
        Args:
            multi_solitons (List[Dict[str, Any]]): List of multi-soliton solutions.
            
        Returns:
            Dict[str, Any]: Comprehensive interaction analysis.
        """
        try:
            if not multi_solitons:
                return {"total_interactions": 0, "stability_analysis": {}}
            
            # Extract all soliton parameters
            all_solitons = []
            for solution in multi_solitons:
                if solution.get("num_solitons", 0) > 1:
                    for i in range(1, solution["num_solitons"] + 1):
                        soliton_key = f"soliton_{i}"
                        if soliton_key in solution:
                            all_solitons.append(solution[soliton_key])
            
            if len(all_solitons) < 2:
                return {"total_interactions": 0, "stability_analysis": {}}
            
            # Compute pairwise interactions
            pairwise_interactions = []
            for i in range(len(all_solitons)):
                for j in range(i + 1, len(all_solitons)):
                    sol1 = all_solitons[i]
                    sol2 = all_solitons[j]
                    
                    interaction = self.compute_soliton_interaction_strength(
                        sol1["amplitude"], sol1["width"], sol1["position"],
                        sol2["amplitude"], sol2["width"], sol2["position"]
                    )
                    pairwise_interactions.append({
                        "soliton_pair": (i, j),
                        "interaction_strength": interaction,
                        "distance": abs(sol2["position"] - sol1["position"])
                    })
            
            # Compute collective properties
            total_interaction = sum(interaction["interaction_strength"] for interaction in pairwise_interactions)
            average_interaction = total_interaction / len(pairwise_interactions) if pairwise_interactions else 0
            
            # Stability analysis
            stable_pairs = sum(1 for interaction in pairwise_interactions if interaction["interaction_strength"] > 0)
            stability_ratio = stable_pairs / len(pairwise_interactions) if pairwise_interactions else 0
            
            return {
                "total_interactions": len(pairwise_interactions),
                "total_interaction_strength": total_interaction,
                "average_interaction_strength": average_interaction,
                "stable_pairs": stable_pairs,
                "stability_ratio": stability_ratio,
                "pairwise_interactions": pairwise_interactions,
                "collective_stability": stability_ratio > 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Soliton interaction analysis failed: {e}")
            return {"total_interactions": 0, "stability_analysis": {}}
    
    def analyze_two_soliton_stability(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float) -> Dict[str, Any]:
        """
        Analyze stability of two-soliton configuration.
        
        Physical Meaning:
            Determines the stability properties of the two-soliton system,
            including binding energy and stability criteria.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            
        Returns:
            Dict[str, Any]: Stability analysis results.
        """
        try:
            # Compute binding energy
            individual_energy1 = amp1 ** 2 / (2 * width1 ** 2)
            individual_energy2 = amp2 ** 2 / (2 * width2 ** 2)
            interaction_energy = self.compute_soliton_interaction_strength(amp1, width1, pos1, amp2, width2, pos2)
            
            binding_energy = individual_energy1 + individual_energy2 - interaction_energy
            
            # Stability criteria
            distance = abs(pos2 - pos1)
            critical_distance = (width1 + width2) / 2
            
            is_stable = binding_energy > 0 and distance > critical_distance
            is_bound = binding_energy > 0
            
            return {
                "binding_energy": binding_energy,
                "individual_energies": [individual_energy1, individual_energy2],
                "interaction_energy": interaction_energy,
                "is_stable": is_stable,
                "is_bound": is_bound,
                "critical_distance": critical_distance,
                "actual_distance": distance,
                "stability_ratio": binding_energy / (individual_energy1 + individual_energy2)
            }
            
        except Exception as e:
            self.logger.error(f"Two-soliton stability analysis failed: {e}")
            return {"is_stable": False, "binding_energy": 0.0}
    
    def analyze_three_soliton_interactions(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, amp3: float, width3: float, pos3: float) -> Dict[str, Any]:
        """
        Analyze interactions in three-soliton system.
        
        Physical Meaning:
            Analyzes all pairwise and three-body interactions in the
            three-soliton system, including stability and binding properties.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            amp3, width3, pos3 (float): Third soliton parameters.
            
        Returns:
            Dict[str, Any]: Complete interaction analysis.
        """
        try:
            # Pairwise interactions
            interaction_12 = self.compute_soliton_interaction_strength(amp1, width1, pos1, amp2, width2, pos2)
            interaction_13 = self.compute_soliton_interaction_strength(amp1, width1, pos1, amp3, width3, pos3)
            interaction_23 = self.compute_soliton_interaction_strength(amp2, width2, pos2, amp3, width3, pos3)
            
            # Three-body interaction
            distances = [abs(pos2 - pos1), abs(pos3 - pos1), abs(pos3 - pos2)]
            total_distance = sum(distances)
            effective_width = (width1 + width2 + width3) / 3
            
            three_body_interaction = self.three_body_strength * amp1 * amp2 * amp3 * self._step_resonator_interaction(total_distance, effective_width)
            
            # Stability analysis
            total_interaction = interaction_12 + interaction_13 + interaction_23 + three_body_interaction
            individual_energies = [
                amp1 ** 2 / (2 * width1 ** 2),
                amp2 ** 2 / (2 * width2 ** 2),
                amp3 ** 2 / (2 * width3 ** 2)
            ]
            
            binding_energy = sum(individual_energies) - total_interaction
            is_stable = binding_energy > 0
            
            return {
                "pairwise_interactions": {
                    "interaction_12": interaction_12,
                    "interaction_13": interaction_13,
                    "interaction_23": interaction_23
                },
                "three_body_interaction": three_body_interaction,
                "total_interaction": total_interaction,
                "individual_energies": individual_energies,
                "binding_energy": binding_energy,
                "is_stable": is_stable,
                "distances": distances,
                "interaction_ratios": {
                    "pairwise_to_three_body": (interaction_12 + interaction_13 + interaction_23) / three_body_interaction if three_body_interaction > 0 else float('inf'),
                    "strongest_pairwise": max(interaction_12, interaction_13, interaction_23),
                    "weakest_pairwise": min(interaction_12, interaction_13, interaction_23)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Three-soliton interaction analysis failed: {e}")
            return {"is_stable": False, "binding_energy": 0.0}
    
    def analyze_three_soliton_stability(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, amp3: float, width3: float, pos3: float) -> Dict[str, Any]:
        """
        Analyze stability of three-soliton configuration.
        
        Physical Meaning:
            Determines the stability properties of the three-soliton system,
            including binding energy, stability criteria, and mode analysis.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            amp3, width3, pos3 (float): Third soliton parameters.
            
        Returns:
            Dict[str, Any]: Comprehensive stability analysis.
        """
        try:
            # Compute all interaction energies
            interaction_analysis = self.analyze_three_soliton_interactions(amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3)
            
            # Stability criteria
            distances = [abs(pos2 - pos1), abs(pos3 - pos1), abs(pos3 - pos2)]
            critical_distances = [
                (width1 + width2) / 2,
                (width1 + width3) / 2,
                (width2 + width3) / 2
            ]
            
            # Check if all pairs are stable
            pairwise_stable = all(d > cd for d, cd in zip(distances, critical_distances))
            
            # Overall stability
            is_stable = interaction_analysis["is_stable"] and pairwise_stable
            
            # Full mode analysis using 7D BVP theory
            mode_analysis = self._compute_full_mode_analysis(amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3)
            total_energy = sum(interaction_analysis["individual_energies"])
            binding_ratio = interaction_analysis["binding_energy"] / total_energy if total_energy > 0 else 0
            
            return {
                "is_stable": is_stable,
                "pairwise_stable": pairwise_stable,
                "binding_energy": interaction_analysis["binding_energy"],
                "binding_ratio": binding_ratio,
                "critical_distances": critical_distances,
                "actual_distances": distances,
                "stability_margin": min(d / cd for d, cd in zip(distances, critical_distances)) if all(cd > 0 for cd in critical_distances) else 0,
                "interaction_analysis": interaction_analysis,
                "mode_analysis": mode_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Three-soliton stability analysis failed: {e}")
            return {"is_stable": False, "binding_energy": 0.0}
    
    def _compute_full_mode_analysis(self, amp1: float, width1: float, pos1: float, amp2: float, width2: float, pos2: float, amp3: float, width3: float, pos3: float) -> Dict[str, Any]:
        """
        Compute full mode analysis for three-soliton system using 7D BVP theory.
        
        Physical Meaning:
            Performs complete mode analysis of the three-soliton system,
            including collective modes, stability eigenvalues, and
            interaction-induced mode splitting.
            
        Mathematical Foundation:
            Computes the full eigenvalue spectrum of the three-soliton
            system using 7D fractional Laplacian equations and
            soliton-soliton interaction potentials.
            
        Args:
            amp1, width1, pos1 (float): First soliton parameters.
            amp2, width2, pos2 (float): Second soliton parameters.
            amp3, width3, pos3 (float): Third soliton parameters.
            
        Returns:
            Dict[str, Any]: Complete mode analysis results.
        """
        try:
            # Setup spatial grid for mode analysis
            x = np.linspace(-20.0, 20.0, 400)
            dx = x[1] - x[0]
            
            # Compute individual soliton profiles
            profile1 = amp1 * np.exp(-((x - pos1) ** 2) / (2 * width1 ** 2))
            profile2 = amp2 * np.exp(-((x - pos2) ** 2) / (2 * width2 ** 2))
            profile3 = amp3 * np.exp(-((x - pos3) ** 2) / (2 * width3 ** 2))
            total_profile = profile1 + profile2 + profile3
            
            # Compute interaction potential matrix
            interaction_matrix = self._compute_interaction_matrix(x, amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3)
            
            # Compute kinetic energy matrix (fractional Laplacian)
            kinetic_matrix = self._compute_kinetic_matrix(x)
            
            # Total Hamiltonian matrix
            hamiltonian_matrix = kinetic_matrix + interaction_matrix
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
            
            # Analyze mode properties
            stable_modes = sum(1 for ev in eigenvalues if ev > 0)
            unstable_modes = sum(1 for ev in eigenvalues if ev < 0)
            zero_modes = sum(1 for ev in eigenvalues if abs(ev) < 1e-10)
            
            # Compute collective mode frequencies
            collective_frequencies = np.sqrt(np.abs(eigenvalues[eigenvalues > 0]))
            
            # Compute mode participation ratios
            participation_ratios = self._compute_mode_participation_ratios(eigenvectors, profile1, profile2, profile3)
            
            # Compute interaction-induced mode splitting
            mode_splitting = self._compute_mode_splitting(eigenvalues, amp1, width1, pos1, amp2, width2, pos2, amp3, width3, pos3)
            
            return {
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "stable_modes": stable_modes,
                "unstable_modes": unstable_modes,
                "zero_modes": zero_modes,
                "collective_frequencies": collective_frequencies,
                "participation_ratios": participation_ratios,
                "mode_splitting": mode_splitting,
                "hamiltonian_matrix": hamiltonian_matrix,
                "interaction_matrix": interaction_matrix,
                "kinetic_matrix": kinetic_matrix
            }
            
        except Exception as e:
            self.logger.error(f"Full mode analysis computation failed: {e}")
            return {
                "eigenvalues": np.array([]),
                "eigenvectors": np.array([]),
                "stable_modes": 0,
                "unstable_modes": 0,
                "zero_modes": 0,
                "collective_frequencies": np.array([]),
                "participation_ratios": {},
                "mode_splitting": 0.0
            }
