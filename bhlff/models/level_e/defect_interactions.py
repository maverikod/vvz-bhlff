"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Defect interactions implementation for Level E experiments in 7D phase field theory.

This module implements interactions between topological defects including
forces, annihilation, and multi-defect systems.

Theoretical Background:
    Defect interactions are governed by Green functions and depend on
    the topological charges and separations between defects. Defects
    can attract, repel, or annihilate depending on their charges.

Mathematical Foundation:
    Interaction potential: U_int = Σᵢⱼ qᵢqⱼ G(rᵢⱼ) where G is the Green
    function and rᵢⱼ is the separation between defects i and j.

Example:
    >>> system = MultiDefectSystem(domain, physics_params)
    >>> forces = system.compute_interaction_forces()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class DefectInteractions:
    """
    Defect interactions calculator for topological defects.

    Physical Meaning:
        Implements interactions between topological defects including
        forces, potentials, and annihilation processes.

    Mathematical Foundation:
        Computes interaction forces based on Green functions and
        topological charges: Fᵢ = -∇ᵢ Σⱼ qᵢqⱼ G(rᵢⱼ).
    """

    def __init__(self, domain: "Domain", physics_params: Dict[str, Any]):
        """
        Initialize defect interactions calculator.

        Physical Meaning:
            Sets up the computational framework for defect interactions
            including Green functions, interaction potentials, and
            annihilation processes.

        Args:
            domain: Computational domain
            physics_params: Physical parameters
        """
        self.domain = domain
        self.params = physics_params
        self._setup_interaction_parameters()

    def _setup_interaction_parameters(self) -> None:
        """
        Setup parameters for defect interactions.
        
        Physical Meaning:
            Initializes the physical parameters required for
            defect interaction calculations including interaction
            strength, range, and Green function parameters.
        """
        self.interaction_strength = self.params.get("interaction_strength", 1.0)
        self.interaction_range = self.params.get("interaction_range", 1.0)
        self.screening_length = self.params.get("screening_length", 0.5)
        self.cutoff_radius = self.params.get("cutoff_radius", 0.1)
        
        # Green function parameters
        self.green_prefactor = self.interaction_strength / (4 * np.pi)
        self.screening_factor = 1.0 / self.screening_length

    def compute_interaction_forces(
        self, 
        positions: List[np.ndarray], 
        charges: List[int]
    ) -> List[np.ndarray]:
        """
        Compute interaction forces between defects.

        Physical Meaning:
            Calculates the forces acting on each defect due to
            interactions with all other defects in the system.

        Mathematical Foundation:
            Fᵢ = -∇ᵢ Σⱼ qᵢqⱼ G(rᵢⱼ) where G is the Green function
            and the sum is over all other defects j ≠ i.

        Args:
            positions: List of defect positions
            charges: List of defect charges

        Returns:
            List of force vectors for each defect
        """
        n_defects = len(positions)
        forces = [np.zeros(3) for _ in range(n_defects)]

        # Compute pairwise interactions
        for i in range(n_defects):
            for j in range(n_defects):
                if i != j:
                    # Compute separation vector
                    r_ij = positions[j] - positions[i]
                    r_magnitude = np.linalg.norm(r_ij)

                    # Skip if defects are too close
                    if r_magnitude < self.cutoff_radius:
                        continue

                    # Compute interaction force
                    force_ij = self._compute_pair_force(
                        r_ij, r_magnitude, charges[i], charges[j]
                    )
                    forces[i] += force_ij

        return forces

    def _compute_pair_force(
        self, 
        r_ij: np.ndarray, 
        r_magnitude: float, 
        charge_i: int, 
        charge_j: int
    ) -> np.ndarray:
        """
        Compute force between defect pair.

        Physical Meaning:
            Calculates the force between two defects based on
            their charges and separation.

        Mathematical Foundation:
            F = -qᵢqⱼ ∇G(r) where G is the Green function.

        Args:
            r_ij: Separation vector from defect i to defect j
            r_magnitude: Magnitude of separation
            charge_i: Charge of defect i
            charge_j: Charge of defect j

        Returns:
            Force vector on defect i due to defect j
        """
        # Compute Green function and its gradient
        green_value, green_gradient = self._compute_green_function(r_magnitude)

        # Force magnitude
        force_magnitude = charge_i * charge_j * green_gradient

        # Force direction (along separation vector)
        if r_magnitude > 1e-10:
            force_direction = r_ij / r_magnitude
        else:
            force_direction = np.zeros(3)

        # Total force
        force = force_magnitude * force_direction

        return force

    def _compute_green_function(self, r: float) -> Tuple[float, float]:
        """
        Compute Green function and its gradient.

        Physical Meaning:
            Calculates the Green function G(r) and its gradient
            for defect interactions. The Green function represents
            the potential due to a point source.

        Mathematical Foundation:
            G(r) = (1/4πr) exp(-r/λ) where λ is the screening length.

        Args:
            r: Distance from source

        Returns:
            Tuple of (Green function value, gradient)
        """
        if r < self.cutoff_radius:
            # Regularize at small distances
            r = self.cutoff_radius

        # Screened Coulomb potential
        screening_factor = np.exp(-r * self.screening_factor)
        
        # Green function value
        green_value = self.green_prefactor * screening_factor / r
        
        # Green function gradient
        green_gradient = -self.green_prefactor * screening_factor * (
            1/r**2 + self.screening_factor/r
        )

        return green_value, green_gradient

    def simulate_defect_annihilation(
        self, 
        defect_pair: List[int],
        positions: List[np.ndarray],
        charges: List[int]
    ) -> Dict[str, Any]:
        """
        Simulate annihilation of defect-antidefect pair.

        Physical Meaning:
            Models the process where a defect and antidefect approach
            and annihilate, releasing energy and creating topological
            transitions in the field.

        Mathematical Foundation:
            Annihilation occurs when defects of opposite charge
            approach within a critical distance, leading to
            energy release and field relaxation.

        Args:
            defect_pair: Indices of defect and antidefect
            positions: Current defect positions
            charges: Current defect charges

        Returns:
            Dictionary containing annihilation results
        """
        i, j = defect_pair
        
        # Check if defects have opposite charges
        if charges[i] * charges[j] >= 0:
            return {
                "annihilated": False,
                "reason": "Defects have same sign charges"
            }

        # Compute separation
        r_ij = positions[j] - positions[i]
        r_magnitude = np.linalg.norm(r_ij)

        # Check if defects are close enough for annihilation
        annihilation_radius = self.params.get("annihilation_radius", 0.2)
        
        if r_magnitude > annihilation_radius:
            return {
                "annihilated": False,
                "reason": f"Defects too far apart: {r_magnitude:.3f} > {annihilation_radius}"
            }

        # Compute annihilation energy
        annihilation_energy = self._compute_annihilation_energy(charges[i], charges[j], r_magnitude)

        # Compute energy release rate
        energy_release_rate = self._compute_energy_release_rate(annihilation_energy)

        # Simulate field relaxation
        relaxation_time = self._compute_relaxation_time(r_magnitude)

        return {
            "annihilated": True,
            "annihilation_energy": annihilation_energy,
            "energy_release_rate": energy_release_rate,
            "relaxation_time": relaxation_time,
            "final_separation": r_magnitude
        }

    def _compute_annihilation_energy(
        self, 
        charge1: int, 
        charge2: int, 
        separation: float
    ) -> float:
        """
        Compute energy released during annihilation.
        
        Physical Meaning:
            Calculates the energy released when a defect-antidefect
            pair annihilates, based on their charges and separation.
        """
        # Energy scales with charge magnitude
        charge_factor = abs(charge1 * charge2)
        
        # Energy decreases with separation
        separation_factor = np.exp(-separation / self.screening_length)
        
        # Total annihilation energy
        energy = charge_factor * self.interaction_strength * separation_factor
        
        return energy

    def _compute_energy_release_rate(self, annihilation_energy: float) -> float:
        """
        Compute rate of energy release during annihilation.
        
        Physical Meaning:
            Calculates how quickly energy is released during
            the annihilation process.
        """
        # Energy release rate depends on annihilation energy
        release_rate = annihilation_energy / self.params.get("annihilation_time", 1.0)
        
        return release_rate

    def _compute_relaxation_time(self, separation: float) -> float:
        """
        Compute field relaxation time after annihilation.
        
        Physical Meaning:
            Calculates the time required for the field to relax
            to its new configuration after defect annihilation.
        """
        # Relaxation time increases with initial separation
        base_time = self.params.get("base_relaxation_time", 0.1)
        separation_factor = 1.0 + separation / self.screening_length
        
        relaxation_time = base_time * separation_factor
        
        return relaxation_time

    def compute_interaction_potential(
        self, 
        positions: List[np.ndarray], 
        charges: List[int]
    ) -> float:
        """
        Compute total interaction potential energy.

        Physical Meaning:
            Calculates the total potential energy of the defect
            system due to all pairwise interactions.

        Mathematical Foundation:
            U = (1/2) Σᵢⱼ qᵢqⱼ G(rᵢⱼ) where the factor 1/2 avoids
            double counting.

        Args:
            positions: List of defect positions
            charges: List of defect charges

        Returns:
            Total interaction potential energy
        """
        n_defects = len(positions)
        total_potential = 0.0

        # Compute pairwise interactions
        for i in range(n_defects):
            for j in range(i + 1, n_defects):
                # Compute separation
                r_ij = positions[j] - positions[i]
                r_magnitude = np.linalg.norm(r_ij)

                # Skip if defects are too close
                if r_magnitude < self.cutoff_radius:
                    continue

                # Compute Green function
                green_value, _ = self._compute_green_function(r_magnitude)

                # Add to total potential
                total_potential += charges[i] * charges[j] * green_value

        return total_potential
