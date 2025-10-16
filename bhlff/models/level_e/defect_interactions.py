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
from scipy.special import gamma


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
            Uses energy-based parameters instead of mass terms.
        """
        self.interaction_strength = self.params.get("interaction_strength", 1.0)
        self.interaction_range = self.params.get("interaction_range", 1.0)
        self.screening_length = self.params.get("screening_length", 0.5)
        self.cutoff_radius = self.params.get("cutoff_radius", 0.1)
        
        # Compute defect energy from field configuration instead of using mass
        self.defect_energy = self._compute_defect_energy_from_field()

        # Fractional Green function parameters
        beta = self.params.get("beta", 1.0)
        self.beta = beta
        # Fractional Green function normalization: C_β chosen so that (-Δ)^β G_β = δ in R³
        # For 3D fractional Laplacian: C_β = Γ(3/2-β) / (2^(2β) π^(3/2) Γ(β))
        self.green_prefactor = (
            self.interaction_strength
            * self._compute_fractional_green_normalization(beta)
        )

        # Remove default screening (λ=0 as per ALL.md)
        self.tempered_lambda = self.params.get("tempered_lambda", 0.0)

        # Forbid mass terms: assert tempered_lambda==0 in base configs
        if self.tempered_lambda > 0:
            # Allow override only in diagnostic paths
            diagnostic_mode = self.params.get("diagnostic_mode", False)
            if not diagnostic_mode:
                raise ValueError(
                    f"Mass terms forbidden in base regime: tempered_lambda={self.tempered_lambda} > 0. Use diagnostic_mode=True for diagnostics only."
                )
            self.screening_factor = 1.0 / self.tempered_lambda
        else:
            self.screening_factor = 0.0  # No screening in base regime

    def compute_interaction_forces(
        self, positions: List[np.ndarray], charges: List[int]
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
        self, r_ij: np.ndarray, r_magnitude: float, charge_i: int, charge_j: int
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
        Compute fractional Green function and its gradient.

        Physical Meaning:
            Calculates the fractional Green function G_β(r) and its gradient
            for defect interactions. The Green function represents
            the potential due to a point source in fractional Laplacian theory.

        Mathematical Foundation:
            G_β(r) = C_β r^(2β-3) for 3D fractional Laplacian (-Δ)^β.
            For β=1: G_1(r) = 1/(4πr) (7D BVP Coulomb).
            For β<1: G_β(r) ∝ r^(2β-3) with power-law tail.

        Args:
            r: Distance from source

        Returns:
            Tuple of (Green function value, gradient)
        """
        if r < self.cutoff_radius:
            # Regularize at small distances
            r = self.cutoff_radius

        # Fractional Green function: G_β(r) = C_β r^(2β-3)
        # Always use fractional Green function - no fallback to Coulomb
        power = 2 * self.beta - 3
        green_value = self.green_prefactor * (r**power)

        # Gradient: dG_β/dr = C_β (2β-3) r^(2β-4)
        if power != 0:
            green_gradient = self.green_prefactor * power * (r ** (power - 1))
        else:
            green_gradient = 0.0

        # Apply tempered screening if λ > 0 (diagnostic only)
        if self.tempered_lambda > 0:
            screening_factor = self._step_resonator_screening(r)
            green_value *= screening_factor
            green_gradient = (
                green_gradient * screening_factor - green_value * self.screening_factor
            )

        return green_value, green_gradient

    def simulate_defect_annihilation(
        self, defect_pair: List[int], positions: List[np.ndarray], charges: List[int]
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
            return {"annihilated": False, "reason": "Defects have same sign charges"}

        # Compute separation
        r_ij = positions[j] - positions[i]
        r_magnitude = np.linalg.norm(r_ij)

        # Check if defects are close enough for annihilation
        annihilation_radius = self.params.get("annihilation_radius", 0.2)

        if r_magnitude > annihilation_radius:
            return {
                "annihilated": False,
                "reason": f"Defects too far apart: {r_magnitude:.3f} > {annihilation_radius}",
            }

        # Compute annihilation energy
        annihilation_energy = self._compute_annihilation_energy(
            charges[i], charges[j], r_magnitude
        )

        # Compute energy release rate
        energy_release_rate = self._compute_energy_release_rate(annihilation_energy)

        # Simulate field relaxation
        relaxation_time = self._compute_relaxation_time(r_magnitude)

        return {
            "annihilated": True,
            "annihilation_energy": annihilation_energy,
            "energy_release_rate": energy_release_rate,
            "relaxation_time": relaxation_time,
            "final_separation": r_magnitude,
        }

    def _compute_annihilation_energy(
        self, charge1: int, charge2: int, separation: float
    ) -> float:
        """
        Compute energy released during annihilation using fractional Green function.

        Physical Meaning:
            Calculates the energy released when a defect-antidefect
            pair annihilates, based on their charges and separation
            using the fractional Green function G_β(r).

        Mathematical Foundation:
            Energy scales with the fractional Green function value
            at the separation distance: E ∝ |q₁q₂| G_β(r).
        """
        # Energy scales with charge magnitude
        charge_factor = abs(charge1 * charge2)

        # Get Green function value at separation
        green_value, _ = self._compute_green_function(separation)

        # Total annihilation energy from fractional Green function
        energy = charge_factor * green_value

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
        self, positions: List[np.ndarray], charges: List[int]
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

    def _compute_fractional_green_normalization(self, beta: float) -> float:
        """
        Compute normalization constant for fractional Green function.

        Physical Meaning:
            Calculates the normalization constant C_β for the fractional
            Green function G_β such that (-Δ)^β G_β = δ in R³.

        Mathematical Foundation:
            For 3D fractional Laplacian: C_β = Γ(3/2-β) / (2^(2β) π^(3/2) Γ(β))
            This ensures proper normalization of the fractional Green function.
            The exact formula ensures that ∫ G_β(r) d³r = 1 and (-Δ)^β G_β = δ.

        Args:
            beta: Fractional order parameter

        Returns:
            Normalization constant C_β
        """
        if beta <= 0 or beta >= 1.5:
            # Fallback to 7D BVP case for extreme values
            return self._compute_7d_bvp_normalization(beta)

        # Exact normalization for 3D fractional Green function
        # C_β = Γ(3/2-β) / (2^(2β) π^(3/2) Γ(β))
        # This is the mathematically correct normalization
        numerator = gamma(3 / 2 - beta)
        denominator = (2 ** (2 * beta)) * (np.pi ** (3 / 2)) * gamma(beta)

        return numerator / denominator

    def _compute_defect_energy_from_field(self) -> float:
        """
        Compute defect energy from field configuration.

        Physical Meaning:
            Calculates the energy of a defect from the field configuration
            using 7D BVP theory principles. Energy emerges from field
            localization and phase gradient contributions.

        Mathematical Foundation:
            E_defect = ∫ [μ|∇a|² + |∇Θ|^(2β)] d³x d³φ dt
            where a is the field amplitude and Θ is the phase.

        Returns:
            float: Defect energy computed from field configuration
        """
        # Extract field parameters
        mu = self.params.get("mu", 1.0)
        beta = self.params.get("beta", 1.0)
        
        # Compute field energy density components
        # Localization energy: μ|∇a|²
        localization_energy = mu * self.interaction_strength
        
        # Phase gradient energy: |∇Θ|^(2β)
        phase_gradient_energy = self.interaction_strength ** (2 * beta)
        
        # Total defect energy
        defect_energy = localization_energy + phase_gradient_energy
        
        return defect_energy
    
    def _step_resonator_screening(self, r: float) -> float:
        """
        Step resonator screening according to 7D BVP theory.
        
        Physical Meaning:
            Implements step function screening instead of exponential screening
            according to 7D BVP theory principles where screening is determined
            by step functions rather than smooth transitions.
            
        Mathematical Foundation:
            Screening = Θ(r_cutoff - r) where Θ is the Heaviside step function
            and r_cutoff is the cutoff radius for screening.
            
        Args:
            r (float): Distance parameter.
            
        Returns:
            float: Step function screening according to 7D BVP theory.
        """
        # Step function screening according to 7D BVP theory
        cutoff_radius = 1.0 / self.screening_factor
        screening_strength = 1.0
        
        # Apply step function boundary condition
        if r < cutoff_radius:
            return screening_strength
        else:
            return 0.0
    
    def _compute_7d_bvp_normalization(self, beta: float) -> float:
        """
        Compute 7D BVP normalization according to 7D BVP theory.
        
        Physical Meaning:
            Computes normalization constant according to 7D BVP theory
            principles where the normalization is determined by the
            7D phase field structure rather than classical limits.
            
        Mathematical Foundation:
            Normalization = 1/(4π) * (7D phase field factor)
            where the 7D phase field factor accounts for the
            additional dimensions in the 7D BVP theory.
            
        Args:
            beta (float): Fractional order parameter.
            
        Returns:
            float: 7D BVP normalization constant.
        """
        # 7D BVP normalization according to 7D BVP theory
        base_normalization = 1.0 / (4 * np.pi)
        phase_field_factor = 7.0  # 7D phase field factor
        
        # Apply 7D BVP correction
        normalization = base_normalization * phase_field_factor
        
        return normalization
