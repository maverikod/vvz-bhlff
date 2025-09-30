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
        Compute BVP flux at boundary in full 7D space-time.
        
        Physical Meaning:
            Computes the BVP flux at the outer boundary in 7D space-time M₇,
            representing the energy flow into the system from the BVP field.
            The flux includes contributions from all 7 dimensions:
            - 3 spatial coordinates (x, y, z)
            - 3 phase coordinates (φ₁, φ₂, φ₃) 
            - 1 temporal coordinate (t)
            
        Mathematical Foundation:
            The BVP flux in 7D is computed as:
            F = ∫_∂Ω₇ (1/2) Re[E × H*] · n dS₇
            where E and H are derived from the 7D envelope field,
            and n is the outward normal in 7D space.
            
        Args:
            envelope (np.ndarray): 7D envelope field with shape
                (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t)
            
        Returns:
            float: Computed BVP flux at boundary in 7D space-time.
        """
        # Use the latest time slice
        a_t = envelope[..., -1]
        differentials = self.domain_7d.get_differentials()
        
        # 7D differentials
        dx = differentials["dx"]
        dy = differentials["dy"] 
        dz = differentials["dz"]
        dphi1 = differentials["dphi_1"]
        dphi2 = differentials["dphi_2"]
        dphi3 = differentials["dphi_3"]

        # Spatial gradients (axes 0,1,2)
        grad_x = np.gradient(a_t, dx, axis=0)
        grad_y = np.gradient(a_t, dy, axis=1)
        grad_z = np.gradient(a_t, dz, axis=2)
        
        # Phase gradients (axes 3,4,5) - U(1)³ structure
        grad_phi1 = np.gradient(a_t, dphi1, axis=3)
        grad_phi2 = np.gradient(a_t, dphi2, axis=4)
        grad_phi3 = np.gradient(a_t, dphi3, axis=5)

        # 7D current density components j = Im(a* · ∇a)
        jx = np.imag(np.conj(a_t) * grad_x)
        jy = np.imag(np.conj(a_t) * grad_y)
        jz = np.imag(np.conj(a_t) * grad_z)
        jphi1 = np.imag(np.conj(a_t) * grad_phi1)
        jphi2 = np.imag(np.conj(a_t) * grad_phi2)
        jphi3 = np.imag(np.conj(a_t) * grad_phi3)

        # 7D surface elements for all boundary faces
        dS_x = dy * dz * dphi1 * dphi2 * dphi3
        dS_y = dx * dz * dphi1 * dphi2 * dphi3
        dS_z = dx * dy * dphi1 * dphi2 * dphi3
        dS_phi1 = dx * dy * dz * dphi2 * dphi3
        dS_phi2 = dx * dy * dz * dphi1 * dphi3
        dS_phi3 = dx * dy * dz * dphi1 * dphi2

        # Flux through spatial boundary faces
        flux_x_neg = -np.sum(jx[0, ...]) * dS_x
        flux_x_pos =  np.sum(jx[-1, ...]) * dS_x
        flux_y_neg = -np.sum(jy[:, 0, ...]) * dS_y
        flux_y_pos =  np.sum(jy[:, -1, ...]) * dS_y
        flux_z_neg = -np.sum(jz[:, :, 0, ...]) * dS_z
        flux_z_pos =  np.sum(jz[:, :, -1, ...]) * dS_z
        
        # Flux through phase boundary faces (U(1)³ periodic boundaries)
        flux_phi1_neg = -np.sum(jphi1[:, :, :, 0, ...]) * dS_phi1
        flux_phi1_pos =  np.sum(jphi1[:, :, :, -1, ...]) * dS_phi1
        flux_phi2_neg = -np.sum(jphi2[:, :, :, :, 0, ...]) * dS_phi2
        flux_phi2_pos =  np.sum(jphi2[:, :, :, :, -1, ...]) * dS_phi2
        flux_phi3_neg = -np.sum(jphi3[:, :, :, :, :, 0]) * dS_phi3
        flux_phi3_pos =  np.sum(jphi3[:, :, :, :, :, -1]) * dS_phi3

        # Total 7D flux
        total_flux = (
            flux_x_neg + flux_x_pos +
            flux_y_neg + flux_y_pos +
            flux_z_neg + flux_z_pos +
            flux_phi1_neg + flux_phi1_pos +
            flux_phi2_neg + flux_phi2_pos +
            flux_phi3_neg + flux_phi3_pos
        )

        return float(total_flux)
    
    def _compute_core_energy_growth(self, envelope: np.ndarray) -> float:
        """
        Compute growth of static core energy in 7D space-time.
        
        Physical Meaning:
            Computes the growth of static core energy in 7D space-time M₇,
            representing the energy stored in the core region of the BVP field.
            This includes contributions from all 7 dimensions according to
            the 7D phase field theory.
            
        Mathematical Foundation:
            The core energy growth in 7D is computed as:
            E_core = ∫_core (1/2)[f_φ²|∇_xΘ|² + f_φ²|∇_φΘ|² + β₄(ΔΘ)² + γ₆|∇Θ|⁶ + ...] dV₇
            where Θ is the 7D phase vector, f_φ is the phase field constant,
            and the integral includes all 7 dimensions.
            
        Args:
            envelope (np.ndarray): 7D envelope field with shape
                (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t)
            
        Returns:
            float: Computed core energy growth in 7D space-time.
        """
        differentials = self.domain_7d.get_differentials()
        dx = differentials["dx"]
        dy = differentials["dy"]
        dz = differentials["dz"]
        dphi1 = differentials["dphi_1"]
        dphi2 = differentials["dphi_2"]
        dphi3 = differentials["dphi_3"]
        dt = self.domain_7d.temporal_config.dt

        # Use last and previous time slices (if available)
        a_t = envelope[..., -1]
        if envelope.shape[-1] > 1:
            a_prev = envelope[..., -2]
        else:
            a_prev = np.zeros_like(a_t)

        # 7D energy density parameters from theory
        f_phi = float(self.config.get("f_phi", 1.0))  # Phase field constant
        k0 = float(self.config.get("k0", 1.0))        # Wave number
        beta4 = float(self.config.get("beta4", 0.1))  # Fourth-order coefficient
        gamma6 = float(self.config.get("gamma6", 0.01))  # Sixth-order coefficient

        def energy_density_7d(a: np.ndarray) -> np.ndarray:
            """
            Compute 7D energy density according to theory.
            
            Physical Meaning:
                Implements the 7D energy functional:
                E[Θ] = f_φ²|∇_xΘ|² + f_φ²|∇_φΘ|² + β₄(ΔΘ)² + γ₆|∇Θ|⁶
            """
            # Spatial gradients (axes 0,1,2)
            grad_x = np.gradient(a, dx, axis=0)
            grad_y = np.gradient(a, dy, axis=1)
            grad_z = np.gradient(a, dz, axis=2)
            
            # Phase gradients (axes 3,4,5) - U(1)³ structure
            grad_phi1 = np.gradient(a, dphi1, axis=3)
            grad_phi2 = np.gradient(a, dphi2, axis=4)
            grad_phi3 = np.gradient(a, dphi3, axis=5)
            
            # Spatial gradient magnitude squared
            grad_spatial_sq = np.abs(grad_x)**2 + np.abs(grad_y)**2 + np.abs(grad_z)**2
            
            # Phase gradient magnitude squared
            grad_phase_sq = np.abs(grad_phi1)**2 + np.abs(grad_phi2)**2 + np.abs(grad_phi3)**2
            
            # Total gradient magnitude
            grad_total_sq = grad_spatial_sq + grad_phase_sq
            
            # Laplacian (second derivatives)
            laplacian_x = np.gradient(grad_x, dx, axis=0)
            laplacian_y = np.gradient(grad_y, dy, axis=1)
            laplacian_z = np.gradient(grad_z, dz, axis=2)
            laplacian_phi1 = np.gradient(grad_phi1, dphi1, axis=3)
            laplacian_phi2 = np.gradient(grad_phi2, dphi2, axis=4)
            laplacian_phi3 = np.gradient(grad_phi3, dphi3, axis=5)
            
            laplacian_sq = (np.abs(laplacian_x)**2 + np.abs(laplacian_y)**2 + 
                           np.abs(laplacian_z)**2 + np.abs(laplacian_phi1)**2 + 
                           np.abs(laplacian_phi2)**2 + np.abs(laplacian_phi3)**2)
            
            # 7D energy density according to theory
            energy_density = (
                f_phi**2 * grad_spatial_sq +      # f_φ²|∇_xΘ|²
                f_phi**2 * grad_phase_sq +        # f_φ²|∇_φΘ|²
                beta4 * laplacian_sq +            # β₄(ΔΘ)²
                gamma6 * (grad_total_sq**3) +     # γ₆|∇Θ|⁶
                k0**2 * np.abs(a)**2              # k₀²|a|² (mass term)
            )
            
            return energy_density

        e_t = energy_density_7d(a_t)
        e_prev = energy_density_7d(a_prev)

        # Define core region as high-amplitude region
        amp_t = np.abs(a_t)
        core_mask = amp_t > (0.5 * np.max(amp_t) if np.max(amp_t) > 0 else 0)

        # 7D volume element
        dV7 = dx * dy * dz * dphi1 * dphi2 * dphi3
        E_core_t = np.sum(e_t[core_mask]) * dV7
        E_core_prev = np.sum(e_prev[core_mask]) * dV7

        dE_dt = (E_core_t - E_core_prev) / (dt if dt > 0 else 1.0)
        return float(dE_dt)
    
    def _compute_radiation_losses(self, envelope: np.ndarray) -> float:
        """
        Compute EM/weak radiation and losses in 7D space-time.
        
        Physical Meaning:
            Computes the EM/weak radiation and losses in 7D space-time M₇,
            representing the energy radiated away from the system through
            electromagnetic and weak interactions. This includes contributions
            from all 7 dimensions according to the BVP theory.
            
        Mathematical Foundation:
            The radiation losses in 7D are computed as:
            P_rad = ∫_∂Ω₇ σ|E|² dS₇ + ∫_∂Ω₇ σ_weak|W|² dS₇
            where σ and σ_weak are the electromagnetic and weak conductivities,
            E and W are the electromagnetic and weak fields derived from
            the 7D BVP envelope, and the integral is over the 7D boundary.
            
        Args:
            envelope (np.ndarray): 7D envelope field with shape
                (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t)
            
        Returns:
            float: Computed EM/weak radiation losses in 7D space-time.
        """
        # Get 7D boundary flux components
        outward_spatial, inward_spatial = self._split_boundary_flux_spatial(envelope)
        outward_phase, inward_phase = self._split_boundary_flux_phase(envelope)
        
        # EM radiation losses (outward flux from spatial boundaries)
        em_losses = float(outward_spatial)
        
        # Weak radiation losses (outward flux from phase boundaries)
        weak_losses = float(outward_phase)
        
        # Total radiation losses
        total_radiation_losses = em_losses + weak_losses
        
        return total_radiation_losses
    
    def _compute_reflection(self, envelope: np.ndarray) -> float:
        """
        Compute reflection component in 7D space-time.
        
        Physical Meaning:
            Computes the reflection component in 7D space-time M₇,
            representing the energy reflected back from the boundaries
            due to impedance mismatch and boundary conditions. This includes
            reflections from both spatial and phase boundaries.
            
        Mathematical Foundation:
            The reflection in 7D is computed as:
            R = ∫_∂Ω₇ |r|²|E_inc|² dS₇
            where r is the reflection coefficient, E_inc is the
            incident field amplitude, and the integral is over the 7D boundary.
            
        Args:
            envelope (np.ndarray): 7D envelope field with shape
                (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t)
            
        Returns:
            float: Computed reflection component in 7D space-time.
        """
        # Get 7D boundary flux components
        outward_spatial, inward_spatial = self._split_boundary_flux_spatial(envelope)
        outward_phase, inward_phase = self._split_boundary_flux_phase(envelope)
        
        # Total reflection (inward flux from all boundaries)
        total_reflection = float(-inward_spatial - inward_phase)
        
        return total_reflection

    def _split_boundary_flux_spatial(self, envelope: np.ndarray) -> tuple:
        """
        Split spatial boundary flux into outward and inward components.
        
        Physical Meaning:
            Separates the flux through spatial boundaries (x, y, z) into
            outward (positive) and inward (negative) components for
            EM radiation analysis.
            
        Returns:
            tuple: (outward_spatial, inward_spatial)
        """
        a_t = envelope[..., -1]
        differentials = self.domain_7d.get_differentials()
        dx = differentials["dx"]
        dy = differentials["dy"]
        dz = differentials["dz"]
        dphi1 = differentials["dphi_1"]
        dphi2 = differentials["dphi_2"]
        dphi3 = differentials["dphi_3"]

        # Spatial gradients only
        grad_x = np.gradient(a_t, dx, axis=0)
        grad_y = np.gradient(a_t, dy, axis=1)
        grad_z = np.gradient(a_t, dz, axis=2)
        jx = np.imag(np.conj(a_t) * grad_x)
        jy = np.imag(np.conj(a_t) * grad_y)
        jz = np.imag(np.conj(a_t) * grad_z)

        # 7D surface elements for spatial faces
        dS_x = dy * dz * dphi1 * dphi2 * dphi3
        dS_y = dx * dz * dphi1 * dphi2 * dphi3
        dS_z = dx * dy * dphi1 * dphi2 * dphi3

        spatial_faces = [
            (-jx[0, ...], dS_x),   # -x face (n = -ex)
            ( jx[-1, ...], dS_x),  # +x face
            (-jy[:, 0, ...], dS_y),
            ( jy[:, -1, ...], dS_y),
            (-jz[:, :, 0, ...], dS_z),
            ( jz[:, :, -1, ...], dS_z),
        ]

        outward = 0.0
        inward = 0.0
        for face_flux_density, dS in spatial_faces:
            face_flux = np.sum(face_flux_density) * dS
            if face_flux >= 0:
                outward += face_flux
            else:
                inward += face_flux

        return float(outward), float(inward)
    
    def _split_boundary_flux_phase(self, envelope: np.ndarray) -> tuple:
        """
        Split phase boundary flux into outward and inward components.
        
        Physical Meaning:
            Separates the flux through phase boundaries (φ₁, φ₂, φ₃) into
            outward (positive) and inward (negative) components for
            weak interaction analysis.
            
        Returns:
            tuple: (outward_phase, inward_phase)
        """
        a_t = envelope[..., -1]
        differentials = self.domain_7d.get_differentials()
        dx = differentials["dx"]
        dy = differentials["dy"]
        dz = differentials["dz"]
        dphi1 = differentials["dphi_1"]
        dphi2 = differentials["dphi_2"]
        dphi3 = differentials["dphi_3"]

        # Phase gradients only
        grad_phi1 = np.gradient(a_t, dphi1, axis=3)
        grad_phi2 = np.gradient(a_t, dphi2, axis=4)
        grad_phi3 = np.gradient(a_t, dphi3, axis=5)
        jphi1 = np.imag(np.conj(a_t) * grad_phi1)
        jphi2 = np.imag(np.conj(a_t) * grad_phi2)
        jphi3 = np.imag(np.conj(a_t) * grad_phi3)

        # 7D surface elements for phase faces
        dS_phi1 = dx * dy * dz * dphi2 * dphi3
        dS_phi2 = dx * dy * dz * dphi1 * dphi3
        dS_phi3 = dx * dy * dz * dphi1 * dphi2

        phase_faces = [
            (-jphi1[:, :, :, 0, ...], dS_phi1),   # -φ₁ face
            ( jphi1[:, :, :, -1, ...], dS_phi1),  # +φ₁ face
            (-jphi2[:, :, :, :, 0, ...], dS_phi2),
            ( jphi2[:, :, :, :, -1, ...], dS_phi2),
            (-jphi3[:, :, :, :, :, 0], dS_phi3),
            ( jphi3[:, :, :, :, :, -1], dS_phi3),
        ]

        outward = 0.0
        inward = 0.0
        for face_flux_density, dS in phase_faces:
            face_flux = np.sum(face_flux_density) * dS
            if face_flux >= 0:
                outward += face_flux
            else:
                inward += face_flux

        return float(outward), float(inward)


