"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP flux computation for Power Balance postulate.

This module implements the computation of BVP flux at boundaries in 7D space-time,
providing the foundation for power balance validation.

Physical Meaning:
    Computes the BVP flux at the outer boundary in 7D space-time M₇,
    representing the energy flow into the system from the BVP field.
    The flux includes contributions from all 7 dimensions.

Mathematical Foundation:
    The BVP flux in 7D is computed as:
    F = ∫_∂Ω₇ (1/2) Re[E × H*] · n dS₇
    where E and H are derived from the 7D envelope field.

Example:
    >>> flux_computer = FluxComputer(domain_7d)
    >>> bvp_flux = flux_computer.compute_bvp_flux(envelope)
"""

import numpy as np
from typing import Dict, Any

from ....domain.domain_7d import Domain7D


class FluxComputer:
    """
    BVP flux computation in 7D space-time.

    Physical Meaning:
        Computes the BVP flux at boundaries in 7D space-time M₇,
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
    """

    def __init__(self, domain_7d: Domain7D):
        """
        Initialize flux computer.

        Args:
            domain_7d (Domain7D): 7D computational domain.
        """
        self.domain_7d = domain_7d

    def compute_bvp_flux(self, envelope: np.ndarray) -> float:
        """
        Compute BVP flux at boundary in full 7D space-time.

        Physical Meaning:
            Computes the BVP flux at the outer boundary in 7D space-time M₇,
            representing the energy flow into the system from the BVP field.
            The flux includes contributions from all 7 dimensions.

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
        flux_x_pos = np.sum(jx[-1, ...]) * dS_x
        flux_y_neg = -np.sum(jy[:, 0, ...]) * dS_y
        flux_y_pos = np.sum(jy[:, -1, ...]) * dS_y
        flux_z_neg = -np.sum(jz[:, :, 0, ...]) * dS_z
        flux_z_pos = np.sum(jz[:, :, -1, ...]) * dS_z

        # Flux through phase boundary faces (U(1)³ periodic boundaries)
        flux_phi1_neg = -np.sum(jphi1[:, :, :, 0, ...]) * dS_phi1
        flux_phi1_pos = np.sum(jphi1[:, :, :, -1, ...]) * dS_phi1
        flux_phi2_neg = -np.sum(jphi2[:, :, :, :, 0, ...]) * dS_phi2
        flux_phi2_pos = np.sum(jphi2[:, :, :, :, -1, ...]) * dS_phi2
        flux_phi3_neg = -np.sum(jphi3[:, :, :, :, :, 0]) * dS_phi3
        flux_phi3_pos = np.sum(jphi3[:, :, :, :, :, -1]) * dS_phi3

        # Total 7D flux
        total_flux = (
            flux_x_neg
            + flux_x_pos
            + flux_y_neg
            + flux_y_pos
            + flux_z_neg
            + flux_z_pos
            + flux_phi1_neg
            + flux_phi1_pos
            + flux_phi2_neg
            + flux_phi2_pos
            + flux_phi3_neg
            + flux_phi3_pos
        )

        return float(total_flux)
