"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Radial analysis module for boundary effects.

This module implements radial analysis functionality
for Level C test C1 in 7D phase field theory.

Physical Meaning:
    Analyzes radial profiles for boundary effects,
    including field distribution and concentration patterns.

Example:
    >>> analyzer = RadialAnalyzer(bvp_core)
    >>> results = analyzer.analyze_radial_profile(domain, boundary, field)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

from bhlff.core.bvp import BVPCore
from .data_structures import BoundaryGeometry, RadialProfile


class RadialAnalyzer:
    """
    Radial analysis for boundary effects.

    Physical Meaning:
        Analyzes radial profiles for boundary effects,
        including field distribution and concentration patterns.

    Mathematical Foundation:
        Implements radial analysis:
        - Radial profile: A(r) = (1/4π) ∫_S(r) |a(x)|² dS
        - Local maxima detection in radial profiles
        - Field concentration analysis
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize radial analyzer.

        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def analyze_radial_profile(
        self,
        domain: Dict[str, Any],
        boundary: BoundaryGeometry,
        field: np.ndarray,
    ) -> RadialProfile:
        """
        Analyze radial profile around boundary.

        Physical Meaning:
            Computes the radial profile A(r) of field amplitude
            around the boundary, revealing field distribution
            and concentration patterns.

        Mathematical Foundation:
            A(r) = (1/4π) ∫_S(r) |a(x)|² dS
            where S(r) is the spherical surface at radius r.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            field (np.ndarray): Field configuration.

        Returns:
            RadialProfile: Radial profile analysis.
        """
        # Compute radial distances and amplitudes
        radii, amplitudes = self._compute_radial_profile(
            domain, boundary, field
        )

        # Find local maxima
        local_maxima = self._find_local_maxima(radii, amplitudes)

        # Create radial profile
        profile = RadialProfile(
            radii=radii,
            amplitudes=amplitudes,
            local_maxima=local_maxima,
        )

        return profile

    def _compute_radial_profile(
        self,
        domain: Dict[str, Any],
        boundary: BoundaryGeometry,
        field: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radial profile.

        Physical Meaning:
            Computes the radial profile A(r) by integrating
            the field amplitude over spherical surfaces.

        Mathematical Foundation:
            A(r) = (1/4π) ∫_S(r) |a(x)|² dS
            where S(r) is the spherical surface at radius r.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            field (np.ndarray): Field configuration.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Radii and amplitudes.
        """
        N = domain.get("N", 64)
        L = domain.get("L", 1.0)

        # Create coordinate arrays
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Compute distances from boundary center
        distances = np.sqrt(
            (X - boundary.center[0]) ** 2
            + (Y - boundary.center[1]) ** 2
            + (Z - boundary.center[2]) ** 2
        )

        # Define radial bins
        r_min = 0.0
        r_max = min(boundary.radius * 3, L / 2)
        num_bins = 50
        radii = np.linspace(r_min, r_max, num_bins)

        # Compute amplitudes for each radial bin
        amplitudes = []
        for r in radii:
            amplitude = self._compute_amplitude_at_radius(
                field, distances, r, domain
            )
            amplitudes.append(amplitude)

        amplitudes = np.array(amplitudes)

        return radii, amplitudes

    def _compute_amplitude_at_radius(
        self,
        field: np.ndarray,
        distances: np.ndarray,
        radius: float,
        domain: Dict[str, Any],
    ) -> float:
        """
        Compute amplitude at specific radius.

        Physical Meaning:
            Computes the field amplitude at a specific radius
            by integrating over the spherical surface.

        Mathematical Foundation:
            A(r) = (1/4π) ∫_S(r) |a(x)|² dS
            where S(r) is the spherical surface at radius r.

        Args:
            field (np.ndarray): Field configuration.
            distances (np.ndarray): Distances from boundary center.
            radius (float): Radial distance.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            float: Amplitude at radius.
        """
        # Create spherical shell mask
        shell_mask = self._create_spherical_shell_mask(distances, radius, domain)

        # Compute amplitude in shell
        if np.any(shell_mask):
            amplitude = np.mean(np.abs(field[shell_mask]) ** 2)
        else:
            amplitude = 0.0

        return amplitude

    def _create_spherical_shell_mask(
        self, distances: np.ndarray, radius: float, domain: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create spherical shell mask.

        Physical Meaning:
            Creates a mask for the spherical shell at radius r
            for radial profile computation.

        Args:
            distances (np.ndarray): Distances from boundary center.
            radius (float): Radial distance.
            domain (Dict[str, Any]): Domain parameters.

        Returns:
            np.ndarray: Spherical shell mask.
        """
        # Shell thickness
        dr = domain.get("L", 1.0) / domain.get("N", 64)

        # Create shell mask
        shell_mask = (distances >= radius - dr / 2) & (distances <= radius + dr / 2)

        return shell_mask

    def _find_local_maxima(
        self, radii: np.ndarray, amplitudes: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Find local maxima in radial profile.

        Physical Meaning:
            Identifies local maxima in the radial amplitude profile,
            indicating regions of field concentration.

        Args:
            radii (np.ndarray): Radial distances.
            amplitudes (np.ndarray): Field amplitudes.

        Returns:
            List[Tuple[float, float]]: Local maxima (radius, amplitude).
        """
        maxima = []

        for i in range(1, len(amplitudes) - 1):
            if (amplitudes[i] > amplitudes[i - 1] and 
                amplitudes[i] > amplitudes[i + 1]):
                maxima.append((radii[i], amplitudes[i]))

        return maxima

    def analyze_field_concentration(
        self, profile: RadialProfile, boundary: BoundaryGeometry
    ) -> Dict[str, Any]:
        """
        Analyze field concentration.

        Physical Meaning:
            Analyzes the field concentration patterns
            in the radial profile around the boundary.

        Args:
            profile (RadialProfile): Radial profile.
            boundary (BoundaryGeometry): Boundary geometry.

        Returns:
            Dict[str, Any]: Field concentration analysis.
        """
        # Analyze concentration near boundary
        near_boundary = self._analyze_near_boundary_concentration(
            profile, boundary
        )

        # Analyze concentration far from boundary
        far_boundary = self._analyze_far_boundary_concentration(
            profile, boundary
        )

        # Analyze overall concentration pattern
        overall_pattern = self._analyze_overall_concentration_pattern(profile)

        return {
            "near_boundary": near_boundary,
            "far_boundary": far_boundary,
            "overall_pattern": overall_pattern,
            "concentration_analysis_complete": True,
        }

    def _analyze_near_boundary_concentration(
        self, profile: RadialProfile, boundary: BoundaryGeometry
    ) -> Dict[str, Any]:
        """
        Analyze concentration near boundary.

        Physical Meaning:
            Analyzes field concentration in the region
            near the boundary.

        Args:
            profile (RadialProfile): Radial profile.
            boundary (BoundaryGeometry): Boundary geometry.

        Returns:
            Dict[str, Any]: Near boundary concentration analysis.
        """
        # Define near boundary region
        near_radius = boundary.radius * 1.5
        near_mask = profile.radii <= near_radius

        if np.any(near_mask):
            near_amplitudes = profile.amplitudes[near_mask]
            near_radii = profile.radii[near_mask]

            # Compute concentration metrics
            max_amplitude = np.max(near_amplitudes)
            mean_amplitude = np.mean(near_amplitudes)
            concentration_strength = max_amplitude / mean_amplitude if mean_amplitude > 0 else 0.0

            return {
                "max_amplitude": max_amplitude,
                "mean_amplitude": mean_amplitude,
                "concentration_strength": concentration_strength,
                "region_radius": near_radius,
            }
        else:
            return {
                "max_amplitude": 0.0,
                "mean_amplitude": 0.0,
                "concentration_strength": 0.0,
                "region_radius": near_radius,
            }

    def _analyze_far_boundary_concentration(
        self, profile: RadialProfile, boundary: BoundaryGeometry
    ) -> Dict[str, Any]:
        """
        Analyze concentration far from boundary.

        Physical Meaning:
            Analyzes field concentration in the region
            far from the boundary.

        Args:
            profile (RadialProfile): Radial profile.
            boundary (BoundaryGeometry): Boundary geometry.

        Returns:
            Dict[str, Any]: Far boundary concentration analysis.
        """
        # Define far boundary region
        far_radius = boundary.radius * 1.5
        far_mask = profile.radii > far_radius

        if np.any(far_mask):
            far_amplitudes = profile.amplitudes[far_mask]
            far_radii = profile.radii[far_mask]

            # Compute concentration metrics
            max_amplitude = np.max(far_amplitudes)
            mean_amplitude = np.mean(far_amplitudes)
            concentration_strength = max_amplitude / mean_amplitude if mean_amplitude > 0 else 0.0

            return {
                "max_amplitude": max_amplitude,
                "mean_amplitude": mean_amplitude,
                "concentration_strength": concentration_strength,
                "region_radius": far_radius,
            }
        else:
            return {
                "max_amplitude": 0.0,
                "mean_amplitude": 0.0,
                "concentration_strength": 0.0,
                "region_radius": far_radius,
            }

    def _analyze_overall_concentration_pattern(
        self, profile: RadialProfile
    ) -> Dict[str, Any]:
        """
        Analyze overall concentration pattern.

        Physical Meaning:
            Analyzes the overall field concentration pattern
            in the radial profile.

        Args:
            profile (RadialProfile): Radial profile.

        Returns:
            Dict[str, Any]: Overall concentration pattern analysis.
        """
        # Compute overall metrics
        max_amplitude = profile.max_amplitude
        peak_radius = profile.peak_radius
        profile_width = profile.profile_width
        profile_energy = profile.profile_energy

        # Compute concentration efficiency
        concentration_efficiency = max_amplitude / profile_energy if profile_energy > 0 else 0.0

        # Analyze local maxima
        num_maxima = len(profile.local_maxima)
        maxima_strength = np.mean([amp for _, amp in profile.local_maxima]) if num_maxima > 0 else 0.0

        return {
            "max_amplitude": max_amplitude,
            "peak_radius": peak_radius,
            "profile_width": profile_width,
            "profile_energy": profile_energy,
            "concentration_efficiency": concentration_efficiency,
            "num_maxima": num_maxima,
            "maxima_strength": maxima_strength,
        }
