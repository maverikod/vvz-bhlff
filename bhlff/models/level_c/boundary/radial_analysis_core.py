"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Radial analysis core module.

This module implements core radial analysis functionality for boundary effects
in Level C test C1 of 7D phase field theory.

Physical Meaning:
    Analyzes radial profiles for boundary effects,
    including field distribution and concentration patterns.

Example:
    >>> analyzer = RadialAnalysisCore(bvp_core)
    >>> results = analyzer.analyze_radial_profile(domain, boundary, field)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

from bhlff.core.bvp import BVPCore
from .data_structures import BoundaryGeometry, RadialProfile
from ..cuda import LevelCCUDAProcessor


class RadialAnalysisCore:
    """
    Radial analysis core for boundary effects.

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

        Physical Meaning:
            Sets up the radial analysis system with
            appropriate parameters and methods.

        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

        # Initialize CUDA processor for vectorized operations
        try:
            self.cuda_processor = LevelCCUDAProcessor(bvp_core, use_cuda=True)
            self.use_cuda = self.cuda_processor.cuda_available
        except Exception as e:
            self.logger.warning(f"CUDA processor initialization failed: {e}, using CPU")
            self.cuda_processor = None
            self.use_cuda = False

    def analyze_radial_profile(
        self, domain: Dict[str, Any], boundary: BoundaryGeometry, field: np.ndarray
    ) -> RadialProfile:
        """
        Analyze radial profile.

        Physical Meaning:
            Analyzes radial profile for boundary effects
            including field distribution and concentration patterns.

        Mathematical Foundation:
            Radial profile: A(r) = (1/4π) ∫_S(r) |a(x)|² dS

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            field (np.ndarray): Field data.

        Returns:
            RadialProfile: Radial profile analysis results.
        """
        self.logger.info("Starting radial profile analysis")

        # Use CUDA vectorized computation if available
        if self.use_cuda and self.cuda_processor is not None:
            # Extract center from boundary
            center = (
                boundary.center
                if hasattr(boundary, "center")
                else np.array([domain.get("L", 1.0) / 2] * 3)
            )

            # Define radial range
            r_max = domain.get("L", 1.0) / 2
            num_radii = 50
            radii = np.linspace(0, r_max, num_radii)

            try:
                # Compute radial profile using CUDA vectorized operations
                amplitudes = self.cuda_processor.compute_radial_profile_vectorized(
                    field, center, radii, domain
                )

                # Create radial profile data structure
                radial_profile_data = {
                    "radii": radii,
                    "amplitudes": amplitudes,
                }
            except Exception as e:
                self.logger.warning(
                    f"CUDA radial profile computation failed: {e}, falling back to CPU"
                )
                radial_profile_data = self._compute_radial_profile(
                    domain, boundary, field
                )
        else:
            # CPU computation
            radial_profile_data = self._compute_radial_profile(domain, boundary, field)

        # Find local maxima
        local_maxima = self._find_local_maxima(radial_profile_data)

        # Create radial profile object
        profile = RadialProfile(
            radii=radial_profile_data["radii"],
            amplitudes=radial_profile_data["amplitudes"],
            local_maxima=local_maxima,
            analysis_complete=True,
        )

        self.logger.info("Radial profile analysis completed")
        return profile

    def _compute_radial_profile(
        self, domain: Dict[str, Any], boundary: BoundaryGeometry, field: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute radial profile.

        Physical Meaning:
            Computes radial profile for boundary effects
            analysis.

        Mathematical Foundation:
            Radial profile: A(r) = (1/4π) ∫_S(r) |a(x)|² dS

        Args:
            domain (Dict[str, Any]): Domain parameters.
            boundary (BoundaryGeometry): Boundary geometry.
            field (np.ndarray): Field data.

        Returns:
            Dict[str, Any]: Radial profile data.
        """
        # Extract domain parameters
        N = domain["N"]
        L = domain["L"]

        # Define radial range
        r_max = L / 2
        num_radii = 50
        radii = np.linspace(0, r_max, num_radii)

        # Initialize amplitude array
        amplitudes = np.zeros(num_radii)

        # Compute amplitude at each radius
        for i, r in enumerate(radii):
            amplitudes[i] = self._compute_amplitude_at_radius(domain, field, r)

        return {
            "radii": radii,
            "amplitudes": amplitudes,
        }

    def _compute_amplitude_at_radius(
        self, domain: Dict[str, Any], field: np.ndarray, radius: float
    ) -> float:
        """
        Compute amplitude at specific radius.

        Physical Meaning:
            Computes field amplitude at specific radius
            for radial profile analysis.

        Mathematical Foundation:
            Amplitude at radius: A(r) = (1/4π) ∫_S(r) |a(x)|² dS

        Args:
            domain (Dict[str, Any]): Domain parameters.
            field (np.ndarray): Field data.
            radius (float): Radius for computation.

        Returns:
            float: Amplitude at radius.
        """
        # Create spherical shell mask
        shell_mask = self._create_spherical_shell_mask(domain, radius)

        # Compute amplitude
        amplitude = np.sqrt(np.mean(np.abs(field[shell_mask]) ** 2))

        return amplitude

    def _create_spherical_shell_mask(
        self, domain: Dict[str, Any], radius: float
    ) -> np.ndarray:
        """
        Create spherical shell mask.

        Physical Meaning:
            Creates spherical shell mask for radial
            profile computation.

        Args:
            domain (Dict[str, Any]): Domain parameters.
            radius (float): Shell radius.

        Returns:
            np.ndarray: Spherical shell mask.
        """
        # Extract domain parameters
        N = domain["N"]
        L = domain["L"]

        # Create coordinate arrays
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Calculate distance from center
        center = L / 2
        distance = np.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)

        # Create shell mask
        shell_thickness = L / (2 * N)  # Shell thickness
        shell_mask = (distance >= radius - shell_thickness) & (
            distance <= radius + shell_thickness
        )

        return shell_mask

    def _find_local_maxima(
        self, radial_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find local maxima in radial profile.

        Physical Meaning:
            Finds local maxima in radial profile
            for boundary effects analysis.

        Args:
            radial_profile (Dict[str, Any]): Radial profile data.

        Returns:
            List[Dict[str, Any]]: Local maxima information.
        """
        amplitudes = radial_profile["amplitudes"]
        radii = radial_profile["radii"]

        # Find local maxima
        local_maxima = []
        for i in range(1, len(amplitudes) - 1):
            if (
                amplitudes[i] > amplitudes[i - 1]
                and amplitudes[i] > amplitudes[i + 1]
                and amplitudes[i] > np.mean(amplitudes)
            ):
                maximum = {
                    "radius": radii[i],
                    "amplitude": amplitudes[i],
                    "index": i,
                }
                local_maxima.append(maximum)

        return local_maxima
