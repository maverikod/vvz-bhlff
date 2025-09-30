"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core interface implementation for BVP framework.

This module implements the interface between BVP and core,
providing the necessary data transformations for core calculations.

Theoretical Background:
    The core interface provides the necessary data for core calculations
    including renormalized coefficients c_i^eff(A,∇A), boundary conditions
    (pressure/stiffness), core energy density and gradients, and effective
    parameters for core evolution.

Example:
    >>> core_interface = CoreInterface(bvp_core)
    >>> core_data = core_interface.interface_with_core(envelope)
"""

import numpy as np
from typing import Dict, Any

from ...domain.domain_7d import Domain7D
from ..bvp_core_new import BVPCore


class CoreInterface:
    """
    Interface between BVP and core.

    Physical Meaning:
        Provides the connection between BVP envelope and core.
        This interface implements the data transformations required for
        integrating BVP with core calculations.

    Mathematical Foundation:
        Implements interface functions for core:
        1. Renormalized coefficients c_i^eff(A,∇A) from BVP averaging
        2. Boundary conditions (pressure/stiffness) from BVP field
        3. Core energy density and gradients
        4. Effective parameters for core evolution
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize core interface.

        Physical Meaning:
            Sets up the interface with the BVP core module for
            core calculations.

        Args:
            bvp_core (BVPCore): BVP core module instance.
        """
        self.bvp_core = bvp_core
        self.domain_7d = bvp_core.domain_7d
        self.config = bvp_core.config

    def interface_with_core(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Interface BVP with core.

        Physical Meaning:
            Provides the necessary data for core calculations:
            - Renormalized coefficients c_i^eff(A,∇A) from BVP averaging
            - Boundary conditions (pressure/stiffness) from BVP field
            - Core energy density and gradients
            - Effective parameters for core evolution

        Mathematical Foundation:
            Computes core interface functions through BVP averaging:
            - c_i^eff = c_i + α_i|A|² + β_i|∇A|²/ω₀² + ...
            - Boundary pressure: P_boundary = f(|A|,|∇A|)
            - Core stiffness: K_core = f(|A|,|∇A|)

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            Dict[str, Any]: Core interface data including:
                - renormalized_coefficients (Dict): c_i^eff(A,∇A) coefficients
                - boundary_pressure (np.ndarray): P_boundary pressure
                - core_stiffness (np.ndarray): K_core stiffness
                - energy_density (np.ndarray): Core energy density
                - effective_parameters (Dict): Effective core parameters
        """
        # Compute renormalized coefficients
        renormalized_coefficients = self._compute_renormalized_coefficients(envelope)

        # Compute boundary conditions
        boundary_pressure = self._compute_boundary_pressure(envelope)
        core_stiffness = self._compute_core_stiffness(envelope)

        # Compute energy density
        energy_density = self._compute_core_energy_density(envelope)

        # Compute effective parameters
        effective_parameters = self._compute_effective_parameters(envelope)

        core_data = {
            "renormalized_coefficients": renormalized_coefficients,
            "boundary_pressure": boundary_pressure,
            "core_stiffness": core_stiffness,
            "energy_density": energy_density,
            "effective_parameters": effective_parameters,
        }

        return core_data

    def _compute_renormalized_coefficients(
        self, envelope: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute renormalized coefficients c_i^eff(A,∇A).

        Physical Meaning:
            Computes the renormalized coefficients that result
            from BVP averaging over the high-frequency carrier.

        Returns:
            Dict[str, float]: Renormalized coefficients.
        """
        amplitude = np.abs(envelope)
        mean_amplitude = np.mean(amplitude)

        # Renormalized coefficients
        coefficients = {
            "c2_eff": 1.0 + 0.1 * mean_amplitude**2,
            "c4_eff": 0.1 + 0.01 * mean_amplitude**2,
            "c6_eff": 0.01 + 0.001 * mean_amplitude**2,
        }

        return coefficients

    def _compute_boundary_pressure(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute boundary pressure P_boundary.

        Physical Meaning:
            Computes the boundary pressure that results from
            the BVP field at the boundaries.

        Returns:
            np.ndarray: Boundary pressure.
        """
        amplitude = np.abs(envelope)
        return 0.5 * amplitude**2

    def _compute_core_stiffness(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute core stiffness K_core.

        Physical Meaning:
            Computes the core stiffness that results from
            the BVP field interaction with the core.

        Returns:
            np.ndarray: Core stiffness.
        """
        amplitude = np.abs(envelope)
        return 1.0 + 0.2 * amplitude**2

    def _compute_core_energy_density(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute core energy density.

        Physical Meaning:
            Computes the energy density in the core region
            resulting from the BVP field.

        Returns:
            np.ndarray: Core energy density.
        """
        amplitude = np.abs(envelope)
        return 0.5 * amplitude**2

    def _compute_effective_parameters(self, envelope: np.ndarray) -> Dict[str, float]:
        """
        Compute effective core parameters.

        Physical Meaning:
            Computes the effective parameters for core evolution
            that result from BVP averaging.

        Returns:
            Dict[str, float]: Effective parameters.
        """
        amplitude = np.abs(envelope)
        mean_amplitude = np.mean(amplitude)

        parameters = {
            "effective_mass": 1.0 + 0.1 * mean_amplitude**2,
            "effective_damping": 0.01 + 0.001 * mean_amplitude**2,
            "effective_coupling": 0.1 + 0.01 * mean_amplitude**2,
        }

        return parameters
