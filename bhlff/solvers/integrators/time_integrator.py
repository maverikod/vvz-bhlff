"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Abstract time integrator base class.

This module provides the abstract base class for all time integrators
in the BHLFF framework.

Physical Meaning:
    Time integrators implement numerical methods for advancing phase field
    configurations in time, handling the temporal evolution of the system.

Mathematical Foundation:
    Implements various time integration schemes including explicit, implicit,
    and adaptive methods for solving time-dependent phase field equations.

Example:
    >>> integrator = BVPModulationIntegrator(domain, config)
    >>> field_next = integrator.step(field_current, dt)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

from ...core.domain import Domain


class TimeIntegrator(ABC):
    """
    Abstract base class for time integrators.

    Physical Meaning:
        Provides the fundamental interface for all time integrators in the
        7D phase field theory, representing numerical methods for temporal
        evolution of phase field configurations.

    Mathematical Foundation:
        Time integrators solve time-dependent phase field equations:
        ∂a/∂t = F(a, t)
        where F(a, t) represents the right-hand side of the evolution equation.

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Integrator configuration.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize time integrator.

        Physical Meaning:
            Sets up the time integrator with computational domain and
            configuration parameters for temporal evolution.

        Args:
            domain (Domain): Computational domain for the integrator.
            config (Dict[str, Any]): Integrator configuration parameters.
        """
        self.domain = domain
        self.config = config

    @abstractmethod
    def step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform one time step.

        Physical Meaning:
            Advances the phase field configuration by one time step,
            computing the temporal evolution of the field.

        Mathematical Foundation:
            Solves ∂a/∂t = F(a, t) for one time step:
            a(t + dt) = a(t) + ∫[t to t+dt] F(a, τ) dτ

        Args:
            field (np.ndarray): Current field configuration a(t).
            dt (float): Time step size.

        Returns:
            np.ndarray: Updated field configuration a(t + dt).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step method")

    @abstractmethod
    def get_integrator_type(self) -> str:
        """
        Get the integrator type.

        Physical Meaning:
            Returns the type of time integrator being used.

        Returns:
            str: Integrator type.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement get_integrator_type method"
        )

    def get_domain(self) -> Domain:
        """
        Get the computational domain.

        Physical Meaning:
            Returns the computational domain for the integrator.

        Returns:
            Domain: Computational domain.
        """
        return self.domain

    def get_config(self) -> Dict[str, Any]:
        """
        Get the integrator configuration.

        Physical Meaning:
            Returns the configuration parameters for the integrator.

        Returns:
            Dict[str, Any]: Integrator configuration.
        """
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation of the integrator."""
        return (
            f"{self.__class__.__name__}(domain={self.domain}, "
            f"type={self.get_integrator_type()})"
        )
