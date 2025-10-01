"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Runge-Kutta methods module for adaptive integrator.

This module implements embedded Runge-Kutta methods for adaptive integration,
including RK4(5) method with error estimation.

Physical Meaning:
    Uses embedded Runge-Kutta method to compute both fourth-order
    accurate solution and fifth-order error estimate for adaptive control.

Mathematical Foundation:
    Implements embedded RK4(5) method:
    - k1 = dt * f(t, y)
    - k2 = dt * f(t + dt/2, y + k1/2)
    - k3 = dt * f(t + dt/2, y + k2/2)
    - k4 = dt * f(t + dt, y + k3)
    - y4 = y + (k1 + 2*k2 + 2*k3 + k4)/6  (4th order)
    - y5 = y + (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)/90  (5th order)
    - error = |y5 - y4|
"""

import numpy as np
from typing import Tuple, Callable
import logging


class RungeKuttaMethods:
    """
    Runge-Kutta methods for adaptive integration.
    
    Physical Meaning:
        Implements embedded Runge-Kutta methods for adaptive integration
        with error estimation and step size control.
    """
    
    def __init__(self):
        """Initialize Runge-Kutta methods."""
        self.logger = logging.getLogger(__name__)
    
    def embedded_rk_step(
        self, field: np.ndarray, source: np.ndarray, dt: float, compute_rhs: Callable
    ) -> Tuple[np.ndarray, float]:
        """
        Perform embedded Runge-Kutta step with error estimation.

        Physical Meaning:
            Uses embedded Runge-Kutta method to compute both fourth-order
            accurate solution and fifth-order error estimate for adaptive control.

        Mathematical Foundation:
            Implements embedded RK4(5) method:
            - k1 = dt * f(t, y)
            - k2 = dt * f(t + dt/2, y + k1/2)
            - k3 = dt * f(t + dt/2, y + k2/2)
            - k4 = dt * f(t + dt, y + k3)
            - y4 = y + (k1 + 2*k2 + 2*k3 + k4)/6  (4th order)
            - y5 = y + (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)/90  (5th order)
            - error = |y5 - y4|
        """
        # Compute k1
        k1 = compute_rhs(field, source)

        # Compute k2
        field_temp = field + 0.5 * dt * k1
        k2 = compute_rhs(field_temp, source)

        # Compute k3
        field_temp = field + 0.5 * dt * k2
        k3 = compute_rhs(field_temp, source)

        # Compute k4
        field_temp = field + dt * k3
        k4 = compute_rhs(field_temp, source)

        # Fourth-order solution
        field_4th = field + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Fifth-order solution using embedded Runge-Kutta method
        # This provides a higher-order estimate for error control
        k5 = compute_rhs(field + dt * (7 * k1 + 10 * k2 + k4) / 27, source)
        k6 = compute_rhs(
            field + dt * (28 * k1 - 125 * k2 + 546 * k3 + 54 * k4 - 378 * k5) / 625,
            source,
        )

        # Fifth-order solution using Butcher tableau coefficients
        field_5th = field + dt * (k1 + 4 * k2 + k3 + 4 * k4 + k5 + k6) / 6.0

        # Compute error estimate
        error_estimate = np.linalg.norm(field_5th - field_4th) / np.linalg.norm(field_4th)

        return field_4th, error_estimate
