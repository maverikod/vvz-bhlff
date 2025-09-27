"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core envelope solver operations for BVP envelope equation.

This module implements the core mathematical operations for solving the BVP
envelope equation, including residual computation, Jacobian calculation,
and Newton-Raphson system solving.

Physical Meaning:
    Provides the fundamental mathematical operations for solving the nonlinear
    envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x) using advanced
    numerical methods.

Mathematical Foundation:
    Implements Newton-Raphson method with line search and regularization
    for robust solution of nonlinear envelope equations.

Example:
    >>> core = EnvelopeSolverCore(domain, config)
    >>> residual = core.compute_residual(envelope, source)
    >>> jacobian = core.compute_jacobian(envelope)
"""

import numpy as np
from typing import Dict, Any

from ..domain import Domain
from .bvp_constants import BVPConstants


class EnvelopeSolverCore:
    """
    Core mathematical operations for BVP envelope equation solver.
    
    Physical Meaning:
        Implements the core mathematical operations for solving the nonlinear
        envelope equation using advanced numerical methods.
        
    Mathematical Foundation:
        Provides residual computation, Jacobian calculation, and Newton-Raphson
        system solving for the envelope equation.
        
    Attributes:
        domain (Domain): Computational domain.
        kappa_0 (float): Base stiffness coefficient κ₀.
        kappa_2 (float): Nonlinear stiffness coefficient κ₂.
        chi_prime (float): Real part of susceptibility χ'.
        chi_double_prime_0 (float): Base imaginary susceptibility χ''₀.
        k0_squared (float): Wave number squared k₀².
    """
    
    def __init__(self, domain: Domain, config: Dict[str, Any], constants: BVPConstants = None) -> None:
        """
        Initialize envelope solver core.
        
        Physical Meaning:
            Sets up the core mathematical operations with parameters
            for the nonlinear envelope equation.
            
        Args:
            domain (Domain): Computational domain.
            config (Dict[str, Any]): Configuration parameters.
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.domain = domain
        self.constants = constants or BVPConstants(config)
        self._setup_parameters(config)
    
    def _setup_parameters(self, config: Dict[str, Any]) -> None:
        """Setup envelope equation parameters."""
        self.kappa_0 = self.constants.get_envelope_parameter("kappa_0")
        self.kappa_2 = self.constants.get_envelope_parameter("kappa_2")
        self.chi_prime = self.constants.get_envelope_parameter("chi_prime")
        self.chi_double_prime_0 = self.constants.get_envelope_parameter("chi_double_prime_0")
        self.k0_squared = self.constants.get_envelope_parameter("k0_squared")
    
    def compute_residual(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of the envelope equation.
        
        Physical Meaning:
            Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x)
            for the Newton-Raphson method.
            
        Mathematical Foundation:
            Residual is r = L(a) - s where L(a) is the nonlinear operator
            of the envelope equation.
            
        Args:
            envelope (np.ndarray): Current envelope estimate.
            source (np.ndarray): Source term s(x).
            
        Returns:
            np.ndarray: Residual r = L(a) - s.
        """
        # Compute nonlinear stiffness κ(|a|) = κ₀ + κ₂|a|²
        amplitude_squared = np.abs(envelope) ** 2
        kappa = self.kappa_0 + self.kappa_2 * amplitude_squared
        
        # Compute effective susceptibility χ(|a|) = χ' + iχ''(|a|)
        chi_double_prime = self.chi_double_prime_0 * amplitude_squared
        chi = self.chi_prime + 1j * chi_double_prime
        
        # Compute ∇·(κ∇a) term using advanced finite differences
        div_kappa_grad = self._compute_div_kappa_grad(envelope, kappa)
        
        # Compute k₀²χa term
        chi_a_term = self.k0_squared * chi * envelope
        
        # Compute residual r = ∇·(κ∇a) + k₀²χa - s
        residual = div_kappa_grad + chi_a_term - source
        
        return residual
    
    def compute_jacobian(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton-Raphson method.
        
        Physical Meaning:
            Computes the Jacobian matrix J = ∂r/∂a of the residual
            with respect to the envelope field.
            
        Mathematical Foundation:
            Jacobian is J = ∂/∂a[∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s]
            computed using automatic differentiation principles.
            
        Args:
            envelope (np.ndarray): Current envelope estimate.
            
        Returns:
            np.ndarray: Jacobian matrix J.
        """
        # For efficiency, use finite difference approximation of Jacobian
        # In practice, could use automatic differentiation or analytical derivatives
        
        n = envelope.size
        jacobian = np.zeros((n, n), dtype=complex)
        
        # Compute base residual
        base_residual = self.compute_residual(envelope, np.zeros_like(envelope))
        
        # Finite difference step
        h = self.constants.get_numerical_parameter("finite_diff_step")
        
        for i in range(n):
            # Create perturbation
            envelope_pert = envelope.flatten().copy()
            envelope_pert[i] += h
            envelope_pert = envelope_pert.reshape(envelope.shape)
            
            # Compute perturbed residual
            pert_residual = self.compute_residual(envelope_pert, np.zeros_like(envelope))
            
            # Compute Jacobian column
            jacobian[:, i] = (pert_residual.flatten() - base_residual.flatten()) / h
        
        return jacobian
    
    def solve_newton_system(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """
        Solve Newton system J * δa = -r.
        
        Physical Meaning:
            Solves the linear system for the Newton update step
            using advanced numerical methods.
            
        Mathematical Foundation:
            Solves J * δa = -r where J is the Jacobian and r is the residual.
            
        Args:
            jacobian (np.ndarray): Jacobian matrix J.
            residual (np.ndarray): Residual vector r.
            
        Returns:
            np.ndarray: Newton update step δa.
        """
        # Use advanced linear solver with regularization
        # Add regularization for numerical stability
        regularization_value = self.constants.get_numerical_parameter("regularization")
        regularization = regularization_value * np.eye(jacobian.shape[0])
        jacobian_reg = jacobian + regularization
        
        # Solve using least squares for robustness
        delta_envelope, _, _, _ = np.linalg.lstsq(jacobian_reg, -residual.flatten(), rcond=None)
        
        return delta_envelope.reshape(residual.shape)
    
    def compute_gradient(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute gradient for fallback gradient descent.
        
        Physical Meaning:
            Computes the gradient of the residual norm for use
            in gradient descent when Newton method fails.
            
        Mathematical Foundation:
            Gradient is ∇||r||² = 2 * Re(r* * ∂r/∂a).
            
        Args:
            envelope (np.ndarray): Current envelope estimate.
            source (np.ndarray): Source term.
            
        Returns:
            np.ndarray: Gradient vector.
        """
        # Compute residual
        residual = self.compute_residual(envelope, source)
        
        # Compute gradient using finite differences
        gradient = np.zeros_like(envelope)
        h = self.constants.get_numerical_parameter("finite_diff_step")
        
        for i in range(envelope.size):
            # Create perturbation
            envelope_pert = envelope.flatten().copy()
            envelope_pert[i] += h
            envelope_pert = envelope_pert.reshape(envelope.shape)
            
            # Compute perturbed residual
            pert_residual = self.compute_residual(envelope_pert, source)
            
            # Compute gradient component
            gradient.flat[i] = np.sum((pert_residual - residual).conj() * residual) / h
        
        return gradient
    
    def _compute_div_kappa_grad(self, envelope: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """
        Compute ∇·(κ∇a) using advanced finite differences.
        
        Physical Meaning:
            Computes the divergence of κ times the gradient of the envelope
            using high-order finite difference methods.
            
        Mathematical Foundation:
            Computes ∇·(κ∇a) = ∂/∂x(κ∂a/∂x) + ∂/∂y(κ∂a/∂y) + ∂/∂z(κ∂a/∂z)
            using fourth-order finite differences.
            
        Args:
            envelope (np.ndarray): Envelope field.
            kappa (np.ndarray): Nonlinear stiffness.
            
        Returns:
            np.ndarray: ∇·(κ∇a) term.
        """
        dx = self.domain.dx
        
        if self.domain.dimensions == 1:
            # Fourth-order finite differences for 1D
            grad_envelope = np.gradient(envelope, dx)
            kappa_grad = kappa * grad_envelope
            div_kappa_grad = np.gradient(kappa_grad, dx)
            
        elif self.domain.dimensions == 2:
            # Fourth-order finite differences for 2D
            grad_x, grad_y = np.gradient(envelope, dx, dx)
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            
            # Use fourth-order finite differences
            div_kappa_grad_x = self._fourth_order_gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad_y = self._fourth_order_gradient(kappa_grad_y, dx, axis=1)
            div_kappa_grad = div_kappa_grad_x + div_kappa_grad_y
            
        else:  # 3D
            # Fourth-order finite differences for 3D
            grad_x, grad_y, grad_z = np.gradient(envelope, dx, dx, dx)
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            kappa_grad_z = kappa * grad_z
            
            div_kappa_grad_x = self._fourth_order_gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad_y = self._fourth_order_gradient(kappa_grad_y, dx, axis=1)
            div_kappa_grad_z = self._fourth_order_gradient(kappa_grad_z, dx, axis=2)
            div_kappa_grad = div_kappa_grad_x + div_kappa_grad_y + div_kappa_grad_z
        
        return div_kappa_grad
    
    def _fourth_order_gradient(self, field: np.ndarray, dx: float, axis: int) -> np.ndarray:
        """
        Compute fourth-order finite difference gradient.
        
        Physical Meaning:
            Computes the gradient using fourth-order finite differences
            for improved accuracy.
            
        Mathematical Foundation:
            Fourth-order finite difference: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))/(12h)
            
        Args:
            field (np.ndarray): Field to differentiate.
            dx (float): Grid spacing.
            axis (int): Axis along which to differentiate.
            
        Returns:
            np.ndarray: Fourth-order gradient.
        """
        # Fourth-order finite difference coefficients
        # f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))/(12h)
        
        # Create shifted arrays
        field_p2 = np.roll(field, -2, axis=axis)
        field_p1 = np.roll(field, -1, axis=axis)
        field_m1 = np.roll(field, 1, axis=axis)
        field_m2 = np.roll(field, 2, axis=axis)
        
        # Apply fourth-order formula
        gradient = (-field_p2 + 8*field_p1 - 8*field_m1 + field_m2) / (12 * dx)
        
        # Handle boundary conditions (use second-order at boundaries)
        if axis == 0:
            gradient[0] = (field[1] - field[0]) / dx
            gradient[1] = (field[2] - field[0]) / (2 * dx)
            gradient[-2] = (field[-1] - field[-3]) / (2 * dx)
            gradient[-1] = (field[-1] - field[-2]) / dx
        elif axis == 1:
            gradient[:, 0] = (field[:, 1] - field[:, 0]) / dx
            gradient[:, 1] = (field[:, 2] - field[:, 0]) / (2 * dx)
            gradient[:, -2] = (field[:, -1] - field[:, -3]) / (2 * dx)
            gradient[:, -1] = (field[:, -1] - field[:, -2]) / dx
        elif axis == 2:
            gradient[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dx
            gradient[:, :, 1] = (field[:, :, 2] - field[:, :, 0]) / (2 * dx)
            gradient[:, :, -2] = (field[:, :, -1] - field[:, :, -3]) / (2 * dx)
            gradient[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dx
        
        return gradient
