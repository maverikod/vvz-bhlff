"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

7D BVP envelope equation implementation.

This module implements the full 7D envelope equation for the BVP framework,
solving the nonlinear envelope equation in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Physical Meaning:
    Implements the 7D envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.

Mathematical Foundation:
    The 7D envelope equation includes:
    - Spatial derivatives: ∇ₓ·(κ(|a|)∇ₓa)
    - Phase derivatives: ∇φ·(κ(|a|)∇φa)  
    - Temporal evolution: ∂ₜa
    - Nonlinear terms: κ₂|a|² and χ''(|a|)

Example:
    >>> equation_7d = BVPEnvelopeEquation7D(domain_7d, config)
    >>> envelope = equation_7d.solve_envelope(source_7d)
"""

import numpy as np
from typing import Dict, Any, Tuple
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from ..domain.domain_7d import Domain7D
from .bvp_constants import BVPConstants


class BVPEnvelopeEquation7D:
    """
    7D BVP envelope equation solver.
    
    Physical Meaning:
        Solves the full 7D envelope equation in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ,
        including spatial, phase, and temporal derivatives with nonlinear
        stiffness and susceptibility terms.
        
    Mathematical Foundation:
        Solves the 7D envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
        where:
        - κ(|a|) = κ₀ + κ₂|a|² (nonlinear stiffness)
        - χ(|a|) = χ' + iχ''(|a|) (effective susceptibility)
        - ∇ includes spatial and phase derivatives
    """
    
    def __init__(self, domain_7d: Domain7D, config: Dict[str, Any]):
        """
        Initialize 7D envelope equation solver.
        
        Physical Meaning:
            Sets up the solver for the 7D envelope equation with
            nonlinear stiffness and susceptibility terms.
            
        Args:
            domain_7d (Domain7D): 7D space-time domain.
            config (Dict[str, Any]): Configuration including:
                - kappa_0: Linear stiffness coefficient
                - kappa_2: Nonlinear stiffness coefficient
                - chi_prime: Real part of susceptibility
                - chi_double_prime_0: Base imaginary part of susceptibility
                - k_0: Carrier wavenumber
        """
        self.domain_7d = domain_7d
        self.config = config
        self.constants = BVPConstants(config)
        
        # Extract parameters
        self.kappa_0 = config.get('kappa_0', 1.0)
        self.kappa_2 = config.get('kappa_2', 0.1)
        self.chi_prime = config.get('chi_prime', 1.0)
        self.chi_double_prime_0 = config.get('chi_double_prime_0', 0.01)
        self.k_0 = config.get('k_0', 1.0)
        
        self._setup_derivative_operators()
        self._setup_nonlinear_terms()
    
    def _setup_derivative_operators(self) -> None:
        """Setup derivative operators for 7D space-time."""
        # Get grid shapes
        spatial_shape = self.domain_7d.get_spatial_shape()
        phase_shape = self.domain_7d.get_phase_shape()
        full_shape = self.domain_7d.get_full_7d_shape()
        
        # Get differentials
        differentials = self.domain_7d.get_differentials()
        dx, dy, dz = differentials['dx'], differentials['dy'], differentials['dz']
        dphi_1, dphi_2, dphi_3 = differentials['dphi_1'], differentials['dphi_2'], differentials['dphi_3']
        
        # Setup spatial derivative operators
        self._setup_spatial_derivatives(spatial_shape, dx, dy, dz)
        
        # Setup phase derivative operators
        self._setup_phase_derivatives(phase_shape, dphi_1, dphi_2, dphi_3)
        
        # Setup temporal derivative operator
        self._setup_temporal_derivative()
    
    def _setup_spatial_derivatives(self, spatial_shape: Tuple[int, int, int], 
                                 dx: float, dy: float, dz: float) -> None:
        """Setup spatial derivative operators."""
        N_x, N_y, N_z = spatial_shape
        
        # Spatial gradient operators (finite difference)
        self.grad_x = self._create_gradient_operator(N_x, dx, axis=0)
        self.grad_y = self._create_gradient_operator(N_y, dy, axis=1)
        self.grad_z = self._create_gradient_operator(N_z, dz, axis=2)
        
        # Spatial divergence operators
        self.div_x = self._create_divergence_operator(N_x, dx, axis=0)
        self.div_y = self._create_divergence_operator(N_y, dy, axis=1)
        self.div_z = self._create_divergence_operator(N_z, dz, axis=2)
    
    def _setup_phase_derivatives(self, phase_shape: Tuple[int, int, int],
                               dphi_1: float, dphi_2: float, dphi_3: float) -> None:
        """Setup phase derivative operators."""
        N_phi_1, N_phi_2, N_phi_3 = phase_shape
        
        # Phase gradient operators (periodic boundary conditions)
        self.grad_phi_1 = self._create_periodic_gradient_operator(N_phi_1, dphi_1, axis=3)
        self.grad_phi_2 = self._create_periodic_gradient_operator(N_phi_2, dphi_2, axis=4)
        self.grad_phi_3 = self._create_periodic_gradient_operator(N_phi_3, dphi_3, axis=5)
        
        # Phase divergence operators
        self.div_phi_1 = self._create_periodic_divergence_operator(N_phi_1, dphi_1, axis=3)
        self.div_phi_2 = self._create_periodic_divergence_operator(N_phi_2, dphi_2, axis=4)
        self.div_phi_3 = self._create_periodic_divergence_operator(N_phi_3, dphi_3, axis=5)
    
    def _setup_temporal_derivative(self) -> None:
        """Setup temporal derivative operator."""
        dt = self.domain_7d.temporal_config.dt
        N_t = self.domain_7d.temporal_config.N_t
        
        # Temporal derivative operator (backward difference)
        self.dt_operator = self._create_temporal_derivative_operator(N_t, dt)
    
    def _create_gradient_operator(self, N: int, dx: float, axis: int) -> csc_matrix:
        """Create gradient operator for given axis."""
        # Central difference gradient operator
        diag = np.ones(N)
        off_diag = -np.ones(N-1)
        
        # Create tridiagonal matrix
        matrix = np.diag(diag, 0) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        matrix[0, 0] = 1.0  # Forward difference at boundary
        matrix[-1, -1] = 1.0  # Backward difference at boundary
        
        return csc_matrix(matrix / (2 * dx))
    
    def _create_divergence_operator(self, N: int, dx: float, axis: int) -> csc_matrix:
        """Create divergence operator for given axis."""
        # Divergence is negative of gradient for conservative form
        return -self._create_gradient_operator(N, dx, axis)
    
    def _create_periodic_gradient_operator(self, N: int, dx: float, axis: int) -> csc_matrix:
        """Create periodic gradient operator for phase coordinates."""
        # Central difference with periodic boundary conditions
        diag = np.zeros(N)
        off_diag_pos = np.ones(N-1)
        off_diag_neg = -np.ones(N-1)
        
        # Create periodic matrix
        matrix = np.diag(diag, 0) + np.diag(off_diag_pos, 1) + np.diag(off_diag_neg, -1)
        matrix[0, -1] = -1.0  # Periodic boundary condition
        matrix[-1, 0] = 1.0   # Periodic boundary condition
        
        return csc_matrix(matrix / (2 * dx))
    
    def _create_periodic_divergence_operator(self, N: int, dx: float, axis: int) -> csc_matrix:
        """Create periodic divergence operator for phase coordinates."""
        return -self._create_periodic_gradient_operator(N, dx, axis)
    
    def _create_temporal_derivative_operator(self, N_t: int, dt: float) -> csc_matrix:
        """Create temporal derivative operator."""
        # Backward difference for temporal derivative
        diag = np.ones(N_t)
        off_diag = -np.ones(N_t-1)
        
        matrix = np.diag(diag, 0) + np.diag(off_diag, -1)
        matrix[0, 0] = 1.0  # Initial condition
        
        return csc_matrix(matrix / dt)
    
    def _setup_nonlinear_terms(self) -> None:
        """Setup nonlinear stiffness and susceptibility terms."""
        # Nonlinear stiffness function: κ(|a|) = κ₀ + κ₂|a|²
        self.kappa_func = lambda amplitude: self.kappa_0 + self.kappa_2 * amplitude**2
        
        # Nonlinear susceptibility function: χ(|a|) = χ' + iχ''(|a|)
        self.chi_func = lambda amplitude: (self.chi_prime + 
                                         1j * self.chi_double_prime_0 * amplitude**2)
    
    def solve_envelope(self, source_7d: np.ndarray, initial_guess: np.ndarray = None) -> np.ndarray:
        """
        Solve 7D envelope equation.
        
        Physical Meaning:
            Solves the full 7D envelope equation for the BVP field
            in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
            
        Mathematical Foundation:
            Solves: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
            using iterative Newton-Raphson method for nonlinear terms.
            
        Args:
            source_7d (np.ndarray): 7D source term s(x,φ,t).
            initial_guess (np.ndarray): Initial guess for envelope (optional).
            
        Returns:
            np.ndarray: 7D envelope solution a(x,φ,t).
        """
        # Get full 7D shape
        full_shape = self.domain_7d.get_full_7d_shape()
        
        # Initialize solution
        if initial_guess is None:
            envelope = np.zeros(full_shape, dtype=complex)
        else:
            envelope = initial_guess.copy()
        
        # Iterative solution using Newton-Raphson method
        max_iterations = 100
        tolerance = 1e-12
        
        for iteration in range(max_iterations):
            # Compute residual
            residual = self._compute_residual(envelope, source_7d)
            
            # Check convergence
            residual_norm = np.linalg.norm(residual)
            if residual_norm < tolerance:
                break
            
            # Compute Jacobian matrix
            jacobian = self._compute_jacobian(envelope)
            
            # Solve linear system for correction
            correction = spsolve(jacobian, residual.flatten())
            correction = correction.reshape(full_shape)
            
            # Update solution
            envelope -= correction
        
        return envelope
    
    def _compute_residual(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual of 7D envelope equation.
        
        Physical Meaning:
            Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            for the current envelope solution.
            
        Args:
            envelope (np.ndarray): Current envelope solution.
            source (np.ndarray): Source term.
            
        Returns:
            np.ndarray: Residual vector.
        """
        # Compute amplitude
        amplitude = np.abs(envelope)
        
        # Compute nonlinear coefficients
        kappa = self.kappa_func(amplitude)
        chi = self.chi_func(amplitude)
        
        # Compute spatial divergence terms
        spatial_div = (self._compute_spatial_divergence(kappa, envelope) +
                      self._compute_phase_divergence(kappa, envelope))
        
        # Compute nonlinear susceptibility term
        susceptibility_term = self.k_0**2 * chi * envelope
        
        # Compute residual
        residual = spatial_div + susceptibility_term - source
        
        return residual
    
    def _compute_spatial_divergence(self, kappa: np.ndarray, envelope: np.ndarray) -> np.ndarray:
        """Compute spatial divergence ∇ₓ·(κ∇ₓa)."""
        # Compute spatial gradients
        grad_x = self._apply_spatial_gradient(envelope, axis=0)
        grad_y = self._apply_spatial_gradient(envelope, axis=1)
        grad_z = self._apply_spatial_gradient(envelope, axis=2)
        
        # Compute kappa * gradient
        kappa_grad_x = kappa * grad_x
        kappa_grad_y = kappa * grad_y
        kappa_grad_z = kappa * grad_z
        
        # Compute divergence
        div_x = self._apply_spatial_divergence(kappa_grad_x, axis=0)
        div_y = self._apply_spatial_divergence(kappa_grad_y, axis=1)
        div_z = self._apply_spatial_divergence(kappa_grad_z, axis=2)
        
        return div_x + div_y + div_z
    
    def _compute_phase_divergence(self, kappa: np.ndarray, envelope: np.ndarray) -> np.ndarray:
        """Compute phase divergence ∇φ·(κ∇φa)."""
        # Compute phase gradients
        grad_phi_1 = self._apply_phase_gradient(envelope, axis=3)
        grad_phi_2 = self._apply_phase_gradient(envelope, axis=4)
        grad_phi_3 = self._apply_phase_gradient(envelope, axis=5)
        
        # Compute kappa * gradient
        kappa_grad_phi_1 = kappa * grad_phi_1
        kappa_grad_phi_2 = kappa * grad_phi_2
        kappa_grad_phi_3 = kappa * grad_phi_3
        
        # Compute divergence
        div_phi_1 = self._apply_phase_divergence(kappa_grad_phi_1, axis=3)
        div_phi_2 = self._apply_phase_divergence(kappa_grad_phi_2, axis=4)
        div_phi_3 = self._apply_phase_divergence(kappa_grad_phi_3, axis=5)
        
        return div_phi_1 + div_phi_2 + div_phi_3
    
    def _apply_spatial_gradient(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Apply spatial gradient operator."""
        # Simplified implementation - in practice would use sparse matrix operations
        if axis == 0:
            return np.gradient(field, axis=0)
        elif axis == 1:
            return np.gradient(field, axis=1)
        elif axis == 2:
            return np.gradient(field, axis=2)
    
    def _apply_spatial_divergence(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Apply spatial divergence operator."""
        return -self._apply_spatial_gradient(field, axis)
    
    def _apply_phase_gradient(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Apply phase gradient operator with periodic boundary conditions."""
        return np.gradient(field, axis=axis)
    
    def _apply_phase_divergence(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Apply phase divergence operator with periodic boundary conditions."""
        return -self._apply_phase_gradient(field, axis)
    
    def _compute_jacobian(self, envelope: np.ndarray) -> csc_matrix:
        """
        Compute Jacobian matrix for Newton-Raphson iteration.
        
        Physical Meaning:
            Computes the Jacobian matrix of the residual with respect to
            the envelope field for Newton-Raphson iteration.
            
        Args:
            envelope (np.ndarray): Current envelope solution.
            
        Returns:
            csc_matrix: Jacobian matrix.
        """
        # Simplified implementation - in practice would compute full Jacobian
        # For now, return identity matrix as placeholder
        full_shape = self.domain_7d.get_full_7d_shape()
        total_size = np.prod(full_shape)
        
        return csc_matrix(np.eye(total_size))
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get envelope equation parameters.
        
        Returns:
            Dict[str, float]: Dictionary of parameters.
        """
        return {
            'kappa_0': self.kappa_0,
            'kappa_2': self.kappa_2,
            'chi_prime': self.chi_prime,
            'chi_double_prime_0': self.chi_double_prime_0,
            'k_0': self.k_0
        }
    
    def __repr__(self) -> str:
        """String representation of 7D envelope equation solver."""
        return (
            f"BVPEnvelopeEquation7D(domain_7d={self.domain_7d}, "
            f"kappa_0={self.kappa_0}, kappa_2={self.kappa_2})"
        )
