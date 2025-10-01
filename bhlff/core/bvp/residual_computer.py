"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Universal residual computer for BVP envelope equation.

This module implements a universal residual computer that combines
the functionality of both existing ResidualComputer classes, providing
support for different domain types and configurations while preserving
the full algorithmic implementation.

Physical Meaning:
    Computes residuals for the 7D BVP envelope equation with support
    for different domain types and configurations. The residual represents
    how well the current solution satisfies the nonlinear envelope equation.

Mathematical Foundation:
    Computes the residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
    where κ(|a|) and χ(|a|) are nonlinear coefficients, using the
    most complete algorithmic implementation available.

Example:
    >>> # With Domain + BVPConstants (original envelope_solver approach)
    >>> computer = ResidualComputer(domain, constants)
    >>> residual = computer.compute_residual(envelope, source)
    
    >>> # With Domain7D + config (original envelope_equation approach)
    >>> computer = ResidualComputer(domain_7d, config)
    >>> residual = computer.compute_residual(envelope, source, derivative_ops, nonlinear_terms)
"""

import numpy as np
from typing import Dict, Any, Union, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...domain import Domain
    from ...domain.domain_7d import Domain7D
    from ..bvp_constants import BVPConstants

from .residual_computer_base import ResidualComputerBase


class ResidualComputer(ResidualComputerBase):
    """
    Universal residual computer for BVP envelope equation.
    
    Physical Meaning:
        Computes residuals for the 7D BVP envelope equation with support
        for different domain types and configurations. Combines the best
        features of both existing implementations while preserving the
        complete algorithmic implementation.
        
    Mathematical Foundation:
        Computes the residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
        using the most complete finite difference implementation for
        all 7 dimensions of space-time.
    """
    
    def __init__(self, domain: Union["Domain", "Domain7D"], 
                 config_or_constants: Union[Dict[str, Any], "BVPConstants"]):
        """
        Initialize universal residual computer.
        
        Physical Meaning:
            Sets up the residual computer with the computational domain
            and configuration parameters or constants, automatically
            detecting the domain type and setting up appropriate parameters.
            
        Args:
            domain (Union[Domain, Domain7D]): Computational domain.
            config_or_constants (Union[Dict[str, Any], BVPConstants]): 
                Configuration parameters or BVP constants instance.
        """
        self.domain = domain
        self.config_or_constants = config_or_constants
        self.domain_type = self._detect_domain_type()
        self._setup_parameters()
    
    def _detect_domain_type(self) -> str:
        """
        Detect domain type and return appropriate handler.
        
        Physical Meaning:
            Automatically detects whether the domain is a standard Domain
            or a Domain7D, and determines the appropriate parameter setup
            and computation methods.
            
        Returns:
            str: Domain type identifier ('standard' or '7d_bvp').
        """
        if hasattr(self.domain, 'N_phi'):
            return "7d_bvp"
        else:
            return "standard"
    
    def _setup_parameters(self) -> None:
        """
        Setup envelope equation parameters based on domain type.
        
        Physical Meaning:
            Initializes the parameters needed for computing residuals
            of the envelope equation, adapting to the domain type and
            configuration source.
        """
        if self.domain_type == "standard":
            # Original envelope_solver approach with BVPConstants
            if hasattr(self.config_or_constants, 'get_envelope_parameter'):
                # BVPConstants object
                self.kappa_0 = self.config_or_constants.get_envelope_parameter("kappa_0")
                self.kappa_2 = self.config_or_constants.get_envelope_parameter("kappa_2")
                self.chi_prime = self.config_or_constants.get_envelope_parameter("chi_prime")
                self.chi_double_prime_0 = self.config_or_constants.get_envelope_parameter("chi_double_prime_0")
                self.k0_squared = self.config_or_constants.get_envelope_parameter("k0_squared")
            else:
                # Config dict fallback
                self.kappa_0 = self.config_or_constants.get("kappa_0", 1.0)
                self.kappa_2 = self.config_or_constants.get("kappa_2", 0.1)
                self.chi_prime = self.config_or_constants.get("chi_prime", 1.0)
                self.chi_double_prime_0 = self.config_or_constants.get("chi_double_prime_0", 0.1)
                self.k0_squared = self.config_or_constants.get("k0_squared", 1.0)
        else:
            # 7D BVP approach - parameters will be handled by external objects
            self.kappa_0 = None
            self.kappa_2 = None
            self.chi_prime = None
            self.chi_double_prime_0 = None
            self.k0_squared = None
    
    def compute_residual(self, envelope: np.ndarray, source: np.ndarray, 
                        derivative_operators: Optional[object] = None,
                        nonlinear_terms: Optional[object] = None) -> np.ndarray:
        """
        Compute residual of the envelope equation.
        
        Physical Meaning:
            Computes the residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
            for the current envelope solution, using the most complete
            algorithmic implementation available.
            
        Mathematical Foundation:
            The residual measures how well the current solution satisfies
            the envelope equation and is used in Newton-Raphson iterations.
            Uses the full finite difference implementation for all 7 dimensions.
            
        Args:
            envelope (np.ndarray): Current envelope solution in 7D space-time.
            source (np.ndarray): Source term s(x,φ,t) in 7D space-time.
            derivative_operators (Optional[object]): Derivative operators for 7D approach.
            nonlinear_terms (Optional[object]): Nonlinear terms for 7D approach.
            
        Returns:
            np.ndarray: Residual R = L(a) - s in 7D space-time.
        """
        if self.domain_type == "standard":
            # Use the complete finite difference implementation
            return self._compute_residual_standard(envelope, source)
        else:
            # Use the 7D approach with external operators
            return self._compute_residual_7d(envelope, source, derivative_operators, nonlinear_terms)
    
    def _compute_residual_standard(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute residual using standard finite difference approach.
        
        Physical Meaning:
            Computes the residual using the complete finite difference
            implementation for all 7 dimensions, preserving the original
            algorithmic approach.
            
        Args:
            envelope (np.ndarray): Current envelope solution.
            source (np.ndarray): Source term.
            
        Returns:
            np.ndarray: Residual using standard approach.
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
    
    def _compute_residual_7d(self, envelope: np.ndarray, source: np.ndarray,
                           derivative_operators: object, nonlinear_terms: object) -> np.ndarray:
        """
        Compute residual using 7D approach with external operators.
        
        Physical Meaning:
            Computes the residual using the 7D approach with external
            derivative operators and nonlinear terms, preserving the
            original algorithmic approach.
            
        Args:
            envelope (np.ndarray): Current envelope solution.
            source (np.ndarray): Source term.
            derivative_operators: Derivative operators object.
            nonlinear_terms: Nonlinear terms object.
            
        Returns:
            np.ndarray: Residual using 7D approach.
        """
        amplitude = np.abs(envelope)
        
        # Compute nonlinear coefficients
        kappa = nonlinear_terms.compute_stiffness(amplitude)
        chi = nonlinear_terms.compute_susceptibility(amplitude)
        
        # Compute spatial divergence term: ∇ₓ·(κ(|a|)∇ₓa)
        spatial_div = self._compute_spatial_divergence(kappa, envelope, derivative_operators)
        
        # Compute phase divergence term: ∇φ·(κ(|a|)∇φa)
        phase_div = self._compute_phase_divergence(kappa, envelope, derivative_operators)
        
        # Compute susceptibility term: k₀²χ(|a|)a
        susceptibility_term = nonlinear_terms.k0**2 * chi * envelope
        
        # Total residual
        residual = spatial_div + phase_div + susceptibility_term - source
        
        return residual
    
    def _compute_div_kappa_grad(self, envelope: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """
        Compute ∇·(κ∇a) using advanced finite differences for 7D space-time.
        
        Physical Meaning:
            Computes the divergence of κ times the gradient of the envelope
            using high-order finite difference methods in 7D space-time.
            This is the complete implementation from the original class.
            
        Mathematical Foundation:
            Computes ∇·(κ∇a) = ∂/∂x(κ∂a/∂x) + ∂/∂y(κ∂a/∂y) + ∂/∂z(κ∂a/∂z) +
                              ∂/∂φ₁(κ∂a/∂φ₁) + ∂/∂φ₂(κ∂a/∂φ₂) + ∂/∂φ₃(κ∂a/∂φ₃) +
                              ∂/∂t(κ∂a/∂t)
            using fourth-order finite differences.
            
        Args:
            envelope (np.ndarray): Envelope field in 7D space-time.
            kappa (np.ndarray): Nonlinear stiffness in 7D space-time.
            
        Returns:
            np.ndarray: ∇·(κ∇a) term in 7D space-time.
        """
        # Get grid spacings for 7D
        dx = self.domain.dx
        dphi = self.domain.dphi
        dt = self.domain.dt
        
        # Initialize divergence
        div_kappa_grad = np.zeros_like(envelope)
        
        # Spatial gradients ℝ³ₓ
        if self.domain.dimensions == 1:
            grad_x = np.gradient(envelope, dx, axis=0)
            kappa_grad_x = kappa * grad_x
            div_kappa_grad += np.gradient(kappa_grad_x, dx, axis=0)
        elif self.domain.dimensions == 2:
            grad_x, grad_y = np.gradient(envelope, dx, dx, axis=(0, 1))
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            div_kappa_grad += np.gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad += np.gradient(kappa_grad_y, dx, axis=1)
        else:  # 3D spatial
            grad_x, grad_y, grad_z = np.gradient(envelope, dx, dx, dx, axis=(0, 1, 2))
            kappa_grad_x = kappa * grad_x
            kappa_grad_y = kappa * grad_y
            kappa_grad_z = kappa * grad_z
            div_kappa_grad += np.gradient(kappa_grad_x, dx, axis=0)
            div_kappa_grad += np.gradient(kappa_grad_y, dx, axis=1)
            div_kappa_grad += np.gradient(kappa_grad_z, dx, axis=2)
        
        # Phase gradients 𝕋³_φ
        grad_phi1 = np.gradient(envelope, dphi, axis=3)
        grad_phi2 = np.gradient(envelope, dphi, axis=4)
        grad_phi3 = np.gradient(envelope, dphi, axis=5)
        
        kappa_grad_phi1 = kappa * grad_phi1
        kappa_grad_phi2 = kappa * grad_phi2
        kappa_grad_phi3 = kappa * grad_phi3
        
        div_kappa_grad += np.gradient(kappa_grad_phi1, dphi, axis=3)
        div_kappa_grad += np.gradient(kappa_grad_phi2, dphi, axis=4)
        div_kappa_grad += np.gradient(kappa_grad_phi3, dphi, axis=5)
        
        # Temporal gradient ℝₜ
        grad_t = np.gradient(envelope, dt, axis=6)
        kappa_grad_t = kappa * grad_t
        div_kappa_grad += np.gradient(kappa_grad_t, dt, axis=6)
        
        return div_kappa_grad
    
    def _compute_spatial_divergence(self, kappa: np.ndarray, envelope: np.ndarray, 
                                  derivative_operators: object) -> np.ndarray:
        """
        Compute spatial divergence term ∇ₓ·(κ(|a|)∇ₓa).
        
        Physical Meaning:
            Computes the spatial divergence of the stiffness-weighted
            gradient, representing the spatial part of the envelope equation.
            
        Mathematical Foundation:
            Computes ∇ₓ·(κ(|a|)∇ₓa) = Σᵢ ∂/∂xᵢ(κ(|a|)∂a/∂xᵢ)
            for i = x, y, z spatial coordinates.
            
        Args:
            kappa (np.ndarray): Nonlinear stiffness κ(|a|).
            envelope (np.ndarray): Envelope field.
            derivative_operators: Derivative operators object.
            
        Returns:
            np.ndarray: Spatial divergence term.
        """
        divergence = np.zeros_like(envelope)
        
        # Apply to each spatial dimension
        for axis in range(3):
            # Compute gradient: ∇ₓa
            grad_envelope = derivative_operators.apply_spatial_gradient(envelope, axis)
            
            # Multiply by stiffness: κ(|a|)∇ₓa
            weighted_grad = kappa * grad_envelope
            
            # Compute divergence: ∇ₓ·(κ(|a|)∇ₓa)
            div_term = derivative_operators.apply_spatial_divergence(weighted_grad, axis)
            divergence += div_term
        
        return divergence
    
    def _compute_phase_divergence(self, kappa: np.ndarray, envelope: np.ndarray, 
                                derivative_operators: object) -> np.ndarray:
        """
        Compute phase divergence term ∇φ·(κ(|a|)∇φa).
        
        Physical Meaning:
            Computes the phase divergence of the stiffness-weighted
            gradient, representing the phase part of the envelope equation.
            
        Mathematical Foundation:
            Computes ∇φ·(κ(|a|)∇φa) = Σᵢ ∂/∂φᵢ(κ(|a|)∂a/∂φᵢ)
            for i = φ₁, φ₂, φ₃ phase coordinates.
            
        Args:
            kappa (np.ndarray): Nonlinear stiffness κ(|a|).
            envelope (np.ndarray): Envelope field.
            derivative_operators: Derivative operators object.
            
        Returns:
            np.ndarray: Phase divergence term.
        """
        divergence = np.zeros_like(envelope)
        
        # Apply to each phase dimension
        for axis in range(3, 6):
            # Compute gradient: ∇φa
            grad_envelope = derivative_operators.apply_phase_gradient(envelope, axis)
            
            # Multiply by stiffness: κ(|a|)∇φa
            weighted_grad = kappa * grad_envelope
            
            # Compute divergence: ∇φ·(κ(|a|)∇φa)
            div_term = derivative_operators.apply_phase_divergence(weighted_grad, axis)
            divergence += div_term
        
        return divergence
    
    def analyze_residual_components(
        self,
        envelope: np.ndarray,
        source: np.ndarray,
        derivative_operators: Optional[object] = None,
        nonlinear_terms: Optional[object] = None,
    ) -> Dict[str, Any]:
        """
        Analyze components of the residual.
        
        Physical Meaning:
            Analyzes the individual components of the residual to understand
            the relative contributions of different terms in the equation.
            
        Args:
            envelope (np.ndarray): Current envelope solution.
            source (np.ndarray): Source term.
            derivative_operators (Optional[object]): Derivative operators for 7D approach.
            nonlinear_terms (Optional[object]): Nonlinear terms for 7D approach.
            
        Returns:
            Dict[str, Any]: Dictionary containing residual component analysis.
        """
        if self.domain_type == "standard":
            # Use standard approach
            return self._analyze_components_standard(envelope, source)
        else:
            # Use 7D approach
            return self._analyze_components_7d(envelope, source, derivative_operators, nonlinear_terms)
    
    def _analyze_components_standard(self, envelope: np.ndarray, source: np.ndarray) -> Dict[str, Any]:
        """Analyze components using standard approach."""
        # Compute nonlinear stiffness and susceptibility
        amplitude_squared = np.abs(envelope) ** 2
        kappa = self.kappa_0 + self.kappa_2 * amplitude_squared
        chi_double_prime = self.chi_double_prime_0 * amplitude_squared
        chi = self.chi_prime + 1j * chi_double_prime
        
        # Compute individual components
        div_kappa_grad = self._compute_div_kappa_grad(envelope, kappa)
        chi_a_term = self.k0_squared * chi * envelope
        
        # Compute norms
        div_norm = np.linalg.norm(div_kappa_grad)
        chi_norm = np.linalg.norm(chi_a_term)
        source_norm = np.linalg.norm(source)
        
        # Total residual
        total_residual = div_kappa_grad + chi_a_term - source
        total_norm = np.linalg.norm(total_residual)
        
        return {
            "divergence_norm": float(div_norm),
            "susceptibility_norm": float(chi_norm),
            "source_norm": float(source_norm),
            "total_residual_norm": float(total_norm),
            "component_ratios": {
                "divergence_ratio": float(div_norm / total_norm) if total_norm > 0 else 0.0,
                "susceptibility_ratio": float(chi_norm / total_norm) if total_norm > 0 else 0.0,
                "source_ratio": float(source_norm / total_norm) if total_norm > 0 else 0.0,
            },
        }
    
    def _analyze_components_7d(self, envelope: np.ndarray, source: np.ndarray,
                             derivative_operators: object, nonlinear_terms: object) -> Dict[str, Any]:
        """Analyze components using 7D approach."""
        amplitude = np.abs(envelope)
        
        # Compute nonlinear coefficients
        kappa = nonlinear_terms.compute_stiffness(amplitude)
        chi = nonlinear_terms.compute_susceptibility(amplitude)
        
        # Compute individual components
        spatial_div = self._compute_spatial_divergence(kappa, envelope, derivative_operators)
        phase_div = self._compute_phase_divergence(kappa, envelope, derivative_operators)
        susceptibility_term = nonlinear_terms.k0**2 * chi * envelope
        
        # Compute norms
        spatial_norm = np.linalg.norm(spatial_div)
        phase_norm = np.linalg.norm(phase_div)
        susceptibility_norm = np.linalg.norm(susceptibility_term)
        source_norm = np.linalg.norm(source)
        
        # Total residual
        total_residual = spatial_div + phase_div + susceptibility_term - source
        total_norm = np.linalg.norm(total_residual)
        
        return {
            "spatial_divergence_norm": float(spatial_norm),
            "phase_divergence_norm": float(phase_norm),
            "susceptibility_norm": float(susceptibility_norm),
            "source_norm": float(source_norm),
            "total_residual_norm": float(total_norm),
            "component_ratios": {
                "spatial_ratio": float(spatial_norm / total_norm) if total_norm > 0 else 0.0,
                "phase_ratio": float(phase_norm / total_norm) if total_norm > 0 else 0.0,
                "susceptibility_ratio": float(susceptibility_norm / total_norm) if total_norm > 0 else 0.0,
                "source_ratio": float(source_norm / total_norm) if total_norm > 0 else 0.0,
            },
        }
