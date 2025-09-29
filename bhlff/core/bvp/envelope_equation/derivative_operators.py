"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

7D derivative operators for BVP envelope equation.

This module implements the derivative operators needed for the 7D BVP envelope
equation, including spatial, phase, and temporal derivatives with appropriate
boundary conditions.

Physical Meaning:
    The derivative operators implement the spatial, phase, and temporal
    derivatives required for the 7D envelope equation. Spatial derivatives
    use finite differences, phase derivatives use periodic boundary conditions,
    and temporal derivatives use backward differences.

Mathematical Foundation:
    Implements the derivative operators for:
    - Spatial derivatives: ∇ₓ·(κ(|a|)∇ₓa) with finite differences
    - Phase derivatives: ∇φ·(κ(|a|)∇φa) with periodic boundary conditions
    - Temporal derivatives: ∂ₜa with backward differences

Example:
    >>> operators = DerivativeOperators7D(domain_7d)
    >>> operators.setup_operators()
    >>> gradient = operators.apply_spatial_gradient(field, axis=0)
"""

import numpy as np
from typing import Tuple
from scipy.sparse import csc_matrix

from ...domain.domain_7d import Domain7D


class DerivativeOperators7D:
    """
    7D derivative operators for BVP envelope equation.
    
    Physical Meaning:
        Implements all derivative operators needed for the 7D envelope equation,
        including spatial gradients/divergences, phase gradients/divergences,
        and temporal derivatives with appropriate boundary conditions.
        
    Mathematical Foundation:
        Provides finite difference operators for spatial coordinates,
        periodic operators for phase coordinates, and backward difference
        operators for temporal evolution.
    """
    
    def __init__(self, domain_7d: Domain7D):
        """
        Initialize derivative operators.
        
        Physical Meaning:
            Sets up the derivative operators with the 7D computational domain,
            preparing for the computation of spatial, phase, and temporal
            derivatives in the envelope equation.
            
        Args:
            domain_7d (Domain7D): 7D computational domain.
        """
        self.domain_7d = domain_7d
        self.grad_x = None
        self.grad_y = None
        self.grad_z = None
        self.div_x = None
        self.div_y = None
        self.div_z = None
        self.grad_phi_1 = None
        self.grad_phi_2 = None
        self.grad_phi_3 = None
        self.div_phi_1 = None
        self.div_phi_2 = None
        self.div_phi_3 = None
        self.dt_operator = None
    
    def setup_operators(self) -> None:
        """
        Setup all derivative operators for 7D space-time.
        
        Physical Meaning:
            Initializes all derivative operators including spatial,
            phase, and temporal operators with appropriate boundary
            conditions for the 7D envelope equation.
        """
        # Get grid shapes
        spatial_shape = self.domain_7d.get_spatial_shape()
        phase_shape = self.domain_7d.get_phase_shape()
        
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
        """
        Setup spatial derivative operators.
        
        Physical Meaning:
            Creates finite difference operators for spatial derivatives
            in the x, y, and z directions with appropriate boundary conditions.
            
        Args:
            spatial_shape: Tuple of (N_x, N_y, N_z) grid dimensions.
            dx, dy, dz: Spatial step sizes.
        """
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
        """
        Setup phase derivative operators.
        
        Physical Meaning:
            Creates periodic derivative operators for phase coordinates
            with periodic boundary conditions appropriate for the
            toroidal phase space.
            
        Args:
            phase_shape: Tuple of (N_phi_1, N_phi_2, N_phi_3) grid dimensions.
            dphi_1, dphi_2, dphi_3: Phase step sizes.
        """
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
        """
        Setup temporal derivative operator.
        
        Physical Meaning:
            Creates the temporal derivative operator using backward
            differences for time evolution in the envelope equation.
        """
        dt = self.domain_7d.temporal_config.dt
        N_t = self.domain_7d.temporal_config.N_t
        
        # Temporal derivative operator (backward difference)
        self.dt_operator = self._create_temporal_derivative_operator(N_t, dt)
    
    def _create_gradient_operator(self, N: int, dx: float, axis: int) -> csc_matrix:
        """
        Create gradient operator for given axis.
        
        Physical Meaning:
            Creates a finite difference gradient operator using central
            differences with appropriate boundary conditions.
            
        Args:
            N: Grid size along the axis.
            dx: Step size along the axis.
            axis: Axis index.
            
        Returns:
            csc_matrix: Sparse gradient operator matrix.
        """
        # Central difference gradient operator
        diag = np.ones(N)
        off_diag = -np.ones(N-1)
        
        # Create tridiagonal matrix
        matrix = np.diag(diag, 0) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        matrix[0, 0] = 1.0  # Forward difference at boundary
        matrix[-1, -1] = 1.0  # Backward difference at boundary
        
        return csc_matrix(matrix / (2 * dx))
    
    def _create_divergence_operator(self, N: int, dx: float, axis: int) -> csc_matrix:
        """
        Create divergence operator for given axis.
        
        Physical Meaning:
            Creates a divergence operator as the negative of the gradient
            operator for conservative form of the equations.
            
        Args:
            N: Grid size along the axis.
            dx: Step size along the axis.
            axis: Axis index.
            
        Returns:
            csc_matrix: Sparse divergence operator matrix.
        """
        # Divergence is negative of gradient for conservative form
        return -self._create_gradient_operator(N, dx, axis)
    
    def _create_periodic_gradient_operator(self, N: int, dx: float, axis: int) -> csc_matrix:
        """
        Create periodic gradient operator for phase coordinates.
        
        Physical Meaning:
            Creates a periodic gradient operator using central differences
            with periodic boundary conditions for the toroidal phase space.
            
        Args:
            N: Grid size along the axis.
            dx: Step size along the axis.
            axis: Axis index.
            
        Returns:
            csc_matrix: Sparse periodic gradient operator matrix.
        """
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
        """
        Create periodic divergence operator for phase coordinates.
        
        Physical Meaning:
            Creates a periodic divergence operator as the negative of
            the periodic gradient operator.
            
        Args:
            N: Grid size along the axis.
            dx: Step size along the axis.
            axis: Axis index.
            
        Returns:
            csc_matrix: Sparse periodic divergence operator matrix.
        """
        return -self._create_periodic_gradient_operator(N, dx, axis)
    
    def _create_temporal_derivative_operator(self, N_t: int, dt: float) -> csc_matrix:
        """
        Create temporal derivative operator.
        
        Physical Meaning:
            Creates a temporal derivative operator using backward
            differences for time evolution.
            
        Args:
            N_t: Number of time steps.
            dt: Time step size.
            
        Returns:
            csc_matrix: Sparse temporal derivative operator matrix.
        """
        # Backward difference for temporal derivative
        diag = np.ones(N_t)
        off_diag = -np.ones(N_t-1)
        
        # Create lower triangular matrix
        matrix = np.diag(diag, 0) + np.diag(off_diag, -1)
        matrix[0, 0] = 1.0  # Initial condition
        
        return csc_matrix(matrix / dt)
    
    def apply_spatial_gradient(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Apply spatial gradient operator.
        
        Physical Meaning:
            Applies the spatial gradient operator to compute the gradient
            of the field along the specified spatial axis.
            
        Args:
            field: Field to differentiate.
            axis: Spatial axis (0=x, 1=y, 2=z).
            
        Returns:
            np.ndarray: Gradient of the field.
        """
        if axis == 0:
            return self.grad_x.dot(field.flatten()).reshape(field.shape)
        elif axis == 1:
            return self.grad_y.dot(field.flatten()).reshape(field.shape)
        elif axis == 2:
            return self.grad_z.dot(field.flatten()).reshape(field.shape)
        else:
            raise ValueError(f"Invalid spatial axis: {axis}")
    
    def apply_spatial_divergence(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Apply spatial divergence operator.
        
        Physical Meaning:
            Applies the spatial divergence operator to compute the divergence
            of the field along the specified spatial axis.
            
        Args:
            field: Field to differentiate.
            axis: Spatial axis (0=x, 1=y, 2=z).
            
        Returns:
            np.ndarray: Divergence of the field.
        """
        if axis == 0:
            return self.div_x.dot(field.flatten()).reshape(field.shape)
        elif axis == 1:
            return self.div_y.dot(field.flatten()).reshape(field.shape)
        elif axis == 2:
            return self.div_z.dot(field.flatten()).reshape(field.shape)
        else:
            raise ValueError(f"Invalid spatial axis: {axis}")
    
    def apply_phase_gradient(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Apply phase gradient operator.
        
        Physical Meaning:
            Applies the phase gradient operator to compute the gradient
            of the field along the specified phase axis with periodic
            boundary conditions.
            
        Args:
            field: Field to differentiate.
            axis: Phase axis (3=phi_1, 4=phi_2, 5=phi_3).
            
        Returns:
            np.ndarray: Gradient of the field.
        """
        if axis == 3:
            return self.grad_phi_1.dot(field.flatten()).reshape(field.shape)
        elif axis == 4:
            return self.grad_phi_2.dot(field.flatten()).reshape(field.shape)
        elif axis == 5:
            return self.grad_phi_3.dot(field.flatten()).reshape(field.shape)
        else:
            raise ValueError(f"Invalid phase axis: {axis}")
    
    def apply_phase_divergence(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Apply phase divergence operator.
        
        Physical Meaning:
            Applies the phase divergence operator to compute the divergence
            of the field along the specified phase axis with periodic
            boundary conditions.
            
        Args:
            field: Field to differentiate.
            axis: Phase axis (3=phi_1, 4=phi_2, 5=phi_3).
            
        Returns:
            np.ndarray: Divergence of the field.
        """
        if axis == 3:
            return self.div_phi_1.dot(field.flatten()).reshape(field.shape)
        elif axis == 4:
            return self.div_phi_2.dot(field.flatten()).reshape(field.shape)
        elif axis == 5:
            return self.div_phi_3.dot(field.flatten()).reshape(field.shape)
        else:
            raise ValueError(f"Invalid phase axis: {axis}")
