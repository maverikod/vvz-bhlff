"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Electroweak coupling implementation for U(1)³ phase vector structure.

This module implements electroweak coupling coefficients and current
calculations for the U(1)³ phase vector structure in the BVP framework.

Physical Meaning:
    Implements electromagnetic and weak interaction currents that are
    generated as functionals of the BVP envelope through the U(1)³
    phase structure with proper Weinberg mixing.

Mathematical Foundation:
    Computes electroweak currents:
    - J_EM = g_EM * |A|² * ∇Θ_EM
    - J_weak = g_weak * |A|⁴ * ∇Θ_weak
    where Θ_EM and Θ_weak are combinations of Θ_a components.

Example:
    >>> coupling = ElectroweakCoupling(config)
    >>> currents = coupling.compute_currents(envelope, phase_components)
"""

import numpy as np
from typing import Dict, Any, List
import logging

from bhlff.core.domain import Domain

# CUDA optimization
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    logging.info("CUDA support enabled with CuPy")
except ImportError:
    CUDA_AVAILABLE = False
    logging.warning("CUDA not available, falling back to CPU")


class ElectroweakCoupling:
    """
    Electroweak coupling for U(1)³ phase vector structure.

    Physical Meaning:
        Implements electromagnetic and weak interaction currents
        that are generated as functionals of the BVP envelope
        through the U(1)³ phase structure.

    Mathematical Foundation:
        Computes electroweak currents with proper Weinberg mixing
        and gauge coupling coefficients.

    Attributes:
        config (Dict[str, Any]): Electroweak coupling configuration.
        electroweak_coefficients (Dict[str, float]): Coupling coefficients.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize electroweak coupling.

        Physical Meaning:
            Sets up the coefficients for electroweak currents
            that are generated as functionals of the envelope.

        Args:
            config (Dict[str, Any]): Electroweak configuration including:
                - em_coupling: Electromagnetic coupling strength
                - weak_coupling: Weak interaction coupling strength
                - mixing_angle: Weinberg mixing angle
                - gauge_coupling: Gauge coupling strength
        """
        self.config = config
        
        # CUDA optimization setup
        self.cuda_available = CUDA_AVAILABLE
        self.use_cuda = config.get("use_cuda", True) and self.cuda_available
        self.logger = logging.getLogger(__name__)
        
        if self.use_cuda:
            self.logger.info("ElectroweakCoupling: CUDA optimization enabled")
        else:
            self.logger.info("ElectroweakCoupling: Using CPU computation")
        
        self._setup_electroweak_coefficients()

    def _setup_electroweak_coefficients(self) -> None:
        """
        Setup electroweak coupling coefficients.

        Physical Meaning:
            Initializes the coefficients for electroweak currents
            that are generated as functionals of the envelope.
        """
        electroweak_config = self.config.get("electroweak", {})

        self.electroweak_coefficients = {
            "em_coupling": electroweak_config.get("em_coupling", 1.0),
            "weak_coupling": electroweak_config.get("weak_coupling", 0.1),
            "mixing_angle": electroweak_config.get(
                "mixing_angle", 0.23
            ),  # Weinberg angle
            "gauge_coupling": electroweak_config.get("gauge_coupling", 0.65),
        }

    def compute_electroweak_currents(
        self, envelope: np.ndarray, phase_components: List[np.ndarray], domain: Domain
    ) -> Dict[str, np.ndarray]:
        """
        Compute electroweak currents as functionals of the envelope.

        Physical Meaning:
            Computes electromagnetic and weak currents that are
            generated as functionals of the BVP envelope through
            the U(1)³ phase structure.

        Mathematical Foundation:
            J_EM = g_EM * |A|² * ∇Θ_EM
            J_weak = g_weak * |A|⁴ * ∇Θ_weak
            where Θ_EM and Θ_weak are combinations of Θ_a components.

        Args:
            envelope (np.ndarray): BVP envelope |A|.
            phase_components (List[np.ndarray]): Three U(1) phase components.
            domain (Domain): Computational domain.

        Returns:
            Dict[str, np.ndarray]: Electroweak currents including:
                - em_current: Electromagnetic current
                - weak_current: Weak interaction current
                - mixed_current: Mixed electroweak current
        """
        # Compute phase gradients in 7D space-time
        phase_gradients = []
        for theta_a in phase_components:
            # Transfer to GPU
            theta_a_gpu = self._to_gpu(theta_a)
            
            # Compute gradients in all 7 dimensions
            gradients = []

            # Spatial gradients ℝ³ₓ
            if domain.dimensions >= 1:
                gradients.append(self._cuda_gradient(theta_a_gpu, axis=0))
            if domain.dimensions >= 2:
                gradients.append(self._cuda_gradient(theta_a_gpu, axis=1))
            if domain.dimensions >= 3:
                gradients.append(self._cuda_gradient(theta_a_gpu, axis=2))

            # Phase gradients 𝕋³_φ
            if theta_a.ndim > 3:
                gradients.append(self._cuda_gradient(theta_a_gpu, axis=3))  # φ₁
            if theta_a.ndim > 4:
                gradients.append(self._cuda_gradient(theta_a_gpu, axis=4))  # φ₂
            if theta_a.ndim > 5:
                gradients.append(self._cuda_gradient(theta_a_gpu, axis=5))  # φ₃

            # Temporal gradient ℝₜ
            if theta_a.ndim > 6:
                gradients.append(self._cuda_gradient(theta_a_gpu, axis=6))  # t

            # Compute magnitude of gradient vector in 7D
            grad_theta = self._cuda_sqrt(self._cuda_sum([self._cuda_abs(g) ** 2 for g in gradients], axis=0))
            phase_gradients.append(grad_theta)

        # Transfer envelope to GPU
        envelope_gpu = self._to_gpu(envelope)
        
        # Electromagnetic current (primarily from Θ₁)
        em_gradient = phase_gradients[0]  # Primary EM component
        em_current = (
            self.electroweak_coefficients["em_coupling"] * envelope_gpu**2 * em_gradient
        )

        # Weak current (primarily from Θ₂ and Θ₃)
        weak_gradient = phase_gradients[1] + phase_gradients[2]  # Weak components
        weak_current = (
            self.electroweak_coefficients["weak_coupling"] * envelope_gpu**4 * weak_gradient
        )

        # Mixed electroweak current (Weinberg mixing)
        mixing_angle = self.electroweak_coefficients["mixing_angle"]
        mixed_current = (
            self.electroweak_coefficients["gauge_coupling"]
            * envelope_gpu**3
            * (
                self._cuda_cos(mixing_angle) * em_gradient
                + self._cuda_sin(mixing_angle) * weak_gradient
            )
        )

        # Convert results to CPU
        return {
            "em_current": self._to_cpu(em_current),
            "weak_current": self._to_cpu(weak_current),
            "mixed_current": self._to_cpu(mixed_current),
        }

    def get_electroweak_coefficients(self) -> Dict[str, float]:
        """
        Get electroweak coupling coefficients.

        Physical Meaning:
            Returns the current electroweak coupling coefficients
            used for current calculations.

        Returns:
            Dict[str, float]: Electroweak coupling coefficients.
        """
        return self.electroweak_coefficients.copy()

    def set_electroweak_coefficients(self, coefficients: Dict[str, float]) -> None:
        """
        Set electroweak coupling coefficients.

        Physical Meaning:
            Updates the electroweak coupling coefficients
            used for current calculations.

        Args:
            coefficients (Dict[str, float]): New coupling coefficients.
        """
        self.electroweak_coefficients.update(coefficients)

    def get_weinberg_angle(self) -> float:
        """
        Get the Weinberg mixing angle.

        Physical Meaning:
            Returns the Weinberg mixing angle used for
            electroweak mixing calculations.

        Returns:
            float: Weinberg mixing angle.
        """
        return self.electroweak_coefficients["mixing_angle"]

    def set_weinberg_angle(self, angle: float) -> None:
        """
        Set the Weinberg mixing angle.

        Physical Meaning:
            Updates the Weinberg mixing angle used for
            electroweak mixing calculations.

        Args:
            angle (float): New Weinberg mixing angle.
        """
        self.electroweak_coefficients["mixing_angle"] = angle

    def _to_gpu(self, array: np.ndarray) -> 'cp.ndarray':
        """
        Convert numpy array to GPU array.
        
        Physical Meaning:
            Transfers array to GPU memory for CUDA computation.
            
        Args:
            array (np.ndarray): Input array.
            
        Returns:
            cp.ndarray: GPU array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.asarray(array)
        return array
    
    def _to_cpu(self, array) -> np.ndarray:
        """
        Convert GPU array to numpy array.
        
        Physical Meaning:
            Transfers array from GPU memory to CPU memory.
            
        Args:
            array: Input array (GPU or CPU).
            
        Returns:
            np.ndarray: CPU array.
        """
        if self.use_cuda and CUDA_AVAILABLE and hasattr(array, 'get'):
            return array.get()
        return array
    
    def _cuda_gradient(self, array, axis: int = 0) -> 'cp.ndarray':
        """
        Compute gradient using CUDA.
        
        Physical Meaning:
            Computes gradient using CUDA for optimal performance.
            
        Args:
            array: Input array.
            axis (int): Axis along which to compute gradient.
            
        Returns:
            cp.ndarray: Gradient array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.gradient(array, axis=axis)
        return np.gradient(array, axis=axis)
    
    def _cuda_sqrt(self, array) -> 'cp.ndarray':
        """
        Compute square root using CUDA.
        
        Physical Meaning:
            Computes square root using CUDA for optimal performance.
            
        Args:
            array: Input array.
            
        Returns:
            cp.ndarray: Square root array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.sqrt(array)
        return np.sqrt(array)
    
    def _cuda_sum(self, array, axis=None) -> 'cp.ndarray':
        """
        Compute sum using CUDA.
        
        Physical Meaning:
            Computes sum using CUDA for optimal performance.
            
        Args:
            array: Input array.
            axis: Axis along which to sum.
            
        Returns:
            cp.ndarray: Sum array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.sum(array, axis=axis)
        return np.sum(array, axis=axis)
    
    def _cuda_sin(self, array) -> 'cp.ndarray':
        """
        Compute sine using CUDA.
        
        Physical Meaning:
            Computes sine using CUDA for optimal performance.
            
        Args:
            array: Input array.
            
        Returns:
            cp.ndarray: Sine array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.sin(array)
        return np.sin(array)
    
    def _cuda_cos(self, array) -> 'cp.ndarray':
        """
        Compute cosine using CUDA.
        
        Physical Meaning:
            Computes cosine using CUDA for optimal performance.
            
        Args:
            array: Input array.
            
        Returns:
            cp.ndarray: Cosine array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.cos(array)
        return np.cos(array)
    
    def _cuda_abs(self, array) -> 'cp.ndarray':
        """
        Compute absolute value using CUDA.
        
        Physical Meaning:
            Computes absolute value using CUDA for optimal performance.
            
        Args:
            array: Input array.
            
        Returns:
            cp.ndarray: Absolute value array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.abs(array)
        return np.abs(array)

    def __repr__(self) -> str:
        """String representation of electroweak coupling."""
        cuda_status = "CUDA" if self.use_cuda else "CPU"
        return (
            f"ElectroweakCoupling("
            f"em_coupling={self.electroweak_coefficients['em_coupling']:.3f}, "
            f"weak_coupling={self.electroweak_coefficients['weak_coupling']:.3f}, "
            f"mixing_angle={self.electroweak_coefficients['mixing_angle']:.3f}, "
            f"compute={cuda_status})"
        )
