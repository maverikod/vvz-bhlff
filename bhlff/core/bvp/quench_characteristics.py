"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quench characteristics computation.

This module implements the computation of quench characteristics
such as center of mass, strength, and local frequency analysis
for quench detection in 7D space-time.

Physical Meaning:
    Computes various characteristics of quench events including
    center of mass, strength measures, and local frequency
    analysis to provide comprehensive quench event information.

Mathematical Foundation:
    - Center of mass: Σ(r_i * w_i) / Σ(w_i)
    - Quench strength: max(|A|) within component
    - Local frequency: |dφ/dt| / dt
    - Gradient magnitude: |∇A| in 7D space-time

Example:
    >>> characteristics = QuenchCharacteristics(domain_7d)
    >>> center = characteristics.compute_center_of_mass(component_mask)
    >>> strength = characteristics.compute_quench_strength(component_mask, amplitude)
"""

import numpy as np
from typing import Tuple, Dict, Any

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from ..domain.domain_7d import Domain7D


class QuenchCharacteristics:
    """
    Computer for quench event characteristics.

    Physical Meaning:
        Computes various characteristics of quench events including
        center of mass, strength measures, and local frequency
        analysis to provide comprehensive quench event information.

    Mathematical Foundation:
        - Center of mass: Σ(r_i * w_i) / Σ(w_i)
        - Quench strength: max(|A|) within component
        - Local frequency: |dφ/dt| / dt
        - Gradient magnitude: |∇A| in 7D space-time
    """

    def __init__(self, domain_7d: Domain7D):
        """
        Initialize quench characteristics computer.

        Physical Meaning:
            Sets up the characteristics computer with the computational
            domain to compute quench event properties.

        Args:
            domain_7d (Domain7D): 7D computational domain.
        """
        self.domain_7d = domain_7d
        self.cuda_available = CUDA_AVAILABLE

    def compute_center_of_mass(self, component_mask: np.ndarray) -> Tuple[float, ...]:
        """
        Compute center of mass for a quench component.

        Physical Meaning:
            Calculates the center of mass of a quench component,
            representing the effective location of the quench event
            in 7D space-time.

        Mathematical Foundation:
            Center of mass = Σ(r_i * w_i) / Σ(w_i)
            where r_i are coordinates and w_i are weights (amplitudes).

        Args:
            component_mask (np.ndarray): Binary mask of component.

        Returns:
            Tuple[float, ...]: 7D coordinates of center of mass.
        """
        # Get coordinates of component points
        coords = np.where(component_mask)

        if len(coords[0]) == 0:
            return (0.0,) * 7

        # Compute center of mass (simple average for now)
        center = []
        for axis in range(7):
            center.append(float(np.mean(coords[axis])))

        return tuple(center)

    def compute_quench_strength(
        self, component_mask: np.ndarray, amplitude: np.ndarray
    ) -> float:
        """
        Compute quench strength for a component.

        Physical Meaning:
            Calculates the strength of a quench event based on
            the maximum amplitude within the component region.

        Mathematical Foundation:
            Quench strength = max(|A|) within component
            This represents the peak field strength in the quench region.

        Args:
            component_mask (np.ndarray): Binary mask of component.
            amplitude (np.ndarray): Amplitude field.

        Returns:
            float: Quench strength.
        """
        # Get amplitudes within component
        component_amplitudes = amplitude[component_mask]

        if len(component_amplitudes) == 0:
            return 0.0

        # Return maximum amplitude as quench strength
        return float(np.max(component_amplitudes))

    def compute_local_frequency(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute local frequency from phase evolution.

        Physical Meaning:
            Calculates the local frequency at each point in 7D space-time
            by analyzing the phase evolution of the envelope field.
            This represents the instantaneous frequency of the BVP field.

        Mathematical Foundation:
            ω_local = |dφ/dt| / dt
            where φ is the phase of the envelope and dt is the time step.
            Uses finite differences to approximate the derivative.

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            np.ndarray: Local frequency field with same shape as envelope.
        """
        # Extract phase
        phase = np.angle(envelope)

        # Compute phase difference along time axis
        if envelope.shape[-1] > 1:
            phase_diff = np.diff(phase, axis=-1)

            # Get time step
            dt = self.domain_7d.temporal_config.dt

            # Compute local frequency (avoid division by zero)
            local_frequency = np.abs(phase_diff) / (dt + 1e-12)

            # Pad to match original shape
            local_frequency = np.pad(
                local_frequency,
                [(0, 0)] * (local_frequency.ndim - 1) + [(0, 1)],
                mode="edge",
            )
        else:
            # Single time slice - use zero frequency
            local_frequency = np.zeros_like(phase)

        return local_frequency

    def compute_detuning_strength(
        self, component_mask: np.ndarray, detuning: np.ndarray
    ) -> float:
        """
        Compute detuning strength for a component.

        Physical Meaning:
            Calculates the strength of a detuning quench event based on
            the maximum detuning within the component region.

        Mathematical Foundation:
            Detuning strength = max(|ω_local - ω_0|) within component
            This represents the peak frequency deviation in the quench region.

        Args:
            component_mask (np.ndarray): Binary mask of component.
            detuning (np.ndarray): Detuning field.

        Returns:
            float: Detuning strength.
        """
        # Get detuning values within component
        component_detuning = detuning[component_mask]

        if len(component_detuning) == 0:
            return 0.0

        # Return maximum detuning as quench strength
        return float(np.max(component_detuning))

    def compute_7d_gradient_magnitude(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute 7D gradient magnitude of envelope field.

        Physical Meaning:
            Calculates the magnitude of the gradient in all 7 dimensions
            (3 spatial + 3 phase + 1 temporal), representing the rate
            of change of the envelope field in 7D space-time.

        Mathematical Foundation:
            ∇A = (∂A/∂x, ∂A/∂y, ∂A/∂z, ∂A/∂φ₁, ∂A/∂φ₂, ∂A/∂φ₃, ∂A/∂t)
            |∇A| = √(Σ|∂A/∂xᵢ|²)
            Uses finite differences to approximate partial derivatives.

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            np.ndarray: Gradient magnitude field with same shape as envelope.
        """
        # Get differentials for all 7 dimensions
        differentials = self.domain_7d.get_differentials()

        # Compute gradients in all 7 dimensions
        gradients = []

        # Spatial gradients (x, y, z)
        for axis, dx in enumerate(
            [differentials["dx"], differentials["dy"], differentials["dz"]]
        ):
            grad = np.gradient(envelope, dx, axis=axis)
            gradients.append(grad)

        # Phase gradients (φ₁, φ₂, φ₃)
        for axis, dphi in enumerate(
            [differentials["dphi_1"], differentials["dphi_2"], differentials["dphi_3"]]
        ):
            grad = np.gradient(envelope, dphi, axis=axis + 3)
            gradients.append(grad)

        # Temporal gradient (t)
        if envelope.shape[-1] > 1:
            dt = differentials.get("dt", 1.0)
            grad_t = np.gradient(envelope, dt, axis=-1)
            gradients.append(grad_t)
        else:
            # Single time slice - zero temporal gradient
            grad_t = np.zeros_like(envelope)
            gradients.append(grad_t)

        # Compute gradient magnitude
        grad_magnitude = np.sqrt(sum(np.abs(grad) ** 2 for grad in gradients))

        return grad_magnitude

    def compute_gradient_strength(
        self, component_mask: np.ndarray, gradient_magnitude: np.ndarray
    ) -> float:
        """
        Compute gradient strength for a component.

        Physical Meaning:
            Calculates the strength of a gradient quench event based on
            the maximum gradient magnitude within the component region.

        Mathematical Foundation:
            Gradient strength = max(|∇A|) within component
            This represents the peak gradient magnitude in the quench region.

        Args:
            component_mask (np.ndarray): Binary mask of component.
            gradient_magnitude (np.ndarray): Gradient magnitude field.

        Returns:
            float: Gradient strength.
        """
        # Get gradient magnitudes within component
        component_gradients = gradient_magnitude[component_mask]

        if len(component_gradients) == 0:
            return 0.0

        # Return maximum gradient magnitude as quench strength
        return float(np.max(component_gradients))

    def compute_center_of_mass_cuda(self, component_mask_gpu) -> Tuple[float, ...]:
        """
        Compute center of mass for a quench component using CUDA.

        Physical Meaning:
            Calculates the center of mass of a quench component on GPU,
            representing the effective location of the quench event
            in 7D space-time.

        Args:
            component_mask_gpu: GPU array of component mask.

        Returns:
            Tuple[float, ...]: 7D coordinates of center of mass.
        """
        if not self.cuda_available:
            # Fallback to CPU if CUDA not available
            component_mask_cpu = (
                cp.asnumpy(component_mask_gpu)
                if hasattr(component_mask_gpu, "get")
                else component_mask_gpu
            )
            return self.compute_center_of_mass(component_mask_cpu)

        # Ensure component_mask_gpu is CuPy array
        if not hasattr(component_mask_gpu, "get"):
            component_mask_gpu = cp.asarray(component_mask_gpu)

        # Get coordinates of component points on GPU
        coords = cp.where(component_mask_gpu)

        if len(coords[0]) == 0:
            return (0.0,) * 7

        # Compute center of mass on GPU
        center = []
        for axis in range(7):
            center.append(float(cp.mean(coords[axis])))

        return tuple(center)

    def compute_quench_strength_cuda(self, component_mask_gpu, amplitude_gpu) -> float:
        """
        Compute quench strength for a component using CUDA.

        Physical Meaning:
            Calculates the strength of a quench event based on
            the maximum amplitude within the component region on GPU.

        Args:
            component_mask_gpu: GPU array of component mask.
            amplitude_gpu: GPU array of amplitude field.

        Returns:
            float: Quench strength.
        """
        if not self.cuda_available:
            # Fallback to CPU if CUDA not available
            component_mask_cpu = (
                cp.asnumpy(component_mask_gpu)
                if hasattr(component_mask_gpu, "get")
                else component_mask_gpu
            )
            amplitude_cpu = (
                cp.asnumpy(amplitude_gpu)
                if hasattr(amplitude_gpu, "get")
                else amplitude_gpu
            )
            return self.compute_quench_strength(component_mask_cpu, amplitude_cpu)

        # Ensure arrays are CuPy arrays
        if not hasattr(component_mask_gpu, "get"):
            component_mask_gpu = cp.asarray(component_mask_gpu)
        if not hasattr(amplitude_gpu, "get"):
            amplitude_gpu = cp.asarray(amplitude_gpu)

        # Get amplitudes within component on GPU
        component_amplitudes = amplitude_gpu[component_mask_gpu]

        if len(component_amplitudes) == 0:
            return 0.0

        # Return maximum amplitude as quench strength
        return float(cp.max(component_amplitudes))

    def compute_local_frequency_cuda(self, envelope_gpu) -> np.ndarray:
        """
        Compute local frequency from phase evolution using CUDA.

        Physical Meaning:
            Calculates the local frequency at each point in 7D space-time
            by analyzing the phase evolution of the envelope field on GPU.

        Args:
            envelope_gpu: GPU array of 7D envelope field.

        Returns:
            np.ndarray: Local frequency field (transferred to CPU).
        """
        if not self.cuda_available:
            # Fallback to CPU if CUDA not available
            envelope_cpu = (
                cp.asnumpy(envelope_gpu)
                if hasattr(envelope_gpu, "get")
                else envelope_gpu
            )
            return self.compute_local_frequency(envelope_cpu)

        # Ensure envelope_gpu is CuPy array
        if not hasattr(envelope_gpu, "get"):
            envelope_gpu = cp.asarray(envelope_gpu)

        # Extract phase on GPU
        phase = cp.angle(envelope_gpu)

        # Compute phase difference along time axis
        if envelope_gpu.shape[-1] > 1:
            phase_diff = cp.diff(phase, axis=-1)

            # Get time step
            dt = self.domain_7d.temporal_config.dt

            # Compute local frequency (avoid division by zero)
            local_frequency = cp.abs(phase_diff) / (dt + 1e-12)

            # Pad to match original shape
            local_frequency = cp.pad(
                local_frequency,
                [(0, 0)] * (local_frequency.ndim - 1) + [(0, 1)],
                mode="edge",
            )
        else:
            # Single time slice - use zero frequency
            local_frequency = cp.zeros_like(phase)

        return local_frequency

    def compute_detuning_strength_cuda(self, component_mask_gpu, detuning_gpu) -> float:
        """
        Compute detuning strength for a component using CUDA.

        Physical Meaning:
            Calculates the strength of a detuning quench event based on
            the maximum detuning within the component region on GPU.

        Args:
            component_mask_gpu: GPU array of component mask.
            detuning_gpu: GPU array of detuning field.

        Returns:
            float: Detuning strength.
        """
        if not self.cuda_available:
            # Fallback to CPU if CUDA not available
            component_mask_cpu = (
                cp.asnumpy(component_mask_gpu)
                if hasattr(component_mask_gpu, "get")
                else component_mask_gpu
            )
            detuning_cpu = (
                cp.asnumpy(detuning_gpu)
                if hasattr(detuning_gpu, "get")
                else detuning_gpu
            )
            return self.compute_detuning_strength(component_mask_cpu, detuning_cpu)

        # Ensure arrays are CuPy arrays
        if not hasattr(component_mask_gpu, "get"):
            component_mask_gpu = cp.asarray(component_mask_gpu)
        if not hasattr(detuning_gpu, "get"):
            detuning_gpu = cp.asarray(detuning_gpu)

        # Get detuning values within component on GPU
        component_detuning = detuning_gpu[component_mask_gpu]

        if len(component_detuning) == 0:
            return 0.0

        # Return maximum detuning as quench strength
        return float(cp.max(component_detuning))

    def compute_7d_gradient_magnitude_cuda(self, envelope_gpu) -> np.ndarray:
        """
        Compute 7D gradient magnitude of envelope field using CUDA.

        Physical Meaning:
            Calculates the magnitude of the gradient in all 7 dimensions
            (3 spatial + 3 phase + 1 temporal) on GPU, representing the rate
            of change of the envelope field in 7D space-time.

        Args:
            envelope_gpu: GPU array of 7D envelope field.

        Returns:
            np.ndarray: Gradient magnitude field (transferred to CPU).
        """
        if not self.cuda_available:
            # Fallback to CPU if CUDA not available
            envelope_cpu = cp.asnumpy(envelope_gpu)
            return self.compute_7d_gradient_magnitude(envelope_cpu)

        # Get differentials for all 7 dimensions
        differentials = self.domain_7d.get_differentials()

        # Compute gradients in all 7 dimensions on GPU
        gradients = []

        # Spatial gradients (x, y, z)
        for axis, dx in enumerate(
            [differentials["dx"], differentials["dy"], differentials["dz"]]
        ):
            grad = cp.gradient(envelope_gpu, dx, axis=axis)
            gradients.append(grad)

        # Phase gradients (φ₁, φ₂, φ₃)
        for axis, dphi in enumerate(
            [differentials["dphi_1"], differentials["dphi_2"], differentials["dphi_3"]]
        ):
            grad = cp.gradient(envelope_gpu, dphi, axis=axis + 3)
            gradients.append(grad)

        # Temporal gradient (t)
        if envelope_gpu.shape[-1] > 1:
            dt = differentials.get("dt", 1.0)
            grad_t = cp.gradient(envelope_gpu, dt, axis=-1)
            gradients.append(grad_t)
        else:
            # Single time slice - zero temporal gradient
            grad_t = cp.zeros_like(envelope_gpu)
            gradients.append(grad_t)

        # Compute gradient magnitude on GPU
        grad_magnitude = cp.sqrt(sum(cp.abs(grad) ** 2 for grad in gradients))

        return grad_magnitude

    def compute_gradient_strength_cuda(
        self, component_mask_gpu, gradient_magnitude_gpu
    ) -> float:
        """
        Compute gradient strength for a component using CUDA.

        Physical Meaning:
            Calculates the strength of a gradient quench event based on
            the maximum gradient magnitude within the component region on GPU.

        Args:
            component_mask_gpu: GPU array of component mask.
            gradient_magnitude_gpu: GPU array of gradient magnitude field.

        Returns:
            float: Gradient strength.
        """
        if not self.cuda_available:
            # Fallback to CPU if CUDA not available
            component_mask_cpu = (
                cp.asnumpy(component_mask_gpu)
                if hasattr(component_mask_gpu, "get")
                else component_mask_gpu
            )
            gradient_magnitude_cpu = (
                cp.asnumpy(gradient_magnitude_gpu)
                if hasattr(gradient_magnitude_gpu, "get")
                else gradient_magnitude_gpu
            )
            return self.compute_gradient_strength(
                component_mask_cpu, gradient_magnitude_cpu
            )

        # Ensure arrays are CuPy arrays
        if not hasattr(component_mask_gpu, "get"):
            component_mask_gpu = cp.asarray(component_mask_gpu)
        if not hasattr(gradient_magnitude_gpu, "get"):
            gradient_magnitude_gpu = cp.asarray(gradient_magnitude_gpu)

        # Get gradient magnitudes within component on GPU
        component_gradients = gradient_magnitude_gpu[component_mask_gpu]

        if len(component_gradients) == 0:
            return 0.0

        # Return maximum gradient magnitude as quench strength
        return float(cp.max(component_gradients))
