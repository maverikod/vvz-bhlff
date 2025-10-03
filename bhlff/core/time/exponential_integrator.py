"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

DEPRECATED: Classical exponential integrator replaced by BVP envelope integrator.

This module is deprecated and replaced by BVPEnvelopeIntegrator which
implements the BVP envelope modulation approach instead of classical
exponential solutions.

Physical Meaning:
    Classical exponential solutions contradict BVP theory where all
    observed "modes" are envelope modulations and beatings of the
    Base High-Frequency Field.

Mathematical Foundation:
    Classical exponential solutions are replaced by envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    where κ(|a|) = κ₀ + κ₂|a|² and χ(|a|) = χ' + iχ''(|a|)

DEPRECATED: Use BVPEnvelopeIntegrator instead.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging

from .base_integrator import BaseTimeIntegrator
from .memory_kernel import MemoryKernel
from .quench_detector import QuenchDetector
from ..fft import SpectralOperations


class BVPExponentialIntegrator(BaseTimeIntegrator):
    """
    DEPRECATED: Classical exponential integrator replaced by BVP envelope integrator.

    Physical Meaning:
        Classical exponential solutions contradict BVP theory where all
        observed "modes" are envelope modulations and beatings of the
        Base High-Frequency Field.

    Mathematical Foundation:
        Classical exponential solutions are replaced by envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)

    DEPRECATED: Use BVPEnvelopeIntegrator instead.

    Attributes:
        domain (Domain): Computational domain.
        parameters (Parameters): Physics parameters.
        _spectral_ops (SpectralOperations): Spectral operations for FFT.
        _spectral_coeffs (np.ndarray): Pre-computed spectral coefficients.
        _memory_kernel (Optional[MemoryKernel]): Memory kernel for non-local effects.
        _quench_detector (Optional[QuenchDetector]): Quench detection system.
    """

    def __init__(self, domain, parameters) -> None:
        """
        Initialize BVP exponential integrator.

        Physical Meaning:
            Sets up the exponential integrator with the computational domain
            and physics parameters, pre-computing spectral coefficients
            for efficient integration.

        Args:
            domain (Domain): Computational domain for the simulation.
            parameters (Parameters): Physics parameters controlling
                the behavior of the phase field system.
        """
        super().__init__(domain, parameters)

        # Initialize spectral operations
        self._spectral_ops = SpectralOperations(domain, parameters.precision)

        # Pre-compute spectral coefficients
        self._spectral_coeffs = None
        self._setup_spectral_coefficients()

        self._initialized = True
        self.logger.info("BVP Exponential integrator initialized")

    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for exponential integrator.

        Physical Meaning:
            Pre-computes the spectral representation of the operator
            ν|k|^(2β) + λ for efficient exponential integration.
        """
        # Get wave vectors
        wave_vectors = self._spectral_ops._get_wave_vectors()

        # Compute wave vector magnitudes
        k_magnitude_squared = np.zeros(self.domain.shape)
        
        # Create meshgrids for each dimension
        for i, k_vec in enumerate(wave_vectors):
            if i < 3:  # Spatial dimensions
                # Create 7D array by broadcasting
                k_7d = np.zeros(self.domain.shape)
                for j in range(self.domain.N):
                    for k in range(self.domain.N):
                        for l in range(self.domain.N):
                            k_7d[j, k, l, :, :, :, :] = k_vec[j] if i == 0 else (k_vec[k] if i == 1 else k_vec[l])
                k_magnitude_squared += k_7d**2
            elif i < 6:  # Phase dimensions
                # Create 7D array by broadcasting
                k_7d = np.zeros(self.domain.shape)
                for j in range(self.domain.N_phi):
                    k_7d[:, :, :, j, :, :, :] = k_vec[j] if i == 3 else (k_vec[j] if i == 4 else k_vec[j])
                k_magnitude_squared += k_7d**2
            else:  # Temporal dimension
                # Create 7D array by broadcasting
                k_7d = np.zeros(self.domain.shape)
                for j in range(self.domain.N_t):
                    k_7d[:, :, :, :, :, :, j] = k_vec[j]
                k_magnitude_squared += k_7d**2
        
        k_magnitude = np.sqrt(k_magnitude_squared)

        # Compute spectral coefficients: ν|k|^(2β) + λ
        self._spectral_coeffs = (
            self.parameters.nu * (k_magnitude ** (2 * self.parameters.beta))
            + self.parameters.lambda_param
        )

        # Handle k=0 mode
        k_zero_mask = k_magnitude == 0
        if self.parameters.lambda_param > 0:
            self._spectral_coeffs[k_zero_mask] = self.parameters.lambda_param
        else:
            # For λ=0, k=0 mode should be handled separately
            self._spectral_coeffs[k_zero_mask] = (
                1e-12  # Small value to avoid division by zero
            )

        self.logger.info("Spectral coefficients computed")

    def integrate(
        self,
        initial_field: np.ndarray,
        source_field: np.ndarray,
        time_steps: np.ndarray,
    ) -> np.ndarray:
        """
        Integrate the dynamic equation over time using exponential method.

        Physical Meaning:
            Solves the dynamic phase field equation over the specified
            time steps using the exponential integrator, providing
            optimal accuracy for BVP problems.

        Mathematical Foundation:
            For each time step, applies the exponential solution:
            â(k,t+dt) = â(k,t)e^(-(ν|k|^(2β)+λ)dt) + ŝ(k,t)/(ν|k|^(2β)+λ)(1-e^(-(ν|k|^(2β)+λ)dt))

        Args:
            initial_field (np.ndarray): Initial field configuration a(x,φ,0).
            source_field (np.ndarray): Source term s(x,φ,t) over time.
            time_steps (np.ndarray): Time points for integration.

        Returns:
            np.ndarray: Field evolution a(x,φ,t) over time.
        """
        if not self._initialized:
            raise RuntimeError("Integrator not initialized")

        # Validate inputs
        if initial_field.shape != self.domain.shape:
            raise ValueError(
                f"Initial field shape {initial_field.shape} incompatible with domain {self.domain.shape}"
            )

        if source_field.shape != (len(time_steps),) + self.domain.shape:
            raise ValueError(
                f"Source field shape {source_field.shape} incompatible with time steps and domain"
            )

        # Initialize result array
        result = np.zeros((len(time_steps),) + self.domain.shape, dtype=np.complex128)
        result[0] = initial_field.copy()

        # Current field state
        current_field = initial_field.copy()

        # Integrate over time steps
        for i in range(1, len(time_steps)):
            dt = time_steps[i] - time_steps[i - 1]
            current_source = source_field[i]

            # Perform single step
            current_field = self.step(current_field, current_source, dt)
            result[i] = current_field.copy()

            # Check for quench events
            if self._quench_detector is not None and self._quench_detector.detect_quench(current_field, time_steps[i]):
                self.logger.warning(f"Quench detected at t={time_steps[i]:.3f}")
                # DEPRECATED: Classical exponential integrator - use BVPEnvelopeIntegrator instead

        self.logger.info(f"Integration completed over {len(time_steps)} time steps")
        return result

    def step(
        self, current_field: np.ndarray, source_field: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Perform a single time step using exponential method.

        Physical Meaning:
            Advances the field configuration by one time step using the
            exponential integrator, providing optimal accuracy for
            BVP problems.

        Mathematical Foundation:
            Applies the exponential solution:
            â(k,t+dt) = â(k,t)e^(-(ν|k|^(2β)+λ)dt) + ŝ(k,t)/(ν|k|^(2β)+λ)(1-e^(-(ν|k|^(2β)+λ)dt))

        Args:
            current_field (np.ndarray): Current field configuration.
            source_field (np.ndarray): Source term at current time.
            dt (float): Time step size.

        Returns:
            np.ndarray: Field configuration at next time step.
        """
        if not self._initialized:
            raise RuntimeError("Integrator not initialized")

        # Transform to spectral space
        current_spectral = self._spectral_ops.forward_fft(current_field, "ortho")
        source_spectral = self._spectral_ops.forward_fft(source_field, "ortho")

        # Apply memory kernel effects if present
        if self._memory_kernel is not None:
            memory_contribution = self._memory_kernel.get_memory_contribution()
            memory_spectral = self._spectral_ops.forward_fft(
                memory_contribution, "ortho"
            )
            source_spectral += memory_spectral

        # Exponential integration in spectral space
        # â(k,t+dt) = â(k,t)e^(-(ν|k|^(2β)+λ)dt) + ŝ(k,t)/(ν|k|^(2β)+λ)(1-e^(-(ν|k|^(2β)+λ)dt))
        decay_factor = np.exp(-self._spectral_coeffs * dt)

        # Handle division by zero for k=0 mode when λ=0
        source_factor = np.zeros_like(self._spectral_coeffs)
        nonzero_mask = self._spectral_coeffs != 0
        source_factor[nonzero_mask] = (
            1 - decay_factor[nonzero_mask]
        ) / self._spectral_coeffs[nonzero_mask]
        # For k=0 mode with λ=0, source_factor = dt (limit as λ→0)
        source_factor[~nonzero_mask] = dt

        next_spectral = (
            current_spectral * decay_factor + source_spectral * source_factor
        )

        # Transform back to real space
        next_field = self._spectral_ops.inverse_fft(next_spectral, "ortho")

        # Evolve memory kernel if present
        if self._memory_kernel is not None:
            self._memory_kernel.evolve(current_field, dt)

        return next_field

    def integrate_harmonic_source(
        self,
        initial_field: np.ndarray,
        source_amplitude: np.ndarray,
        frequency: float,
        time_steps: np.ndarray,
    ) -> np.ndarray:
        """
        Integrate with harmonic source using exact solution.

        Physical Meaning:
            Solves the dynamic equation with harmonic source s(x,t) = s₀(x)e^(-iωt)
            using the exact analytical solution for optimal accuracy.

        Mathematical Foundation:
            For harmonic source, exact solution:
            â(k,t) = â₀(k)e^(-(ν|k|^(2β)+λ)t) + ŝ₀(k)/(ν|k|^(2β)+λ+iω)(1-e^(-(ν|k|^(2β)+λ+iω)t))

        Args:
            initial_field (np.ndarray): Initial field configuration.
            source_amplitude (np.ndarray): Source amplitude s₀(x).
            frequency (float): Source frequency ω.
            time_steps (np.ndarray): Time points for integration.

        Returns:
            np.ndarray: Field evolution over time.
        """
        if not self._initialized:
            raise RuntimeError("Integrator not initialized")

        # Transform to spectral space
        initial_spectral = self._spectral_ops.forward_fft(initial_field, "ortho")
        source_spectral = self._spectral_ops.forward_fft(source_amplitude, "ortho")

        # Compute complex spectral coefficients
        complex_coeffs = self._spectral_coeffs + 1j * frequency

        # Initialize result
        result = np.zeros((len(time_steps),) + self.domain.shape, dtype=np.complex128)

        # Apply exact solution for each time step
        for i, t in enumerate(time_steps):
            # Exact solution: â(k,t) = â₀(k)e^(-(ν|k|^(2β)+λ)t) + ŝ₀(k)/(ν|k|^(2β)+λ+iω)(1-e^(-(ν|k|^(2β)+λ+iω)t))
            decay_factor = np.exp(-self._spectral_coeffs * t)
            harmonic_factor = (1 - np.exp(-complex_coeffs * t)) / complex_coeffs

            field_spectral = (
                initial_spectral * decay_factor + source_spectral * harmonic_factor
            )

            # Transform back to real space
            result[i] = self._spectral_ops.inverse_fft(field_spectral, "ortho")

        self.logger.info(f"Harmonic integration completed for frequency ω={frequency}")
        return result

    def __repr__(self) -> str:
        """String representation of integrator."""
        return (
            f"BVPExponentialIntegrator("
            f"domain={self.domain.shape}, "
            f"nu={self.parameters.nu}, "
            f"beta={self.parameters.beta}, "
            f"lambda={self.parameters.lambda_param})"
        )
