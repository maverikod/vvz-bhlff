"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP source envelope and carrier generation implementation.

This module provides envelope and carrier generation functionality for
BVP-modulated sources in the 7D phase field theory.

Physical Meaning:
    BVP source envelope and carrier generation creates the envelope modulation
    and high-frequency carrier components for BVP-modulated sources.

Mathematical Foundation:
    Implements envelope and carrier generation:
    - Envelope: A(x) = A₀ * f(x)
    - Carrier: exp(iω₀t) with frequency ω₀

Example:
    >>> envelope_generator = BVPSourceEnvelope(domain, config)
    >>> envelope = envelope_generator.generate_envelope()
    >>> carrier = envelope_generator.generate_carrier()
"""

import numpy as np
from typing import Dict, Any, Optional

from ..domain import Domain
from .block_7d_expansion import (
    expand_spatial_to_7d,
    generate_7d_block_on_device,
)

# Import CUDA availability check
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class BVPSourceEnvelope:
    """
    BVP source envelope and carrier generator.

    Physical Meaning:
        Generates envelope modulation and high-frequency carrier components
        for BVP-modulated sources in phase field evolution.

    Mathematical Foundation:
        Implements envelope and carrier generation:
        - Envelope: A(x) = A₀ * f(x)
        - Carrier: exp(iω₀t) with frequency ω₀

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Envelope generator configuration.
        carrier_frequency (float): High-frequency carrier frequency.
        envelope_amplitude (float): Envelope amplitude.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize BVP source envelope generator.

        Physical Meaning:
            Sets up the BVP source envelope generator with domain and
            configuration for generating envelope and carrier components.

        Args:
            domain (Domain): Computational domain for envelope generation.
            config (Dict[str, Any]): Envelope generator configuration.
        """
        self.domain = domain
        self.config = config
        self.carrier_frequency = config.get("carrier_frequency", 1.85e43)
        self.envelope_amplitude = config.get("envelope_amplitude", 1.0)

    def generate_envelope(self) -> np.ndarray:
        """
        Generate envelope modulation.

        Physical Meaning:
            Creates the envelope modulation A(x) that modulates the
            base source in the BVP framework.

        Mathematical Foundation:
            Envelope: A(x) = A₀ * f(x)
            where A₀ is the envelope amplitude and f(x) is the spatial
            distribution function.

        Returns:
            np.ndarray: Envelope modulation field.
        """
        # Get envelope parameters
        envelope_type = self.config.get("envelope_type", "constant")

        # Create coordinate arrays
        x = np.linspace(0, 1, self.domain.N)
        y = np.linspace(0, 1, self.domain.N)
        z = np.linspace(0, 1, self.domain.N)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Generate envelope based on type
        if envelope_type == "constant":
            envelope = self.envelope_amplitude * np.ones_like(X)

        elif envelope_type == "gaussian":
            # Gaussian envelope
            center = self.config.get("envelope_center", [0.5, 0.5, 0.5])
            width = self.config.get("envelope_width", 0.2)

            dx = X - center[0]
            dy = Y - center[1]
            dz = Z - center[2]
            r_squared = dx**2 + dy**2 + dz**2

            envelope = self.envelope_amplitude * self._step_resonator_envelope(
                r_squared, width
            )

        elif envelope_type == "sine":
            # Sine wave envelope
            kx = self.config.get("envelope_kx", 2 * np.pi)
            ky = self.config.get("envelope_ky", 2 * np.pi)
            kz = self.config.get("envelope_kz", 2 * np.pi)

            envelope = self.envelope_amplitude * (
                np.sin(kx * X) * np.sin(ky * Y) * np.sin(kz * Z)
            )

        elif envelope_type == "exponential":
            # Exponential envelope
            decay_rate = self.config.get("envelope_decay_rate", 1.0)
            center = self.config.get("envelope_center", [0.5, 0.5, 0.5])

            dx = X - center[0]
            dy = Y - center[1]
            dz = Z - center[2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)

            envelope = self.envelope_amplitude * self._step_resonator_decay(
                r, decay_rate
            )

        else:
            # Default to constant envelope
            envelope = self.envelope_amplitude * np.ones_like(X)

        return envelope

    def generate_carrier(self, time: float = 0.0, use_cuda: bool = False) -> np.ndarray:
        """
        Generate high-frequency carrier with explicit 7D construction.

        Physical Meaning:
            Creates the high-frequency carrier component exp(iω₀t) that
            modulates the source in the BVP framework. For 7D domains,
            generates carrier with explicit phase and time dimensions.

        Mathematical Foundation:
            Carrier: exp(iω₀t)
            where ω₀ is the carrier frequency and t is time. For 7D domains,
            carrier is expanded to 7D with explicit phase and time extents.

        Args:
            time (float): Time for carrier generation.
            use_cuda (bool): Whether to use CUDA for generation (if available).

        Returns:
            np.ndarray: Carrier component (complex, 3D or 7D depending on domain).
        """
        # Generate carrier phase
        carrier_phase = self.carrier_frequency * time

        # Create carrier as complex exponential
        carrier = np.exp(1j * carrier_phase)

        # For 7D domains, expand to 7D with explicit phase/time dimensions
        if self.domain.dimensions == 7:
            # Create 3D spatial carrier block
            carrier_3d = carrier * np.ones(
                (self.domain.N, self.domain.N, self.domain.N), dtype=complex
            )
            # Expand to 7D with explicit phase and time extents
            carrier_field = expand_spatial_to_7d(
                carrier_3d,
                N_phi=self.domain.N_phi,
                N_t=self.domain.N_t,
                use_cuda=use_cuda,
            )
        else:
            # For non-7D domains, return 3D carrier
            carrier_field = carrier * np.ones(
                (self.domain.N, self.domain.N, self.domain.N), dtype=complex
            )

        return carrier_field

    def generate_modulated_source(
        self, base_source: np.ndarray, time: float = 0.0, use_cuda: bool = False
    ) -> np.ndarray:
        """
        Generate BVP-modulated source with explicit 7D construction and block processing.

        Physical Meaning:
            Creates the complete BVP-modulated source by combining the
            base source, envelope, and carrier components using explicit
            7D block construction that respects the 7D space-time structure
            M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ. For large arrays, uses block-wise processing
            to manage memory constraints (80% GPU memory limit).

        Mathematical Foundation:
            BVP-modulated source: s(x,φ,t) = s₀(x) * A(x) * exp(iω₀t)
            where s₀(x) is the base source, A(x) is the envelope, and
            exp(iω₀t) is the carrier. Components are explicitly expanded
            to 7D with concrete phase and time extents using block-wise
            processing for large arrays.

        Args:
            base_source (np.ndarray): Base source field (3D or 7D).
            time (float): Time for carrier generation.
            use_cuda (bool): Whether to use CUDA for expansion (if available).

        Returns:
            np.ndarray: BVP-modulated source field with shape matching domain.shape.
        """
        # Generate envelope and carrier (these are 3D initially)
        envelope = self.generate_envelope()
        carrier = self.generate_carrier(time, use_cuda=use_cuda)

        # Check if we need to expand to 7D
        if base_source.shape != self.domain.shape:
            # Expand 3D components to 7D using explicit construction
            # Get phase and time dimensions from domain
            N_phi = self.domain.N_phi
            N_t = self.domain.N_t

            # Explicit 7D expansion with concrete phase/time extents
            # Uses block processing automatically if needed (80% GPU memory limit)
            envelope_7d = expand_spatial_to_7d(
                envelope, N_phi=N_phi, N_t=N_t, use_cuda=use_cuda, optimize_block_size=True
            )
            carrier_7d = expand_spatial_to_7d(
                carrier, N_phi=N_phi, N_t=N_t, use_cuda=use_cuda, optimize_block_size=True
            )

            # Expand base_source if needed
            if base_source.shape == envelope.shape:
                base_source_7d = expand_spatial_to_7d(
                    base_source, N_phi=N_phi, N_t=N_t, use_cuda=use_cuda, optimize_block_size=True
                )
            else:
                base_source_7d = base_source

            # Combine components (vectorized operation, preserves 7D structure)
            if use_cuda and CUDA_AVAILABLE:
                import cupy as cp
                if isinstance(base_source_7d, np.ndarray):
                    base_source_7d = cp.asarray(base_source_7d)
                if isinstance(envelope_7d, np.ndarray):
                    envelope_7d = cp.asarray(envelope_7d)
                if isinstance(carrier_7d, np.ndarray):
                    carrier_7d = cp.asarray(carrier_7d)
                modulated_source = base_source_7d * envelope_7d * carrier_7d
                # Synchronize CUDA operations
                cp.cuda.Stream.null.synchronize()
            else:
                modulated_source = base_source_7d * envelope_7d * carrier_7d
        else:
            # Already 7D, just combine (vectorized operation)
            if use_cuda and CUDA_AVAILABLE:
                import cupy as cp
                if isinstance(base_source, np.ndarray):
                    base_source = cp.asarray(base_source)
                if isinstance(envelope, np.ndarray):
                    envelope = cp.asarray(envelope)
                if isinstance(carrier, np.ndarray):
                    carrier = cp.asarray(carrier)
                modulated_source = base_source * envelope * carrier
                cp.cuda.Stream.null.synchronize()
            else:
                modulated_source = base_source * envelope * carrier

        return modulated_source

    def get_envelope_info(self) -> Dict[str, Any]:
        """
        Get envelope information.

        Physical Meaning:
            Returns information about the envelope generation including
            parameters and configuration.

        Returns:
            Dict[str, Any]: Envelope information.
        """
        return {
            "envelope_amplitude": self.envelope_amplitude,
            "envelope_type": self.config.get("envelope_type", "constant"),
            "carrier_frequency": self.carrier_frequency,
            "supported_envelope_types": ["constant", "gaussian", "sine", "exponential"],
        }

    def get_carrier_info(self) -> Dict[str, Any]:
        """
        Get carrier information.

        Physical Meaning:
            Returns information about the carrier generation including
            frequency and mathematical description.

        Returns:
            Dict[str, Any]: Carrier information.
        """
        return {
            "carrier_frequency": self.carrier_frequency,
            "carrier_formula": "exp(iω₀t)",
            "frequency_units": "rad/s",
            "wavelength": (
                2 * np.pi / self.carrier_frequency if self.carrier_frequency > 0 else 0
            ),
        }

    def _step_resonator_envelope(
        self, r_squared: np.ndarray, width: float
    ) -> np.ndarray:
        """
        Step resonator envelope according to 7D BVP theory.

        Physical Meaning:
            Implements step function envelope instead of exponential decay
            according to 7D BVP theory principles.
        """
        cutoff_radius_squared = (width * 2.0) ** 2  # 2-sigma cutoff
        return np.where(r_squared < cutoff_radius_squared, 1.0, 0.0)

    def _step_resonator_decay(self, r: np.ndarray, decay_rate: float) -> np.ndarray:
        """
        Step resonator decay according to 7D BVP theory.

        Physical Meaning:
            Implements step function decay instead of exponential decay
            according to 7D BVP theory principles.
        """
        cutoff_radius = 1.0 / decay_rate  # Inverse decay rate cutoff
        return np.where(r < cutoff_radius, 1.0, 0.0)
