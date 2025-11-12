"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic source generation methods for BVP source generators.

This module provides basic source generation methods as a mixin class.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...arrays import FieldArray

try:
    import cupy as cp
except Exception:
    cp = None


class BVPSourceGeneratorsBasicMixin:
    """Mixin providing basic source generation methods."""
    
    def generate_gaussian_source(self) -> 'FieldArray':
        """
        Generate Gaussian source.
        
        Physical Meaning:
            Creates a Gaussian source distribution centered at a specified
            location with given width and amplitude.
        """
        from ...arrays import FieldArray
        # Get Gaussian parameters
        amplitude = self.config.get("gaussian_amplitude", 1.0)
        center = self.config.get("gaussian_center", [0.5, 0.5, 0.5])
        width = self.config.get("gaussian_width", 0.1)

        xp = cp if self.use_cuda else np

        # Create coordinate arrays
        x = xp.linspace(0, 1, self.domain.N)
        y = xp.linspace(0, 1, self.domain.N)
        z = xp.linspace(0, 1, self.domain.N)

        X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

        # Compute distances from center
        dx = X - center[0]
        dy = Y - center[1]
        dz = Z - center[2]
        r_squared = dx**2 + dy**2 + dz**2

        # Generate step resonator source
        source = amplitude * self._step_resonator_source(r_squared, width, xp=xp)

        if self.use_cuda:
            source = cp.asnumpy(source)

        # Return as FieldArray for transparent swap support
        return FieldArray(array=source)
    
    def generate_point_source(self) -> 'FieldArray':
        """
        Generate point source.
        
        Physical Meaning:
            Creates a point source at a specified location with given
            amplitude, representing a localized excitation.
        """
        from ...arrays import FieldArray
        # Get point source parameters
        amplitude = self.config.get("point_amplitude", 1.0)
        location = self.config.get("point_location", [0.5, 0.5, 0.5])

        # Create coordinate arrays
        x = np.linspace(0, 1, self.domain.N)
        y = np.linspace(0, 1, self.domain.N)
        z = np.linspace(0, 1, self.domain.N)

        # Find closest grid points to source location
        i = int(location[0] * (self.domain.N - 1))
        j = int(location[1] * (self.domain.N - 1))
        k = int(location[2] * (self.domain.N - 1))

        # Create point source
        source = np.zeros((self.domain.N, self.domain.N, self.domain.N))
        source[i, j, k] = amplitude

        # Return as FieldArray for transparent swap support
        return FieldArray(array=source)
    
    def generate_distributed_source(self) -> 'FieldArray':
        """
        Generate distributed source.
        
        Physical Meaning:
            Creates a distributed source with specified spatial distribution
            and amplitude profile.
        """
        from ...arrays import FieldArray
        # Get distributed source parameters
        amplitude = self.config.get("distributed_amplitude", 1.0)
        distribution_type = self.config.get("distribution_type", "sine")

        xp = cp if self.use_cuda else np

        # Create coordinate arrays
        x = xp.linspace(0, 1, self.domain.N)
        y = xp.linspace(0, 1, self.domain.N)
        z = xp.linspace(0, 1, self.domain.N)

        X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

        # Generate distributed source based on type
        if distribution_type == "sine":
            # Sine wave distribution
            kx = self.config.get("sine_kx", 2 * np.pi)
            ky = self.config.get("sine_ky", 2 * np.pi)
            kz = self.config.get("sine_kz", 2 * np.pi)

            source = amplitude * (xp.sin(kx * X) * xp.sin(ky * Y) * xp.sin(kz * Z))

        elif distribution_type == "cosine":
            # Cosine wave distribution
            kx = self.config.get("cosine_kx", 2 * np.pi)
            ky = self.config.get("cosine_ky", 2 * np.pi)
            kz = self.config.get("cosine_kz", 2 * np.pi)

            source = amplitude * (xp.cos(kx * X) * xp.cos(ky * Y) * xp.cos(kz * Z))

        elif distribution_type == "polynomial":
            # Polynomial distribution
            order = self.config.get("polynomial_order", 2)

            source = amplitude * (X**order + Y**order + Z**order)

        else:
            # Default to constant distribution
            source = amplitude * xp.ones_like(X)

        if self.use_cuda:
            source = cp.asnumpy(source)

        # Return as FieldArray for transparent swap support
        return FieldArray(array=source)
    
    def generate_plane_wave_source(self) -> 'FieldArray':
        """
        Generate plane wave source.
        
        Physical Meaning:
            Creates a plane wave source with specified wave vector (mode)
            and amplitude, representing a monochromatic excitation.
            
        Mathematical Foundation:
            Plane wave has the form:
            s(x) = A * exp(i * k · x)
            where k is the wave vector and A is the amplitude.
            
        Returns:
            FieldArray: Plane wave source field.
        """
        from ...arrays import FieldArray
        # Get plane wave parameters
        amplitude = self.config.get("plane_wave_amplitude", 1.0)
        mode = self.config.get("plane_wave_mode", [1, 0, 0])  # Default mode (1,0,0)
        
        # Convert mode to tuple if needed
        if isinstance(mode, (list, np.ndarray)):
            mode = tuple(int(m) for m in mode)
        
        xp = cp if self.use_cuda else np
        
        # Create grid indices (0 to N-1) for plane wave generation
        # This matches the formula used in tests: phase = sum(2π * m_i * g_i / N)
        grid = xp.meshgrid(
            *[xp.arange(self.domain.N) for _ in range(3)],
            indexing="ij"
        )
        
        # Compute phase: k · x = 2π * sum(m_i * g_i / N)
        # where g_i are grid indices (0 to N-1) and m_i are mode components
        phase = xp.zeros((self.domain.N, self.domain.N, self.domain.N), dtype=xp.float64)
        for i, (m_i, g_i) in enumerate(zip(mode, grid)):
            phase += (2.0 * xp.pi * m_i * g_i) / self.domain.N
        
        # Generate plane wave: A * exp(i * phase)
        source = amplitude * xp.exp(1j * phase)
        
        if self.use_cuda:
            source = cp.asnumpy(source)
        
        # Return as FieldArray for transparent swap support
        return FieldArray(array=source.astype(np.complex128))

