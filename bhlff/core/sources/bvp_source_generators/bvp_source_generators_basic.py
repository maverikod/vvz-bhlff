"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic source generation methods for BVP source generators.

This module provides basic source generation methods as a mixin class.
"""

import numpy as np
from typing import Callable, Optional, Tuple

from ...arrays import FieldArray
from ..blocked_field_generator import BlockedFieldGenerator
from ...domain.optimal_block_size_calculator import OptimalBlockSizeCalculator

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
    cp = None


class BVPSourceGeneratorsBasicMixin:
    """Mixin providing basic source generation methods."""
    
    _block_size_calculator: Optional[OptimalBlockSizeCalculator] = None

    def generate_gaussian_source(self) -> 'FieldArray':
        """
        Generate Gaussian source.
        
        Physical Meaning:
            Creates a Gaussian source distribution centered at a specified
            location with given width and amplitude.
        """
        self._ensure_cuda()

        amplitude = self.config.get("gaussian_amplitude", 1.0)
        center = self.config.get("gaussian_center", [0.5, 0.5, 0.5])
        width = self.config.get("gaussian_width", 0.1)

        swap_threshold = self.config.get("swap_threshold_gb")

        def gaussian_block(domain, slice_config, runtime_config):
            start = tuple(slice_config["start"])
            shape = tuple(slice_config["shape"])
            xp = cp
            x, y, z = self._create_spatial_axes(start, shape, xp, normalized=True)
            X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")
            dx = X - center[0]
            dy = Y - center[1]
            dz = Z - center[2]
            r_squared = dx**2 + dy**2 + dz**2
            spatial_block = amplitude * self._step_resonator_source(r_squared, width, xp=xp)
            return self._expand_spatial_block(spatial_block, shape, xp)

        return self._materialize_field_array(gaussian_block, swap_threshold_gb=swap_threshold)
    
    def generate_point_source(self) -> 'FieldArray':
        """
        Generate point source.
        
        Physical Meaning:
            Creates a point source at a specified location with given
            amplitude, representing a localized excitation.
        """
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

        # Create point source (3D spatial)
        source_3d = np.zeros((self.domain.N, self.domain.N, self.domain.N), dtype=np.complex128)
        source_3d[i, j, k] = amplitude

        # Determine target shape based on domain
        # Framework automatically generates correct dimensionality
        if hasattr(self.domain, 'dimensions') and self.domain.dimensions == 7:
            # Expand to 7D for 7D domain
            from ...sources.block_7d_expansion import expand_spatial_to_7d
            N_phi = getattr(self.domain, 'N_phi', 1)
            N_t = getattr(self.domain, 'N_t', 1)
            source = expand_spatial_to_7d(
                source_3d, N_phi, N_t, use_cuda=self.use_cuda, optimize_block_size=True
            )
        else:
            # Keep 3D for non-7D domain
            source = source_3d

        return FieldArray(array=source)
    
    def generate_distributed_source(self) -> 'FieldArray':
        """
        Generate distributed source.
        
        Physical Meaning:
            Creates a distributed source with specified spatial distribution
            and amplitude profile.
        """
        self._ensure_cuda()

        amplitude = self.config.get("distributed_amplitude", 1.0)
        distribution_type = self.config.get("distribution_type", "sine")

        def distributed_block(domain, slice_config, runtime_config):
            start = tuple(slice_config["start"])
            shape = tuple(slice_config["shape"])
            xp = cp
            x, y, z = self._create_spatial_axes(start, shape, xp, normalized=True)
            X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

            if distribution_type == "sine":
                kx = self.config.get("sine_kx", 2 * np.pi)
                ky = self.config.get("sine_ky", 2 * np.pi)
                kz = self.config.get("sine_kz", 2 * np.pi)
                spatial_block = amplitude * (xp.sin(kx * X) * xp.sin(ky * Y) * xp.sin(kz * Z))
            elif distribution_type == "cosine":
                kx = self.config.get("cosine_kx", 2 * np.pi)
                ky = self.config.get("cosine_ky", 2 * np.pi)
                kz = self.config.get("cosine_kz", 2 * np.pi)
                spatial_block = amplitude * (xp.cos(kx * X) * xp.cos(ky * Y) * xp.cos(kz * Z))
            elif distribution_type == "polynomial":
                order = self.config.get("polynomial_order", 2)
                spatial_block = amplitude * (X**order + Y**order + Z**order)
            else:
                spatial_block = amplitude * xp.ones_like(X)

            return self._expand_spatial_block(spatial_block, shape, xp)

        return self._materialize_field_array(distributed_block, swap_threshold_gb=swap_threshold)
    
    def generate_plane_wave_source(self) -> 'FieldArray':
        """
        Generate plane wave source.
        
        Physical Meaning:
            Creates a plane wave source with specified wave vector (mode)
            and amplitude, representing a monochromatic excitation.
            For 7D domains, generates 7D field directly matching domain shape.
            
        Mathematical Foundation:
            Plane wave has the form:
            s(x) = A * exp(i * k · x)
            where k is the wave vector and A is the amplitude.
            For 7D: s(x,φ,t) = A * exp(i * k · x) (constant across phase/time).
            
        Returns:
            FieldArray: Plane wave source field with shape matching domain.shape.
        """
        self._ensure_cuda()

        amplitude = self.config.get("plane_wave_amplitude", 1.0)
        mode = self.config.get("plane_wave_mode", [1, 0, 0])  # Default mode (1,0,0)
        
        # Convert mode to tuple if needed
        if isinstance(mode, (list, np.ndarray)):
            mode = tuple(int(m) for m in mode)
        
        grid_size = getattr(self.domain, "N", self.domain.shape[0])

        swap_threshold = self.config.get("swap_threshold_gb")

        def plane_wave_block(domain, slice_config, runtime_config):
            start = tuple(slice_config["start"])
            shape = tuple(slice_config["shape"])
            xp = cp
            indices = self._create_spatial_axes(start, shape, xp, normalized=False)
            grid = xp.meshgrid(*indices, indexing="ij")
            phase = xp.zeros(shape[:3], dtype=xp.float64)
            for m_i, g_i in zip(mode, grid):
                phase += (2.0 * xp.pi * m_i * g_i) / max(1, grid_size)
            spatial_block = amplitude * xp.exp(1j * phase)
            return self._expand_spatial_block(spatial_block, shape, xp)

        return self._materialize_field_array(plane_wave_block, swap_threshold_gb=swap_threshold)

    # ------------------------------------------------------------------ #
    # Helper methods for block-aware generation                          #
    # ------------------------------------------------------------------ #

    def _ensure_cuda(self) -> None:
        if not self.use_cuda or not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA is required for source generation. CPU fallback is not supported."
            )

    def _materialize_field_array(
        self,
        block_fn: Callable,
        dtype: np.dtype = np.complex128,
        swap_threshold_gb: Optional[float] = None,
    ) -> FieldArray:
        block_size = self._get_block_size(dtype)
        generator = BlockedFieldGenerator(
            domain=self.domain,
            field_generator=block_fn,
            block_size=tuple(block_size),
            config=self.config,
            use_cuda=True,
        )
        return FieldArray.from_block_generator(
            block_generator=generator,
            dtype=dtype,
            swap_threshold_gb=swap_threshold_gb,
        )

    def _get_block_size(self, dtype: np.dtype) -> Tuple[int, ...]:
        if self._block_size_calculator is None:
            self._block_size_calculator = OptimalBlockSizeCalculator(gpu_memory_ratio=0.8)
        return self._block_size_calculator.calculate_for_7d(
            domain_shape=tuple(self.domain.shape),
            dtype=dtype,
        )

    def _create_spatial_axes(
        self,
        start: Tuple[int, ...],
        shape: Tuple[int, ...],
        xp,
        *,
        normalized: bool,
    ):
        axes = []
        grid_size = getattr(self.domain, "N", shape[0])
        denom = max(1, grid_size - 1)
        for axis in range(min(3, len(shape))):
            axis_start = start[axis]
            axis_len = shape[axis]
            values = xp.arange(axis_start, axis_start + axis_len, dtype=xp.float64)
            if normalized:
                values = values / denom
            axes.append(values)
        return axes

    def _expand_spatial_block(self, spatial_block, block_shape, xp):
        spatial_shape = block_shape[: spatial_block.ndim]
        target_shape = block_shape
        if len(target_shape) <= spatial_block.ndim:
            return spatial_block.astype(xp.complex128)

        reshape_dims = spatial_shape + (1,) * (len(target_shape) - spatial_block.ndim)
        expanded = spatial_block.reshape(reshape_dims)
        broadcast_shape = (1, 1, 1) + tuple(target_shape[3:])
        ones = xp.ones(broadcast_shape, dtype=spatial_block.dtype)
        result = (expanded * ones).astype(xp.complex128)
        return result

