"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP source generators implementation.

This module provides various source generators for BVP-modulated sources
in the 7D phase field theory.

Physical Meaning:
    BVP source generators create different types of base sources that
    can be modulated by the BVP framework, including Gaussian, point,
    and distributed sources.

Mathematical Foundation:
    Implements various source generation methods:
    - Gaussian sources: s(x) = A * exp(-|x-x₀|²/σ²)
    - Point sources: s(x) = A * δ(x-x₀)
    - Distributed sources: s(x) = A * f(x)

Example:
    >>> generators = BVPSourceGenerators(domain, config)
    >>> gaussian_source = generators.generate_gaussian_source()
"""

import numpy as np
from typing import Dict, Any

from ..domain import Domain

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False


class BVPSourceGenerators:
    """
    BVP source generators for various source types.

    Physical Meaning:
        Generates different types of base sources that can be modulated
        by the BVP framework for phase field evolution.

    Mathematical Foundation:
        Implements various source generation methods for creating
        base source terms in the BVP-modulated source equation.

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Source generator configuration.
    """

    def __init__(self, domain: Domain, config: Dict[str, Any]) -> None:
        """
        Initialize BVP source generators.

        Physical Meaning:
            Sets up the BVP source generators with domain and configuration
            for generating various types of base sources.

        Args:
            domain (Domain): Computational domain for source generation.
            config (Dict[str, Any]): Source generator configuration.
        """
        self.domain = domain
        self.config = config
        self.use_cuda = bool(config.get("use_cuda", False)) and CUDA_AVAILABLE

    def generate_gaussian_source(self) -> np.ndarray:
        """
        Generate Gaussian source.

        Physical Meaning:
            Creates a Gaussian source distribution centered at a specified
            location with given width and amplitude.

        Mathematical Foundation:
            Gaussian source: s(x) = A * exp(-|x-x₀|²/σ²)
            where A is amplitude, x₀ is center, and σ is width.

        Returns:
            np.ndarray: Gaussian source field.
        """
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

        return source

    def generate_point_source(self) -> np.ndarray:
        """
        Generate point source.

        Physical Meaning:
            Creates a point source at a specified location with given
            amplitude, representing a localized excitation.

        Mathematical Foundation:
            Point source: s(x) = A * δ(x-x₀)
            where A is amplitude and x₀ is the source location.

        Returns:
            np.ndarray: Point source field.
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

        # Create point source
        source = np.zeros((self.domain.N, self.domain.N, self.domain.N))
        source[i, j, k] = amplitude

        return source

    def generate_distributed_source(self) -> np.ndarray:
        """
        Generate distributed source.

        Physical Meaning:
            Creates a distributed source with specified spatial distribution
            and amplitude profile.

        Mathematical Foundation:
            Distributed source: s(x) = A * f(x)
            where A is amplitude and f(x) is the spatial distribution function.

        Returns:
            np.ndarray: Distributed source field.
        """
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

        return source

    def generate_base_source(self, source_type: str) -> np.ndarray:
        """
        Generate base source of specified type.

        Physical Meaning:
            Generates a base source of the specified type for BVP modulation.

        Args:
            source_type (str): Type of source to generate.

        Returns:
            np.ndarray: Base source field.

        Raises:
            ValueError: If source type is not supported.
        """
        if source_type == "gaussian":
            return self.generate_gaussian_source()
        elif source_type == "point":
            return self.generate_point_source()
        elif source_type == "distributed":
            return self.generate_distributed_source()
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def get_supported_source_types(self) -> list:
        """
        Get supported source types.

        Physical Meaning:
            Returns the list of supported source types for BVP modulation.

        Returns:
            list: Supported source types.
        """
        return ["gaussian", "point", "distributed"]

    def get_source_info(self, source_type: str) -> Dict[str, Any]:
        """
        Get information about source type.

        Physical Meaning:
            Returns information about the specified source type including
            parameters and mathematical description.

        Args:
            source_type (str): Source type to get information about.

        Returns:
            Dict[str, Any]: Source type information.
        """
        source_info = {
            "gaussian": {
                "description": "Gaussian source distribution",
                "formula": "s(x) = A * exp(-|x-x₀|²/σ²)",
                "parameters": [
                    "gaussian_amplitude",
                    "gaussian_center",
                    "gaussian_width",
                ],
            },
            "point": {
                "description": "Point source at specified location",
                "formula": "s(x) = A * δ(x-x₀)",
                "parameters": ["point_amplitude", "point_location"],
            },
            "distributed": {
                "description": "Distributed source with spatial distribution",
                "formula": "s(x) = A * f(x)",
                "parameters": ["distributed_amplitude", "distribution_type"],
            },
        }

        return source_info.get(source_type, {})

    def generate_topological_substrate(
        self, defect_config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate topological substrate with defects and resonator walls.

        Physical Meaning:
            Creates the fundamental 7D BVP substrate based on topological defects
            that form semi-transparent resonator walls. This is the primary structure
            that determines field behavior, not derived from wave patterns.

        Mathematical Foundation:
            The substrate S(x,φ,t) represents permeability/loss/phase-shift field
            in 7D space-time with discrete layers and geometric decay of transparency.
            Defects create walls with thickness ~1-2 voxels and quantized radii R_n.

        Args:
            defect_config (Dict[str, Any]): Configuration for defect generation:
                - defect_type: "line", "surface", "junction", "dislocation"
                - defect_density: density of defects per unit volume
                - core_radius: radius of defect core
                - wall_thickness: thickness of resonator walls
                - transparency: base transparency of walls (0-1)
                - regularization: smoothing parameter for walls

        Returns:
            np.ndarray: 7D substrate field S(x,φ,t) with shape (N, N, N, N_phi, N_phi, N_phi, N_t)
        """
        # Get defect parameters
        defect_type = defect_config.get("defect_type", "line")
        defect_density = defect_config.get("defect_density", 0.1)
        core_radius = defect_config.get("core_radius", 0.05)
        wall_thickness = defect_config.get("wall_thickness", 0.02)
        transparency = defect_config.get("transparency", 0.3)
        regularization = defect_config.get("regularization", 0.01)

        # Create 7D coordinate arrays
        shape = (
            self.domain.N,
            self.domain.N,
            self.domain.N,
            self.domain.N_phi,
            self.domain.N_phi,
            self.domain.N_phi,
            self.domain.N_t,
        )

        xp = cp if self.use_cuda else np
        # Initialize substrate with base transparency
        substrate = xp.full(shape, transparency, dtype=xp.float64)

        # Generate defects based on type
        if defect_type == "line":
            substrate = self._add_line_defects(
                substrate, defect_density, core_radius, wall_thickness
            )
        elif defect_type == "surface":
            substrate = self._add_surface_defects(
                substrate, defect_density, core_radius, wall_thickness
            )
        elif defect_type == "junction":
            substrate = self._add_junction_defects(
                substrate, defect_density, core_radius, wall_thickness
            )
        elif defect_type == "dislocation":
            substrate = self._add_dislocation_defects(
                substrate, defect_density, core_radius, wall_thickness
            )

        # Apply regularization to smooth walls
        if regularization > 0:
            # Regularize on CPU for now to avoid complex GPU pipeline
            substrate_np = cp.asnumpy(substrate) if self.use_cuda else substrate
            substrate_np = self._regularize_walls(substrate_np, regularization)
            substrate = cp.asarray(substrate_np) if self.use_cuda else substrate_np

        if self.use_cuda:
            return cp.asnumpy(substrate)
        return substrate

    def compose_multiscale_substrate(
        self, base_substrate: np.ndarray, layer_config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compose multiscale substrate with discrete layers and geometric decay.

        Physical Meaning:
            Creates discrete layers with quantized radii R_n and geometric decay
            of transparency q between layers. This implements the stepwise structure
            of 7D BVP theory.

        Mathematical Foundation:
            Layers have radii R_n = πn/k with transparency T_n = T_0 * q^n
            where q < 1 is the geometric decay factor.

        Args:
            base_substrate (np.ndarray): Base substrate from defect generation
            layer_config (Dict[str, Any]): Layer configuration:
                - num_layers: number of discrete layers
                - base_radius: base radius R_0
                - wave_number: wave number k for R_n = πn/k
                - decay_factor: geometric decay factor q
                - center: center coordinates for radial layers

        Returns:
            np.ndarray: Multiscale substrate with discrete layers
        """
        num_layers = layer_config.get("num_layers", 4)
        base_radius = layer_config.get("base_radius", 0.1)
        wave_number = layer_config.get("wave_number", 2.0)
        decay_factor = layer_config.get("decay_factor", 0.7)
        center = layer_config.get("center", [0.5, 0.5, 0.5])

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
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Create discrete layers with geometric decay
        multiscale_substrate = (
            cp.asarray(base_substrate)
            if (self.use_cuda and not isinstance(base_substrate, cp.ndarray))
            else base_substrate
        ).copy()

        for n in range(1, num_layers + 1):
            # Quantized radius
            R_n = base_radius + (np.pi * n) / wave_number

            # Geometric decay of transparency
            T_n = decay_factor**n

            # Create layer wall (semi-transparent barrier)
            wall_mask = self._create_layer_wall(
                r, R_n, 0.02, xp=xp
            )  # 0.02 wall thickness
            # Expand wall mask (3D) to 7D for broadcasting
            wall_mask_7d = wall_mask[
                ..., xp.newaxis, xp.newaxis, xp.newaxis, xp.newaxis
            ]
            multiscale_substrate = xp.where(wall_mask_7d, T_n, multiscale_substrate)

        if self.use_cuda:
            return cp.asnumpy(multiscale_substrate)
        return multiscale_substrate

    def _add_line_defects(
        self,
        substrate: np.ndarray,
        density: float,
        core_radius: float,
        wall_thickness: float,
    ) -> np.ndarray:
        """Add line defects (strings) to substrate."""
        # Simple implementation: add vertical line defects
        num_defects = int(density * self.domain.N**2)

        for _ in range(num_defects):
            # Random line position
            i = np.random.randint(0, self.domain.N)
            j = np.random.randint(0, self.domain.N)

            # Create line defect (low transparency)
            substrate[i, j, :, :, :, :, :] *= 0.1

        return substrate

    def _add_surface_defects(
        self,
        substrate: np.ndarray,
        density: float,
        core_radius: float,
        wall_thickness: float,
    ) -> np.ndarray:
        """Add surface defects (domain walls) to substrate."""
        # Simple implementation: add planar defects
        num_defects = int(density * self.domain.N)

        for _ in range(num_defects):
            # Random plane position
            i = np.random.randint(0, self.domain.N)

            # Create surface defect
            substrate[i, :, :, :, :, :, :] *= 0.2

        return substrate

    def _add_junction_defects(
        self,
        substrate: np.ndarray,
        density: float,
        core_radius: float,
        wall_thickness: float,
    ) -> np.ndarray:
        """Add junction defects to substrate."""
        # Simple implementation: add point-like junction defects
        num_defects = int(density * self.domain.N**3)

        for _ in range(num_defects):
            # Random junction position
            i = np.random.randint(0, self.domain.N)
            j = np.random.randint(0, self.domain.N)
            k = np.random.randint(0, self.domain.N)

            # Create junction defect (very low transparency)
            substrate[i, j, k, :, :, :, :] *= 0.05

        return substrate

    def _add_dislocation_defects(
        self,
        substrate: np.ndarray,
        density: float,
        core_radius: float,
        wall_thickness: float,
    ) -> np.ndarray:
        """Add dislocation defects in phase space to substrate."""
        # Simple implementation: add phase-space dislocations
        num_defects = int(density * self.domain.N_phi**3)

        for _ in range(num_defects):
            # Random dislocation position in phase space
            phi1 = np.random.randint(0, self.domain.N_phi)
            phi2 = np.random.randint(0, self.domain.N_phi)
            phi3 = np.random.randint(0, self.domain.N_phi)

            # Create dislocation defect
            substrate[:, :, :, phi1, phi2, phi3, :] *= 0.15

        return substrate

    def _create_layer_wall(
        self, r: np.ndarray, radius: float, thickness: float, xp=np
    ) -> np.ndarray:
        """Create a layer wall at specified radius with given thickness."""
        # Create smooth wall using sigmoid function
        wall_center = radius
        wall_width = thickness

        # Sigmoid function for smooth wall
        wall_mask = 1.0 / (1.0 + xp.exp(-(r - wall_center) / wall_width))

        # Threshold to get binary wall
        return wall_mask > 0.5

    def _regularize_walls(
        self, substrate: np.ndarray, regularization: float
    ) -> np.ndarray:
        """Apply regularization to smooth wall boundaries."""
        # Simple Gaussian smoothing
        from scipy import ndimage

        # Apply 3D Gaussian filter to spatial dimensions only
        smoothed = substrate.copy()
        for t in range(substrate.shape[6]):
            for phi3 in range(substrate.shape[5]):
                for phi2 in range(substrate.shape[4]):
                    for phi1 in range(substrate.shape[3]):
                        smoothed[:, :, :, phi1, phi2, phi3, t] = (
                            ndimage.gaussian_filter(
                                substrate[:, :, :, phi1, phi2, phi3, t],
                                sigma=regularization,
                            )
                        )

        return smoothed

    def _step_resonator_source(
        self, r_squared: np.ndarray, width: float, xp=np
    ) -> np.ndarray:
        """
        Step resonator source according to 7D BVP theory.

        Physical Meaning:
            Implements step function source instead of exponential decay
            according to 7D BVP theory principles.
        """
        cutoff_radius_squared = (width * 2.0) ** 2  # 2-sigma cutoff
        return xp.where(r_squared < cutoff_radius_squared, 1.0, 0.0)
