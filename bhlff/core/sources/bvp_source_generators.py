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
        
        # Create coordinate arrays
        x = np.linspace(0, 1, self.domain.N)
        y = np.linspace(0, 1, self.domain.N)
        z = np.linspace(0, 1, self.domain.N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Compute distances from center
        dx = X - center[0]
        dy = Y - center[1]
        dz = Z - center[2]
        r_squared = dx**2 + dy**2 + dz**2
        
        # Generate Gaussian source
        source = amplitude * np.exp(-r_squared / (2 * width**2))
        
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
        
        # Create coordinate arrays
        x = np.linspace(0, 1, self.domain.N)
        y = np.linspace(0, 1, self.domain.N)
        z = np.linspace(0, 1, self.domain.N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Generate distributed source based on type
        if distribution_type == "sine":
            # Sine wave distribution
            kx = self.config.get("sine_kx", 2 * np.pi)
            ky = self.config.get("sine_ky", 2 * np.pi)
            kz = self.config.get("sine_kz", 2 * np.pi)
            
            source = amplitude * (np.sin(kx * X) * np.sin(ky * Y) * np.sin(kz * Z))
            
        elif distribution_type == "cosine":
            # Cosine wave distribution
            kx = self.config.get("cosine_kx", 2 * np.pi)
            ky = self.config.get("cosine_ky", 2 * np.pi)
            kz = self.config.get("cosine_kz", 2 * np.pi)
            
            source = amplitude * (np.cos(kx * X) * np.cos(ky * Y) * np.cos(kz * Z))
            
        elif distribution_type == "polynomial":
            # Polynomial distribution
            order = self.config.get("polynomial_order", 2)
            
            source = amplitude * (X**order + Y**order + Z**order)
            
        else:
            # Default to constant distribution
            source = amplitude * np.ones_like(X)
        
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
                "parameters": ["gaussian_amplitude", "gaussian_center", "gaussian_width"]
            },
            "point": {
                "description": "Point source at specified location",
                "formula": "s(x) = A * δ(x-x₀)",
                "parameters": ["point_amplitude", "point_location"]
            },
            "distributed": {
                "description": "Distributed source with spatial distribution",
                "formula": "s(x) = A * f(x)",
                "parameters": ["distributed_amplitude", "distribution_type"]
            }
        }
        
        return source_info.get(source_type, {})
