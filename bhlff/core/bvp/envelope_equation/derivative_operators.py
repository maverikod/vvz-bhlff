"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

7D derivative operators for BVP envelope equation.

This module provides the main DerivativeOperators7D class that coordinates
all derivative operations needed for the 7D BVP envelope equation.

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

from .derivative_operators_facade import DerivativeOperators7D
