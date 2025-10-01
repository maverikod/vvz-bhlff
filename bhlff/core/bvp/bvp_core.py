"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Core module - Unified interface.

This module provides the unified BVP Core interface using the facade pattern.
The implementation has been consolidated into a single, well-organized
interface that follows the 1 class = 1 file principle.

Physical Meaning:
    The BVP core serves as the central backbone of the entire system, where
    all observed particles and fields are manifestations of envelope
    modulations and beatings of the high-frequency carrier field.

Mathematical Foundation:
    BVP implements the envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.

Example:
    >>> from bhlff.core.bvp.bvp_core import BVPCore
    >>> bvp_core = BVPCore(domain, config, domain_7d)
    >>> envelope = bvp_core.solve_envelope(source)
    >>> quenches = bvp_core.detect_quenches(envelope)
"""

# Import from the consolidated facade structure
from .bvp_core import BVPCore
from .bvp_core import BVPCoreOperations, BVPCore7DInterface

# Re-export for unified interface
__all__ = ["BVPCore", "BVPCoreOperations", "BVPCore7DInterface"]
