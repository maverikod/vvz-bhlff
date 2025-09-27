"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Sources package for BHLFF framework.

This package provides source term implementations for the 7D phase field
theory.

Physical Meaning:
    Sources implement various types of source terms for phase field equations,
    including BVP-modulated sources, quench sources, and harmonic sources.

Mathematical Foundation:
    Implements source terms s(x) for phase field equations including
    BVP-modulated sources and quench event sources.
"""

from .source import Source
from .bvp_source import BVPSource

__all__ = [
    "Source",
    "BVPSource",
]
