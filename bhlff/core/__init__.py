"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core module for BHLFF package.

This module contains the fundamental components of the BHLFF framework,
including domain definitions, parameter management, BVP core, and base classes
for all computational components.

Physical Meaning:
    The core module provides the mathematical foundation for the 7D phase
    field theory implementation, including the computational domain,
    parameter validation, BVP framework, and base interfaces for all "
    "solvers and models.

Mathematical Foundation:
    Core components implement the fundamental mathematical structures
    required for solving the fractional Riesz operator and related
    equations in 7D space-time, with BVP as the central backbone.
"""

from .domain import Domain
from .parameters import Parameters
from .bvp_core import BVPCore
from .quench_detector import QuenchDetector
from .bvp_interface import BVPInterface
from .base.abstract_solver import AbstractSolver
from .base.field import Field

__all__ = [
    "Domain",
    "Parameters",
    "BVPCore",
    "QuenchDetector",
    "BVPInterface",
    "AbstractSolver",
    "Field",
]
