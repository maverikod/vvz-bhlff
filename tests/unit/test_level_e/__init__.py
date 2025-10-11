"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for Level E experiments in 7D phase field theory.

This package contains comprehensive physical tests for Level E models,
including soliton models, defect models, and phase transitions.

Theoretical Background:
    Tests verify the physical correctness of Level E implementations
    against known theoretical results and physical constraints.

Example:
    >>> pytest tests/unit/test_level_e/ -v
"""

from .test_soliton_physics_basic import TestSolitonPhysicsBasic
from .test_soliton_energy_physics_basic import TestSolitonEnergyPhysicsBasic
from .test_soliton_topology_physics_basic import TestSolitonTopologyPhysicsBasic

__all__ = [
    "TestSolitonPhysicsBasic",
    "TestSolitonEnergyPhysicsBasic",
    "TestSolitonTopologyPhysicsBasic",
]
