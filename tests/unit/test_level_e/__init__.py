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

# Safe, optional re-exports to avoid breaking test discovery if files/classes
# are renamed or temporarily unavailable. This prevents ImportError on package import.
try:  # pragma: no cover
    from .test_soliton_physics_basic import TestSolitonPhysicsBasic  # type: ignore
except Exception:  # pragma: no cover
    TestSolitonPhysicsBasic = None  # type: ignore

try:  # pragma: no cover
    from .test_soliton_energy_physics_basic import TestSolitonEnergyPhysicsBasic  # type: ignore
except Exception:  # pragma: no cover
    TestSolitonEnergyPhysicsBasic = None  # type: ignore

try:  # pragma: no cover
    from .test_soliton_topology_physics_basic import TestSolitonTopologyPhysicsBasic  # type: ignore
except Exception:  # pragma: no cover
    TestSolitonTopologyPhysicsBasic = None  # type: ignore

__all__ = [
    name
    for name in (
        "TestSolitonPhysicsBasic",
        "TestSolitonEnergyPhysicsBasic",
        "TestSolitonTopologyPhysicsBasic",
    )
    if globals().get(name) is not None
]
