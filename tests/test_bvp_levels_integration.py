"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Framework integration tests for all levels A-G.

This module implements comprehensive tests for BVP framework integration
across all levels A-G, ensuring that all levels use BVP envelope equation,
integrate with BVP quench detection, utilize BVP impedance calculation,
implement U(1)³ phase vector, and replace classical patterns with BVP modulations.

Physical Meaning:
    Tests validate that the BVP framework serves as the central backbone
    for all levels A-G, replacing classical patterns with BVP-modulational
    approach throughout the entire system.

Mathematical Foundation:
    Tests validate BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    integration across all levels with consistent quench detection,
    impedance calculation, and U(1)³ phase structure.

Example:
    >>> pytest tests/test_bvp_levels_integration.py -v
"""

# Import all level-specific test modules
from .test_bvp_level_a_integration import TestBVPLevelAIntegration
from .test_bvp_level_b_integration import TestBVPLevelBIntegration
from .test_bvp_level_c_integration import TestBVPLevelCIntegration
from .test_bvp_level_d_integration import TestBVPLevelDIntegration
from .test_bvp_level_e_integration import TestBVPLevelEIntegration
from .test_bvp_level_f_integration import TestBVPLevelFIntegration
from .test_bvp_level_g_integration import TestBVPLevelGIntegration