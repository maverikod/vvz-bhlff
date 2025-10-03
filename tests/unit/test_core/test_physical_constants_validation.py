"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for constants in 7D BVP theory.

This module provides physical validation tests for constants,
ensuring they satisfy physical constraints and theoretical requirements.
"""

# Import all individual test modules
from tests.unit.test_core.test_bvp_constants_physics import TestBVPConstantsPhysics
from tests.unit.test_core.test_frequency_dependent_properties_physics import TestFrequencyDependentPropertiesPhysics
from tests.unit.test_core.test_nonlinear_coefficients_physics import TestNonlinearCoefficientsPhysics
from tests.unit.test_core.test_renormalized_coefficients_physics import TestRenormalizedCoefficientsPhysics