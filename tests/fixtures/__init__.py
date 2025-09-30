"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test fixtures for BHLFF.

This module provides test fixtures for domains, fields, and parameters
used in testing the 7D phase field theory implementation.
"""

from .domains import test_domain, test_domain_7d
from .fields import test_field, test_source_field
from .parameters import test_physics_params, test_solver_params

__all__ = [
    "test_domain",
    "test_domain_7d", 
    "test_field",
    "test_source_field",
    "test_physics_params",
    "test_solver_params",
]
