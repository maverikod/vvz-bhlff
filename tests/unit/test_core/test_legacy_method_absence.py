"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests to ensure legacy 'basic' methods are not used in production paths.
"""

from bhlff.core.fft.bvp_basic.bvp_basic_core import BVPCoreSolver


def test_no_basic_method_name_remaining():
    methods = [m for m in dir(BVPCoreSolver) if not m.startswith("_")]
    # Allow only the renamed legacy compatibility method
    disallowed = [m for m in methods if "solve_envelope_basic" in m]
    assert len(disallowed) == 0
