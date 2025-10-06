"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple tests for maximum code coverage - Facade.

This module provides a facade for simple tests that focus on covering as much code
as possible without complex logic that might fail.
"""

# Import all coverage test modules
from test_domain_coverage import TestDomainCoverage
from test_bvp_constants_coverage import TestBVPConstantsCoverage
from test_bvp_postulate_coverage import TestBVPPostulateCoverage
from test_fft_coverage import TestFFTCoverage
from test_operators_coverage import TestOperatorsCoverage
from test_sources_coverage import TestSourcesCoverage
from test_solvers_coverage import TestSolversCoverage

# Export all test classes
__all__ = [
    "TestDomainCoverage",
    "TestBVPConstantsCoverage",
    "TestBVPPostulateCoverage",
    "TestFFTCoverage",
    "TestOperatorsCoverage",
    "TestSourcesCoverage",
    "TestSolversCoverage",
]
