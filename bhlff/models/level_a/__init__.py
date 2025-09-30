"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level A: Basic solvers and validation.

This module implements Level A functionality for basic solver validation,
scaling analysis, and benchmarking in the 7D phase field theory.
"""

from .validation import SolverValidator
from .scaling import ScalingAnalyzer
from .benchmarks import BenchmarkRunner

__all__ = ["SolverValidator", "ScalingAnalyzer", "BenchmarkRunner"]
