"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Command-line interface for BHLFF.

This module provides CLI tools for running experiments, analyzing
results, and generating reports for the 7D phase field theory.
"""

from .main import main
from .run import run_experiment
from .analyze import analyze_results
from .report import generate_report

__all__ = ["main", "run_experiment", "analyze_results", "generate_report"]
