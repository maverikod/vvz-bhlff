"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Utilities package for BHLFF 7D phase field theory.

This package provides utility modules for configuration management,
I/O operations, mathematical utilities, visualization, analysis,
and reporting for the 7D phase field theory implementation.
"""

from .config import ConfigLoader, ConfigValidator, ConfigDefaults
from .io import HDF5Handler, NumPyHandler, JSONHandler
from .math import Interpolator, Integrator, Statistics
from .visualization import Plotter, Animator, Exporter3D
from .analysis import StatisticsAnalyzer, TheoryComparator, QualityMetrics
from .reporting import ReportGenerator, TemplateManager, ReportExporter

__all__ = [
    "ConfigLoader",
    "ConfigValidator",
    "ConfigDefaults",
    "HDF5Handler",
    "NumPyHandler",
    "JSONHandler",
    "Interpolator",
    "Integrator",
    "Statistics",
    "Plotter",
    "Animator",
    "Exporter3D",
    "StatisticsAnalyzer",
    "TheoryComparator",
    "QualityMetrics",
    "ReportGenerator",
    "TemplateManager",
    "ReportExporter",
]
