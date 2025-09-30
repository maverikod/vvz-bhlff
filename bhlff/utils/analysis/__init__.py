"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Analysis utilities for BHLFF.

This module provides analysis tools for statistical analysis,
theory comparison, and quality metrics for phase field data.
"""

from .statistics import StatisticsAnalyzer
from .comparison import TheoryComparator
from .quality import QualityMetrics

__all__ = ["StatisticsAnalyzer", "TheoryComparator", "QualityMetrics"]
