"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced power law analysis facade for BVP framework.

This module provides a unified interface for power law analysis,
delegating to specialized modules for different aspects of analysis.
"""

from .power_law import PowerLawCore

# Alias for backward compatibility
PowerLawAnalysis = PowerLawCore

__all__ = [
    'PowerLawCore',
    'PowerLawAnalysis'
]