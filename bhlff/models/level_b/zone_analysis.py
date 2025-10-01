"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Zone analysis module for Level B.

This module implements zone analysis operations for Level B
of the 7D phase field theory, focusing on zone identification and classification.

Physical Meaning:
    Analyzes zone separation in the BVP field including core, transition,
    and tail regions, providing spatial analysis of field structure.

Mathematical Foundation:
    Implements zone analysis including:
    - Zone boundary identification
    - Zone classification based on field properties
    - Zone property analysis
    - Transition region identification

Example:
    >>> analyzer = ZoneAnalysis(bvp_core)
    >>> zones = analyzer.identify_zone_boundaries(envelope)
"""

from .zone_analysis.zone_analysis import ZoneAnalysis
