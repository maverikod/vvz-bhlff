"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Node analysis module for Level B.

This module implements node analysis operations for Level B
of the 7D phase field theory, focusing on node identification and classification.

Physical Meaning:
    Analyzes node structures in the BVP field including saddle nodes,
    source nodes, and sink nodes, providing topological analysis
    of the field structure.

Mathematical Foundation:
    Implements node analysis including:
    - Node identification using gradient analysis
    - Node classification based on local field properties
    - Topological charge computation
    - Node density analysis

Example:
    >>> analyzer = NodeAnalysis(bvp_core)
    >>> nodes = analyzer.identify_nodes(envelope)
"""

from .node_analysis.node_analysis import NodeAnalysis
