"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

FFT butterfly operations computation for optimized FFT operations.

This module provides butterfly operations computation for efficient
spectral operations in the 7D phase field theory.

Physical Meaning:
    Butterfly operations are the fundamental building blocks of FFT
    algorithms, implementing the divide-and-conquer approach.

Mathematical Foundation:
    Butterfly operations implement the Cooley-Tukey FFT algorithm
    using divide-and-conquer decomposition.

Example:
    >>> butterfly_computer = FFTButterflyComputer(domain)
    >>> butterfly_tables = butterfly_computer.compute_butterfly_tables_1d()
"""

import numpy as np
from typing import Dict, Any

from ..domain import Domain


class FFTButterflyComputer:
    """
    FFT butterfly operations computer for optimized FFT operations.

    Physical Meaning:
        Computes butterfly operation patterns for efficient FFT
        computation using divide-and-conquer algorithms.

    Mathematical Foundation:
        Implements butterfly operations for the Cooley-Tukey FFT
        algorithm with optimized memory access patterns.

    Attributes:
        domain (Domain): Computational domain.
    """

    def __init__(self, domain: Domain) -> None:
        """
        Initialize FFT butterfly operations computer.

        Physical Meaning:
            Sets up the butterfly operations computer for efficient
            computation of FFT operation patterns.

        Args:
            domain (Domain): Computational domain for FFT operations.
        """
        self.domain = domain

    def compute_butterfly_tables_1d(self) -> Dict[str, Any]:
        """
        Compute butterfly operation tables for 1D FFT.
        
        Physical Meaning:
            Pre-computes butterfly operation patterns for efficient
            FFT computation using divide-and-conquer algorithms.
            
        Returns:
            Dict[str, Any]: Butterfly operation tables.
        """
        N = self.domain.N
        log2N = int(np.log2(N))
        
        # Compute bit-reversal table
        bit_reverse = np.zeros(N, dtype=int)
        for i in range(N):
            bit_reverse[i] = int(format(i, f'0{log2N}b')[::-1], 2)
        
        # Compute butterfly patterns
        butterfly_patterns = []
        for stage in range(log2N):
            stage_pattern = []
            step = 2 ** stage
            for i in range(0, N, 2 * step):
                for j in range(step):
                    stage_pattern.append((i + j, i + j + step))
            butterfly_patterns.append(stage_pattern)
        
        return {
            "bit_reverse": bit_reverse,
            "butterfly_patterns": butterfly_patterns,
            "log2N": log2N,
        }

    def compute_butterfly_tables_2d(self) -> Dict[str, Any]:
        """
        Compute butterfly operation tables for 2D FFT.
        
        Physical Meaning:
            Pre-computes butterfly operation patterns for 2D FFT
            using row-column decomposition.
            
        Returns:
            Dict[str, Any]: 2D butterfly operation tables.
        """
        return {
            "row": self.compute_butterfly_tables_1d(),
            "column": self.compute_butterfly_tables_1d(),
        }

    def compute_butterfly_tables_3d(self) -> Dict[str, Any]:
        """
        Compute butterfly operation tables for 3D FFT.
        
        Physical Meaning:
            Pre-computes butterfly operation patterns for 3D FFT
            using multi-dimensional decomposition.
            
        Returns:
            Dict[str, Any]: 3D butterfly operation tables.
        """
        return {
            "x": self.compute_butterfly_tables_1d(),
            "y": self.compute_butterfly_tables_1d(),
            "z": self.compute_butterfly_tables_1d(),
        }
