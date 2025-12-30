"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Utility functions for creating proper 7D domains in tests.

Physical Meaning:
    Provides standardized functions for creating full 7D domains
    with square grids for proper testing of 7D phase field theory.
    
Mathematical Foundation:
    For proper 7D scale invariance testing, all dimensions must:
    1. Have the same resolution (square grid)
    2. Scale proportionally with L
    3. Participate in spectral operations correctly

Example:
    >>> from tests.unit.test_level_a.test_utils_7d_domain import create_7d_domain_square
    >>> domain = create_7d_domain_square(L=1.0, N=64)
    >>> assert domain.N_phi == 64  # Square grid
    >>> assert domain.N_t == 64   # Square grid
"""

import numpy as np
from bhlff.core.domain import Domain


def create_7d_domain_square(L: float, N: int, T: float = None) -> Domain:
    """
    Create proper 7D domain with square grid.
    
    Physical Meaning:
        Creates full 7D domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ with square grid
        (N_phi=N, N_t=N) for proper testing of 7D phase field theory.
        
    Mathematical Foundation:
        For proper 7D scale invariance testing, all dimensions must:
        1. Have the same resolution (square grid)
        2. Scale proportionally with L
        3. Participate in spectral operations correctly
        
    Args:
        L (float): Spatial domain size.
        N (int): Spatial resolution (will be used for all dimensions).
        T (float, optional): Temporal domain size. Defaults to 2π.
        
    Returns:
        Domain: Full 7D domain with square grid.
    """
    if T is None:
        T = 2 * np.pi  # Default period for phase coordinates
    
    return Domain(
        L=L,
        N=N,
        N_phi=N,       # ✓ Square grid
        N_t=N,         # ✓ Square grid
        T=T,
        dimensions=7,
    )


def create_7d_domain_minimal(L: float, N: int, T: float = 1.0) -> Domain:
    """
    Create minimal 7D domain for fast tests (N_phi=2, N_t=2).
    
    Physical Meaning:
        Creates minimal 7D domain for fast unit tests where
        full resolution is not critical.
        
    Args:
        L (float): Spatial domain size.
        N (int): Spatial resolution.
        T (float): Temporal domain size.
        
    Returns:
        Domain: Minimal 7D domain.
    """
    return Domain(
        L=L,
        N=N,
        N_phi=2,       # Minimal for fast tests
        N_t=2,         # Minimal for fast tests
        T=T,
        dimensions=7,
    )

