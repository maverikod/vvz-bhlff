"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Semi-transparent step resonator boundary operator for 7D phase field fields.

Physical Meaning:
    Implements partial reflection and transmission at domain boundaries to
    model semi-transparent resonator walls. No exponential attenuation is
    used; energy exchange occurs via boundary mixing with reflection (R) and
    transmission (T) coefficients.

Mathematical Foundation:
    For a given spatial axis a and boundary indices 0 and -1, apply:
        f[0]  <- R * f[0]  + T * f[1]
        f[-1] <- R * f[-1] + T * f[-2]
    where 0 <= R <= 1, 0 <= T <= 1 and typically R + T <= 1 (leaky walls).

Usage:
    apply_step_resonator(field, axes=(0,1,2), R=0.1, T=0.9)
"""

from typing import Iterable, Tuple
import numpy as np


def apply_step_resonator(
    field: np.ndarray,
    axes: Iterable[int] = (0, 1, 2),
    R: float | np.ndarray = 0.1,
    T: float | np.ndarray = 0.9,
) -> np.ndarray:
    """
    Apply semi-transparent step resonator boundary conditions in-place.

    Args:
        field: N-dimensional complex or real field (supports 7D arrays)
        axes: axes along which to apply boundary mixing (e.g., spatial axes)
        R: reflection coefficient at the wall (0..1)
        T: transmission coefficient from the interior cell (0..1)

    Returns:
        np.ndarray: The same field array with updated boundary values.
    """
    # Allow scalar or frequency-dependent arrays for R/T
    if np.isscalar(R):
        if not 0.0 <= float(R) <= 1.0:
            raise ValueError("R must be in [0,1]")
    if np.isscalar(T):
        if not 0.0 <= float(T) <= 1.0:
            raise ValueError("T must be in [0,1]")

    # Work on a view to avoid copies
    updated = field

    for axis in axes:
        if updated.shape[axis] < 2:
            continue  # cannot apply mixing on degenerate axis

        # Build index tuples for boundary and neighbor positions
        slicer_low = [slice(None)] * updated.ndim
        slicer_low_neighbor = [slice(None)] * updated.ndim
        slicer_high = [slice(None)] * updated.ndim
        slicer_high_neighbor = [slice(None)] * updated.ndim

        slicer_low[axis] = 0
        slicer_low_neighbor[axis] = 1
        slicer_high[axis] = -1
        slicer_high_neighbor[axis] = -2

        low = tuple(slicer_low)
        low_n = tuple(slicer_low_neighbor)
        high = tuple(slicer_high)
        high_n = tuple(slicer_high_neighbor)

        # Apply mixing at both boundaries
        # If R/T are arrays (e.g., frequency/axis-dependent), rely on numpy broadcasting
        updated[low] = (R * updated[low]) + (T * updated[low_n])
        updated[high] = (R * updated[high]) + (T * updated[high_n])

    return updated
