"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Robust fitting utilities for power-law scaling in 7D BVP analysis.

This module provides robust log-log regression with outlier suppression
and binning suitable for stable estimation of scaling exponents.

Physical Meaning:
    Robust exponent estimation is critical for accurate characterization
    of scaling behavior in 7D BVP fields, where heavy tails and localized
    defects introduce strong outliers and leverage points.

Example:
    >>> slope = robust_loglog_slope(x, y)
"""

from __future__ import annotations

import numpy as np


def robust_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute robust slope of log(y) vs log(x) using outlier suppression.

    Physical Meaning:
        Estimates scaling exponents from noisy, heavy-tailed data while
        suppressing outliers typical for BVP fields near criticality.

    Mathematical Foundation:
        - Filter to positive finite pairs
        - Log-transform
        - IQR-based trimming on both axes
        - Quantile-binning by x with median aggregation
        - OLS on binned medians combined with median adjacent-bin slopes

    Args:
        x (np.ndarray): Control parameter values (positive).
        y (np.ndarray): Measured response values (positive).

    Returns:
        float: Robust slope d log(y) / d log(x).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        raise ValueError("insufficient data for robust log-log fit")
    lx = np.log(x)
    ly = np.log(y)

    def _trim(v: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return (v >= lo) & (v <= hi)

    m = _trim(lx) & _trim(ly)
    lx = lx[m]
    ly = ly[m]
    if lx.size < 3:
        raise ValueError("insufficient trimmed data for robust log-log fit")

    nbins = min(12, max(4, lx.size // 4))
    q = np.linspace(0, 1, nbins + 1)
    bins = np.quantile(lx, q)
    bins = np.unique(bins)
    if bins.size < 3:
        return float(np.polyfit(lx, ly, 1)[0])

    binned_x = []
    binned_y = []
    for i in range(bins.size - 1):
        sel = (lx >= bins[i]) & (lx <= bins[i + 1])
        if np.any(sel):
            binned_x.append(np.median(lx[sel]))
            binned_y.append(np.median(ly[sel]))
    bx = np.asarray(binned_x)
    by = np.asarray(binned_y)
    if bx.size < 3:
        return float(np.polyfit(lx, ly, 1)[0])

    slope = float(np.polyfit(bx, by, 1)[0])
    adj_slopes = []
    for i in range(bx.size - 1):
        dx = bx[i + 1] - bx[i]
        if dx != 0:
            adj_slopes.append((by[i + 1] - by[i]) / dx)
    if len(adj_slopes) >= 2:
        slope = 0.5 * (slope + float(np.median(adj_slopes)))
    return slope
