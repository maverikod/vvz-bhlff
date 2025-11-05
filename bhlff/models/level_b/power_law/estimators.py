"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Block-aware estimators for critical exponents (nu, beta, gamma).

This module implements robust estimators that work on 7D blocks, avoid
biased global flattening, and leverage CUDA where available.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .block_utils import iter_blocks
from .robust_fit import robust_loglog_slope


def estimate_nu_from_correlation_length(bvp_core: Any, amplitude: np.ndarray) -> float:
    """
    Estimate ν via block-aware scaling of correlation length: ξ ~ |t|^{-ν}.

    Control parameter t is taken as the deviation of the block mean amplitude
    from the global mean. Correlation length is computed per block.
    """
    from .correlation_analysis import CorrelationAnalysis

    A_c = float(np.mean(amplitude))
    corr = CorrelationAnalysis(bvp_core)
    t_vals = []
    xi_vals = []
    total_elems = amplitude.size
    min_block_elems = max(16, min(262144, int(0.002 * total_elems)))

    for block in iter_blocks(amplitude):
        block_arr = amplitude[block]
        if block_arr.size < min_block_elems:
            continue
        A_b = float(np.mean(block_arr))
        t = abs(A_b - A_c)
        if t <= 0:
            continue
        try:
            c7 = corr._compute_7d_correlation_function(block_arr)
            lens = corr._compute_7d_correlation_lengths(c7)
            if not lens:
                continue
            xi = float(np.mean(list(lens.values())))
            if xi > 0 and np.isfinite(xi):
                t_vals.append(t)
                xi_vals.append(xi)
        except Exception:
            continue
    if len(t_vals) < 3:
        raise ValueError("insufficient block data for ν estimate")
    slope = robust_loglog_slope(np.asarray(t_vals), np.asarray(xi_vals))
    nu = -slope
    return float(max(0.1, min(2.0, nu)))


def estimate_beta_from_tail(amplitude: np.ndarray) -> float:
    """
    Estimate β from the tail CCDF across blocks without blind flattening.

    Uses per-block CCDF p(>A) and aggregates on a common amplitude grid by
    averaging CCDFs across blocks. For a power-law tail, CCDF ∼ A^{-β}.
    """
    tail_grid = []
    per_block_ccdf = []
    total_elems = amplitude.size
    min_block_elems = max(64, min(524288, int(0.004 * total_elems)))

    for block in iter_blocks(amplitude):
        block_arr = amplitude[block]
        flat = block_arr.reshape(-1)
        flat = flat[np.isfinite(flat) & (flat > 0)]
        if flat.size < min_block_elems:
            continue
        hi_adj = max(99.0, 100.0 - 100.0 / max(3.0, np.sqrt(flat.size)))
        q_lo = np.percentile(flat, 80.0)
        q_hi = np.percentile(flat, hi_adj)
        if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
            continue
        n_grid = int(np.clip(np.sqrt(flat.size), 8, 64))
        grid = np.linspace(q_lo, q_hi, n_grid)
        tail_grid.append(grid)
    if not tail_grid:
        raise ValueError("insufficient block data for β estimate")
    grid = np.unique(np.round(np.median(np.vstack(tail_grid), axis=0), 12))
    if grid.size < 6:
        raise ValueError("insufficient tail grid for β estimate")

    for block in iter_blocks(amplitude):
        block_arr = amplitude[block]
        flat = block_arr.reshape(-1)
        flat = flat[np.isfinite(flat) & (flat > 0)]
        if flat.size < min_block_elems:
            continue
        try:
            from bhlff.utils.cuda_utils import get_global_backend, CUDABackend

            backend = get_global_backend()
            if isinstance(backend, CUDABackend):
                import cupy as cp

                v = cp.asarray(flat)
                g = cp.asarray(grid)
                ccdf_gpu = (v[None, :] > g[:, None]).mean(axis=1)
                ccdf = cp.asnumpy(ccdf_gpu)
            else:
                v = flat[None, :]
                g = grid[:, None]
                ccdf = (v > g).mean(axis=1)
        except Exception:
            v = flat[None, :]
            g = grid[:, None]
            ccdf = (v > g).mean(axis=1)
        mask = (ccdf > 0) & np.isfinite(ccdf)
        if np.any(mask):
            per_block_ccdf.append(ccdf)
    if len(per_block_ccdf) < 2:
        raise ValueError("insufficient CCDF blocks for β estimate")
    mean_ccdf = np.mean(np.vstack(per_block_ccdf), axis=0)
    slope = robust_loglog_slope(grid, mean_ccdf)
    beta = -slope
    return float(max(0.1, min(2.0, beta)))


def estimate_chi_from_variance(amplitude: np.ndarray) -> float:
    """
    Estimate γ (susceptibility exponent) from block variance scaling.

    For each block, compute χ_block = Var(A_block)/Mean(A_block) and control
    parameter t_block = |Mean(A_block) - A_c|, then fit χ ∼ t^{-γ} robustly.
    """
    A_c = float(np.mean(amplitude))
    t_vals = []
    chi_vals = []
    total_elems = amplitude.size
    min_block_elems = max(32, min(262144, int(0.0025 * total_elems)))

    for block in iter_blocks(amplitude):
        block_arr = amplitude[block]
        if block_arr.size < min_block_elems:
            continue
        m = float(np.mean(block_arr))
        v = float(np.var(block_arr))
        if m <= 0 or not np.isfinite(m) or v <= 0 or not np.isfinite(v):
            continue
        t = abs(m - A_c)
        if t <= 0:
            continue
        chi = v / m
        if chi > 0 and np.isfinite(chi):
            t_vals.append(t)
            chi_vals.append(chi)
    if len(t_vals) < 3:
        raise ValueError("insufficient block data for γ estimate")
    slope = robust_loglog_slope(np.asarray(t_vals), np.asarray(chi_vals))
    gamma = -slope
    return float(max(0.5, min(2.0, gamma)))
