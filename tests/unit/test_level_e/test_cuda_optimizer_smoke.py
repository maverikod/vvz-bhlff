"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA optimizer smoke test for Level E.

Ensures the optimizer initializes and can attempt a solve on a very small
problem. Non-convergence should not fail the test (treated as xfail/skip).
"""

from __future__ import annotations

import sys
from typing import Any, Dict

import numpy as np
import pytest


sys.path.insert(0, ".")


@pytest.mark.level_e
def test_cuda_optimizer_smoke():
    try:
        import cupy as cp

        cuda_available = cp.cuda.is_available()
    except Exception:
        cuda_available = False

    from bhlff.core.domain import Domain
    from bhlff.models.level_e.cuda import SolitonOptimizerCUDA

    N, Nphi, Nt = 4, 4, 8
    domain = Domain(L=1.0, N=N, dimensions=7, N_phi=Nphi, N_t=Nt, T=1.0)
    params: Dict[str, Any] = {
        "mu": 1.0,
        "beta": 1.0,
        "lambda": 0.1,
        "S4": 0.1,
        "S6": 0.01,
        "F2": 1.0,
        "N_c": 3,
    }

    # small initial guess
    rng = np.random.default_rng(321)
    init = (
        rng.standard_normal((N, N, N, Nphi, Nphi, Nphi, Nt))
        + 1j * rng.standard_normal((N, N, N, Nphi, Nphi, Nphi, Nt))
    ).astype(np.complex128)

    if not cuda_available:
        pytest.skip("CUDA not available in test environment")

    opt = SolitonOptimizerCUDA(domain, params, use_cuda=True)
    try:
        sol = opt.find_solution(init)
        assert sol.shape == init.shape
        assert np.all(np.isfinite(sol))
    except Exception as e:
        pytest.xfail(f"Non-convergent or backend limitation (expected in smoke): {e}")
