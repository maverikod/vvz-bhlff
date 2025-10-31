"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

GPU/CPU energy consistency tests for Level E CUDA modules.

Checks that total energy computed via CUDA path matches CPU fallback
within a reasonable numerical tolerance on a small 7D field.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import pytest


sys.path.insert(0, ".")


@pytest.mark.level_e
def test_cuda_cpu_energy_consistency():
    try:
        import cupy as cp

        cuda_available = cp.cuda.is_available()
    except Exception:
        cuda_available = False

    from bhlff.core.domain import Domain
    from bhlff.models.level_e.cuda import SolitonEnergyCalculatorCUDA

    # Small domain for fast test
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

    rng = np.random.default_rng(123)
    field = (
        rng.standard_normal((N, N, N, Nphi, Nphi, Nphi, Nt))
        + 1j * rng.standard_normal((N, N, N, Nphi, Nphi, Nphi, Nt))
    ).astype(np.complex128)

    # CPU energy (fallback)
    calc_cpu = SolitonEnergyCalculatorCUDA(domain, params, use_cuda=False)
    E_cpu = float(calc_cpu.compute_total_energy(field))

    assert np.isfinite(E_cpu)

    # If CUDA available, compare energies
    if cuda_available:
        calc_gpu = SolitonEnergyCalculatorCUDA(domain, params, use_cuda=True)
        E_gpu = float(calc_gpu.compute_total_energy(field))
        assert np.isfinite(E_gpu)
        # Strict parity
        assert np.isclose(E_gpu, E_cpu, rtol=1e-6, atol=1e-9), (E_gpu, E_cpu)
    else:
        pytest.skip("CUDA not available in test environment")
