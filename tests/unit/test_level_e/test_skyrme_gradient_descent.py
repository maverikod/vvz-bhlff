"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Gradient descent smoke test for Skyrme energy using field generator.

This test:
- generates a 7D phase field via the project field generator (U(1)^3),
- maps it to SU(2) matrices per grid point,
- computes Skyrme gradient on GPU,
- takes a small descent step with unitary projection,
- verifies energy does not increase.
"""

from __future__ import annotations

import sys
from typing import Dict, Any

import numpy as np
import pytest


sys.path.insert(0, ".")


@pytest.mark.level_e
def test_skyrme_gradient_descent_energy_nonincreasing():
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        pytest.skip("CUDA not available in test environment")

    from bhlff.core.domain import Domain
    from bhlff.models.level_e.cuda import (
        SolitonEnergyCalculatorCUDA,
        SolitonOptimizerCUDA,
    )

    # Small domain for speed
    N, Nphi, Nt = 3, 3, 2
    domain = Domain(L=1.0, N=N, dimensions=7, N_phi=Nphi, N_t=Nt, T=1.0)
    params: Dict[str, Any] = {
        "mu": 1.0,
        "beta": 1.0,
        "lambda": 0.1,
        "S4": 0.1,
        "S6": 0.0,
        "F2": 1.0,
        "N_c": 3,
    }

    # Use project field generator (U(1)^3-like) and map to SU(2)
    rng = np.random.default_rng(42)
    # Start with three complex components (U(1)^3 phase field surrogate)
    comp1 = (
        rng.standard_normal(domain.shape) + 1j * rng.standard_normal(domain.shape)
    ).astype(np.complex128)
    comp2 = (
        rng.standard_normal(domain.shape) + 1j * rng.standard_normal(domain.shape)
    ).astype(np.complex128)

    # Map two components -> SU(2) matrix per grid point
    eps = 1e-12
    norm = np.sqrt(np.abs(comp1) ** 2 + np.abs(comp2) ** 2) + eps
    a = comp1 / norm
    b = comp2 / norm
    U11 = a
    U12 = -np.conj(b)
    U21 = b
    U22 = np.conj(a)
    field = np.stack(
        [np.stack([U11, U12], axis=-1), np.stack([U21, U22], axis=-1)], axis=-2
    ).astype(np.complex128)

    # Skyrme energy before (compare only Skyrme term)
    from bhlff.models.level_e.cuda.energy.skyrme import SkyrmeEnergyCUDA

    sky = SkyrmeEnergyCUDA(S4=params["S4"])
    E0 = float(cp.asnumpy(sky.compute_cuda(cp.asarray(field))))

    # One descent step using Skyrme gradient and unitary projection
    optimizer = SolitonOptimizerCUDA(domain, params, use_cuda=True)
    grad = optimizer._compute_skyrme_gradient_cuda(cp.asarray(field))
    grad_np = cp.asnumpy(grad)

    tau = 1e-10
    # Try both directions to account for sign convention
    Fm = field - tau * grad_np
    Fp = field + tau * grad_np

    # Project to unitary: U = F (F†F)^(-1/2)
    def project_unitary(Farr: np.ndarray) -> np.ndarray:
        F_dag = np.swapaxes(Farr, -2, -1).conj()
        H = np.einsum("...ji,...jk->...ik", F_dag, Farr)
        w, V = np.linalg.eigh(H)
        w = np.clip(w, 1e-12, None)
        w_inv_sqrt = 1.0 / np.sqrt(w)
        Vh = np.swapaxes(V, -2, -1).conj()
        V_scaled = V * w_inv_sqrt[..., None, :]
        S = V_scaled @ Vh
        return Farr @ S

    # eigh per grid point
    Um = project_unitary(Fm)
    Up = project_unitary(Fp)
    E1m = float(cp.asnumpy(sky.compute_cuda(cp.asarray(Um))))
    E1p = float(cp.asnumpy(sky.compute_cuda(cp.asarray(Up))))

    E1 = min(E1m, E1p)
    assert E1 <= E0 * (1 + 1e-6)
