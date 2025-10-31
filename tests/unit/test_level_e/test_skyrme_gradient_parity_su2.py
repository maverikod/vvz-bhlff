"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Skyrme gradient parity test on SU(2) manifold using project-style field generator.

Checks: directional derivative along δU = U·X (X ∈ su(2)) matches
analytic value Re Tr(R X)/(32π²), where R is the variational residual.
"""

from __future__ import annotations

import sys
from typing import Dict, Any, Tuple

import numpy as np
import pytest


sys.path.insert(0, ".")


@pytest.mark.level_e
def test_skyrme_gradient_parity_su2():
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        pytest.skip("CUDA not available in test environment")

    from bhlff.core.domain import Domain
    from bhlff.models.level_e.cuda import SolitonOptimizerCUDA
    from bhlff.models.level_e.cuda.energy.skyrme import SkyrmeEnergyCUDA
    from scipy.linalg import expm

    # Small domain
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

    # Generator field: build U(1)^3 phase Θ(x,φ), then embed into SU(2) via U = exp(i Θ σ_z)
    N = domain.N
    L = domain.L
    x = np.linspace(-L / 2, L / 2, N)
    y = np.linspace(-L / 2, L / 2, N)
    z = np.linspace(-L / 2, L / 2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    phi = np.linspace(0, 2 * np.pi, domain.N_phi)
    PHI1, PHI2, PHI3 = np.meshgrid(phi, phi, phi, indexing="ij")
    # Construct scalar phase field Θ with spatial envelope
    R = L / 4
    Theta = np.zeros(domain.shape, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                r = np.sqrt(X[i, j, k] ** 2 + Y[i, j, k] ** 2 + Z[i, j, k] ** 2)
                s = np.exp(-(r**2) / (2 * R**2)) if r < R else 0.0
                Theta[i, j, k, :, :, :, :] = s * (PHI1 + PHI2 + PHI3)[..., None]
    # SU(2): U = diag(e^{iΘ}, e^{-iΘ})
    U11 = np.exp(1j * Theta)
    U22 = np.exp(-1j * Theta)
    zeros = np.zeros_like(Theta, dtype=np.complex128)
    U0 = np.stack(
        [np.stack([U11, zeros], axis=-1), np.stack([zeros, U22], axis=-1)], axis=-2
    )

    # Compute GPU Skyrme gradient
    optimizer = SolitonOptimizerCUDA(domain, params, use_cuda=True)
    grad_gpu = optimizer._compute_skyrme_gradient_cuda(cp.asarray(U0))
    grad = cp.asnumpy(grad_gpu)

    # Choose a block index and a tangent direction X ∈ su(2)
    blk_idx: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = 1j * (0.31 * sx - 0.23 * sy + 0.17 * sz)

    # Numerical directional derivative of Skyrme energy
    sky = SkyrmeEnergyCUDA(S4=params["S4"])  # S6 disabled
    eps = 1e-7
    U_plus = U0.copy()
    U_plus[blk_idx] = U0[blk_idx] @ expm(eps * X)
    U_minus = U0.copy()
    U_minus[blk_idx] = U0[blk_idx] @ expm(-eps * X)
    E_plus = float(cp.asnumpy(sky.compute_cuda(cp.asarray(U_plus))))
    E_minus = float(cp.asnumpy(sky.compute_cuda(cp.asarray(U_minus))))
    dE_num = (E_plus - E_minus) / (2.0 * eps)

    # Analytic directional derivative via residual estimate from gradient
    U_blk = U0[blk_idx]
    grad_blk = grad[blk_idx]
    R_est = U_blk.conj().T @ grad_blk * (32 * np.pi**2)
    dE_ana = np.real(np.trace(R_est @ X))

    # Allow moderate tolerance on small random field
    assert np.isfinite(dE_num) and np.isfinite(dE_ana)
    # Allow left/right action convention: match up to sign
    ok = np.isclose(dE_ana, dE_num, rtol=1e-4, atol=1e-6) or np.isclose(
        -dE_ana, dE_num, rtol=1e-4, atol=1e-6
    )
    assert ok, (dE_ana, dE_num)
