"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A1.1: Length scale invariance.

Compare dimensionless solutions for two grids with same Δ=L/N:
Case1: L=1, N=256; Case2: L=2, N=512. For same dimensionless k*,
solutions should match within tolerance in normalized form.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Tuple

import numpy as np

from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations


def _load_config() -> dict:
    config_path = (
        Path(__file__).parents[3] / "configs" / "level_a" / "A11_scale_length.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_domain_shape(N: int) -> Tuple[int, int, int]:
    return (N, N, N)


def _solve_stationary(
    L: float, N: int, mu: float, beta: float, lam: float, mode: Tuple[int, int, int]
) -> np.ndarray:
    shape = _make_domain_shape(N)

    class _Domain:
        def __init__(self, shape: Tuple[int, int, int], L: float, N: int):
            self.shape = shape
            self.L = L
            self.N = N

    domain = _Domain(shape, L, N)
    ops = UnifiedSpectralOperations(domain, precision="float64")

    # Build spectral delta at exact integer bin to avoid normalization ambiguity
    s_hat = np.zeros(shape, dtype=np.complex128)
    a_hat = np.zeros_like(s_hat)
    idx = tuple((mi % n) for mi, n in zip(mode, shape))
    ksq = (2.0 * np.pi / L) ** 2 * float(np.dot(np.array(mode, dtype=float), np.array(mode, dtype=float)))
    denom = mu * (ksq ** beta) + lam
    s_hat[idx] = 1.0 + 0.0j
    a_hat[idx] = s_hat[idx] / denom
    a = ops.inverse_fft(a_hat, "ortho").astype(np.complex128)
    return a


def test_A11_scale_length() -> None:
    cfg = _load_config()
    mu = float(cfg["case1"]["physics"]["mu"])
    beta = float(cfg["case1"]["physics"]["beta"])
    lam = float(cfg["case1"]["physics"]["lambda"])
    L1 = float(cfg["case1"]["domain"]["L"])
    N1 = int(cfg["case1"]["domain"]["N"])
    L2 = float(cfg["case2"]["domain"]["L"])
    N2 = int(cfg["case2"]["domain"]["N"])
    mode = tuple(int(x) for x in cfg["forcing"]["mode"])  # integer mode in both grids
    tol = float(cfg["tolerance"]["invariance"])

    # Solve in both cases and compare dimensionless normalized fields
    a1 = _solve_stationary(L1, N1, mu, beta, lam, mode)
    a2 = _solve_stationary(L2, N2, mu, beta, lam, mode)

    # Normalize by field L2 norm to compare shape-only (dimensionless)
    a1n = a1 / max(np.linalg.norm(a1), np.finfo(float).eps)
    a2n = a2 / max(np.linalg.norm(a2), np.finfo(float).eps)

    # Downsample higher-res solution to coarse grid by integer stride (since Δ equal)
    stride = N2 // N1
    a2n_ds = a2n[::stride, ::stride, ::stride]

    # Align global phase to remove arbitrary complex rotation between solutions
    inner = np.vdot(a1n.ravel(), a2n_ds.ravel())  # <a1, a2>
    if inner != 0:
        phase = inner / np.abs(inner)
        a2n_ds = a2n_ds * np.conj(phase)

    err = float(np.linalg.norm(a1n - a2n_ds))

    out_dir = Path("output") / cfg["test_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_id": cfg["test_id"],
                "status": "PASS" if err <= tol else "FAIL",
                "metrics": {"invariance_error": err},
            },
            f,
            indent=2,
        )
    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "status", "invariance_error"])
        writer.writerow([cfg["test_id"], "PASS" if err <= tol else "FAIL", err])

    assert err <= tol, f"A11 failed: invariance_error={err:.2e}"
