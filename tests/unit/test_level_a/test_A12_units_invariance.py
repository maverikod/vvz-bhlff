"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A1.2: Units invariance.

Emulate change of base units (L0,T0,A0) keeping dimensionless parameters (ν̃, λ̃, ŝ̃)
constant; compare normalized solutions.
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
        Path(__file__).parents[3] / "configs" / "level_a" / "A12_units_invariance.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_domain_shape(N: int) -> Tuple[int, int, int]:
    return (N, N, N)


def _solve_with_units(
    L: float,
    N: int,
    nu_dimless: float,
    beta: float,
    lam_dimless: float,
    mode: Tuple[int, int, int],
    L0: float,
    T0: float,
    A0: float,
) -> np.ndarray:
    """
    Keep dimensionless ν̃, λ̃ fixed; convert to dimensional ν, λ for solver given base units.
    For spectral operator here, use effective ν = ν̃ / T0 * L0^(2β) (up to constant factors);
    since we compare normalized fields, proportionality suffices.
    """
    shape = _make_domain_shape(N)

    class _Domain:
        def __init__(self, shape: Tuple[int, int, int], L: float, N: int):
            self.shape = shape
            self.L = L
            self.N = N

    domain = _Domain(shape, L, N)
    ops = UnifiedSpectralOperations(domain, precision="float64")

    # Build plane wave source with amplitude scaled by A0 (dimensionful amplitude)
    grid = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    m = np.array(mode)
    phase = sum((2.0 * np.pi * mi * gi) / n for mi, gi, n in zip(m, grid, shape))
    s = (A0 * np.exp(1j * phase)).astype(np.complex128)

    s_hat = ops.forward_fft(s, "ortho")

    # Effective dimensional parameters from dimensionless (proportional forms)
    nu_eff = nu_dimless * (L0 ** (2.0 * beta)) / max(T0, np.finfo(float).tiny)
    lam_eff = lam_dimless / max(T0, np.finfo(float).tiny)

    # Only mode bin non-zero; compute denominator at that mode
    a_hat = np.zeros_like(s_hat)
    idx = tuple((mi % n) for mi, n in zip(mode, shape))
    ksq = (2.0 * np.pi / L) ** 2 * float(np.dot(np.array(mode, dtype=float), np.array(mode, dtype=float)))
    denom = nu_eff * (ksq ** beta) + lam_eff
    a_hat[idx] = s_hat[idx] / denom
    a = ops.inverse_fft(a_hat, "ortho").astype(np.complex128)
    return a


def test_A12_units_invariance() -> None:
    cfg = _load_config()
    L = float(cfg["domain"]["L"])
    N = int(cfg["domain"]["N"])
    nu_dimless = float(cfg["physics_dimless"]["nu"])
    beta = float(cfg["physics_dimless"]["beta"])
    lam_dimless = float(cfg["physics_dimless"]["lambda"])
    mode = tuple(int(x) for x in cfg["forcing"]["mode"])
    tol = float(cfg["tolerance"]["invariance"])

    u1 = cfg["units1"]
    u2 = cfg["units2"]

    a1 = _solve_with_units(
        L, N, nu_dimless, beta, lam_dimless, mode, u1["L0"], u1["T0"], u1["A0"]
    )
    a2 = _solve_with_units(
        L, N, nu_dimless, beta, lam_dimless, mode, u2["L0"], u2["T0"], u2["A0"]
    )

    # Normalize by amplitude scale to compare dimensionless fields
    a1n = a1 / max(np.linalg.norm(a1), np.finfo(float).eps)
    a2n = a2 / max(np.linalg.norm(a2), np.finfo(float).eps)

    err = float(np.linalg.norm(a1n - a2n))

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

    assert err <= tol, f"A12 failed: invariance_error={err:.2e}"
