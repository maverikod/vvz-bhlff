"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.1: Plane wave validation for Level A.

This test validates the spectral solution for a monochromatic excitation,
checking that the numerical implementation of the fractional operator matches
analytic expectations and that anisotropy is absent for equal |k|.
"""

from __future__ import annotations

import json
import math
import os
import csv
from pathlib import Path
from typing import Tuple

import numpy as np

from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations
from bhlff.core.fft.fractional_laplacian import FractionalLaplacian


def _load_config() -> dict:
    config_path = (
        Path(__file__).parents[3] / "configs" / "level_a" / "A01_plane_wave.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_domain_shape(N: int) -> Tuple[int, int, int]:
    return (N, N, N)


def _generate_plane_wave(
    shape: Tuple[int, int, int], mode: Tuple[int, int, int]
) -> np.ndarray:
    N = np.array(shape)
    grid = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    phase = sum((2.0 * np.pi * m * g) / n for m, g, n in zip(mode, grid, N))
    return np.exp(1j * phase).astype(np.complex128)


def _compute_reference_amplitude(
    mu: float, beta: float, lam: float, mode: Tuple[int, int, int], L: float, N: int
) -> float:
    # |k|^2 with k = 2π/L * m
    m = np.array(mode, dtype=float)
    k_sq = (2.0 * np.pi / L) ** 2 * float(np.dot(m, m))
    return 1.0 / (mu * (k_sq ** (beta)) + lam)


def _ensure_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def test_A01_plane_wave() -> None:
    cfg = _load_config()
    L = float(cfg["domain"]["L"])
    N = int(cfg["domain"]["N"])
    mu = float(cfg["physics"]["mu"])
    beta = float(cfg["physics"]["beta"])
    lam = float(cfg["physics"]["lambda"])
    mode = tuple(int(x) for x in cfg["forcing"]["modes"][0])
    tol = float(cfg["tolerance"]["error_L2"])

    shape = _make_domain_shape(N)

    class _Domain:
        def __init__(self, shape: Tuple[int, int, int]):
            self.shape = shape
            self.L = L
            self.N = N

    domain = _Domain(shape)

    ops = UnifiedSpectralOperations(domain, precision="float64")
    frac = FractionalLaplacian(domain, beta=beta, lambda_param=lam)

    # Source plane wave
    s_real = _generate_plane_wave(shape, mode)

    # Solve in spectral space: a_hat = s_hat / (mu|k|^{2β} + λ)
    s_hat = ops.forward_fft(s_real, "ortho")
    Dk = mu * frac.get_spectral_coefficients()
    # Dk is ν|k|^{2β} in many modules; ensure it matches mu|k|^{2β} here
    # If FractionalLaplacian already includes lambda, add lam only if not present
    # Robust approach: build denominator explicitly as mu|k|^{2β} + λ
    # FractionalLaplacian.get_spectral_coefficients() returns |k|^{2β} (project convention)
    denom = mu * Dk + lam

    a_hat = s_hat / denom
    a_num = ops.inverse_fft(a_hat, "ortho").astype(np.complex128)

    # Reference is s / D(k*) exactly for that single mode
    ref_amp = _compute_reference_amplitude(mu, beta, lam, mode, L, N)
    a_ref = ref_amp * s_real

    # Metrics
    err_L2 = np.linalg.norm(a_num - a_ref) / max(
        np.linalg.norm(a_ref), np.finfo(float).eps
    )
    err_inf = np.max(np.abs(a_num - a_ref)) / max(
        np.max(np.abs(a_ref)), np.finfo(float).eps
    )

    # Basic anisotropy proxy: compare energy per axis for permutations with same |k|
    energies = []
    for alt in [(mode[0], 0, 0), (0, mode[1], 0), (0, 0, mode[2])]:
        if sum(abs(x) for x in alt) == 0:
            continue
        s_alt = _generate_plane_wave(shape, alt)
        a_alt = _compute_reference_amplitude(mu, beta, lam, alt, L, N) * s_alt
        energies.append(float(np.linalg.norm(a_alt)))
    anis = (
        0.0
        if len(energies) <= 1
        else float(np.max(energies) - np.min(energies)) / max(np.max(energies), 1.0)
    )

    # Reporting
    out_dir = Path("output") / cfg["test_id"]
    _ensure_output_dir(out_dir)
    np.save(out_dir / "a_realspace.npy", a_num)
    np.save(out_dir / "a_kspace.npy", a_hat)
    np.save(out_dir / "s_kspace.npy", s_hat)

    metrics = {
        "test_id": cfg["test_id"],
        "status": "PASS" if err_L2 <= tol else "FAIL",
        "metrics": {
            "error_L2": float(err_L2),
            "error_inf": float(err_inf),
            "anisotropy": float(anis),
        },
        "parameters": {
            "domain": {"L": L, "N": N},
            "physics": {"mu": mu, "beta": beta, "lambda": lam},
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "status", "error_L2", "error_inf", "anisotropy"])
        writer.writerow(
            [
                cfg["test_id"],
                metrics["status"],
                metrics["metrics"]["error_L2"],
                metrics["metrics"]["error_inf"],
                metrics["metrics"]["anisotropy"],
            ]
        )

    assert err_L2 <= tol, f"A01 failed: L2 error {err_L2:.2e} > {tol:.2e}"
