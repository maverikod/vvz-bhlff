"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.3: Zero mode handling for lambda=0.

Validates proper handling of the singular k=0 mode when lambda=0:
1) ŝ(0) = 0 — solver should proceed without division by zero.
2) ŝ(0) ≠ 0 — must raise a clear exception.
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
        Path(__file__).parents[3] / "configs" / "level_a" / "A03_zero_mode.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_domain_shape(N: int) -> Tuple[int, int, int]:
    return (N, N, N)


def _ensure_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _solve_stationary(
    domain, mu: float, beta: float, lam: float, source: np.ndarray
) -> np.ndarray:
    """Spectral stationary solver a_hat = s_hat / (mu|k|^{2β} + λ) with k=0 safety."""
    ops = UnifiedSpectralOperations(domain, precision="float64")

    s_hat = ops.forward_fft(source, "ortho")
    # Build a_hat only at non-zero bins
    a_hat = np.zeros_like(s_hat)
    nz = np.argwhere(np.abs(s_hat) > 0)
    for idx in map(tuple, nz):
        # Guard division by zero at k=0 when lambda=0
        if lam == 0.0 and all(i == 0 for i in idx):
            raise ZeroDivisionError("lambda=0 with non-zero zero-mode in source: ŝ(0)≠0")
        # Map to integer mode vector m
        m = [i if i <= n // 2 else i - n for i, n in zip(idx, source.shape)]
        ksq = (2.0 * np.pi / domain.L) ** 2 * float(sum(mi * mi for mi in m))
        denom = mu * (ksq ** beta) + lam
        if denom != 0.0:
            a_hat[idx] = s_hat[idx] / denom

    return ops.inverse_fft(a_hat, "ortho").astype(np.complex128)


def test_A03_zero_mode() -> None:
    cfg = _load_config()
    L = float(cfg["domain"]["L"])
    N = int(cfg["domain"]["N"])
    mu = float(cfg["physics"]["mu"])
    beta = float(cfg["physics"]["beta"])
    lam = float(cfg["physics"]["lambda"])  # 0.0

    class _Domain:
        def __init__(self, shape: Tuple[int, int, int]):
            self.shape = shape
            self.L = L
            self.N = N

    shape = _make_domain_shape(N)
    domain = _Domain(shape)

    out_dir = Path("output") / cfg["test_id"]
    _ensure_output_dir(out_dir)

    # Case 1: lambda=0, ŝ(0)=0
    # Build spectral source with zero DC and transform to real space
    ops = UnifiedSpectralOperations(domain, precision="float64")
    s1_hat = np.zeros(shape, dtype=np.complex128)
    s1_hat[1, 0, 0] = 1.0
    s1_hat[-1, 0, 0] = 1.0
    source1 = ops.inverse_fft(s1_hat, "ortho").astype(np.complex128)

    a1 = _solve_stationary(domain, mu, beta, lam, source1)
    # Residual r = L_beta a - s in spectral space
    a1_hat = ops.forward_fft(a1, "ortho")
    s1_hat = ops.forward_fft(source1, "ortho")
    # Residual only at excited bins
    r1_hat = np.zeros_like(a1_hat)
    nz = np.argwhere(np.abs(s1_hat) > 0)
    for idx in map(tuple, nz):
        m = [i if i <= n // 2 else i - n for i, n in zip(idx, source1.shape)]
        ksq = (2.0 * np.pi / domain.L) ** 2 * float(sum(mi * mi for mi in m))
        denom = mu * (ksq ** beta) + lam
        r1_hat[idx] = denom * a1_hat[idx] - s1_hat[idx]
    res1 = float(
        np.linalg.norm(r1_hat) / max(np.linalg.norm(s1_hat), np.finfo(float).eps)
    )

    # Case 2: lambda=0, ŝ(0)≠0 — expect exception
    source2 = np.zeros(shape, dtype=np.complex128)
    source2[0, 0, 0] = 1.0
    raised = False
    try:
        _ = _solve_stationary(domain, mu, beta, lam, source2)
    except ZeroDivisionError:
        raised = True

    metrics = {
        "test_id": cfg["test_id"],
        "status": (
            "PASS"
            if (res1 <= float(cfg["tolerance"]["residual"]) and raised)
            else "FAIL"
        ),
        "metrics": {"residual_norm": res1, "exception_raised": raised},
        "parameters": {
            "domain": {"L": L, "N": N},
            "physics": {"mu": mu, "beta": beta, "lambda": lam},
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "status", "residual_norm", "exception_raised"])
        writer.writerow(
            [
                cfg["test_id"],
                metrics["status"],
                metrics["metrics"]["residual_norm"],
                metrics["metrics"]["exception_raised"],
            ]
        )

    assert (
        metrics["status"] == "PASS"
    ), f"A03 failed: residual={res1:.2e}, exception_raised={raised}"
