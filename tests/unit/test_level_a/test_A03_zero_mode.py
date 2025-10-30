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
from bhlff.core.fft.fractional_laplacian import FractionalLaplacian


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
    frac = FractionalLaplacian(domain, beta=beta, lambda_param=lam)

    s_hat = ops.forward_fft(source, "ortho")
    Dk = mu * frac.get_spectral_coefficients() + lam

    # k=0 is at index (0,0,0) in our fft conventions (ij indexing with np.fftfreq mesh)
    # Guard for division by zero when lambda==0 and s_hat(0)!=0
    zero_idx = tuple(0 for _ in domain.shape)
    if lam == 0.0:
        if np.abs(s_hat[zero_idx]) > 0:
            raise ZeroDivisionError(
                "lambda=0 with non-zero zero-mode in source: ŝ(0)≠0"
            )
        # Set denominator at zero mode to 1 to avoid NaN (since s_hat(0)=0 this is safe)
        Dk = Dk.copy()
        Dk[zero_idx] = 1.0

    a_hat = s_hat / Dk
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
    source1 = np.zeros(shape, dtype=np.complex128)
    # Add non-zero modes only (e.g., a single non-zero spatial frequency)
    source1[1, 0, 0] = 1.0
    source1[-1, 0, 0] = 1.0  # maintain hermitian symmetry for real part if needed

    a1 = _solve_stationary(domain, mu, beta, lam, source1)
    # Residual r = L_beta a - s in spectral space
    ops = UnifiedSpectralOperations(domain, precision="float64")
    frac = FractionalLaplacian(domain, beta=beta, lambda_param=lam)
    a1_hat = ops.forward_fft(a1, "ortho")
    s1_hat = ops.forward_fft(source1, "ortho")
    D = mu * frac.get_spectral_coefficients() + lam
    r1_hat = D * a1_hat - s1_hat
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
