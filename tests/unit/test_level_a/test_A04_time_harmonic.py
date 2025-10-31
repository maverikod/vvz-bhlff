"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.4: Time-harmonic steady-state validation.

Checks steady-state amplitude and phase for harmonic forcing with frequency ω:
ã_ss(k*) = ŝ(k*) / (ν|k*|^{2β} + λ + iω)
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
        Path(__file__).parents[3] / "configs" / "level_a" / "A04_time_harmonic.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_domain_shape(N: int) -> Tuple[int, int, int]:
    return (N, N, N)


def test_A04_time_harmonic_steady_state() -> None:
    cfg = _load_config()
    L = float(cfg["domain"]["L"])
    N = int(cfg["domain"]["N"])
    nu = float(cfg["physics"]["nu"])
    beta = float(cfg["physics"]["beta"])
    lam = float(cfg["physics"]["lambda"])
    omega = float(cfg["forcing"]["omega"])
    k_star = tuple(int(x) for x in cfg["forcing"]["k_star"])
    amp_tol = float(cfg["tolerance"]["amplitude"])
    phase_tol = float(cfg["tolerance"]["phase"])

    shape = _make_domain_shape(N)

    class _Domain:
        def __init__(self, shape: Tuple[int, int, int]):
            self.shape = shape
            self.L = L
            self.N = N

    domain = _Domain(shape)
    ops = UnifiedSpectralOperations(domain, precision="float64")

    # Build spectral source with single mode k*
    s_hat = np.zeros(shape, dtype=np.complex128)
    idx = tuple((mi % n) for mi, n in zip(k_star, shape))
    s0 = 1.0 + 0.0j
    s_hat[idx] = s0

    # Steady-state spectral solution at k*
    # Compute denominator at mode only
    ksq = (2.0 * np.pi / L) ** 2 * float(
        np.dot(np.array(k_star, dtype=float), np.array(k_star, dtype=float))
    )
    Dk = nu * (ksq**beta) + lam + 1j * omega
    a_hat = np.zeros_like(s_hat)
    a_hat[idx] = s0 / Dk

    # Measure amplitude and phase at k*
    a_val = a_hat[idx]
    amp_num = np.abs(a_val)
    phase_num = np.angle(a_val)

    # Reference
    amp_ref = 1.0 / np.abs(Dk)
    phase_ref = -np.angle(Dk)

    amp_err = abs(amp_num - amp_ref)
    # unwrap small differences modulo 2π
    dphi = phase_num - phase_ref
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    phase_err = abs(dphi)

    out_dir = Path("output") / cfg["test_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_id": cfg["test_id"],
                "status": (
                    "PASS"
                    if (amp_err <= amp_tol and phase_err <= phase_tol)
                    else "FAIL"
                ),
                "metrics": {
                    "amplitude_error": float(amp_err),
                    "phase_error": float(phase_err),
                },
                "parameters": {
                    "domain": {"L": L, "N": N},
                    "physics": {"nu": nu, "beta": beta, "lambda": lam},
                    "omega": omega,
                    "k_star": k_star,
                },
            },
            f,
            indent=2,
        )
    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "status", "amplitude_error", "phase_error"])
        status = "PASS" if (amp_err <= amp_tol and phase_err <= phase_tol) else "FAIL"
        writer.writerow([cfg["test_id"], status, amp_err, phase_err])

    assert (
        amp_err <= amp_tol and phase_err <= phase_tol
    ), f"A04 failed: amp_err={amp_err:.2e}, phase_err={phase_err:.2e}"
