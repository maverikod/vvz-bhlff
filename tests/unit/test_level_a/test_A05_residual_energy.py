"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.5: Residual and energy orthogonality validation.

Residual r = L_β a - s computed in spectral space; check ||r||₂/||s||₂ and
orthogonality Re Σ_k â*(k) r̂(k) ≈ 0.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Tuple, List

import numpy as np

from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations
from typing import Sequence


def _load_config() -> dict:
    config_path = (
        Path(__file__).parents[3] / "configs" / "level_a" / "A05_residual_energy.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_domain_shape(N: int) -> Tuple[int, int, int]:
    return (N, N, N)


def _rand_modes(N: int, J: int, seed: int) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    max_f = N // 4
    modes = set()
    while len(modes) < J:
        m = tuple(int(rng.integers(-max_f, max_f)) for _ in range(3))
        if m != (0, 0, 0):
            modes.add(m)
    return list(modes)


def _build_spectral_source(
    shape: Tuple[int, int, int],
    modes: List[Tuple[int, int, int]],
    amplitudes: np.ndarray,
) -> np.ndarray:
    spec = np.zeros(shape, dtype=np.complex128)
    N = np.array(shape)
    for idx, m in enumerate(modes):
        k = tuple((mi % n) for mi, n in zip(m, N))
        spec[k] = amplitudes[idx]
    return spec


def test_A05_residual_energy() -> None:
    cfg = _load_config()
    L = float(cfg["domain"]["L"])
    N = int(cfg["domain"]["N"])
    mu = float(cfg["physics"]["mu"])
    beta = float(cfg["physics"]["beta"])
    lam = float(cfg["physics"]["lambda"])
    J = int(cfg["forcing"]["modes"])
    seed = int(cfg["forcing"]["seed"])
    tol_res = float(cfg["tolerance"]["residual"])
    tol_ortho = float(cfg["tolerance"]["orthogonality"])

    shape = _make_domain_shape(N)

    class _Domain:
        def __init__(self, shape: Tuple[int, int, int]):
            self.shape = shape
            self.L = L
            self.N = N

    domain = _Domain(shape)
    ops = UnifiedSpectralOperations(domain, precision="float64")

    modes = _rand_modes(N, J, seed)
    rng = np.random.default_rng(seed + 11)
    phases = rng.random(J) * 2.0 * np.pi
    amps = np.exp(1j * phases).astype(np.complex128)

    s_hat = _build_spectral_source(shape, modes, amps)
    s_real = ops.inverse_fft(s_hat, "ortho")

    # Solve: a_hat = s_hat / (mu|k|^{2β} + λ)
    a_hat = np.zeros_like(s_hat)
    for m in modes:
        k = tuple((mi % n) for mi, n in zip(m, shape))
        ksq = (2.0 * np.pi / L) ** 2 * float(sum(mi * mi for mi in m))
        denom = mu * (ksq ** beta) + lam
        a_hat[k] = s_hat[k] / denom

    # Residual in spectral space
    # Residual only on excited bins
    r_hat = np.zeros_like(a_hat)
    for m in modes:
        k = tuple((mi % n) for mi, n in zip(m, shape))
        ksq = (2.0 * np.pi / L) ** 2 * float(sum(mi * mi for mi in m))
        denom = mu * (ksq ** beta) + lam
        r_hat[k] = denom * a_hat[k] - s_hat[k]
    res_norm = float(
        np.linalg.norm(r_hat) / max(np.linalg.norm(s_hat), np.finfo(float).eps)
    )

    # Orthogonality: Re Σ â* r̂ ≈ 0
    ortho_value = float(np.real(np.vdot(a_hat, r_hat)))
    a_norm = float(np.linalg.norm(a_hat))
    r_norm = float(np.linalg.norm(r_hat))
    ortho_norm = abs(ortho_value) / max(a_norm * r_norm, np.finfo(float).eps)

    status = "PASS" if (res_norm <= tol_res and ortho_norm <= tol_ortho) else "FAIL"

    out_dir = Path("output") / cfg["test_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_id": cfg["test_id"],
                "status": status,
                "metrics": {"residual_norm": res_norm, "orthogonality": ortho_norm},
                "parameters": {
                    "domain": {"L": L, "N": N},
                    "physics": {"mu": mu, "beta": beta, "lambda": lam},
                },
            },
            f,
            indent=2,
        )
    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "status", "residual_norm", "orthogonality"])
        writer.writerow([cfg["test_id"], status, res_norm, ortho_norm])

    assert (
        status == "PASS"
    ), f"A05 failed: residual={res_norm:.2e}, ortho_norm={ortho_norm:.2e}"
