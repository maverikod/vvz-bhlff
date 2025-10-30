"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.2: Multi-plane periodic source validation.

Validates superposition and absence of aliasing for multiple monochromatic modes:
- Build a spectral source with J random modes below Nyquist.
- Solve â(k_j) = c_j / D(k_j), others ~ 0 within tolerance.
- Check L2 error and spurious modes.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Tuple, List

import numpy as np

from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations


def _load_config() -> dict:
    config_path = (
        Path(__file__).parents[3] / "configs" / "level_a" / "A02_multi_plane.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_domain_shape(N: int) -> Tuple[int, int, int]:
    return (N, N, N)


def _rand_modes(N: int, J: int, seed: int) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    # modes from [-N/4, N/4) to stay well within Nyquist and avoid wrap-around
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
    # Create spectral array with non-zero entries only at exact integer frequency bins
    spec = np.zeros(shape, dtype=np.complex128)
    N = np.array(shape)
    center = tuple(0 for _ in shape)  # using unshifted indexing consistent with ops
    for idx, m in enumerate(modes):
        # Map negative frequencies to equivalent FFT bins via modulo
        k = tuple((mi % n) for mi, n in zip(m, N))
        spec[k] = amplitudes[idx]
    return spec


def test_A02_multi_plane() -> None:
    cfg = _load_config()
    L = float(cfg["domain"]["L"])
    N = int(cfg["domain"]["N"])
    mu = float(cfg["physics"]["mu"])
    beta = float(cfg["physics"]["beta"])
    lam = float(cfg["physics"]["lambda"])
    J = int(cfg["forcing"]["modes"])
    seed = int(cfg["forcing"]["seed"])
    tol = float(cfg["tolerance"]["error_L2"])
    tol_sp = float(cfg["tolerance"].get("spurious", tol))

    shape = _make_domain_shape(N)

    class _Domain:
        def __init__(self, shape: Tuple[int, int, int]):
            self.shape = shape
            self.L = L
            self.N = N

    domain = _Domain(shape)
    ops = UnifiedSpectralOperations(domain, precision="float64")

    # Random modes and complex amplitudes on the unit circle
    modes = _rand_modes(N, J, seed)
    rng = np.random.default_rng(seed + 1)
    phases = rng.random(J) * 2.0 * np.pi
    amps = np.exp(1j * phases).astype(np.complex128)

    # Build spectral source and its real-space version
    s_hat = _build_spectral_source(shape, modes, amps)
    s_real = ops.inverse_fft(s_hat, "ortho")

    # Solve in spectral space: a_hat = s_hat / (mu|k|^{2β} + λ)
    a_hat = np.zeros_like(s_hat)
    # only at selected modes
    for idx, m in enumerate(modes):
        k = tuple((mi % n) for mi, n in zip(m, shape))
        ksq = (2.0 * np.pi / L) ** 2 * float(sum(mi * mi for mi in m))
        denom = mu * (ksq ** beta) + lam
        a_hat[k] = s_hat[k] / denom

    # Transform back to real space and compute forward to check aliasing
    a_real = ops.inverse_fft(a_hat, "ortho")
    a_hat_back = ops.forward_fft(a_real, "ortho")

    # Metrics: error vs exact at target bins, and spurious energy elsewhere
    err_bins = []
    for m in modes:
        k = tuple((mi % n) for mi, n in zip(m, shape))
        err_bins.append(np.abs(a_hat_back[k] - a_hat[k]))
    err_target = float(np.linalg.norm(err_bins))

    # Spurious energy (aliasing) at bins not in modes
    mask = np.zeros(shape, dtype=bool)
    for m in modes:
        mask[tuple((mi % n) for mi, n in zip(m, shape))] = True
    spurious = float(np.linalg.norm(a_hat_back[~mask]))

    # Reporting
    out_dir = Path("output") / cfg["test_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "a_realspace.npy", a_real)
    np.save(out_dir / "a_kspace.npy", a_hat_back)
    np.save(out_dir / "s_kspace.npy", s_hat)

    status = "PASS" if (err_target <= tol and spurious <= tol_sp) else "FAIL"
    metrics = {
        "test_id": cfg["test_id"],
        "status": status,
        "metrics": {"target_error": err_target, "spurious_energy": spurious},
        "parameters": {
            "domain": {"L": L, "N": N},
            "physics": {"mu": mu, "beta": beta, "lambda": lam},
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "status", "target_error", "spurious_energy"])
        writer.writerow(
            [
                cfg["test_id"],
                status,
                metrics["metrics"]["target_error"],
                metrics["metrics"]["spurious_energy"],
            ]
        )

    assert (
        status == "PASS"
    ), f"A02 failed: target_error={err_target:.2e}, spurious={spurious:.2e}"
