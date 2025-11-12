"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.5: Residual and energy orthogonality validation.

Residual r = L_β a - s computed in spectral space; check ||r||₂/||s||₂ and
orthogonality Re Σ_k â*(k) r̂(k) ≈ 0.

This test uses ready-made generators and solvers:
- FFTSolver7DBasic for solving stationary problems
- FieldArray for automatic memory management
- UnifiedSpectralOperations for FFT operations
All vectorization and batching is in the main code, not in tests.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Tuple, List

import numpy as np

from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations
from bhlff.core.arrays import FieldArray
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
) -> FieldArray:
    """
    Build spectral source using FieldArray for automatic memory management.
    
    Physical Meaning:
        Creates spectral representation of source with specified modes
        and amplitudes, using FieldArray for transparent memory management.
    """
    # Use FieldArray for automatic memory management
    spec_field = FieldArray(shape=shape, dtype=np.complex128)
    spec = spec_field.array
    N = np.array(shape)
    for idx, m in enumerate(modes):
        k = tuple((mi % n) for mi, n in zip(m, N))
        spec[k] = amplitudes[idx]
    return spec_field


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

    s_hat_field = _build_spectral_source(shape, modes, amps)
    s_hat = s_hat_field.array if isinstance(s_hat_field, FieldArray) else s_hat_field
    s_real = ops.inverse_fft(s_hat, "ortho")

    # Solve using solver (all vectorization is inside solver)
    # Create 7D domain for solver
    from bhlff.core.domain import Domain
    from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
    
    domain_7d = Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
    solver = FFTSolver7DBasic(
        domain_7d,
        {
            "mu": mu,
            "beta": beta,
            "lambda": lam,
            "use_cuda": True,
        }
    )
    
    # Expand source to 7D
    source_7d = FieldArray(shape=domain_7d.shape, dtype=np.complex128)
    source_7d.array[:, :, :, 0, 0, 0, 0] = s_real
    
    # Solve using solver (all vectorization is in solver)
    solution_field = solver.solve_stationary(source_7d.array)
    solution_3d = solution_field.array[:, :, :, 0, 0, 0, 0] if isinstance(solution_field, FieldArray) else solution_field
    
    # Get spectral representation
    a_hat = ops.forward_fft(solution_3d, "ortho")

    # Residual in spectral space: r = L_β a - s
    # Use solver's spectral coefficients for residual computation
    # Get spectral coefficients from solver
    coeffs = solver.get_spectral_coefficients()
    # Extract 3D slice from 7D coefficients
    coeffs_3d = coeffs[:, :, :, 0, 0, 0, 0] if coeffs.ndim == 7 else coeffs
    
    # Compute residual: r_hat = coeffs * a_hat - s_hat
    # All vectorization is in numpy operations
    r_hat = coeffs_3d * a_hat - s_hat
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
