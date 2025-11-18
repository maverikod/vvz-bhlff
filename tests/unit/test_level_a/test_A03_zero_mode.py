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
from bhlff.core.domain import Domain
from bhlff.core.sources.bvp_source_generators import BVPSourceGenerators
from bhlff.core.arrays import FieldArray
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic


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
    domain_7d: Domain, mu: float, beta: float, lam: float, source_7d: np.ndarray
) -> np.ndarray:
    """
    Spectral stationary solver using FFTSolver7DBasic.
    
    Physical Meaning:
        Solves the fractional Laplacian equation L_β a = s using the ready-made
        FFTSolver7DBasic, which handles all 7D operations, block processing,
        vectorization, and memory management automatically.
        
    Args:
        domain_7d (Domain): 7D domain for the problem.
        mu (float): Diffusion coefficient.
        beta (float): Fractional order.
        lam (float): Damping parameter.
        source_7d (np.ndarray): 7D source field.
        
    Returns:
        np.ndarray: 7D solution field (complex for complex sources).
    """
    # Use ready-made solver - framework handles everything automatically
    solver = FFTSolver7DBasic(
        domain_7d,
        {
            "mu": mu,
            "beta": beta,
            "lambda": lam,
            "use_cuda": True,  # Framework automatically handles CUDA/CPU
        }
    )
    
    # Solve stationary problem - returns FieldArray
    solution_field = solver.solve_stationary(source_7d)
    
    # Extract array from FieldArray
    if isinstance(solution_field, FieldArray):
        solution_array = solution_field.array
    else:
        solution_array = solution_field
    
    # Return as complex array (preserves phase information)
    return solution_array.astype(np.complex128)


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
    # Build plane wave source with mode (1,0,0) using generators
    # Framework automatically handles all operations
    domain_7d = Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
    
    source_config1 = {
        "plane_wave_amplitude": 1.0,
        "plane_wave_mode": [1, 0, 0],  # Non-zero mode, zero DC
        "use_cuda": True,  # Framework automatically handles CUDA/CPU
    }
    
    generators1 = BVPSourceGenerators(domain_7d, source_config1)
    source1_field = generators1.generate_plane_wave_source()
    
    # Extract 3D spatial slice
    if isinstance(source1_field, FieldArray):
        source1_array = source1_field.array
    else:
        source1_array = source1_field
    source1_3d = source1_array[:, :, :, 0, 0, 0, 0] if source1_array.ndim == 7 else source1_array
    source1_3d = source1_3d.astype(np.complex128)
    
    # Expand 3D source to 7D for FFT operations with 7D domain
    # Framework expects 7D arrays for 7D domain
    source1_7d = np.zeros(domain_7d.shape, dtype=np.complex128)
    source1_7d[:, :, :, 0, 0, 0, 0] = source1_3d
    source1 = source1_7d
    
    # Create UnifiedSpectralOperations for FFT operations
    ops = UnifiedSpectralOperations(domain_7d, precision="float64")
    
    # Verify DC component is zero (required for lambda=0 case)
    # Framework automatically ensures this for plane wave with non-zero mode
    s1_hat_check = ops.forward_fft(source1, "ortho")
    dc_component = s1_hat_check[0, 0, 0, 0, 0, 0, 0]
    if abs(dc_component) > 1e-10:
        # Force DC to zero if it's not already zero (due to numerical errors)
        source1_hat = ops.forward_fft(source1, "ortho")
        source1_hat[0, 0, 0, 0, 0, 0, 0] = 0.0
        source1 = ops.inverse_fft(source1_hat, "ortho")
        # Extract 3D slice for _solve_stationary
        source1_3d = source1[:, :, :, 0, 0, 0, 0] if source1.ndim == 7 else source1
        if np.isrealobj(source1_3d):
            source1_3d = source1_3d.real
        # Update 7D source
        source1[:, :, :, 0, 0, 0, 0] = source1_3d

    # Use 7D source and 7D domain for _solve_stationary
    # Framework automatically handles all operations
    a1_7d = _solve_stationary(domain_7d, mu, beta, lam, source1)
    # Residual r = L_beta a - s in spectral space
    a1_hat = ops.forward_fft(a1_7d, "ortho")
    s1_hat = ops.forward_fft(source1, "ortho")
    # Residual only at excited bins
    r1_hat = np.zeros_like(a1_hat)
    nz = np.argwhere(np.abs(s1_hat) > 0)
    for idx in map(tuple, nz):
        m = [i if i <= n // 2 else i - n for i, n in zip(idx, source1.shape)]
        ksq = (2.0 * np.pi / domain_7d.L) ** 2 * float(sum(mi * mi for mi in m))
        denom = mu * (ksq**beta) + lam
        r1_hat[idx] = denom * a1_hat[idx] - s1_hat[idx]
    res1 = float(
        np.linalg.norm(r1_hat) / max(np.linalg.norm(s1_hat), np.finfo(float).eps)
    )

    # Case 2: lambda=0, ŝ(0)≠0 — expect exception
    # Build constant source (DC component) using generators
    # This should raise exception when lambda=0
    source_config2 = {
        "gaussian_amplitude": 1.0,
        "gaussian_center": [0.5, 0.5, 0.5],
        "gaussian_width": 1.0,  # Very wide Gaussian approximates constant
        "use_cuda": True,  # Framework automatically handles CUDA/CPU
    }
    
    generators2 = BVPSourceGenerators(domain_7d, source_config2)
    source2_field = generators2.generate_gaussian_source()
    
    # Extract 7D source (keep full 7D for solver)
    if isinstance(source2_field, FieldArray):
        source2_array = source2_field.array
    else:
        source2_array = source2_field
    source2_7d = source2_array if source2_array.ndim == 7 else source2_array
    source2_7d = source2_7d.astype(np.complex128)
    
    # Normalize to have non-zero DC component
    source2_3d = source2_7d[:, :, :, 0, 0, 0, 0] if source2_7d.ndim == 7 else source2_7d
    mean_3d = np.mean(source2_3d)
    if mean_3d != 0:
        source2_7d = source2_7d / mean_3d
    
    raised = False
    try:
        _ = _solve_stationary(domain_7d, mu, beta, lam, source2_7d)
    except (ZeroDivisionError, ValueError, RuntimeError) as e:
        # Accept any exception that indicates zero-mode problem
        if "zero-mode" in str(e).lower() or "lambda=0" in str(e).lower() or "ŝ(0)" in str(e):
            raised = True
        else:
            raise

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
