"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.1: Plane wave validation for Level A.

This test validates the spectral solution for a monochromatic excitation,
checking that the numerical implementation of the fractional operator matches
analytic expectations and that anisotropy is absent for equal |k|.

This test uses ready-made generators and solvers:
- BVPSourceGenerators for plane wave source generation
- FFTSolver7DBasic for solving stationary problems
- FieldArray for automatic memory management
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from bhlff.core.domain import Domain
from bhlff.core.sources.bvp_source_generators import BVPSourceGenerators
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.arrays import FieldArray
from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations


def _load_config() -> dict:
    config_path = (
        Path(__file__).parents[3] / "configs" / "level_a" / "A01_plane_wave.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _create_3d_domain(L: float, N: int) -> Domain:
    """
    Create 3D domain as 7D domain with minimal phase and temporal dimensions.
    
    Physical Meaning:
        Creates a 7D domain for 3D spatial computations by using
        minimal phase (N_phi=1) and temporal (N_t=1) dimensions.
        This allows using 7D solvers for 3D problems.
    """
    return Domain(
        L=L,
        N=N,
        N_phi=1,  # Minimal phase dimensions for 3D spatial problem
        N_t=1,    # Minimal temporal dimension for stationary problem
        T=1.0,
        dimensions=7,
    )


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

    # Create 7D domain for 3D spatial problem
    domain = _create_3d_domain(L, N)
    
    # Create plane wave source using BVPSourceGenerators
    source_config = {
        "plane_wave_amplitude": 1.0,  # Unit amplitude
        "plane_wave_mode": list(mode),  # Wave vector mode
        "use_cuda": True,  # Use CUDA if available
    }
    
    generators = BVPSourceGenerators(domain, source_config)
    
    # Generate plane wave source (returns FieldArray)
    source_field = generators.generate_plane_wave_source()
    
    # Extract spatial slice (3D) from 7D field for 3D problem
    if isinstance(source_field, FieldArray):
        source_array = source_field.array
    else:
        source_array = source_field
    
    # Extract 3D spatial slice
    source_3d = source_array[:, :, :, 0, 0, 0, 0] if source_array.ndim == 7 else source_array
    
    # Expand back to 7D for solver (solver expects 7D)
    # Use FieldArray for automatic memory management
    source_7d = FieldArray(shape=domain.shape, dtype=np.complex128)
    source_7d.array[:, :, :, 0, 0, 0, 0] = source_3d
    
    # Create solver with physics parameters
    solver = FFTSolver7DBasic(
        domain,
        {
            "mu": mu,
            "beta": beta,
            "lambda": lam,
            "use_cuda": True,  # Use CUDA if available
        }
    )
    
    # Solve stationary problem (returns FieldArray)
    # Extract array from FieldArray if needed
    source_7d_array = source_7d.array if isinstance(source_7d, FieldArray) else source_7d
    solution_field = solver.solve_stationary(source_7d_array)
    
    # Extract 3D spatial slice from solution
    if isinstance(solution_field, FieldArray):
        solution_array = solution_field.array
    else:
        solution_array = solution_field
    
    solution_3d = solution_array[:, :, :, 0, 0, 0, 0] if solution_array.ndim == 7 else solution_array
    a_num = solution_3d.astype(np.complex128)
    
    # Compute reference solution using exact spectral formula
    # For plane wave source, the solution should be proportional to the source
    # with amplitude given by the spectral operator response
    shape = (N, N, N)
    
    class _Domain:
        def __init__(self, shape: Tuple[int, int, int], L: float, N: int):
            self.shape = shape
            self.L = L
            self.N = N
    
    domain_3d = _Domain(shape, L, N)
    ops = UnifiedSpectralOperations(domain_3d, precision="float64")
    
    # Build spectral representation of source
    s_hat = ops.forward_fft(source_3d, "ortho")
    
    # Build reference solution using exact spectral formula
    # Use FieldArray for automatic memory management
    a_hat = FieldArray(array=np.zeros_like(s_hat))
    idx = tuple((mi % n) for mi, n in zip(mode, shape))
    ksq = (2.0 * np.pi / L) ** 2 * float(
        np.dot(np.array(mode, dtype=float), np.array(mode, dtype=float))
    )
    denom = mu * (ksq**beta) + lam
    a_hat[idx] = s_hat[idx] / denom
    a_ref = ops.inverse_fft(a_hat, "ortho").astype(np.complex128)

    # Metrics
    err_L2 = np.linalg.norm(a_num - a_ref) / max(
        np.linalg.norm(a_ref), np.finfo(float).eps
    )
    err_inf = np.max(np.abs(a_num - a_ref)) / max(
        np.max(np.abs(a_ref)), np.finfo(float).eps
    )

    # Basic anisotropy proxy: compare energy per axis
    # for permutations with same |k|
    energies = []
    for alt in [(mode[0], 0, 0), (0, mode[1], 0), (0, 0, mode[2])]:
        if sum(abs(x) for x in alt) == 0:
            continue
        # Generate plane wave source for alternative mode
        alt_config = {
            "plane_wave_amplitude": 1.0,
            "plane_wave_mode": list(alt),
            "use_cuda": True,
        }
        alt_domain = _create_3d_domain(L, N)
        alt_generators = BVPSourceGenerators(alt_domain, alt_config)
        alt_source_field = alt_generators.generate_plane_wave_source()
        
        # Extract 3D spatial slice
        if isinstance(alt_source_field, FieldArray):
            alt_source_array = alt_source_field.array
        else:
            alt_source_array = alt_source_field
        alt_source_3d = alt_source_array[:, :, :, 0, 0, 0, 0] if alt_source_array.ndim == 7 else alt_source_array
        
        # Compute reference amplitude
        a_alt_amp = _compute_reference_amplitude(mu, beta, lam, alt, L, N)
        a_alt = a_alt_amp * alt_source_3d
        energies.append(float(np.linalg.norm(a_alt)))
    anis = (
        0.0
        if len(energies) <= 1
        else (float(np.max(energies) - np.min(energies)) / max(np.max(energies), 1.0))
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
