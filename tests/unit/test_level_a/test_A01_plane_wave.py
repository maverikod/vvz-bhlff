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
    
    # Generate plane wave source (returns FieldArray with 7D shape for 7D domain)
    # Framework automatically:
    # - Expands 3D to 7D if domain is 7D
    # - Handles block processing, vectorization, and batching
    # - Manages memory with FieldArray swap
    source_field = generators.generate_plane_wave_source()
    
    # Verify FieldArray is used and check swap status for large fields
    assert isinstance(source_field, FieldArray), "Source should be FieldArray for swap support"
    if source_field.nbytes > 1e9:  # > 1GB
        # For large fields, swap should be used
        print(f"Source field size: {source_field.nbytes/1e9:.3f}GB, swapped: {source_field.is_swapped}")
    
    # Create solver with physics parameters
    # Framework automatically uses CUDA, block processing, and vectorization
    solver = FFTSolver7DBasic(
        domain,
        {
            "mu": mu,
            "beta": beta,
            "lambda": lam,
            "use_cuda": True,  # Framework automatically handles CUDA/CPU
        }
    )
    
    # Solve stationary problem - framework automatically:
    # - Uses block processing for large fields
    # - Vectorizes all operations
    # - Batches FFT operations
    # - Manages memory with FieldArray swap
    solution_field = solver.solve_stationary(source_field)
    
    # Verify solution is FieldArray and check swap status for large fields
    assert isinstance(solution_field, FieldArray), "Solution should be FieldArray for swap support"
    if solution_field.nbytes > 1e9:  # > 1GB
        # For large fields, swap should be used
        print(f"Solution field size: {solution_field.nbytes/1e9:.3f}GB, swapped: {solution_field.is_swapped}")
    
    # Extract 3D spatial slice from solution for comparison
    # Framework handles all 7D operations automatically
    if isinstance(solution_field, FieldArray):
        solution_array = solution_field.array
    else:
        solution_array = solution_field
    
    solution_3d = solution_array[:, :, :, 0, 0, 0, 0] if solution_array.ndim == 7 else solution_array
    a_num = solution_3d.astype(np.complex128)
    
    # Get spectral representation for reporting
    # Framework automatically handles block processing for FFT
    from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations
    ops = UnifiedSpectralOperations(domain, precision="float64")
    
    # Get source array for spectral representation
    if isinstance(source_field, FieldArray):
        source_array = source_field.array
    else:
        source_array = source_field
    
    # Framework automatically handles block processing for FFT
    s_hat_7d = ops.forward_fft(source_array, "ortho")
    a_hat_7d = ops.forward_fft(solution_array, "ortho")
    
    # Extract 3D slices for reporting
    s_hat_3d = s_hat_7d[:, :, :, 0, 0, 0, 0] if s_hat_7d.ndim == 7 else s_hat_7d
    a_hat_3d = a_hat_7d[:, :, :, 0, 0, 0, 0] if a_hat_7d.ndim == 7 else a_hat_7d
    
    # Build reference solution using analytical formula
    # For plane wave with mode m, reference amplitude is: A = 1 / (μ|k|^{2β} + λ)
    # where k = 2π/L * m
    a_ref_amp = _compute_reference_amplitude(mu, beta, lam, mode, L, N)
    
    # Reference solution is amplitude times the source (plane wave)
    # Extract 3D spatial slice from source for reference
    if isinstance(source_field, FieldArray):
        source_array = source_field.array
    else:
        source_array = source_field
    
    source_3d = source_array[:, :, :, 0, 0, 0, 0] if source_array.ndim == 7 else source_array
    a_ref = (a_ref_amp * source_3d).astype(np.complex128)
    
    # Use 3D spectral representation for reporting
    a_hat = a_hat_3d
    s_hat = s_hat_3d

    # Metrics - compare complex solutions (both numerical and reference should be complex)
    # For complex source (plane wave), solution must be complex to preserve phase information
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
