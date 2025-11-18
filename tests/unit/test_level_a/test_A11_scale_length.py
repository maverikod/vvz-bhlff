"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A1.1: Length scale invariance.

Compare dimensionless solutions for two grids with same Δ=L/N:
Case1: L=1, N=256; Case2: L=2, N=512. For same dimensionless k*,
solutions should match within tolerance in normalized form.

This test uses ready-made generators and solvers:
- BVPSourceGenerators for Gaussian source generation
- FFTSolver7DBasic for solving stationary problems
- FieldArray for automatic memory management
"""

from __future__ import annotations

import json
import csv
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
        Path(__file__).parents[3] / "configs" / "level_a" / "A11_scale_length.json"
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


def _solve_stationary(
    L: float, N: int, mu: float, beta: float, lam: float
) -> FieldArray:
    """
    Solve stationary problem with properly scaled Gaussian source.
    
    Physical Meaning:
        Solves the fractional Laplacian equation L_β a = s for a Gaussian
        source that scales correctly with L to maintain dimensionless invariance.
        Uses ready-made generators and solvers for consistency.
    
    Args:
        L (float): Domain size.
        N (int): Number of grid points per spatial dimension.
        mu (float): Diffusion coefficient.
        beta (float): Fractional order.
        lam (float): Damping parameter.
        
    Returns:
        FieldArray: Solution field with automatic memory management.
    """
    # Create 7D domain for 3D spatial problem
    domain = _create_3d_domain(L, N)
    
    # Create Gaussian source using BVPSourceGenerators
    # BVPSourceGenerators uses normalized coordinates [0, 1]
    # For dimensionless invariance, physical width σ_phys = L * σ_norm should be proportional to L
    # So σ_norm should be constant (same relative width σ_phys/L = const)
    # Use σ_norm = 1/8 for both cases to ensure σ_phys = L/8
    source_config = {
        "gaussian_amplitude": 1.0,
        "gaussian_center": [0.5, 0.5, 0.5],  # Center at 0.5 in normalized coordinates [0, 1]
        "gaussian_width": 0.125,  # σ_norm = 1/8 in normalized coordinates, so σ_phys = L/8
        "use_cuda": True,  # Use CUDA if available
    }
    
    generators = BVPSourceGenerators(domain, source_config)
    
    # Generate Gaussian source (returns FieldArray)
    source_field = generators.generate_gaussian_source()
    
    # Extract spatial slice (3D) from 7D field for 3D problem
    # For 3D domain with N_phi=1, N_t=1, spatial slice is [:, :, :, 0, 0, 0, 0]
    if isinstance(source_field, FieldArray):
        source_array = source_field.array
    else:
        source_array = source_field
    
    # Extract 3D spatial slice
    source_3d = source_array[:, :, :, 0, 0, 0, 0] if source_array.ndim == 7 else source_array
    
    # Remove mean for lambda=0 compatibility
    source_3d = source_3d - np.mean(source_3d)
    
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
    
    # Return as FieldArray for automatic memory management
    return FieldArray(array=solution_3d.astype(np.complex128))


def test_A11_scale_length() -> None:
    cfg = _load_config()
    mu = float(cfg["case1"]["physics"]["mu"])
    beta = float(cfg["case1"]["physics"]["beta"])
    lam = float(cfg["case1"]["physics"]["lambda"])
    L1 = float(cfg["case1"]["domain"]["L"])
    N1 = int(cfg["case1"]["domain"]["N"])
    L2 = float(cfg["case2"]["domain"]["L"])
    N2 = int(cfg["case2"]["domain"]["N"])
    tol = float(cfg["tolerance"]["invariance"])

    # Solve in both cases with properly scaled sources
    # Sources are scaled to maintain dimensionless invariance (Gaussian with σ ∝ L)
    a1 = _solve_stationary(L1, N1, mu, beta, lam)
    a2 = _solve_stationary(L2, N2, mu, beta, lam)

    # Compare solutions in spectral domain for accurate invariance check
    # This avoids interpolation errors and directly compares spectral structure
    
    # Extract arrays from FieldArray if needed
    if isinstance(a1, FieldArray):
        a1_array = a1.array
    else:
        a1_array = a1
    
    if isinstance(a2, FieldArray):
        a2_array = a2.array
    else:
        a2_array = a2
    
    # Create domains for spectral operations
    domain1 = _create_3d_domain(L1, N1)
    domain2 = _create_3d_domain(L2, N2)
    
    # Expand 3D arrays to 7D for UnifiedSpectralOperations
    # UnifiedSpectralOperations expects 7D shape matching domain
    a1_7d = FieldArray(shape=domain1.shape, dtype=np.complex128)
    a1_7d.array[:, :, :, 0, 0, 0, 0] = a1_array
    
    a2_7d = FieldArray(shape=domain2.shape, dtype=np.complex128)
    a2_7d.array[:, :, :, 0, 0, 0, 0] = a2_array
    
    ops1 = UnifiedSpectralOperations(domain1, precision="float64")
    ops2 = UnifiedSpectralOperations(domain2, precision="float64")
    
    # Get spectral representations (extract 3D slice from 7D result)
    a1_hat_7d = ops1.forward_fft(a1_7d.array, "ortho")
    a2_hat_7d = ops2.forward_fft(a2_7d.array, "ortho")
    
    # Extract 3D spatial slice from 7D spectral representation
    a1_hat = a1_hat_7d[:, :, :, 0, 0, 0, 0] if a1_hat_7d.ndim == 7 else a1_hat_7d
    a2_hat = a2_hat_7d[:, :, :, 0, 0, 0, 0] if a2_hat_7d.ndim == 7 else a2_hat_7d
    
    # Compare solutions in physical space after proper downsampling
    # Use spectral downsampling: extract low-frequency components and apply inverse FFT
    # This preserves spectral structure better than simple spatial downsampling
    
    # For proper downsampling, we need to extract low-frequency components correctly
    # FFT frequencies: [0, 1, 2, ..., N/2-1, -N/2, ..., -2, -1] for even N
    # For downsampling by factor 2 (N2=128 -> N1=64), we need:
    # - Positive frequencies: [0, 1, 2, ..., N1/2-1] = [0, 1, ..., 31]
    # - Negative frequencies: [-N1/2, ..., -2, -1] = [-32, ..., -2, -1]
    # But FFT stores them as: [0, 1, ..., 31, -64, ..., -33, -32, ..., -2, -1]
    # So we need: [0:N1//2] and [N2-N1//2:N2]
    
    # Extract low-frequency components correctly
    # For each axis: take [0:N1//2] and [N2-N1//2:N2], then concatenate
    a2_hat_ds = np.zeros((N1, N1, N1), dtype=a2_hat.dtype)
    
    # Extract positive frequencies [0, 1, ..., N1//2-1]
    a2_hat_ds[:N1//2, :N1//2, :N1//2] = a2_hat[:N1//2, :N1//2, :N1//2]
    
    # Extract negative frequencies [-N1//2, ..., -2, -1]
    # In FFT, these are stored at indices [N2-N1//2:N2]
    a2_hat_ds[N1//2:, :N1//2, :N1//2] = a2_hat[N2-N1//2:, :N1//2, :N1//2]
    a2_hat_ds[:N1//2, N1//2:, :N1//2] = a2_hat[:N1//2, N2-N1//2:, :N1//2]
    a2_hat_ds[:N1//2, :N1//2, N1//2:] = a2_hat[:N1//2, :N1//2, N2-N1//2:]
    a2_hat_ds[N1//2:, N1//2:, :N1//2] = a2_hat[N2-N1//2:, N2-N1//2:, :N1//2]
    a2_hat_ds[N1//2:, :N1//2, N1//2:] = a2_hat[N2-N1//2:, :N1//2, N2-N1//2:]
    a2_hat_ds[:N1//2, N1//2:, N1//2:] = a2_hat[:N1//2, N2-N1//2:, N2-N1//2:]
    a2_hat_ds[N1//2:, N1//2:, N1//2:] = a2_hat[N2-N1//2:, N2-N1//2:, N2-N1//2:]
    
    # Apply inverse FFT on smaller grid (domain1) to get downsampled solution
    a2_ds_7d = ops1.inverse_fft(
        FieldArray(array=a2_hat_ds.reshape(domain1.shape)).array, "ortho"
    )
    a2_ds = a2_ds_7d[:, :, :, 0, 0, 0, 0] if a2_ds_7d.ndim == 7 else a2_ds_7d
    
    # Normalize solutions for comparison
    a1_norm = a1_array / max(np.linalg.norm(a1_array), np.finfo(float).eps)
    a2_ds_norm = a2_ds / max(np.linalg.norm(a2_ds), np.finfo(float).eps)
    
    # Align phase using correlation
    a1_flat = a1_norm.reshape(-1)
    a2_flat = a2_ds_norm.reshape(-1)
    
    correlation = np.vdot(a1_flat, a2_flat)
    if np.abs(correlation) > np.finfo(float).eps:
        phase_optimal = correlation / np.abs(correlation)
        a2_ds_aligned = a2_ds_norm * phase_optimal
    else:
        a2_ds_aligned = a2_ds_norm
    
    # Compute relative error in physical space
    norm_diff = np.linalg.norm(a1_norm - a2_ds_aligned)
    norm_ref = np.linalg.norm(a1_norm)
    err = float(norm_diff / max(norm_ref, np.finfo(float).eps))

    out_dir = Path("output") / cfg["test_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_id": cfg["test_id"],
                "status": "PASS" if err <= tol else "FAIL",
                "metrics": {"invariance_error": err},
            },
            f,
            indent=2,
        )
    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "status", "invariance_error"])
        writer.writerow([cfg["test_id"], "PASS" if err <= tol else "FAIL", err])

    assert err <= tol, f"A11 failed: invariance_error={err:.2e}"
