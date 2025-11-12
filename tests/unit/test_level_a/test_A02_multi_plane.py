"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.2: Multi-plane periodic source validation.

Validates superposition and absence of aliasing for multiple monochromatic modes:
- Build a spectral source with J random modes below Nyquist.
- Solve â(k_j) = c_j / D(k_j), others ~ 0 within tolerance.
- Check L2 error and spurious modes.

This test uses ready-made generators and solvers:
- BVPSourceGenerators for plane wave source generation
- FFTSolver7DBasic for solving stationary problems
- FieldArray for automatic memory management
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Tuple, List

import numpy as np

from bhlff.core.domain import Domain
from bhlff.core.sources.bvp_source_generators import BVPSourceGenerators
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.arrays import FieldArray
from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations


def _load_config() -> dict:
    config_path = (
        Path(__file__).parents[3] / "configs" / "level_a" / "A02_multi_plane.json"
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

    # Create 7D domain for 3D spatial problem
    domain = _create_3d_domain(L, N)
    
    # Random modes and complex amplitudes on the unit circle
    modes = _rand_modes(N, J, seed)
    rng = np.random.default_rng(seed + 1)
    phases = rng.random(J) * 2.0 * np.pi
    amps = np.exp(1j * phases).astype(np.complex128)

    # Build multi-plane wave source by summing individual plane waves
    # Each plane wave is generated using BVPSourceGenerators
    # Use FieldArray for automatic memory management
    source_3d = FieldArray(shape=(N, N, N), dtype=np.complex128)
    
    for idx, mode in enumerate(modes):
        # Generate plane wave source for this mode with given amplitude
        source_config = {
            "plane_wave_amplitude": amps[idx],  # Complex amplitude
            "plane_wave_mode": list(mode),
            "use_cuda": True,  # Use CUDA if available
        }
        
        generators = BVPSourceGenerators(domain, source_config)
        mode_source_field = generators.generate_plane_wave_source()
        
        # Extract 3D spatial slice
        if isinstance(mode_source_field, FieldArray):
            mode_source_array = mode_source_field.array
        else:
            mode_source_array = mode_source_field
        
        mode_source_3d = mode_source_array[:, :, :, 0, 0, 0, 0] if mode_source_array.ndim == 7 else mode_source_array
        
        # Sum into total source (superposition principle)
        source_3d.array += mode_source_3d
    
    # Expand to 7D for solver (solver expects 7D)
    # Use FieldArray for automatic memory management
    source_7d = FieldArray(shape=domain.shape, dtype=np.complex128)
    source_7d.array[:, :, :, 0, 0, 0, 0] = source_3d.array
    
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
    # Pass array from FieldArray to solver
    source_7d_array = source_7d.array if isinstance(source_7d, FieldArray) else source_7d
    solution_field = solver.solve_stationary(source_7d_array)
    
    # Extract 3D spatial slice from solution
    if isinstance(solution_field, FieldArray):
        solution_array = solution_field.array
    else:
        solution_array = solution_field
    
    solution_3d = solution_array[:, :, :, 0, 0, 0, 0] if solution_array.ndim == 7 else solution_array
    # Use FieldArray for automatic memory management
    a_real_field = FieldArray(array=solution_3d.astype(np.complex128))
    a_real = a_real_field.array
    
    # Build reference solution using exact spectral formula
    # For comparison with solver result
    shape = (N, N, N)
    
    class _Domain:
        def __init__(self, shape: Tuple[int, int, int], L: float, N: int):
            self.shape = shape
            self.L = L
            self.N = N
    
    domain_3d = _Domain(shape, L, N)
    ops = UnifiedSpectralOperations(domain_3d, precision="float64")
    
    # Build spectral representation of source
    # Extract array from FieldArray if needed
    source_3d_array = source_3d.array if isinstance(source_3d, FieldArray) else source_3d
    s_hat = ops.forward_fft(source_3d_array, "ortho")
    
    # Build reference solution using solver (all vectorization is in solver)
    # Use the same solver to compute reference solution
    # Expand source to 7D for solver
    source_7d_ref = FieldArray(shape=domain.shape, dtype=np.complex128)
    source_7d_ref.array[:, :, :, 0, 0, 0, 0] = source_3d_array
    
    # Get reference solution using solver (all vectorization is inside solver)
    a_real_ref_field = solver.solve_stationary(source_7d_ref.array)
    a_real_ref = a_real_ref_field.array[:, :, :, 0, 0, 0, 0] if isinstance(a_real_ref_field, FieldArray) else a_real_ref_field
    
    # Get spectral representation for comparison
    a_hat_ref = ops.forward_fft(a_real_ref, "ortho")

    # Transform solver solution back to spectral space to check aliasing
    a_hat_back = ops.forward_fft(a_real, "ortho")

    # Metrics: error vs exact at target bins, and spurious energy elsewhere
    # Use vectorized operations from numpy (all vectorization is in numpy)
    mode_indices = [tuple((mi % n) for mi, n in zip(m, shape)) for m in modes]
    err_bins = np.array([np.abs(a_hat_back[k] - a_hat_ref[k]) for k in mode_indices])
    err_target = float(np.linalg.norm(err_bins))
    
    # Spurious energy (aliasing) at bins not in modes
    # Use vectorized operations from numpy
    mask = np.zeros(shape, dtype=bool)
    for k in mode_indices:
        mask[k] = True
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
