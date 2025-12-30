"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Deep analysis of A1.1 scale length invariance test failure.

This script performs comprehensive analysis of the downsampling problem
in the A1.1 test to identify root causes of the 2.91e-02 error.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.sources.bvp_source_generators import BVPSourceGenerators
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.core.arrays import FieldArray
from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations


def _create_3d_domain(L: float, N: int) -> Domain:
    """Create 3D domain as 7D domain with minimal phase and temporal dimensions."""
    return Domain(
        L=L,
        N=N,
        N_phi=1,
        N_t=1,
        T=1.0,
        dimensions=7,
    )


def _solve_stationary(
    L: float, N: int, mu: float, beta: float, lam: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve stationary problem and return solution, spectral representation, and source.
    
    Returns:
        (solution_3d, solution_spectral_3d, source_3d)
    """
    domain = _create_3d_domain(L, N)
    
    source_config = {
        "gaussian_amplitude": 1.0,
        "gaussian_center": [0.5, 0.5, 0.5],
        "gaussian_width": 0.125,  # σ_norm = 1/8
        "use_cuda": True,
    }
    
    generators = BVPSourceGenerators(domain, source_config)
    source_field = generators.generate_gaussian_source()
    
    if isinstance(source_field, FieldArray):
        source_array = source_field.array
    else:
        source_array = source_field
    
    source_3d = source_array[:, :, :, 0, 0, 0, 0] if source_array.ndim == 7 else source_array
    source_3d = source_3d - np.mean(source_3d)
    
    source_7d = FieldArray(shape=domain.shape, dtype=np.complex128)
    source_7d.array[:, :, :, 0, 0, 0, 0] = source_3d
    
    solver = FFTSolver7DBasic(
        domain,
        {
            "mu": mu,
            "beta": beta,
            "lambda": lam,
            "use_cuda": True,
        }
    )
    
    solution_field = solver.solve_stationary(source_7d.array)
    
    if isinstance(solution_field, FieldArray):
        solution_array = solution_field.array
    else:
        solution_array = solution_field
    
    solution_3d = solution_array[:, :, :, 0, 0, 0, 0] if solution_array.ndim == 7 else solution_array
    
    # Get spectral representation
    ops = UnifiedSpectralOperations(domain, precision="float64")
    solution_7d = FieldArray(shape=domain.shape, dtype=np.complex128)
    solution_7d.array[:, :, :, 0, 0, 0, 0] = solution_3d
    solution_spectral_7d = ops.forward_fft(solution_7d.array, "ortho")
    solution_spectral_3d = solution_spectral_7d[:, :, :, 0, 0, 0, 0] if solution_spectral_7d.ndim == 7 else solution_spectral_7d
    
    return solution_3d.astype(np.complex128), solution_spectral_3d, source_3d


def analyze_fft_frequency_layout(N: int) -> Dict[str, Any]:
    """
    Analyze FFT frequency layout for understanding downsampling.
    
    Physical Meaning:
        FFT stores frequencies in a specific order that depends on N.
        Understanding this layout is crucial for correct downsampling.
    """
    kx = np.fft.fftfreq(N, 1.0/N)
    
    # FFT frequency order: [0, 1, 2, ..., N/2-1, -N/2, ..., -2, -1]
    positive_freqs = kx[:N//2]
    negative_freqs = kx[N//2:]
    
    return {
        "N": N,
        "positive_indices": list(range(N//2)),
        "negative_indices": list(range(N//2, N)),
        "positive_freqs": positive_freqs,
        "negative_freqs": negative_freqs,
        "frequency_order": "FFT order: [0, 1, ..., N/2-1, -N/2, ..., -2, -1]",
    }


def analyze_downsampling_method(
    a2_hat: np.ndarray, N1: int, N2: int
) -> Dict[str, Any]:
    """
    Analyze different downsampling methods and their correctness.
    
    Physical Meaning:
        Compares different approaches to spectral downsampling to identify
        which method preserves the spectral structure correctly.
    """
    results = {}
    
    # Method 1: Current implementation (extract specific indices)
    a2_hat_ds_method1 = np.zeros((N1, N1, N1), dtype=a2_hat.dtype)
    
    # Extract positive frequencies [0, 1, ..., N1//2-1]
    a2_hat_ds_method1[:N1//2, :N1//2, :N1//2] = a2_hat[:N1//2, :N1//2, :N1//2]
    
    # Extract negative frequencies [-N1//2, ..., -2, -1]
    # In FFT, these are stored at indices [N2-N1//2:N2]
    a2_hat_ds_method1[N1//2:, :N1//2, :N1//2] = a2_hat[N2-N1//2:, :N1//2, :N1//2]
    a2_hat_ds_method1[:N1//2, N1//2:, :N1//2] = a2_hat[:N1//2, N2-N1//2:, :N1//2]
    a2_hat_ds_method1[:N1//2, :N1//2, N1//2:] = a2_hat[:N1//2, :N1//2, N2-N1//2:]
    a2_hat_ds_method1[N1//2:, N1//2:, :N1//2] = a2_hat[N2-N1//2:, N2-N1//2:, :N1//2]
    a2_hat_ds_method1[N1//2:, :N1//2, N1//2:] = a2_hat[N2-N1//2:, :N1//2, N2-N1//2:]
    a2_hat_ds_method1[:N1//2, N1//2:, N1//2:] = a2_hat[:N1//2, N2-N1//2:, N2-N1//2:]
    a2_hat_ds_method1[N1//2:, N1//2:, N1//2:] = a2_hat[N2-N1//2:, N2-N1//2:, N2-N1//2:]
    
    results["method1_current"] = {
        "description": "Current implementation: extract [0:N1//2] and [N2-N1//2:N2]",
        "array": a2_hat_ds_method1,
        "energy": np.sum(np.abs(a2_hat_ds_method1)**2),
    }
    
    # Method 2: Proper frequency mapping using fftfreq
    # Map frequencies from N2 grid to N1 grid
    kx2 = np.fft.fftfreq(N2, 1.0/N2)
    kx1 = np.fft.fftfreq(N1, 1.0/N1)
    
    a2_hat_ds_method2 = np.zeros((N1, N1, N1), dtype=a2_hat.dtype)
    
    # For each frequency in N1 grid, find corresponding frequency in N2 grid
    for i1 in range(N1):
        k1 = kx1[i1]
        # Find closest frequency in N2 grid
        i2_closest = np.argmin(np.abs(kx2 - k1))
        
        for j1 in range(N1):
            k1_y = kx1[j1]
            j2_closest = np.argmin(np.abs(kx2 - k1_y))
            
            for k1_idx in range(N1):
                k1_z = kx1[k1_idx]
                k2_closest = np.argmin(np.abs(kx2 - k1_z))
                
                a2_hat_ds_method2[i1, j1, k1_idx] = a2_hat[i2_closest, j2_closest, k2_closest]
    
    results["method2_frequency_mapping"] = {
        "description": "Frequency mapping: map each N1 frequency to closest N2 frequency",
        "array": a2_hat_ds_method2,
        "energy": np.sum(np.abs(a2_hat_ds_method2)**2),
    }
    
    # Method 3: Extract low-frequency block correctly
    # For downsampling by factor 2, we need frequencies in range [-N1/2, N1/2)
    a2_hat_ds_method3 = np.zeros((N1, N1, N1), dtype=a2_hat.dtype)
    
    # Positive frequencies: [0, 1, ..., N1//2-1]
    a2_hat_ds_method3[:N1//2, :N1//2, :N1//2] = a2_hat[:N1//2, :N1//2, :N1//2]
    
    # Negative frequencies: need to map correctly
    # For N2=128, negative frequencies are at indices [64:128] = [-64, -63, ..., -1]
    # For N1=64, we need frequencies [-32, -31, ..., -1]
    # These correspond to indices [N2-N1//2:N2] = [96:128] in N2 grid
    # But wait - that's wrong! For N2=128, index 96 corresponds to frequency -32
    # Actually: fftfreq(128) gives: [0, 1, ..., 63, -64, -63, ..., -1]
    # So index 96 = 96-128 = -32, which is correct!
    
    # Extract negative frequencies correctly
    neg_start = N2 - N1//2
    neg_end = N2
    a2_hat_ds_method3[N1//2:, :, :] = a2_hat[neg_start:neg_end, :, :]
    a2_hat_ds_method3[:, N1//2:, :] = a2_hat[:, neg_start:neg_end, :]
    a2_hat_ds_method3[:, :, N1//2:] = a2_hat[:, :, neg_start:neg_end]
    
    # But this overwrites previous assignments! Need to be more careful
    # Actually, we need to handle all 8 octants separately
    a2_hat_ds_method3 = np.zeros((N1, N1, N1), dtype=a2_hat.dtype)
    
    # Octant 1: [0:N1//2, 0:N1//2, 0:N1//2] - all positive
    a2_hat_ds_method3[:N1//2, :N1//2, :N1//2] = a2_hat[:N1//2, :N1//2, :N1//2]
    
    # Octant 2: [N1//2:N1, 0:N1//2, 0:N1//2] - x negative, y,z positive
    a2_hat_ds_method3[N1//2:, :N1//2, :N1//2] = a2_hat[N2-N1//2:, :N1//2, :N1//2]
    
    # Octant 3: [0:N1//2, N1//2:N1, 0:N1//2] - y negative, x,z positive
    a2_hat_ds_method3[:N1//2, N1//2:, :N1//2] = a2_hat[:N1//2, N2-N1//2:, :N1//2]
    
    # Octant 4: [0:N1//2, 0:N1//2, N1//2:N1] - z negative, x,y positive
    a2_hat_ds_method3[:N1//2, :N1//2, N1//2:] = a2_hat[:N1//2, :N1//2, N2-N1//2:]
    
    # Octant 5: [N1//2:N1, N1//2:N1, 0:N1//2] - x,y negative, z positive
    a2_hat_ds_method3[N1//2:, N1//2:, :N1//2] = a2_hat[N2-N1//2:, N2-N1//2:, :N1//2]
    
    # Octant 6: [N1//2:N1, 0:N1//2, N1//2:N1] - x,z negative, y positive
    a2_hat_ds_method3[N1//2:, :N1//2, N1//2:] = a2_hat[N2-N1//2:, :N1//2, N2-N1//2:]
    
    # Octant 7: [0:N1//2, N1//2:N1, N1//2:N1] - y,z negative, x positive
    a2_hat_ds_method3[:N1//2, N1//2:, N1//2:] = a2_hat[:N1//2, N2-N1//2:, N2-N1//2:]
    
    # Octant 8: [N1//2:N1, N1//2:N1, N1//2:N1] - all negative
    a2_hat_ds_method3[N1//2:, N1//2:, N1//2:] = a2_hat[N2-N1//2:, N2-N1//2:, N2-N1//2:]
    
    results["method3_octant_extraction"] = {
        "description": "Octant-by-octant extraction (current method refined)",
        "array": a2_hat_ds_method3,
        "energy": np.sum(np.abs(a2_hat_ds_method3)**2),
    }
    
    return results


def analyze_physical_scaling(
    a1: np.ndarray, a2: np.ndarray, L1: float, L2: float, N1: int, N2: int
) -> Dict[str, Any]:
    """
    Analyze physical scaling of solutions.
    
    Physical Meaning:
        For dimensionless invariance, solutions should scale correctly with L.
        This function checks if the scaling is correct.
    """
    # Physical coordinates
    dx1 = L1 / N1
    dx2 = L2 / N2
    
    # Check if grid spacing is the same
    spacing_match = abs(dx1 - dx2) < 1e-12
    
    # Check solution norms
    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    
    # For dimensionless solutions, norms should be comparable
    # But they may differ due to different grid sizes
    
    # Check if solutions have similar structure
    # Normalize both
    a1_norm = a1 / norm1 if norm1 > 0 else a1
    a2_norm = a2 / norm2 if norm2 > 0 else a2
    
    # Downsample a2 to match a1 grid
    stride = N2 // N1
    a2_ds_spatial = a2_norm[::stride, ::stride, ::stride]
    
    # Compare normalized solutions
    if a2_ds_spatial.shape == a1_norm.shape:
        diff_spatial = np.linalg.norm(a1_norm - a2_ds_spatial) / np.linalg.norm(a1_norm)
    else:
        diff_spatial = np.nan
    
    return {
        "dx1": dx1,
        "dx2": dx2,
        "spacing_match": spacing_match,
        "norm1": norm1,
        "norm2": norm2,
        "norm_ratio": norm2 / norm1 if norm1 > 0 else np.nan,
        "spatial_downsampling_error": diff_spatial,
    }


def main():
    """Main analysis function."""
    print("=" * 80)
    print("A1.1 Scale Length Invariance - Deep Analysis")
    print("=" * 80)
    
    # Load config
    config_path = Path("configs/level_a/A11_scale_length.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    L1 = float(cfg["case1"]["domain"]["L"])
    N1 = int(cfg["case1"]["domain"]["N"])
    L2 = float(cfg["case2"]["domain"]["L"])
    N2 = int(cfg["case2"]["domain"]["N"])
    mu = float(cfg["case1"]["physics"]["mu"])
    beta = float(cfg["case1"]["physics"]["beta"])
    lam = float(cfg["case1"]["physics"]["lambda"])
    
    print(f"\nTest Parameters:")
    print(f"  Case 1: L={L1}, N={N1}, Δ={L1/N1:.6f}")
    print(f"  Case 2: L={L2}, N={N2}, Δ={L2/N2:.6f}")
    print(f"  Physics: μ={mu}, β={beta}, λ={lam}")
    print(f"  Grid spacing match: {'✓' if abs(L1/N1 - L2/N2) < 1e-12 else '✗'}")
    
    # Solve both cases
    print("\n" + "=" * 80)
    print("1. Solving Case 1...")
    a1, a1_hat, s1 = _solve_stationary(L1, N1, mu, beta, lam)
    print(f"   Solution shape: {a1.shape}")
    print(f"   ||a1||₂ = {np.linalg.norm(a1):.6e}")
    print(f"   ||â1||₂ = {np.linalg.norm(a1_hat):.6e}")
    print(f"   Source ||s1||₂ = {np.linalg.norm(s1):.6e}")
    
    print("\n" + "=" * 80)
    print("2. Solving Case 2...")
    a2, a2_hat, s2 = _solve_stationary(L2, N2, mu, beta, lam)
    print(f"   Solution shape: {a2.shape}")
    print(f"   ||a2||₂ = {np.linalg.norm(a2):.6e}")
    print(f"   ||â2||₂ = {np.linalg.norm(a2_hat):.6e}")
    print(f"   Source ||s2||₂ = {np.linalg.norm(s2):.6e}")
    
    # Analyze FFT frequency layout
    print("\n" + "=" * 80)
    print("3. FFT Frequency Layout Analysis")
    layout1 = analyze_fft_frequency_layout(N1)
    layout2 = analyze_fft_frequency_layout(N2)
    print(f"   N1={N1}: positive indices [0:{N1//2-1}], negative indices [{N1//2}:{N1-1}]")
    print(f"   N2={N2}: positive indices [0:{N2//2-1}], negative indices [{N2//2}:{N2-1}]")
    print(f"   For downsampling: need N1 frequencies from N2 grid")
    print(f"   Positive: [0:{N1//2}] from N2 [0:{N1//2}] ✓")
    print(f"   Negative: [{N1//2}:{N1}] from N2 [{N2-N1//2}:{N2}] = [{N2-N1//2}:{N2}]")
    
    # Analyze downsampling methods
    print("\n" + "=" * 80)
    print("4. Downsampling Method Analysis")
    downsampling_results = analyze_downsampling_method(a2_hat, N1, N2)
    for method_name, method_data in downsampling_results.items():
        print(f"\n   {method_name}:")
        print(f"     Description: {method_data['description']}")
        print(f"     Energy: {method_data['energy']:.6e}")
    
    # Analyze physical scaling
    print("\n" + "=" * 80)
    print("5. Physical Scaling Analysis")
    scaling_analysis = analyze_physical_scaling(a1, a2, L1, L2, N1, N2)
    for key, value in scaling_analysis.items():
        if isinstance(value, bool):
            print(f"   {key}: {'✓' if value else '✗'}")
        else:
            print(f"   {key}: {value:.6e}" if isinstance(value, float) else f"   {key}: {value}")
    
    # Test current downsampling method
    print("\n" + "=" * 80)
    print("6. Testing Current Downsampling Implementation")
    domain1 = _create_3d_domain(L1, N1)
    ops1 = UnifiedSpectralOperations(domain1, precision="float64")
    
    # Use method 3 (octant extraction)
    a2_hat_ds = downsampling_results["method3_octant_extraction"]["array"]
    
    # Apply inverse FFT
    a2_ds_7d = ops1.inverse_fft(
        FieldArray(array=a2_hat_ds.reshape(domain1.shape)).array, "ortho"
    )
    a2_ds = a2_ds_7d[:, :, :, 0, 0, 0, 0] if a2_ds_7d.ndim == 7 else a2_ds_7d
    
    # Normalize
    a1_norm = a1 / np.linalg.norm(a1)
    a2_ds_norm = a2_ds / np.linalg.norm(a2_ds)
    
    # Phase alignment
    a1_flat = a1_norm.reshape(-1)
    a2_flat = a2_ds_norm.reshape(-1)
    correlation = np.vdot(a1_flat, a2_flat)
    if np.abs(correlation) > np.finfo(float).eps:
        phase_optimal = correlation / np.abs(correlation)
        a2_ds_aligned = a2_ds_norm * phase_optimal
    else:
        a2_ds_aligned = a2_ds_norm
    
    # Compute error
    err = np.linalg.norm(a1_norm - a2_ds_aligned) / np.linalg.norm(a1_norm)
    print(f"   Current method error: {err:.2e}")
    print(f"   Required tolerance: 1e-12")
    print(f"   Status: {'✓ PASS' if err <= 1e-12 else '✗ FAIL'}")
    
    # Analyze error components
    print("\n" + "=" * 80)
    print("7. Error Component Analysis")
    diff = a1_norm - a2_ds_aligned
    print(f"   Max absolute difference: {np.max(np.abs(diff)):.6e}")
    print(f"   Mean absolute difference: {np.mean(np.abs(diff)):.6e}")
    print(f"   Relative error at each point:")
    rel_errors = np.abs(diff) / (np.abs(a1_norm) + 1e-15)
    print(f"     Max: {np.max(rel_errors):.6e}")
    print(f"     Mean: {np.mean(rel_errors):.6e}")
    print(f"     Median: {np.median(rel_errors):.6e}")
    
    # Check spectral consistency
    print("\n" + "=" * 80)
    print("8. Spectral Consistency Check")
    a1_hat_ds = ops1.forward_fft(
        FieldArray(array=a1.reshape(domain1.shape)).array, "ortho"
    )[:, :, :, 0, 0, 0, 0] if domain1.shape[0] == N1 else ops1.forward_fft(
        FieldArray(array=a1.reshape(domain1.shape)).array, "ortho"
    )
    
    # Compare spectral representations
    spectral_diff = np.abs(a1_hat_ds - a2_hat_ds)
    print(f"   Spectral difference ||â1 - â2_ds||₂: {np.linalg.norm(spectral_diff):.6e}")
    print(f"   Max spectral difference: {np.max(spectral_diff):.6e}")
    
    # Check low-frequency modes specifically
    low_freq_mask = np.zeros_like(a1_hat_ds, dtype=bool)
    low_freq_mask[:N1//4, :N1//4, :N1//4] = True
    low_freq_mask[N1-N1//4:, :N1//4, :N1//4] = True
    low_freq_mask[:N1//4, N1-N1//4:, :N1//4] = True
    low_freq_mask[:N1//4, :N1//4, N1-N1//4:] = True
    
    low_freq_diff = np.linalg.norm(spectral_diff[low_freq_mask])
    print(f"   Low-frequency difference: {low_freq_diff:.6e}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
