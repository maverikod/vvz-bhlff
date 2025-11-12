"""
Diagnostic script for A1.1 scale length invariance test.

Analyzes why the test fails with ~141% error.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
from bhlff.core.fft.unified_spectral_operations import UnifiedSpectralOperations


def _make_domain_shape(N: int) -> Tuple[int, int, int]:
    return (N, N, N)


def _solve_stationary(
    L: float, N: int, mu: float, beta: float, lam: float, mode: Tuple[int, int, int]
) -> np.ndarray:
    shape = _make_domain_shape(N)

    class _Domain:
        def __init__(self, shape: Tuple[int, int, int], L: float, N: int):
            self.shape = shape
            self.L = L
            self.N = N

    domain = _Domain(shape, L, N)
    ops = UnifiedSpectralOperations(domain, precision="float64")

    # Build spectral delta at exact integer bin
    s_hat = np.zeros(shape, dtype=np.complex128)
    a_hat = np.zeros_like(s_hat)
    idx = tuple((mi % n) for mi, n in zip(mode, shape))
    ksq = (2.0 * np.pi / L) ** 2 * float(
        np.dot(np.array(mode, dtype=float), np.array(mode, dtype=float))
    )
    denom = mu * (ksq**beta) + lam
    s_hat[idx] = 1.0 + 0.0j
    a_hat[idx] = s_hat[idx] / denom
    a = ops.inverse_fft(a_hat, "ortho").astype(np.complex128)
    return a, a_hat, ksq, denom


def analyze_a11_problem():
    """Detailed analysis of A1.1 invariance problem."""
    
    # Test parameters
    L1, N1 = 1.0, 128
    L2, N2 = 2.0, 256
    mu, beta, lam = 1.0, 1.0, 0.1
    mode1 = (4, 0, 0)
    mode2 = (8, 0, 0)  # mode1 * (L2/L1)
    
    print("=" * 80)
    print("A1.1 Scale Length Invariance - Detailed Analysis")
    print("=" * 80)
    print(f"\nCase 1: L={L1}, N={N1}, mode={mode1}")
    print(f"Case 2: L={L2}, N={N2}, mode={mode2}")
    print(f"Physics: μ={mu}, β={beta}, λ={lam}")
    print(f"Grid spacing: Δ1={L1/N1:.6f}, Δ2={L2/N2:.6f} (should be equal)")
    
    # Solve both cases
    print("\n" + "=" * 80)
    print("1. Solving Case 1...")
    a1, a1_hat, ksq1, denom1 = _solve_stationary(L1, N1, mu, beta, lam, mode1)
    
    print(f"   Solution shape: {a1.shape}")
    print(f"   k² = {ksq1:.6f}")
    print(f"   Denominator D = {denom1:.6f}")
    print(f"   |a1| range: [{np.min(np.abs(a1)):.6e}, {np.max(np.abs(a1)):.6e}]")
    print(f"   ||a1||₂ = {np.linalg.norm(a1):.6e}")
    print(f"   Spectral: |â1[{mode1}]| = {np.abs(a1_hat[mode1]):.6e}")
    
    print("\n" + "=" * 80)
    print("2. Solving Case 2...")
    a2, a2_hat, ksq2, denom2 = _solve_stationary(L2, N2, mu, beta, lam, mode2)
    
    print(f"   Solution shape: {a2.shape}")
    print(f"   k² = {ksq2:.6f}")
    print(f"   Denominator D = {denom2:.6f}")
    print(f"   |a2| range: [{np.min(np.abs(a2)):.6e}, {np.max(np.abs(a2)):.6e}]")
    print(f"   ||a2||₂ = {np.linalg.norm(a2):.6e}")
    print(f"   Spectral: |â2[{mode2}]| = {np.abs(a2_hat[mode2]):.6e}")
    
    # Check dimensionless wave numbers
    print("\n" + "=" * 80)
    print("3. Dimensionless Wave Numbers")
    k_dimless1 = 2.0 * np.pi / L1 * np.linalg.norm(mode1)
    k_dimless2 = 2.0 * np.pi / L2 * np.linalg.norm(mode2)
    print(f"   k*₁ = 2π/L₁ * |m₁| = {k_dimless1:.6f}")
    print(f"   k*₂ = 2π/L₂ * |m₂| = {k_dimless2:.6f}")
    print(f"   Difference: |k*₁ - k*₂| = {abs(k_dimless1 - k_dimless2):.6e}")
    print(f"   Should be equal: {'✓' if abs(k_dimless1 - k_dimless2) < 1e-10 else '✗'}")
    
    # Check denominators
    print("\n" + "=" * 80)
    print("4. Spectral Denominators")
    print(f"   D₁ = μ(k²₁)^β + λ = {denom1:.6f}")
    print(f"   D₂ = μ(k²₂)^β + λ = {denom2:.6f}")
    print(f"   Ratio D₂/D₁ = {denom2/denom1:.6f}")
    print(f"   Expected: D₂/D₁ = (k²₂/k²₁)^β = {(ksq2/ksq1)**beta:.6f}")
    
    # Normalize solutions
    print("\n" + "=" * 80)
    print("5. Normalized Solutions")
    a1n = a1 / np.linalg.norm(a1)
    a2n = a2 / np.linalg.norm(a2)
    print(f"   ||a1n||₂ = {np.linalg.norm(a1n):.6f}")
    print(f"   ||a2n||₂ = {np.linalg.norm(a2n):.6f}")
    
    # Downsample a2
    print("\n" + "=" * 80)
    print("6. Downsampling Analysis")
    stride = N2 // N1
    print(f"   Stride = N₂/N₁ = {stride}")
    a2n_ds = a2n[::stride, ::stride, ::stride]
    print(f"   Downsampled shape: {a2n_ds.shape} (should match {a1n.shape})")
    print(f"   Shape match: {'✓' if a2n_ds.shape == a1n.shape else '✗'}")
    
    # Check if downsampling preserves structure
    print("\n   Checking downsampling quality...")
    a2n_ds_renorm = a2n_ds / np.linalg.norm(a2n_ds)
    
    # Phase alignment
    print("\n" + "=" * 80)
    print("7. Phase Alignment")
    a1_flat = a1n.reshape(-1)
    a2_flat = a2n_ds_renorm.reshape(-1)
    
    correlation = np.vdot(a1_flat, a2_flat)
    print(f"   Correlation <a1n, a2n_ds> = {correlation:.6e}")
    print(f"   |correlation| = {np.abs(correlation):.6e}")
    
    if np.abs(correlation) > np.finfo(float).eps:
        phase_optimal = correlation / np.abs(correlation)
        print(f"   Optimal phase: {phase_optimal:.6e}")
        print(f"   |phase| = {np.abs(phase_optimal):.6f}")
        a2n_aligned = a2n_ds_renorm * phase_optimal
    else:
        print("   WARNING: Correlation too small, using identity phase")
        a2n_aligned = a2n_ds_renorm
    
    # Final error
    print("\n" + "=" * 80)
    print("8. Final Comparison")
    diff = a1n - a2n_aligned
    norm_diff = np.linalg.norm(diff)
    norm_ref = np.linalg.norm(a1n)
    err = norm_diff / norm_ref
    print(f"   ||a1n - a2n_aligned||₂ = {norm_diff:.6e}")
    print(f"   ||a1n||₂ = {norm_ref:.6f}")
    print(f"   Relative error = {err:.6e} ({err*100:.2f}%)")
    
    # Sample comparison
    print("\n" + "=" * 80)
    print("9. Sample Point Comparison")
    sample_points = [(0, 0, 0), (N1//4, N1//4, N1//4), (N1//2, N1//2, N1//2)]
    for i, j, k in sample_points:
        val1 = a1n[i, j, k]
        val2 = a2n_aligned[i, j, k] if i < a2n_aligned.shape[0] else 0
        diff_val = val1 - val2
        print(f"   Point ({i},{j},{k}): a1n={val1:.6e}, a2n={val2:.6e}, diff={diff_val:.6e}")
    
    # Spectral comparison
    print("\n" + "=" * 80)
    print("10. Spectral Domain Comparison")
    # Transform normalized solutions back to spectral
    class _Domain1:
        def __init__(self):
            self.shape = a1n.shape
            self.L = L1
            self.N = N1
    class _Domain2:
        def __init__(self):
            self.shape = a2n_ds_renorm.shape
            self.L = L2
            self.N = N1  # After downsampling
    
    domain1 = _Domain1()
    domain2 = _Domain2()
    ops1 = UnifiedSpectralOperations(domain1, precision="float64")
    ops2 = UnifiedSpectralOperations(domain2, precision="float64")
    
    a1n_hat = ops1.forward_fft(a1n, "ortho")
    a2n_hat = ops2.forward_fft(a2n_ds_renorm, "ortho")
    
    print(f"   |â1n[{mode1}]| = {np.abs(a1n_hat[mode1]):.6e}")
    print(f"   |â2n[{mode1}]| = {np.abs(a2n_hat[mode1]):.6e}")
    print(f"   Ratio = {np.abs(a2n_hat[mode1]) / max(np.abs(a1n_hat[mode1]), 1e-15):.6f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_a11_problem()

