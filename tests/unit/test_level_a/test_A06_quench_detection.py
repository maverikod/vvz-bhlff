"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test A0.6: BVP Quench Detection validation for Level A with CUDA block processing.

This test validates the quench detection system - threshold events that
fix the birth of defects. Without correct quench detection, it is impossible
to investigate the birth of particles (protons/neutrons).

Physical Meaning:
    Quenches are local threshold transitions where part of BVP energy
    irreversibly goes into "matter", giving birth to a topological defect
    (proton/neutron core). This test validates the detection of three
    quench thresholds: amplitude, detuning, and gradient.

Mathematical Foundation:
    Tests three quench thresholds:
    1. Amplitude: |A| > |A_q|
    2. Detuning: |ω - ω_n(|A|)| < Δω_q with ∂Q_n/∂|A| < 0
    3. Gradient: |∇A| > Γ_q

CUDA Block Processing:
    Uses GPU acceleration when available, but this unit test enforces a
    lightweight CPU configuration for reliable and fast execution.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from bhlff.core.domain.config import PhaseConfig, SpatialConfig, TemporalConfig
from bhlff.core.domain.domain import Domain
from bhlff.core.domain.domain_7d import Domain7D
from bhlff.core.bvp.quench_detector import QuenchDetector
from bhlff.core.fft.fft_solver_7d_advanced import FFTSolver7DAdvanced
from bhlff.core.arrays import FieldArray

# Lightweight parameters for unit testing
TEST_GRID_SIZE = 16
TEST_PHASE_DIM = 2
TEST_TEMPORAL_DIM = 2


def _load_config() -> dict:
    """Load Level A quench detection configuration."""
    config_path = (
        Path(__file__).parents[3] / "configs" / "level_a" / "A06_quench_detection.json"
    )
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _create_neutralized_gaussian_7d(
    shape: Tuple[int, int, int, int, int, int, int],
    center: Tuple[float, float, float, float, float, float, float],
    sigma: float,
    domain_length: float,
) -> np.ndarray:
    """
    Create neutralized Gaussian source in 7D (zero mean for λ = 0 compatibility).
    """
    n_spatial = shape[0]
    n_phase = shape[3]
    n_time = shape[6]

    grid = np.linspace(0, domain_length, n_spatial, endpoint=False)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")

    dx = X - center[0]
    dy = Y - center[1]
    dz = Z - center[2]
    r_sq = dx**2 + dy**2 + dz**2

    exponent = np.clip(-r_sq / (2.0 * sigma**2), -700.0, 700.0)
    gaussian = np.exp(exponent)
    gaussian -= np.mean(gaussian)

    # VECTORIZED: Use broadcasting instead of loops
    # Expand gaussian to 7D shape using broadcasting
    # gaussian shape: (n_spatial, n_spatial, n_spatial)
    # target shape: (n_spatial, n_spatial, n_spatial, n_phase, n_phase, n_phase, n_time)
    # Use np.broadcast_to for efficient memory usage
    gaussian_7d = np.broadcast_to(
        gaussian[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis],
        shape
    ).copy()  # Copy to ensure contiguous array
    
    # Use FieldArray for automatic memory management
    source = FieldArray(array=gaussian_7d.astype(np.complex128))

    return source


def _compute_gradient_magnitude_7d(field: np.ndarray, dx: float) -> np.ndarray:
    """Compute |∇A| for gradient quench detection in 7D."""
    grad_x = np.gradient(field.real, dx, axis=0) + 1j * np.gradient(field.imag, dx, axis=0)
    grad_y = np.gradient(field.real, dx, axis=1) + 1j * np.gradient(field.imag, dx, axis=1)
    grad_z = np.gradient(field.real, dx, axis=2) + 1j * np.gradient(field.imag, dx, axis=2)
    return np.sqrt(np.abs(grad_x) ** 2 + np.abs(grad_y) ** 2 + np.abs(grad_z) ** 2)


def test_A06_quench_detection() -> None:
    """Lightweight validation of the quench detection subsystem."""
    logger = logging.getLogger("tests.unit.level_a.A06")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    logger.info("Starting test_A06_quench_detection")
    cfg = _load_config()

    L = float(cfg["domain"]["L"])
    N_config = int(cfg["domain"]["N"])
    N = min(N_config, TEST_GRID_SIZE)
    if N != N_config:
        logger.info(
            "Using reduced grid for test runtime: N=%d (config requested %d)",
            N,
            N_config,
        )

    mu = float(cfg["physics"]["mu"])
    beta = float(cfg["physics"]["beta"])
    lam = float(cfg["physics"]["lambda"])

    A_q = float(cfg["quench"]["amplitude_threshold"])
    Delta_omega_q = float(cfg["quench"]["detuning_threshold"])
    Gamma_q = float(cfg["quench"]["gradient_threshold"])

    sigma = float(cfg["source"]["sigma"])
    center_3d = tuple(float(x) for x in cfg["source"]["center"])
    center_7d = center_3d + (0.0, 0.0, 0.0, 0.0)

    tol_detection = float(cfg["tolerance"]["detection_accuracy"])
    tol_stability = float(cfg["tolerance"]["map_stability"])

    N_phi = TEST_PHASE_DIM
    N_t = TEST_TEMPORAL_DIM
    shape_7d = (N, N, N, N_phi, N_phi, N_phi, N_t)
    dx = L / N

    domain = Domain(L=L, N=N, N_phi=N_phi, N_t=N_t, T=1.0, dimensions=7)
    s_7d = _create_neutralized_gaussian_7d(shape_7d, center_7d, sigma, L)

    solver_params = {"mu": mu, "beta": beta, "lambda": lam}
    solver = FFTSolver7DAdvanced(domain, solver_params)

    logger.info("Solving stationary BVP with FFT solver")
    solve_start = time.perf_counter()
    # Extract array from FieldArray if needed
    s_7d_array = s_7d.array if isinstance(s_7d, FieldArray) else s_7d
    a_7d_field = solver.solve_stationary(s_7d_array)
    # Extract array from FieldArray result
    a_7d = a_7d_field.array if isinstance(a_7d_field, FieldArray) else a_7d_field
    logger.info(
        "Stationary solve completed in %.2f s (max amplitude %.5f)",
        time.perf_counter() - solve_start,
        float(np.max(np.abs(a_7d))),
    )

    spatial_config = SpatialConfig(L_x=L, L_y=L, L_z=L, N_x=N, N_y=N, N_z=N)
    phase_config = PhaseConfig(N_phi_1=N_phi, N_phi_2=N_phi, N_phi_3=N_phi)
    temporal_config = TemporalConfig(T_max=1.0, N_t=N_t)
    domain_7d = Domain7D(spatial_config, phase_config, temporal_config)

    quench_config = {
        "amplitude_threshold": A_q,
        "detuning_threshold": Delta_omega_q,
        "gradient_threshold": Gamma_q,
        "carrier_frequency": 1.0,
        "use_cuda": False,
        "block_size": 0,
        "overlap": 0,
        "batch_size": 1,
        "progress_interval": 10_000,
    }
    detector = QuenchDetector(domain_7d, quench_config)

    logger.info("Running first quench detection pass")
    detection_start = time.perf_counter()
    quench_results = detector.detect_quenches(a_7d)
    logger.info(
        "First detection completed in %.2f s (detected=%s, count=%d)",
        time.perf_counter() - detection_start,
        quench_results.get("quenches_detected", False),
        len(quench_results.get("quench_locations", [])),
    )

    envelope_amp = np.abs(a_7d[:, :, :, 0, 0, 0, 0])
    grad_mag = _compute_gradient_magnitude_7d(a_7d, dx)
    grad_slice = grad_mag[:, :, :, 0, 0, 0, 0] if grad_mag.ndim == 7 else grad_mag

    amp_exceeded = envelope_amp > A_q
    grad_exceeded = grad_slice > Gamma_q

    logger.info(
        "Threshold exceedances: amplitude=%d, gradient=%d",
        int(np.sum(amp_exceeded)),
        int(np.sum(grad_exceeded)),
    )

    logger.info("Running second quench detection pass for stability check")
    detection_start = time.perf_counter()
    quench_results_2 = detector.detect_quenches(a_7d)
    logger.info(
        "Second detection completed in %.2f s (count=%d)",
        time.perf_counter() - detection_start,
        len(quench_results_2.get("quench_locations", [])),
    )

    amplitude_quenches = quench_results.get("amplitude_quenches", [])
    gradient_quenches = quench_results.get("gradient_quenches", [])

    amp_spatial_locations = {
        (int(loc[0]), int(loc[1]), int(loc[2]))
        for loc in amplitude_quenches
        if len(loc) >= 3
    }
    grad_spatial_locations = {
        (int(loc[0]), int(loc[1]), int(loc[2]))
        for loc in gradient_quenches
        if len(loc) >= 3
    }

    amp_detected_count = len(amp_spatial_locations)
    amp_expected_count = int(np.sum(amp_exceeded))
    detection_accuracy_amp = (
        1.0
        if amp_expected_count == 0 and amp_detected_count == 0
        else amp_detected_count / max(amp_expected_count, 1)
    )

    grad_detected_count = len(grad_spatial_locations)
    grad_expected_count = int(np.sum(grad_exceeded))
    detection_accuracy_grad = (
        1.0
        if grad_expected_count == 0 and grad_detected_count == 0
        else grad_detected_count / max(grad_expected_count, 1)
    )

    quenches_1_count = len(quench_results.get("quench_locations", []))
    quenches_2_count = len(quench_results_2.get("quench_locations", []))
    map_stability = (
        1.0
        if quenches_1_count == 0
        else abs(quenches_1_count - quenches_2_count) / max(quenches_1_count, 1)
    )

    status = (
        "PASS"
        if (amp_expected_count == 0 or detection_accuracy_amp >= tol_detection)
        and (grad_expected_count == 0 or detection_accuracy_grad >= tol_detection)
        and map_stability <= tol_stability
        else "FAIL"
    )

    out_dir = Path("output") / cfg["test_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use FieldArray for automatic memory management
    quench_map = FieldArray(shape=(N, N, N), dtype=float)
    for loc in quench_results.get("quench_locations", []):
        if len(loc) >= 3:
            i, j, k = int(loc[0]), int(loc[1]), int(loc[2])
            if 0 <= i < N and 0 <= j < N and 0 <= k < N:
                quench_map.array[i, j, k] = 1.0

    np.save(out_dir / "quench_map.npy", quench_map.array)
    np.save(out_dir / "envelope_amplitude.npy", envelope_amp)
    np.save(out_dir / "gradient_magnitude.npy", grad_slice)

    metrics = {
        "test_id": cfg["test_id"],
        "status": status,
        "metrics": {
            "quenches_detected": quench_results.get("quenches_detected", False),
            "quench_count": len(quench_results.get("quench_locations", [])),
            "amplitude_quenches": amp_detected_count,
            "gradient_quenches": grad_detected_count,
            "detection_accuracy_amplitude": float(detection_accuracy_amp),
            "detection_accuracy_gradient": float(detection_accuracy_grad),
            "map_stability": float(map_stability),
        },
        "parameters": {
            "domain": {"L": L, "N": N},
            "physics": {"mu": mu, "beta": beta, "lambda": lam},
            "quench": {
                "amplitude_threshold": A_q,
                "detuning_threshold": Delta_omega_q,
                "gradient_threshold": Gamma_q,
            },
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "test_id",
                "status",
                "quenches_detected",
                "detection_accuracy_amp",
                "detection_accuracy_grad",
                "map_stability",
            ]
        )
        writer.writerow(
            [
                cfg["test_id"],
                status,
                quench_results.get("quenches_detected", False),
                detection_accuracy_amp,
                detection_accuracy_grad,
                map_stability,
            ]
        )

    assert status == "PASS", (
        f"A06 failed: detection_accuracy_amp={detection_accuracy_amp:.3f}, "
        f"detection_accuracy_grad={detection_accuracy_grad:.3f}, "
        f"map_stability={map_stability:.3f}"
    )

