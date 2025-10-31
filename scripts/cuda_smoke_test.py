"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA smoke test script for Level E modules.

This script verifies that the virtual environment is active, CuPy sees the GPU,
and runs a small total energy computation using `SolitonEnergyCalculatorCUDA`.

Theoretical Background:
    Confirms that the CUDA backend and block processing path are operational
    for 7D phase-field energy calculations on a small synthetic field.

Example:
    Activate venv and run:
        $ . .venv/bin/activate
        $ python scripts/cuda_smoke_test.py --N 8 --phi 8 --t 8 -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA smoke test: GPU availability and small energy computation"
    )
    parser.add_argument("--N", type=int, default=8, help="Spatial grid size (per axis)")
    parser.add_argument("--phi", type=int, default=8, help="Phase grid size (per axis)")
    parser.add_argument("--t", type=int, default=8, help="Temporal grid size")
    parser.add_argument(
        "--precision",
        choices=["float64", "float32"],
        default="float64",
        help="Field dtype precision",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("cuda_smoke")

    # Ensure project root on sys.path
    sys.path.insert(0, ".")

    try:
        import cupy as cp

        cuda_ok = cp.cuda.is_available()
        dev_count = cp.cuda.runtime.getDeviceCount() if cuda_ok else 0
        mem_info = cp.cuda.Device(0).mem_info if cuda_ok else (0, 0)
        logger.info(
            "CuPy: %s | CUDA available: %s | devices: %s",
            cp.__version__,
            cuda_ok,
            dev_count,
        )
        if cuda_ok:
            logger.info("GPU[0] mem (free,total) bytes: %s", mem_info)
        else:
            print("[ERROR] CUDA not available in current environment.")
            return 2
    except Exception as e:  # pragma: no cover (env specific)
        print(f"[ERROR] CuPy import/initialization failed: {e}")
        return 2

    try:
        from bhlff.core.domain import Domain
        from bhlff.models.level_e.cuda import SolitonEnergyCalculatorCUDA
    except Exception as e:
        print(f"[ERROR] Project imports failed: {e}")
        return 2

    N = int(args.N)
    N_phi = int(args.phi)
    N_t = int(args.t)
    dtype = np.complex128 if args.precision == "float64" else np.complex64

    # Domain and parameters
    domain = Domain(L=1.0, N=N, dimensions=7, N_phi=N_phi, N_t=N_t, T=1.0)
    physics_params: Dict[str, Any] = {
        "mu": 1.0,
        "beta": 1.0,
        "lambda": 0.1,
        "S4": 0.1,
        "S6": 0.01,
        "F2": 1.0,
        "N_c": 3,
    }

    print("[INFO] Creating small synthetic 7D field...")
    shape = (N, N, N, N_phi, N_phi, N_phi, N_t)
    rng = np.random.default_rng(42)
    field = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)

    print("[INFO] Initializing CUDA energy calculator (with block processing)...")
    calc = SolitonEnergyCalculatorCUDA(domain, physics_params, use_cuda=True)

    print("[INFO] Computing total energy (this will print progress if verbose)...")
    total_energy = calc.compute_total_energy(field)
    print(f"[RESULT] Total energy: {float(total_energy):.6e}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
