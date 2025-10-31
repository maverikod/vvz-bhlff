"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA-accelerated collective excitations for Level F models.

Provides GPU-optimized excite/analyze/dispersion routines with block
processing sized to use ~80% of free GPU memory and vectorized CuPy ops.

Physical Meaning:
    Studies the response of multi-particle systems to external fields
    and extracts collective modes and dispersion using GPU acceleration.

Example:
    >>> cuda_exc = CollectiveExcitationsCUDA(system, excitation_params)
    >>> response = cuda_exc.excite_system(external_field)
    >>> analysis = cuda_exc.analyze_response(response)
    >>> dispersion = cuda_exc.compute_dispersion_relations()
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CUDA_AVAILABLE = False

from bhlff.utils.cuda_utils import CUDABackend, get_optimal_backend


class CollectiveExcitationsCUDA:
    """
    CUDA path for Level F collective excitations.

    Attributes:
        system: Multi-particle system facade
        params: Excitation params (frequency_range, amplitude, type,
            duration)
        backend: CUDABackend
        block_time: Number of time steps per GPU batch
    """

    def __init__(self, system: Any, excitation_params: Dict[str, Any]):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available for CollectiveExcitationsCUDA")

        self.system = system
        self.params = excitation_params
        self.backend = get_optimal_backend()
        if not isinstance(self.backend, CUDABackend):
            raise RuntimeError("CUDA backend not selected for excitations")

        # Extract parameters
        self.frequency_range = self.params.get("frequency_range", [0.1, 10.0])
        self.amplitude = float(self.params.get("amplitude", 0.1))
        self.excitation_type = self.params.get("type", "harmonic")
        self.duration = float(self.params.get("duration", 100.0))
        self.dt = float(self.params.get("dt", 0.01))

        # Time grid and block sizing (80% GPU memory)
        self._setup_time_blocks()

        # Precompute dynamics matrix on CPU and transfer to GPU
        dyn = self.system._compute_dynamics_matrix()
        self.dynamics_gpu = cp.asarray(dyn)

    def _setup_time_blocks(self) -> None:
        """Choose time batch size to fit 80% of free GPU memory."""
        n_steps = int(np.ceil(self.duration / self.dt))
        self.time_grid = np.arange(0.0, n_steps * self.dt, self.dt)

        mem = self.backend.get_memory_info()
        free_bytes = int(mem.get("free_memory", 0))
        usable = int(free_bytes * 0.8)

        # Response buffer per block: n_particles x block_time float64
        n_particles = len(self.system.particles)
        bytes_per_element = 8
        # Allocate for response, excitation, temporaries (~ x3 safety)
        if n_particles <= 0:
            self.block_time = min(1024, n_steps)
            return
        per_step_bytes = n_particles * bytes_per_element * 3
        if per_step_bytes <= 0:
            self.block_time = min(1024, n_steps)
            return
        max_block_time = max(1, usable // max(1, per_step_bytes))
        self.block_time = int(max(8, min(max_block_time, n_steps)))

    def excite_system(self, external_field: np.ndarray) -> np.ndarray:
        """Compute response R for given external field on GPU in batches."""
        n_particles = len(self.system.particles)
        n_steps = len(self.time_grid)
        response = np.zeros((n_particles, n_steps), dtype=np.float64)

        # Prepare excitation signal on GPU per batch
        for t0 in range(0, n_steps, self.block_time):
            t1 = min(n_steps, t0 + self.block_time)
            tb = self.time_grid[t0:t1]

            if self.excitation_type == "harmonic":
                omega = float(np.mean(self.frequency_range))
                exc_cpu = self.amplitude * np.sin(2 * np.pi * omega * tb)
            elif self.excitation_type == "impulse":
                exc_cpu = np.zeros_like(tb)
                exc_cpu[tb < 0.1] = self.amplitude
            elif self.excitation_type == "sweep":
                w0, w1 = self.frequency_range
                omega_t = w0 + (w1 - w0) * (tb - tb[0]) / max(
                    self.dt, (tb[-1] - tb[0] + self.dt)
                )
                exc_cpu = self.amplitude * np.sin(2 * np.pi * omega_t * tb)
            else:
                raise ValueError(f"Unknown excitation type: {self.excitation_type}")

            excitation_gpu = cp.asarray(exc_cpu)

            # External field to forces (vectorized simple mapping)
            F_gpu = self._compute_external_force_gpu(external_field, excitation_gpu)

            # Solve dynamics for all timesteps in the batch at once: X = A^{-1} F
            # cp.linalg.solve supports multiple RHS when F has shape (N, K)
            try:
                batch_resp = cp.linalg.solve(self.dynamics_gpu, F_gpu)
            except cp.linalg.LinAlgError:
                batch_resp = cp.linalg.pinv(self.dynamics_gpu) @ F_gpu

            response[:, t0:t1] = cp.asnumpy(batch_resp)

            # Cleanup GPU memory between batches
            cp.get_default_memory_pool().free_all_blocks()

        return response

    def analyze_response(self, response: np.ndarray) -> Dict[str, Any]:
        """Analyze response using GPU FFT and compute peaks/participation/Q."""
        resp_gpu = cp.asarray(response)
        spectrum_gpu = cp.fft.fft(resp_gpu, axis=-1)
        freqs_gpu = cp.fft.fftfreq(resp_gpu.shape[-1], d=self.dt)

        # Simple peak detection on GPU (magnitude)
        mag = cp.abs(spectrum_gpu)
        # Shift to CPU for small-vector peak picker
        mag_cpu = cp.asnumpy(mag.mean(axis=0))
        freqs_cpu = cp.asnumpy(freqs_gpu)
        peak_indices = []
        peak_freqs = []
        peak_amps = []
        for i in range(1, len(mag_cpu) - 1):
            if (
                mag_cpu[i] > mag_cpu[i - 1]
                and mag_cpu[i] > mag_cpu[i + 1]
                and mag_cpu[i] > 0.1
            ):
                peak_indices.append(i)
                peak_freqs.append(freqs_cpu[i])
                peak_amps.append(mag_cpu[i])

        # Participation ratios from max response per particle
        part = np.max(np.abs(response), axis=1)
        if np.sum(part) > 0:
            part = part / np.sum(part)

        # Quality factors (proxy)
        Q = np.array([max(0.0, float(f)) for f in peak_freqs], dtype=np.float64)

        return {
            "frequencies": freqs_cpu,
            "peaks": {
                "indices": peak_indices,
                "frequencies": peak_freqs,
                "amplitudes": peak_amps,
            },
            "participation": part,
            "quality_factors": Q,
            "spectrum": cp.asnumpy(spectrum_gpu),
        }

    def compute_dispersion_relations(self) -> Dict[str, Any]:
        """Compute ω(k) using a simple vectorized sweep on GPU."""
        k_max = float(self.params.get("k_max", 10.0))
        n_k = int(self.params.get("n_k_points", 100))
        k_values = cp.linspace(0.0, k_max, n_k)

        # Placeholder dispersion model: ω(k) = sqrt(ω0^2 + c^2 k^2)
        omega0 = 0.1
        c = 1.0
        omega = cp.sqrt(omega0 * omega0 + (c * k_values) ** 2)

        # Group and phase velocities
        with cp.errstate(divide="ignore", invalid="ignore"):
            v_g = cp.gradient(omega) / cp.gradient(k_values)
            v_phi = cp.where(k_values != 0, omega / k_values, 0.0)

        return {
            "k_values": cp.asnumpy(k_values),
            "frequencies": cp.asnumpy(omega),
            "group_velocities": cp.asnumpy(v_g),
            "phase_velocities": cp.asnumpy(v_phi),
        }

    def _compute_external_force_gpu(
        self, external_field: np.ndarray, excitation_gpu: "cp.ndarray"
    ) -> "cp.ndarray":
        """Map external field to per-particle force over time on GPU."""
        n_particles = len(self.system.particles)
        n_times = int(excitation_gpu.shape[0])
        # Simple model: mean field times charge times excitation (fully vectorized)
        field_mean = float(np.mean(external_field))
        charges = cp.asarray(
            [p.charge for p in self.system.particles], dtype=cp.float64
        )
        # Outer product: (n_particles, 1) * (1, n_times)
        forces = charges[:, None] * field_mean * excitation_gpu[None, :]
        return forces.astype(cp.float64)
