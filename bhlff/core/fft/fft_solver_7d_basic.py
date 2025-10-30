"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic 7D FFT solver (full-array FFT) for the fractional Riesz operator.

Brief description of the module's purpose and its role in the 7D phase field theory.

Detailed description of the module's functionality, including:
- Physical meaning and theoretical background
- Key algorithms and methods implemented
- Dependencies and relationships with other modules
- Usage examples and typical workflows

Theoretical Background:
    Solves the stationary 7D fractional Riesz equation in spectral space:
    â(k) = ŝ(k) / (μ|k|^(2β) + λ) with orthonormal FFT normalization.

Example:
    >>> solver = FFTSolver7DBasic(domain, {"mu": 1.0, "beta": 1.0, "lambda": 0.0})
    >>> sol = solver.solve_stationary(source)
"""

from typing import Dict, Any
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
    cp = None  # type: ignore


class FFTSolver7DBasic:
    """Full-array FFT-based 7D solver (no batching)."""

    def __init__(self, domain: "Domain", parameters: Dict[str, Any]):
        self.domain = domain
        self.mu = float(parameters.get("mu", 1.0))
        self.beta = float(parameters.get("beta", 1.0))
        self.lmbda = float(parameters.get("lambda", 0.0))
        self.use_cuda = bool(parameters.get("use_cuda", True)) and CUDA_AVAILABLE
        self._xp = cp if self.use_cuda else np
        self._coeffs = None  # type: ignore
        self._build_spectral_coefficients()

    def solve_stationary(self, source_field: np.ndarray) -> np.ndarray:
        xp = self._xp
        src = xp.asarray(source_field) if self.use_cuda else source_field
        s_hat = xp.fft.fftn(src, norm="ortho")
        a_hat = s_hat / self._coeffs
        a = xp.fft.ifftn(a_hat, norm="ortho").real
        return cp.asnumpy(a) if self.use_cuda else a

    def get_spectral_coefficients(self) -> np.ndarray:
        return cp.asnumpy(self._coeffs) if self.use_cuda else self._coeffs  # type: ignore

    def _build_spectral_coefficients(self) -> None:
        xp = self._xp
        N = self.domain.N
        Np = self.domain.N_phi
        Nt = self.domain.N_t
        kx = xp.fft.fftfreq(N)
        ky = xp.fft.fftfreq(N)
        kz = xp.fft.fftfreq(N)
        p = xp.fft.fftfreq(Np) * (2 * xp.pi)
        kt = xp.fft.fftfreq(Nt)
        KX, KY, KZ = xp.meshgrid(kx, ky, kz, indexing="ij")
        P1, P2, P3 = xp.meshgrid(p, p, p, indexing="ij")
        KX7 = KX[:, :, :, None, None, None, None]
        KY7 = KY[:, :, :, None, None, None, None]
        KZ7 = KZ[:, :, :, None, None, None, None]
        P17 = P1[None, None, None, :, None, None, None]
        P27 = P2[None, None, None, None, :, None, None]
        P37 = P3[None, None, None, None, None, :, None]
        KT7 = kt[None, None, None, None, None, None, :]
        k2 = KX7 * KX7 + KY7 * KY7 + KZ7 * KZ7 + P17 * P17 + P27 * P27 + P37 * P37 + KT7 * KT7
        abs_k_2beta = xp.power(k2 + 0.0, self.beta)
        D = self.mu * abs_k_2beta + self.lmbda
        if self.lmbda == 0.0:
            D[(k2 == 0)] = 1.0
        self._coeffs = D.astype(xp.float64)

