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

    def __init__(self, domain: "Domain", parameters: Any):
        self.domain = domain

        # Support both dict-like parameters and dataclass-style Parameters7DBVP
        if isinstance(parameters, dict):
            mu = parameters.get("mu", 1.0)
            beta = parameters.get("beta", 1.0)
            lambda_param = parameters.get("lambda", parameters.get("lambda_param", 0.0))
            use_cuda_flag = parameters.get("use_cuda", True)
        else:
            # Fallback to attribute extraction
            mu = getattr(parameters, "mu", 1.0)
            beta = getattr(parameters, "beta", 1.0)
            lambda_param = getattr(parameters, "lambda_param", getattr(parameters, "lambda", 0.0))
            use_cuda_flag = getattr(parameters, "use_cuda", True)

        self.mu = float(mu)
        self.beta = float(beta)
        self.lmbda = float(lambda_param)
        self.use_cuda = bool(use_cuda_flag) and CUDA_AVAILABLE
        self._xp = cp if self.use_cuda else np
        self._coeffs = None  # type: ignore
        self._build_spectral_coefficients()

    def solve_stationary(self, source_field: np.ndarray) -> np.ndarray:
        xp = self._xp
        src = xp.asarray(source_field) if self.use_cuda else source_field
        # Validate shape
        if src is None:
            raise ValueError("source_field must not be None")
        if tuple(src.shape) != tuple(getattr(self.domain, "shape")):
            raise ValueError(
                f"Source shape {src.shape} incompatible with domain shape {self.domain.shape}"
            )
        s_hat = xp.fft.fftn(src, norm="ortho")
        a_hat = s_hat / self._coeffs
        a = xp.fft.ifftn(a_hat, norm="ortho").real
        return cp.asnumpy(a) if self.use_cuda else a

    # Backward-compatible API expected by tests
    def solve(self, source_field: np.ndarray) -> np.ndarray:
        return self.solve_stationary(source_field)

    def get_spectral_coefficients(self) -> np.ndarray:
        return cp.asnumpy(self._coeffs) if self.use_cuda else self._coeffs  # type: ignore

    def get_info(self) -> dict:
        """
        Return solver diagnostic information.

        Returns:
            dict: Basic metadata about domain and parameters.
        """
        return {
            "solver_type": "FFTSolver7DBasic",
            "domain_shape": tuple(getattr(self.domain, "shape")),
            "mu": self.mu,
            "beta": self.beta,
            "lambda": self.lmbda,
            "use_cuda": self.use_cuda,
        }

    def _build_spectral_coefficients(self) -> None:
        xp = self._xp
        # Support both classic Domain (N, N_phi, N_t) and Domain7DBVP (N_spatial, N_phase, N_t)
        N = getattr(self.domain, "N", getattr(self.domain, "N_spatial", None))
        Np = getattr(self.domain, "N_phi", getattr(self.domain, "N_phase", None))
        Nt = getattr(self.domain, "N_t", None)
        if N is None or Np is None or Nt is None:
            raise AttributeError(
                "Domain must define (N or N_spatial), (N_phi or N_phase), and N_t"
            )
        # Physical wave numbers with proper spacing (cycles per unit)
        dx = getattr(self.domain, "dx", 1.0)
        dphi = getattr(self.domain, "dphi", (2 * xp.pi) / Np)
        dt = getattr(self.domain, "dt", 1.0)

        kx = xp.fft.fftfreq(N, d=dx)
        ky = xp.fft.fftfreq(N, d=dx)
        kz = xp.fft.fftfreq(N, d=dx)
        p = xp.fft.fftfreq(Np, d=dphi)
        kt = xp.fft.fftfreq(Nt, d=dt)

        KX7 = kx[:, None, None, None, None, None, None]
        KY7 = ky[None, :, None, None, None, None, None]
        KZ7 = kz[None, None, :, None, None, None, None]
        P17 = p[None, None, None, :, None, None, None]
        P27 = p[None, None, None, None, :, None, None]
        P37 = p[None, None, None, None, None, :, None]
        KT7 = kt[None, None, None, None, None, None, :]

        k2 = (
            KX7 * KX7
            + KY7 * KY7
            + KZ7 * KZ7
            + P17 * P17
            + P27 * P27
            + P37 * P37
            + KT7 * KT7
        )
        abs_k_2beta = xp.power(k2 + 0.0, self.beta)
        D = self.mu * abs_k_2beta + self.lmbda
        if self.lmbda == 0.0:
            D[(k2 == 0)] = 1.0
        self._coeffs = D.astype(xp.float64)
