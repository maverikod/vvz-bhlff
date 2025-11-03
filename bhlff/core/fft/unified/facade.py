"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Facade class for unified spectral operations in 7D BHLFF Framework.
"""

from typing import Any, Dict, Tuple
import os
import logging
import numpy as np

from bhlff.utils.cuda_utils import get_global_backend, CPUBackend
from .plans import setup_fft_plans
from .volume import compute_volume_element
from .wave_vectors import get_wave_vectors, create_wave_vector_grid
from .filters import (
    create_lowpass_filter,
    create_highpass_filter,
    create_bandpass_filter,
)
from .fft_cpu import forward_fft_cpu, inverse_fft_cpu
from .fft_gpu import forward_fft_gpu, inverse_fft_gpu
from .blocked import forward_fft_blocked, inverse_fft_blocked


class UnifiedSpectralOperations:
    """
    Unified spectral operations for 7D phase field calculations.
    """

    def __init__(self, domain: "Domain", precision: str = "float64"):
        self.domain = domain
        self.precision = precision
        self.logger = logging.getLogger(__name__)
        self.backend = get_global_backend()
        # Deterministic Level-A behavior: force CPU backend for low-dimensional domains
        try:
            if hasattr(self.domain, "shape") and len(self.domain.shape) <= 3:
                self.backend = CPUBackend()
        except Exception:
            pass

        ratio_str = os.getenv("BHLFF_GPU_MEMORY_RATIO", "0.8")
        try:
            self._gpu_memory_ratio = float(min(max(float(ratio_str), 0.1), 0.95))
        except Exception:
            self._gpu_memory_ratio = 0.8

        self._fft_plans = setup_fft_plans()
        self.logger.info(
            f"UnifiedSpectralOperations initialized for domain {self.domain.shape}"
        )

    # Public API
    def forward_fft(
        self, field: np.ndarray, normalization: str = "ortho"
    ) -> np.ndarray:
        self._validate_shape(field.shape)
        if self._is_field_too_large(field):
            # try downcast
            try:
                field32 = field.astype(np.complex64, copy=False)
                if not self._is_field_too_large(field32):
                    out32 = self._forward_backend(field32, normalization)
                    return out32.astype(
                        np.complex128 if self.precision == "float64" else np.complex64
                    )
            except Exception:
                pass
            # blocked
            try:
                return forward_fft_blocked(
                    field, normalization, self.domain.shape, self._gpu_memory_ratio
                )
            except Exception:
                pass
            return forward_fft_cpu(field, normalization, self.domain.shape)
        return self._forward_backend(field, normalization)

    def inverse_fft(
        self, spectral_field: np.ndarray, normalization: str = "ortho"
    ) -> np.ndarray:
        self._validate_shape(spectral_field.shape)
        if self._is_field_too_large(spectral_field):
            try:
                spec32 = spectral_field.astype(np.complex64, copy=False)
                if not self._is_field_too_large(spec32):
                    real32 = self._inverse_backend(spec32, normalization)
                    return real32.astype(
                        np.float64 if self.precision == "float64" else np.float32
                    )
            except Exception:
                pass
            try:
                return inverse_fft_blocked(
                    spectral_field,
                    normalization,
                    self.domain.shape,
                    self._gpu_memory_ratio,
                )
            except Exception:
                pass
            return inverse_fft_cpu(spectral_field, normalization, self.domain.shape)
        return self._inverse_backend(spectral_field, normalization)

    def compute_spectral_derivatives(
        self, field: np.ndarray, order: int = 1
    ) -> Dict[str, np.ndarray]:
        field_spectral = self.forward_fft(field, "ortho")
        k_vectors = get_wave_vectors(self.domain.shape)
        derivatives: Dict[str, np.ndarray] = {}
        for i, (axis, k_vec) in enumerate(
            zip(["x", "y", "z", "phi1", "phi2", "phi3", "t"], k_vectors)
        ):
            if i < len(field.shape):
                k_grid = create_wave_vector_grid(k_vec, i, field.shape)
                derivative_spectral = (1j * k_grid) ** order * field_spectral
                derivatives[f"d{axis}"] = self.inverse_fft(derivative_spectral, "ortho")
        return derivatives

    def apply_spectral_filter(
        self, field: np.ndarray, filter_type: str, **kwargs
    ) -> np.ndarray:
        field_spectral = self.forward_fft(field, "ortho")
        if filter_type == "lowpass":
            cutoff = kwargs.get("cutoff", 0.5)
            mask = create_lowpass_filter(self.domain.shape, cutoff)
        elif filter_type == "highpass":
            cutoff = kwargs.get("cutoff", 0.5)
            mask = create_highpass_filter(self.domain.shape, cutoff)
        elif filter_type == "bandpass":
            mask = create_bandpass_filter(
                self.domain.shape,
                kwargs.get("low_cutoff", 0.3),
                kwargs.get("high_cutoff", 0.7),
            )
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        return self.inverse_fft(field_spectral * mask, "ortho")

    def compute_spectral_energy(self, field: np.ndarray) -> np.ndarray:
        field_spectral = self.forward_fft(field, "ortho")
        return np.abs(field_spectral) ** 2

    def get_spectral_info(self) -> Dict[str, Any]:
        return {
            "domain_shape": self.domain.shape,
            "precision": self.precision,
            "volume_element": compute_volume_element(self.domain.shape),
            "fft_plans": self._fft_plans,
            "wave_vectors": [len(k) for k in get_wave_vectors(self.domain.shape)],
        }

    def __repr__(self) -> str:
        return f"UnifiedSpectralOperations(domain={self.domain.shape}, precision={self.precision})"

    # Internals
    def _validate_shape(self, shape: Tuple[int, ...]) -> None:
        if shape != self.domain.shape:
            raise ValueError(
                f"Shape {shape} incompatible with domain {self.domain.shape}"
            )

    def _is_field_too_large(self, field: np.ndarray) -> bool:
        try:
            field_size_mb = field.nbytes / (1024**2)
            fft_memory_needed_mb = field_size_mb * 4
            if hasattr(self.backend, "get_memory_info"):
                memory_info = self.backend.get_memory_info()
                total_memory_mb = memory_info["total_memory"] / (1024**2)
                allowed_mb = total_memory_mb * self._gpu_memory_ratio
                return fft_memory_needed_mb > allowed_mb
            return False
        except Exception:
            return True

    def _forward_backend(self, field: np.ndarray, normalization: str) -> np.ndarray:
        if self.backend.__class__.__name__ == "CPUBackend":
            return forward_fft_cpu(field, normalization, self.domain.shape)
        return forward_fft_gpu(field, normalization, self.domain.shape)

    def _inverse_backend(
        self, spectral_field: np.ndarray, normalization: str
    ) -> np.ndarray:
        if self.backend.__class__.__name__ == "CPUBackend":
            return inverse_fft_cpu(spectral_field, normalization, self.domain.shape)
        return inverse_fft_gpu(spectral_field, normalization, self.domain.shape)
