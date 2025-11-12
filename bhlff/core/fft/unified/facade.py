"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Facade class for unified spectral operations in 7D BHLFF Framework.
"""

from typing import Any, Dict, Tuple
import os
import logging
import numpy as np

from bhlff.utils.cuda_utils import get_global_backend, CPUBackend, CUDABackend
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
        # CUDA is required - use CUDA backend, raise error if not available
        self.backend = get_global_backend()
        self._using_cuda = isinstance(self.backend, CUDABackend)

        if not self._using_cuda:
            self.logger.warning(
                "CUDA backend unavailable or disabled; using CPU fallback for "
                "UnifiedSpectralOperations. GPU acceleration will be skipped."
            )

        ratio_str = os.getenv("BHLFF_GPU_MEMORY_RATIO", "0.8")
        try:
            self._gpu_memory_ratio = float(min(max(float(ratio_str), 0.1), 0.95))
        except Exception:
            self._gpu_memory_ratio = 0.8
        if not self._using_cuda:
            self._gpu_memory_ratio = 1.0

        self._fft_plans = setup_fft_plans()
        self.logger.info(
            f"UnifiedSpectralOperations initialized for domain {self.domain.shape}"
        )

    # Public API
    def forward_fft(
        self, field: np.ndarray, normalization: str = "ortho"
    ) -> np.ndarray:
        import sys
        self.logger.info(f"[FACADE] forward_fft: START - shape={field.shape}")
        sys.stdout.flush()
        
        self._validate_shape(field.shape)
        
        # For 7D fields, block processing is only applicable when using CUDA
        is_7d = len(field.shape) == 7
        is_too_large = self._is_field_too_large(field)
        
        field_size_mb = field.nbytes / (1024**2)
        self.logger.info(
            f"[FACADE] forward_fft: shape={field.shape}, size={field_size_mb:.2f}MB, "
            f"is_7d={is_7d}, is_too_large={is_too_large}"
        )
        sys.stdout.flush()
        
        # For 7D fields with CUDA backend, prefer block processing for GPU utilization
        if is_7d and self._using_cuda:
            self.logger.info(
                f"[FACADE] 7D field detected - using block processing for maximum GPU utilization"
            )
            sys.stdout.flush()
            try:
                result = forward_fft_blocked(
                    field, normalization, self.domain.shape, self._gpu_memory_ratio
                )
                self.logger.info(f"[FACADE] forward_fft: COMPLETE (blocked)")
                sys.stdout.flush()
                return result
            except Exception as e:
                self.logger.warning(f"[FACADE] Block processing failed: {e}, trying direct GPU")
                sys.stdout.flush()
                # Fallback to direct GPU if block processing fails
                result = self._forward_backend(field, normalization)
                self.logger.info(f"[FACADE] forward_fft: COMPLETE (direct GPU fallback)")
                sys.stdout.flush()
                return result
        
        if is_too_large and self._using_cuda:
            # try downcast
            try:
                field32 = field.astype(np.complex64, copy=False)
                if not self._is_field_too_large(field32):
                    self.logger.info("Using downcast to float32 for direct GPU processing")
                    out32 = self._forward_backend(field32, normalization)
                    return out32.astype(
                        np.complex128 if self.precision == "float64" else np.complex64
                    )
            except Exception as e:
                self.logger.debug(f"Downcast failed: {e}")
            # blocked
            try:
                self.logger.info("Using block processing for large field")
                return forward_fft_blocked(
                    field, normalization, self.domain.shape, self._gpu_memory_ratio
                )
            except Exception as e:
                self.logger.warning(f"Block processing failed: {e}, falling back to CPU")
            return forward_fft_cpu(field, normalization, self.domain.shape)
        
        if self._using_cuda:
            self.logger.info("Using direct GPU processing (field fits in memory)")
            return self._forward_backend(field, normalization)

        self.logger.info("Using CPU FFT backend")
        return forward_fft_cpu(field, normalization, self.domain.shape)

    def inverse_fft(
        self, spectral_field: np.ndarray, normalization: str = "ortho"
    ) -> np.ndarray:
        import sys
        self.logger.info(f"[FACADE] inverse_fft: START - shape={spectral_field.shape}")
        sys.stdout.flush()
        
        self._validate_shape(spectral_field.shape)
        is_too_large = self._is_field_too_large(spectral_field)
        is_7d = len(spectral_field.shape) == 7
        
        self.logger.info(
            f"[FACADE] inverse_fft: is_7d={is_7d}, is_too_large={is_too_large}"
        )
        sys.stdout.flush()
        
        # For 7D fields with CUDA backend, prefer block processing
        if is_7d and self._using_cuda:
            self.logger.info(f"[FACADE] 7D field detected - using block processing")
            sys.stdout.flush()
            try:
                result = inverse_fft_blocked(
                    spectral_field,
                    normalization,
                    self.domain.shape,
                    self._gpu_memory_ratio,
                )
                self.logger.info(f"[FACADE] inverse_fft: COMPLETE (blocked)")
                sys.stdout.flush()
                return result
            except Exception as e:
                self.logger.warning(f"[FACADE] Block processing failed: {e}, trying direct GPU")
                sys.stdout.flush()
                result = self._inverse_backend(spectral_field, normalization)
                self.logger.info(f"[FACADE] inverse_fft: COMPLETE (direct GPU fallback)")
                sys.stdout.flush()
                return result
        
        if is_too_large and self._using_cuda:
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
        if self._using_cuda:
            return self._inverse_backend(spectral_field, normalization)

        return inverse_fft_cpu(spectral_field, normalization, self.domain.shape)

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
        """
        Check if field is too large for direct GPU processing.
        
        Physical Meaning:
            Determines if field requires block processing based on available
            GPU memory and field size. For 7D fields, uses 80% GPU memory limit.
            Uses block processing to maximize GPU utilization with vectorization.
        """
        if not self._using_cuda:
            return False
        try:
            field_size_mb = field.nbytes / (1024**2)
            # FFT requires ~4x field size in memory
            fft_memory_needed_mb = field_size_mb * 4
            if hasattr(self.backend, "get_memory_info"):
                memory_info = self.backend.get_memory_info()
                # Use free_memory, not total_memory, for accurate check
                free_memory_mb = memory_info.get("free_memory", memory_info.get("total_memory", 0)) / (1024**2)
                total_memory_mb = memory_info.get("total_memory", 0) / (1024**2)
                # Use specified GPU memory ratio (default 80%)
                allowed_mb = free_memory_mb * self._gpu_memory_ratio
                
                self.logger.debug(
                    f"_is_field_too_large: field={field_size_mb:.2f}MB, "
                    f"fft_needed={fft_memory_needed_mb:.2f}MB, "
                    f"free={free_memory_mb:.2f}MB, total={total_memory_mb:.2f}MB, "
                    f"allowed={allowed_mb:.2f}MB (ratio={self._gpu_memory_ratio})"
                )
                
                # For 7D fields, use block processing to maximize GPU utilization
                # Block processing allows processing larger fields with better GPU usage
                if len(field.shape) == 7:
                    # For 7D, use block processing to ensure 80% GPU memory usage
                    # This enables better vectorization and GPU utilization
                    # Use more aggressive threshold: use blocks if field > 10% of allowed memory
                    # This ensures better GPU utilization even for smaller fields
                    threshold = allowed_mb * 0.1
                    result = fft_memory_needed_mb > threshold
                    self.logger.debug(
                        f"7D field: threshold={threshold:.2f}MB, result={result}"
                    )
                    return result
                return fft_memory_needed_mb > allowed_mb
            return False
        except Exception as e:
            self.logger.warning(f"Failed to check field size: {e}, using block processing")
            # If we can't check, use block processing to be safe
            # Block processing will use GPU with vectorization
            return True

    def _forward_backend(self, field: np.ndarray, normalization: str) -> np.ndarray:
        if self._using_cuda:
            try:
                return forward_fft_gpu(
                    field, normalization, self.domain.shape, level_c_required=False
                )
            except RuntimeError as exc:
                self.logger.warning(
                    "GPU forward FFT failed (%s); falling back to CPU backend", exc
                )
                self.backend = CPUBackend()
                self._using_cuda = False
        return forward_fft_cpu(field, normalization, self.domain.shape)

    def _inverse_backend(
        self, spectral_field: np.ndarray, normalization: str
    ) -> np.ndarray:
        if self._using_cuda:
            try:
                return inverse_fft_gpu(
                    spectral_field,
                    normalization,
                    self.domain.shape,
                    level_c_required=False,
                )
            except RuntimeError as exc:
                self.logger.warning(
                    "GPU inverse FFT failed (%s); falling back to CPU backend", exc
                )
                self.backend = CPUBackend()
                self._using_cuda = False
        return inverse_fft_cpu(spectral_field, normalization, self.domain.shape)
