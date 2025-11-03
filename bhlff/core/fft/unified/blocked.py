"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Blocked processing utilities for unified spectral operations to fit GPU memory.
"""

from typing import Tuple
import numpy as np
from bhlff.utils.cuda_utils import get_global_backend
from .fft_gpu import forward_fft_gpu, inverse_fft_gpu


def compute_block_size(array: np.ndarray, gpu_memory_ratio: float) -> int:
    backend = get_global_backend()
    if not hasattr(backend, "get_memory_info"):
        return array.shape[-1]
    mem = backend.get_memory_info()
    allowed = int(mem["total_memory"] * gpu_memory_ratio)
    slice_bytes = array[..., 0].nbytes
    if slice_bytes == 0:
        return array.shape[-1]
    max_slices = max(1, allowed // (slice_bytes * 4))
    return int(min(array.shape[-1], max_slices))


def forward_fft_blocked(
    field: np.ndarray,
    normalization: str,
    domain_shape: Tuple[int, ...],
    gpu_memory_ratio: float,
) -> np.ndarray:
    t_len = field.shape[-1]
    block = compute_block_size(field, gpu_memory_ratio)
    out = np.empty_like(field)
    start = 0
    while start < t_len:
        end = min(t_len, start + block)
        slab = field[..., start:end]
        # Use per-slab domain shape to keep normalization consistent
        slab_shape = tuple(list(domain_shape[:-1]) + [slab.shape[-1]])
        out[..., start:end] = forward_fft_gpu(slab, normalization, slab_shape)
        start = end
    return out


def inverse_fft_blocked(
    spectral_field: np.ndarray,
    normalization: str,
    domain_shape: Tuple[int, ...],
    gpu_memory_ratio: float,
) -> np.ndarray:
    t_len = spectral_field.shape[-1]
    block = compute_block_size(spectral_field, gpu_memory_ratio)
    out = np.empty_like(spectral_field)
    start = 0
    while start < t_len:
        end = min(t_len, start + block)
        slab = spectral_field[..., start:end]
        # Use per-slab domain shape to keep normalization consistent
        slab_shape = tuple(list(domain_shape[:-1]) + [slab.shape[-1]])
        out[..., start:end] = inverse_fft_gpu(slab, normalization, slab_shape)
        start = end
    return out
