"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

GPU (CUDA) FFT helpers with physics and ortho normalization.
"""

from typing import Tuple
import numpy as np
from bhlff.utils.cuda_utils import get_global_backend
from .volume import compute_volume_element


def forward_fft_gpu(
    field: np.ndarray, normalization: str, domain_shape: Tuple[int, ...]
) -> np.ndarray:
    backend = get_global_backend()
    field_gpu = backend.array(field)
    axes = tuple(range(field.ndim)) if field.ndim > 0 else None
    spec_gpu = backend.fft(field_gpu, axes=axes)
    n_total = np.prod(domain_shape)
    spec_gpu /= np.sqrt(n_total)
    if normalization == "physics":
        spec_gpu *= compute_volume_element(domain_shape)
    elif normalization != "ortho":
        raise ValueError(f"Unsupported normalization type: {normalization}")
    return backend.to_numpy(spec_gpu)


def inverse_fft_gpu(
    spectral_field: np.ndarray, normalization: str, domain_shape: Tuple[int, ...]
) -> np.ndarray:
    backend = get_global_backend()
    spec_gpu = backend.array(spectral_field)
    axes = tuple(range(spectral_field.ndim)) if spectral_field.ndim > 0 else None
    if normalization == "ortho":
        spec_gpu = spec_gpu * np.sqrt(np.prod(domain_shape))
        out_gpu = backend.ifft(spec_gpu, axes=axes)
        return backend.to_numpy(out_gpu)
    if normalization == "physics":
        vol = compute_volume_element(domain_shape)
        spec_gpu = spec_gpu / vol * np.sqrt(np.prod(domain_shape))
        out = backend.to_numpy(backend.ifft(spec_gpu, axes=axes))
        first = out.reshape(-1)[0]
        if np.abs(first) > 0:
            out = out * np.exp(-1j * np.angle(first))
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out
    raise ValueError(f"Unsupported normalization type: {normalization}")
