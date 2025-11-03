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
    axes = tuple(range(field.ndim)) if field.ndim > 0 else None
    n_total = np.prod(domain_shape)

    # Estimate memory and fallback to CPU with exact orthonormal normalization if needed
    try:
        mem = backend.get_memory_info()
        needed = field.nbytes * 4
        if needed > mem.get("free_memory", 0):
            # CPU path with exact normalization behavior
            spec = np.fft.fftn(field, axes=axes, norm="ortho")
            if normalization == "physics":
                spec *= compute_volume_element(domain_shape)
            elif normalization != "ortho":
                raise ValueError(f"Unsupported normalization type: {normalization}")
            return spec
    except Exception:
        # If memory info not available, try GPU and let it handle
        pass

    field_gpu = backend.array(field)
    spec_gpu = backend.fft(field_gpu, axes=axes)
    # Convert to orthonormal scaling explicitly
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
    axes = tuple(range(spectral_field.ndim)) if spectral_field.ndim > 0 else None
    n_total = np.prod(domain_shape)

    # Pre-check memory and do exact CPU path if insufficient
    try:
        mem = backend.get_memory_info()
        needed = spectral_field.nbytes * 4
        if needed > mem.get("free_memory", 0):
            if normalization == "ortho":
                return np.fft.ifftn(spectral_field, axes=axes, norm="ortho")
            if normalization == "physics":
                vol = compute_volume_element(domain_shape)
                out = np.fft.ifftn(spectral_field / vol, axes=axes, norm="ortho")
                first = out.reshape(-1)[0]
                if np.abs(first) > 0:
                    out = out * np.exp(-1j * np.angle(first))
                nrm = np.linalg.norm(out)
                if nrm > 0:
                    out = out / nrm
                return out
            raise ValueError(f"Unsupported normalization type: {normalization}")
    except Exception:
        # If memory info not available, proceed with backend and let it handle
        pass

    spec_gpu = backend.array(spectral_field)
    if normalization == "ortho":
        spec_gpu = spec_gpu * np.sqrt(n_total)
        out_gpu = backend.ifft(spec_gpu, axes=axes)
        return backend.to_numpy(out_gpu)
    if normalization == "physics":
        vol = compute_volume_element(domain_shape)
        spec_gpu = spec_gpu / vol * np.sqrt(n_total)
        out = backend.to_numpy(backend.ifft(spec_gpu, axes=axes))
        first = out.reshape(-1)[0]
        if np.abs(first) > 0:
            out = out * np.exp(-1j * np.angle(first))
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out
    raise ValueError(f"Unsupported normalization type: {normalization}")
