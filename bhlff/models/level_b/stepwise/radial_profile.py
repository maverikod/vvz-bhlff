"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Radial profile computation for stepwise power law analysis.

This module implements radial profile computation methods for analyzing
the stepwise structure of phase fields, supporting both CPU and CUDA acceleration.

Theoretical Background:
    Radial profiles A(r) are computed by averaging field values over
    spherical shells centered at defects, enabling analysis of decay
    behavior and layer structure in 7D space-time.

Example:
    >>> profiler = RadialProfileComputer(use_cuda=True)
    >>> profile = profiler.compute(field, center)
"""

import numpy as np
from typing import Dict, List
import logging

# CUDA support
try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class RadialProfileComputer:
    """
    Radial profile computation with CUDA acceleration.

    Physical Meaning:
        Computes radial profiles by averaging field values over spherical
        shells, providing the basis for analyzing decay behavior and
        layer structure in the phase field.

    Mathematical Foundation:
        For a field a(x), the radial profile A(r) is computed as:
        A(r) = (1/V_r) ∫_{|x-c|=r} |a(x)| dS
        where V_r is the volume of the spherical shell at radius r.
    """

    def __init__(self, use_cuda: bool = True, gpu_memory_ratio: float = 0.8):
        """
        Initialize radial profile computer.

        Physical Meaning:
            Sets up computer with CUDA acceleration for efficient
            computation of radial profiles in 7D phase fields.

        Args:
            use_cuda (bool): Whether to use CUDA acceleration.
            gpu_memory_ratio (float): GPU memory utilization ratio (0-1).
        """
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.gpu_memory_ratio = gpu_memory_ratio
        self.logger = logging.getLogger(__name__)

        if self.use_cuda:
            self.xp = cp
            try:
                from ...utils.cuda_utils import get_global_backend

                self.backend = get_global_backend()
            except ImportError:
                self.backend = None
        else:
            self.xp = np
            self.backend = None

    def compute(self, field: np.ndarray, center: List[float]) -> Dict[str, np.ndarray]:
        """
        Compute radial profile of the field.

        Physical Meaning:
            Computes the radial profile A(r) by averaging the field
            over spherical shells centered at the defect.

        Args:
            field (np.ndarray): 3D or 7D field array.
            center (List[float]): Center coordinates [x, y, z].

        Returns:
            Dict[str, np.ndarray]: Radial profile with 'r' and 'A' arrays.
        """
        if self.use_cuda:
            return self._compute_cuda(field, center)
        else:
            return self._compute_cpu(field, center)

    def compute_substrate(
        self, substrate: np.ndarray, center: List[float]
    ) -> Dict[str, np.ndarray]:
        """
        Compute radial profile of substrate transparency.

        Physical Meaning:
            Computes the radial profile T(r) by averaging the substrate
            transparency over spherical shells centered at the defect
            using vectorized operations for efficiency.

        Args:
            substrate (np.ndarray): 7D substrate field.
            center (List[float]): Center coordinates [x, y, z].

        Returns:
            Dict[str, np.ndarray]: Radial profile with 'r' and 'A' arrays.
        """
        # Always use CUDA if enabled, converting numpy arrays to cupy
        use_cuda_here = self.use_cuda
        xp = self.xp if use_cuda_here else np
        
        # Convert numpy array to cupy if CUDA is enabled
        if use_cuda_here and isinstance(substrate, np.ndarray):
            substrate = xp.asarray(substrate)

        if len(substrate.shape) == 7:
            shape = substrate.shape[:3]
        else:
            shape = substrate.shape[:3]

        x = xp.arange(shape[0], dtype=xp.float32)
        y = xp.arange(shape[1], dtype=xp.float32)
        z = xp.arange(shape[2], dtype=xp.float32)
        X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

        center_array = xp.array(center, dtype=xp.float32)
        distances = xp.sqrt(
            (X - center_array[0]) ** 2
            + (Y - center_array[1]) ** 2
            + (Z - center_array[2]) ** 2
        )

        if len(substrate.shape) == 7:
            center_phi = substrate.shape[3] // 2
            center_t = substrate.shape[6] // 2
            transparency = xp.abs(
                substrate[:, :, :, center_phi, center_phi, center_phi, center_t]
            )
        else:
            transparency = xp.abs(substrate)

        r_max = float(xp.max(distances))
        num_bins = max(20, min(100, int(r_max * 10)))
        r_bins = xp.linspace(0.0, r_max, num_bins + 1)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0

        distances_flat = distances.ravel()
        transparency_flat = transparency.ravel()
        bin_indices = xp.searchsorted(r_bins[1:], distances_flat, side="right")
        bin_indices = xp.clip(bin_indices, 0, num_bins - 1)

        T_radial = xp.zeros(num_bins, dtype=xp.float32)
        if hasattr(xp, "bincount"):
            bin_sums = xp.bincount(
                bin_indices, weights=transparency_flat, minlength=num_bins
            )
            bin_counts = xp.bincount(bin_indices, minlength=num_bins)
            valid_mask = bin_counts > 0
            T_radial[valid_mask] = bin_sums[valid_mask] / bin_counts[valid_mask]
        else:
            for i in range(num_bins):
                mask = bin_indices == i
                if xp.any(mask):
                    T_radial[i] = xp.mean(transparency_flat[mask])

        # Always convert back to numpy for return
        if use_cuda_here:
            T_radial = cp.asnumpy(T_radial)
            r_centers = cp.asnumpy(r_centers)

        return {"r": r_centers, "A": T_radial}

    def _compute_cuda(
        self, field: np.ndarray, center: List[float]
    ) -> Dict[str, np.ndarray]:
        """
        Compute radial profile with CUDA acceleration.

        Physical Meaning:
            Computes radial profile A(r) using CUDA for efficient
            processing of large 7D fields.

        Args:
            field (np.ndarray): Field array (GPU or CPU).
            center (List[float]): Center coordinates [x, y, z].

        Returns:
            Dict[str, np.ndarray]: Radial profile with 'r' and 'A' arrays.
        """
        if len(field.shape) == 7:
            shape = field.shape[:3]
        else:
            shape = field.shape[:3]

        x = self.xp.arange(shape[0], dtype=self.xp.float32)
        y = self.xp.arange(shape[1], dtype=self.xp.float32)
        z = self.xp.arange(shape[2], dtype=self.xp.float32)
        X, Y, Z = self.xp.meshgrid(x, y, z, indexing="ij")

        center_array = self.xp.array(center, dtype=self.xp.float32)
        distances = self.xp.sqrt(
            (X - center_array[0]) ** 2
            + (Y - center_array[1]) ** 2
            + (Z - center_array[2]) ** 2
        )

        field_gpu = self.xp.asarray(field)
        if len(field.shape) == 7:
            center_phi = field.shape[3] // 2
            center_t = field.shape[6] // 2
            amplitude = self.xp.abs(
                field_gpu[:, :, :, center_phi, center_phi, center_phi, center_t]
            )
        else:
            amplitude = self.xp.abs(field_gpu)

        r_max = float(self.xp.max(distances))
        num_bins = min(100, max(20, int(r_max)))
        r_bins = self.xp.linspace(0.0, r_max, num_bins + 1)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0

        distances_flat = distances.ravel()
        amplitude_flat = amplitude.ravel()
        bin_indices = self.xp.searchsorted(r_bins[1:], distances_flat, side="right")
        bin_indices = self.xp.clip(bin_indices, 0, num_bins - 1)

        A_radial = self.xp.zeros(num_bins, dtype=self.xp.float32)
        if hasattr(self.xp, "bincount"):
            bin_sums = self.xp.bincount(
                bin_indices, weights=amplitude_flat, minlength=num_bins
            )
            bin_counts = self.xp.bincount(bin_indices, minlength=num_bins)
            valid_mask = bin_counts > 0
            A_radial[valid_mask] = bin_sums[valid_mask] / bin_counts[valid_mask]
        else:
            for i in range(num_bins):
                mask = bin_indices == i
                if self.xp.any(mask):
                    A_radial[i] = self.xp.mean(amplitude_flat[mask])

        # Always convert back to numpy for return
        if self.use_cuda:
            return {
                "r": cp.asnumpy(r_centers),
                "A": cp.asnumpy(A_radial),
            }
        return {"r": r_centers, "A": A_radial}

    def _compute_cpu(
        self, field: np.ndarray, center: List[float]
    ) -> Dict[str, np.ndarray]:
        """
        Compute radial profile using CPU.

        Physical Meaning:
            Computes radial profile A(r) using CPU operations,
            suitable for smaller fields or when CUDA is unavailable.

        Args:
            field (np.ndarray): Field array.
            center (List[float]): Center coordinates [x, y, z].

        Returns:
            Dict[str, np.ndarray]: Radial profile with 'r' and 'A' arrays.
        """
        if len(field.shape) == 7:
            shape = field.shape[:3]
        else:
            shape = field.shape[:3]

        x = np.arange(shape[0])
        y = np.arange(shape[1])
        z = np.arange(shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        distances = np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )

        if len(field.shape) == 7:
            center_phi = field.shape[3] // 2
            center_t = field.shape[6] // 2
            amplitude = np.abs(
                field[:, :, :, center_phi, center_phi, center_phi, center_t]
            )
        else:
            amplitude = np.abs(field)

        r_max = np.max(distances)
        r_bins = np.linspace(0, r_max, min(100, int(r_max)))
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2

        A_radial = []
        for i in range(len(r_bins) - 1):
            mask = (distances >= r_bins[i]) & (distances < r_bins[i + 1])
            if np.any(mask):
                A_radial.append(np.mean(amplitude[mask]))
            else:
                A_radial.append(0.0)

        return {"r": r_centers, "A": np.array(A_radial)}
