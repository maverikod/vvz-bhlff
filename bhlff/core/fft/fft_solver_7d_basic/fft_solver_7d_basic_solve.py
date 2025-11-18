"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Solving methods for FFT solver 7D basic.

This module provides solving methods as a mixin class.
"""

from typing import Union
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)


class FFTSolver7DBasicSolveMixin:
    """Mixin providing solving methods."""
    
    def solve_stationary(self, source_field: Union[np.ndarray, 'FieldArray']) -> 'FieldArray':
        """
        Solve stationary phase field equation.
        
        Physical Meaning:
            Solves the fractional Laplacian equation L_β a = s for given source,
            returning solution as FieldArray that may be swapped to disk.
        """
        logger.info(f"[SOLVER] solve_stationary: ENTRY - source_field type={type(source_field)}")
        sys.stdout.flush()
        sys.stderr.flush()
        
        from ...arrays.field_array import FieldArray
        
        logger.info(f"[SOLVER] solve_stationary: STEP 0.1: Preparing source field...")
        sys.stdout.flush()
        sys.stderr.flush()
        
        # For FieldArray, pass it directly to forward_fft to enable streaming
        # For regular numpy arrays, convert to complex128
        if isinstance(source_field, FieldArray):
            source_for_fft = source_field
            source_array = source_field.array
            logger.info(f"[SOLVER] solve_stationary: STEP 0.1 COMPLETE: Using FieldArray directly for streaming")
        else:
            source_array = source_field
            # Convert to complex128 for FFT
            if not np.iscomplexobj(source_array):
                source_for_fft = np.asarray(source_array, dtype=np.complex128)
            else:
                source_for_fft = np.asarray(source_array, dtype=np.complex128)
            logger.info(f"[SOLVER] solve_stationary: STEP 0.1 COMPLETE: Using as numpy array")
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Validate shape
        if source_array is None:
            raise ValueError("source_field must not be None")
        if tuple(source_array.shape) != tuple(getattr(self.domain, "shape")):
            raise ValueError(
                f"Source shape {source_array.shape} incompatible with domain shape {self.domain.shape}"
            )
        
        # Always operate through unified ops (handles GPU/CPU and OOM fallback)
        source_size_mb = source_array.nbytes / (1024**2)
        logger.info(
            f"[SOLVER] solve_stationary: START - source {source_array.shape} "
            f"({source_size_mb:.2f}MB), use_cuda={self.use_cuda}, "
            f"is_FieldArray={isinstance(source_field, FieldArray)}"
        )
        sys.stdout.flush()
        
        logger.info(f"[SOLVER] STEP 1: Performing forward FFT...")
        sys.stdout.flush()
        # Pass FieldArray directly to enable streaming for swapped fields
        s_hat = self._ops.forward_fft(source_for_fft, "ortho")
        logger.info(f"[SOLVER] STEP 1 COMPLETE: Forward FFT completed, spectral shape: {s_hat.shape}")
        sys.stdout.flush()
        
        # CRITICAL: Check for zero-mode (DC component) when lambda=0
        # If lambda=0 and source has non-zero DC component, division by zero will occur
        if self.lmbda == 0.0:
            # Check DC component (all indices = 0)
            dc_idx = tuple([0] * len(s_hat.shape))
            dc_component = s_hat[dc_idx]
            if abs(dc_component) > 1e-12:  # Non-zero DC component
                raise ZeroDivisionError(
                    f"lambda=0 with non-zero zero-mode in source: ŝ(0)={dc_component:.6e}≠0. "
                    f"Source must have zero DC component when lambda=0 to avoid division by zero."
                )
        
        # Apply spectral operator - use lazy evaluation if needed
        logger.info(f"[SOLVER] STEP 2: Applying spectral operator...")
        sys.stdout.flush()
        if self._use_lazy_coeffs:
            a_hat = self._apply_spectral_operator_lazy(s_hat)
        else:
            if self._coeffs is None:
                logger.info(f"[SOLVER] Building spectral coefficients...")
                sys.stdout.flush()
                self._build_spectral_coefficients()
            a_hat = s_hat / self._coeffs
        
        logger.info(f"[SOLVER] STEP 2 COMPLETE: Spectral operator applied")
        sys.stdout.flush()
        
        logger.info(f"[SOLVER] STEP 3: Performing inverse FFT...")
        sys.stdout.flush()
        # CRITICAL: Do not take .real here - preserve complex solution for complex sources
        # For complex sources (e.g., plane waves exp(i*k*x)), the solution must be complex
        # to preserve phase information (critical for EM fields, wave physics, etc.)
        # Only take .real if source is explicitly real-valued
        a = self._ops.inverse_fft(a_hat, "ortho")
        # Check if source is real - if so, solution should also be real (within numerical precision)
        if np.isrealobj(source_array):
            # Source is real, so solution should be real (IFFT of Hermitian spectrum)
            # Take real part to remove numerical noise in imaginary part
            a = a.real
        # Otherwise, keep complex solution for complex sources
        logger.info(f"[SOLVER] STEP 3 COMPLETE: Inverse FFT completed, solution shape: {a.shape}, dtype: {a.dtype}")
        sys.stdout.flush()
        
        # Return as FieldArray for transparent swap support
        return FieldArray(array=a)
    
    # Backward-compatible API expected by tests
    def solve(self, source_field: np.ndarray) -> np.ndarray:
        """Backward-compatible solve method."""
        result = self.solve_stationary(source_field)
        if hasattr(result, "array"):
            return result.array
        return result

