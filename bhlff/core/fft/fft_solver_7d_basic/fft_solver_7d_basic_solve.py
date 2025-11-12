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
        
        logger.info(f"[SOLVER] solve_stationary: STEP 0.1: Extracting array from FieldArray if needed...")
        sys.stdout.flush()
        sys.stderr.flush()
        # Extract array from FieldArray if needed
        if isinstance(source_field, FieldArray):
            source_array = source_field.array
            logger.info(f"[SOLVER] solve_stationary: STEP 0.1 COMPLETE: Extracted from FieldArray")
        else:
            source_array = source_field
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
            f"({source_size_mb:.2f}MB), use_cuda={self.use_cuda}"
        )
        sys.stdout.flush()
        
        logger.info(f"[SOLVER] STEP 1: Performing forward FFT...")
        sys.stdout.flush()
        s_hat = self._ops.forward_fft(np.asarray(source_array, dtype=np.complex128), "ortho")
        logger.info(f"[SOLVER] STEP 1 COMPLETE: Forward FFT completed, spectral shape: {s_hat.shape}")
        sys.stdout.flush()
        
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
        a = self._ops.inverse_fft(a_hat, "ortho").real
        logger.info(f"[SOLVER] STEP 3 COMPLETE: Inverse FFT completed, solution shape: {a.shape}")
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

