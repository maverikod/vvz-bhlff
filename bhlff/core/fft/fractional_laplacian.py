"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Fractional Laplacian legacy shim for 7D BHLFF Framework.

This module is deprecated and preserved as a compatibility shim. The single
source of truth for the fractional Laplacian lives in
`bhlff/core/operators/fractional_laplacian.py` and uses the unified spectral
backend with physics-normalized transforms and CUDA-first policy.

Theoretical Background:
    The fractional Laplacian (−Δ)^β is implemented in the operators module.
    This shim delegates to that implementation and emits a DeprecationWarning
    upon instantiation to guide users to update imports.

Example:
    >>> # Prefer this:
    >>> from bhlff.core.operators.fractional_laplacian import FractionalLaplacian
    >>>
    >>> # Legacy import still works but warns:
    >>> from bhlff.core.fft.fractional_laplacian import FractionalLaplacian
"""

from typing import Any
import warnings

from bhlff.core.operators.fractional_laplacian import (
    FractionalLaplacian as _OperatorsFractionalLaplacian,
)


class FractionalLaplacian(_OperatorsFractionalLaplacian):
    """
    Deprecated legacy alias for the fractional Laplacian operator.

    Physical Meaning:
        Delegates to the canonical implementation in
        `bhlff.core.operators.fractional_laplacian.FractionalLaplacian`.

    Deprecation:
        Import from `bhlff.core.operators.fractional_laplacian` instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            (
                "bhlff.core.fft.fractional_laplacian.FractionalLaplacian is deprecated; "
                "use bhlff.core.operators.fractional_laplacian.FractionalLaplacian instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
