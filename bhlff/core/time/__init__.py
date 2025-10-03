"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Time integration module for 7D BVP framework.

This module provides temporal integrators for solving dynamic phase field
equations in 7D space-time, including support for memory kernels and
quench detection.

Physical Meaning:
    Temporal integrators solve the dynamic phase field equation:
    ∂a/∂t + ν(-Δ)^β a + λa = s(x,φ,t)
    where the phase field evolves in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Implements various time integration schemes for the spectral form:
    ∂â/∂t + (ν|k|^(2β) + λ)â = ŝ(k,t)
    with support for memory kernels and quench detection.

Example:
    >>> integrator = BVPExponentialIntegrator(domain, parameters)
    >>> solution = integrator.integrate(source_field, time_steps)
"""

from .base_integrator import BaseTimeIntegrator
from .bvp_envelope_integrator import BVPEnvelopeIntegrator
from .crank_nicolson_integrator import CrankNicolsonIntegrator
from .adaptive_integrator import AdaptiveIntegrator
from .memory_kernel import MemoryKernel
from .quench_detector import QuenchDetector

__all__ = [
    "BaseTimeIntegrator",
    "BVPEnvelopeIntegrator",
    "CrankNicolsonIntegrator",
    "AdaptiveIntegrator",
    "MemoryKernel",
    "QuenchDetector",
]
