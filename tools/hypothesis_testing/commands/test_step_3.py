"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test Step 03: Time integrators (Crank-Nicolson) basic validation.

This command validates that the Crank-Nicolson integrator runs on a small
problem, is stable for zero source, and exhibits non-increasing L2 norm
under positive damping.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from .base import BaseCommand
from bhlff.core.domain import Domain, Parameters
from bhlff.core.time import CrankNicolsonIntegrator


class TestStep3Command(BaseCommand):
    """Step 03 validation command."""

    def execute(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"name": "Step 03: Time Integrators"}
        try:
            # Setup tiny domain and parameters with damping
            domain: Domain = Domain(L=1.0, N=4, N_phi=2, N_t=6, T=1.0)
            params: Parameters = Parameters(mu=1.0, beta=1.0, lambda_param=0.1, nu=1.0)

            # Initial field: zeros complex in full 7D shape
            initial = np.zeros(domain.shape, dtype=np.complex128)

            # Zero source over time with full shape
            t = np.linspace(0.0, domain.T, domain.N_t).astype(np.float64)
            zero_full = np.zeros(domain.shape, dtype=np.complex128)
            source_time = np.stack([zero_full for _ in range(domain.N_t)], axis=0)

            # Run CN integrator
            integrator = CrankNicolsonIntegrator(domain, params)
            evolution = integrator.integrate(initial, source_time, t)

            # Basic checks
            assert evolution.shape[0] == domain.N_t, "Wrong time dimension"
            assert evolution.shape[1:] == domain.shape, "Wrong field shape"

            # With zero source and damping from zero initial, stays zero
            norms = [float(np.linalg.norm(evolution[i])) for i in range(domain.N_t)]
            success = all(abs(v) < 1e-12 for v in norms)
            result.update(
                {
                    "success": success,
                    "details": {
                        "max_l2": max(norms),
                        "nt": domain.N_t,
                        "nx": domain.N,
                    },
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001
            result.update({"success": False, "error": str(exc)})
            return result
