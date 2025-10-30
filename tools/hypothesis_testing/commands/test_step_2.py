"""
Test Step 2: 7D FFT Solver for Fractional Riesz Operator.
"""

from typing import Dict, Any
import numpy as np
from .base import BaseCommand
from bhlff.core.fft.fft_solver_7d import FFTSolver7D


class TestStep2Command(BaseCommand):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.name = "7D FFT Solver - Fractional Riesz"

    def execute(self) -> Dict[str, Any]:
        self.logger.info("Testing Step 2: 7D FFT Solver...")
        try:
            domain = self.create_minimal_domain()
            print(f"Domain: N={domain.N}, N_phi={domain.N_phi}, N_t={domain.N_t}")
            # Create a simple source: delta-like at center in spatial, zeros elsewhere
            shape = (domain.N, domain.N, domain.N, domain.N_phi, domain.N_phi, domain.N_phi, domain.N_t)
            source = np.zeros(shape, dtype=np.float64)
            c = (domain.N // 2, domain.N // 2, domain.N // 2, domain.N_phi // 2, domain.N_phi // 2, domain.N_phi // 2, domain.N_t // 2)
            source[c] = 1.0

            params = {"mu": 1.0, "beta": 1.0, "lambda": 0.0, "use_cuda": True}
            print("Constructing FFTSolver7D...")
            solver = FFTSolver7D(domain, params)
            print("Solving...")
            solution = solver.solve_stationary(source)
            print("Solved.")

            # Validate: Lβ a ≈ s → check in spectral space: ŝ ≈ D â
            s_hat = np.fft.fftn(source, norm="ortho")
            a_hat = np.fft.fftn(solution, norm="ortho")
            D = solver.get_spectral_coefficients()
            residual_hat = s_hat - D * a_hat
            residual = np.linalg.norm(residual_hat.ravel()) / (np.linalg.norm(s_hat.ravel()) + 1e-12)

            success = residual < 1e-6
            print(f"Residual (spectral): {residual:.3e}")
            return {
                "step": 2,
                "name": self.name,
                "success": success,
                "details": {
                    "residual_spectral": residual,
                    "shape": solution.shape,
                    "dtype": str(solution.dtype),
                },
            }
        except Exception as e:
            self.logger.error(f"❌ Step 2 failed: {e}")
            return {"step": 2, "name": self.name, "success": False, "error": str(e)}


